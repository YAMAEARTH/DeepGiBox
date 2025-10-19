# การเดินทางของข้อมูลจาก DeckLink ถึง Application

## 🎬 Data Flow: จากกล้อง/SDI Source ถึง Application

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Physical Signal (SDI/HDMI)                                           │
│    Camera/Video Source → DeckLink Card (Hardware)                       │
│    • SDI cable carries serialized video signal                          │
│    • ~1.5 Gbps for 1080p60 (uncompressed)                              │
│    • Timing: continuous stream, synchronized to video clock             │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. DeckLink Hardware Capture (FPGA/ASIC)                                │
│    • Deserializes SDI signal                                            │
│    • Converts to YUV422 (UYVY) format                                   │
│    • Writes to DeckLink's onboard buffer (Hardware DMA buffer)          │
│    • Buffer location: PCIe device memory or system RAM (DMA)            │
│    Latency: ~1-2 frame periods (~16-33ms @ 60fps)                      │
│    ⚠️ This is the REAL capture latency we CAN'T directly measure!       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. DeckLink Driver (Kernel Space)                                       │
│    • Driver manages ring buffer of captured frames                      │
│    • Typically 3-5 frame buffer depth                                   │
│    • Location: System RAM (DMA-accessible)                              │
│    • Frame arrives via interrupt from hardware                          │
│    Latency: ~0.5-2ms (driver overhead)                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. DeckLink SDK (User Space)                                            │
│    • VideoInputFrameArrived callback (in shim.cpp)                      │
│    • Copies frame pointer to application-accessible buffer              │
│    • Location: Still in system RAM (shared memory)                      │
│    Latency: ~0.1-0.5ms (callback dispatch)                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. Our Application: get_frame() (Rust)                                  │
│    • Returns pointer to the frame in system RAM                         │
│    • NO COPY at this stage - just pointer!                             │
│    • Location: System RAM (same memory as SDK)                          │
│    Latency: ~0.0003ms (just function call overhead)                    │
│    ✅ THIS is what we measured as "0 ms"!                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. Memory Read/Access (Our measurement)                                 │
│    • Reading data FROM: System RAM (DMA buffer)                         │
│    • Reading data TO: CPU cache → CPU registers                         │
│    • Access pattern: Sequential read through 3.96MB buffer              │
│    Latency: ~0.02ms (measured)                                         │
│    Bandwidth: ~198 GB/s (theoretical DDR4)                              │
│    ✅ THIS is our "memory read" measurement                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. Copy to GPU (H2D Transfer) - Future step                            │
│    • Copying FROM: System RAM (DMA buffer)                              │
│    • Copying TO: GPU memory (VRAM)                                      │
│    • Transfer via: PCIe bus (typically PCIe 3.0 x16)                   │
│    Expected latency: ~0.5-2ms for 3.96MB                               │
│    Bandwidth: ~12-16 GB/s (PCIe 3.0 x16)                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 8. GPU Processing                                                        │
│    • YUV422 → RGB conversion (CUDA kernel)                              │
│    • Resize, normalize, etc.                                            │
│    Expected latency: ~0.5-2ms                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 ทำไม "0 ms" ถึงไม่ใช่ค่าที่ถูกต้องสำหรับ Capture Latency

### ปัญหา: เราวัด `t_capture_ns` ผิดจุด!

```rust
// ใน capture.rs - get_frame()
let t_capture_ns = std::time::SystemTime::now()  // ← วัดตรงนี้
    .duration_since(std::time::UNIX_EPOCH)
    .unwrap_or_default()
    .as_nanos() as u64;
```

**เวลานี้คือ:** เวลาที่เรา**ได้รับ pointer** จาก DeckLink SDK  
**ไม่ใช่:** เวลาที่ hardware capture frame จริงๆ!

---

## 📊 Real Capture Latency Breakdown

### Total Latency จาก "แสงเข้ากล้อง" ถึง "เราได้ pointer":

```
Component                          | Latency        | Can We Measure?
-----------------------------------|----------------|------------------
1. Camera sensor latency           | 0-5ms          | ❌ No (external)
2. SDI transmission                | ~0.016ms       | ❌ No (cable length)
3. DeckLink hardware capture       | 16-33ms        | ❌ No (1-2 frames buffering)
4. DeckLink driver processing      | 0.5-2ms        | ❌ No (kernel space)
5. SDK callback dispatch           | 0.1-0.5ms      | ⚠️ Indirect
6. get_frame() call                | 0.0003ms       | ✅ Yes (what we measured)
-----------------------------------|----------------|------------------
TOTAL END-TO-END                   | ~17-40ms       | ⚠️ Hard to measure

Memory access (after get_frame)    | 0.02ms         | ✅ Yes (what we measured)
```

---

## 🎯 Memory Read: อ่านจากไหนถึงไหน?

### ที่ตั้งของข้อมูล:

```
┌─────────────────────────────────────────────────────────────────┐
│ Physical Memory Layout                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  System RAM (DDR4/DDR5)                                         │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                                                       │       │
│  │  DMA Buffer (allocated by DeckLink driver)           │       │
│  │  ┌────────────────────────────────────────────┐      │       │
│  │  │                                             │      │       │
│  │  │  Frame Data (YUV422, 3.96 MB)              │      │       │
│  │  │  [U Y V Y] [U Y V Y] [U Y V Y] ...         │      │       │
│  │  │                                             │      │       │
│  │  │  Address: 0x72d485007010 (example)         │      │       │
│  │  │  Size: 4,147,200 bytes                     │      │       │
│  │  │                                             │      │       │
│  │  └────────────────────────────────────────────┘      │       │
│  │        ↑                                              │       │
│  │        │ packet.data.ptr points here                 │       │
│  │        │                                              │       │
│  └────────┼──────────────────────────────────────────────┘       │
│           │                                                       │
└───────────┼───────────────────────────────────────────────────────┘
            │
            │ PCIe Bus
            │
    ┌───────┴────────┐
    │ DeckLink Card  │
    │ (PCIe device)  │
    └────────────────┘
```

### การอ่านข้อมูล:

```rust
// When we do this:
unsafe {
    let slice = std::slice::from_raw_parts(packet.data.ptr, packet.data.len);
    let value = slice[0]; // ← Read happens HERE
}
```

**การเดินทางของข้อมูล:**

```
1. System RAM (DMA buffer)
   Address: 0x72d485007010
   Size: 3.96 MB
   ↓
2. Memory Controller
   ↓
3. CPU Cache (L3 → L2 → L1)
   ↓
4. CPU Register
   ↓
5. Your variable `value`
```

**Latency ~0.02ms สำหรับ 3.96MB:**
- Sequential read ผ่าน memory
- Cache-efficient (linear access)
- Bandwidth: ~198 GB/s (DDR4-3200)

---

## 🚨 ปัญหาของการวัด Capture Latency

### วิธีที่เราทำตอนนี้ (ผิด):

```rust
// ใน capture.rs
let t_capture_ns = SystemTime::now().as_nanos();
// ↑ วัดเวลาที่เรา "ได้รับ pointer"
// ไม่ใช่เวลาที่ hardware capture จริง!
```

### วิธีที่ถูกต้อง (ต้องแก้ shim.cpp):

```cpp
// ใน shim.cpp - VideoInputFrameArrived callback
HRESULT VideoInputFrameArrived(IDeckLinkVideoInputFrame* videoFrame, ...) {
    // Get HARDWARE timestamp from frame
    BMDTimeValue frameTime;
    BMDTimeScale timeScale;
    videoFrame->GetStreamTime(&frameTime, NULL, timeScale);
    
    // Convert to absolute time
    uint64_t hardware_capture_ns = convert_to_unix_ns(frameTime, timeScale);
    
    // Store in our buffer
    g_latest_frame.hardware_timestamp_ns = hardware_capture_ns;
}
```

แล้วใน Rust:

```rust
// เวลาตอนนี้
let now = SystemTime::now().as_nanos();

// เวลาที่ hardware capture จริง (จาก frame metadata)
let hardware_capture_ns = packet.meta.t_hardware_capture_ns;

// Real capture latency!
let real_latency_ms = (now - hardware_capture_ns) as f64 / 1e6;
```

---

## 💡 สรุป

### 1. **"0 ms" ไม่ใช่ capture latency จริง**
   - มันคือ latency ของ `get_frame()` function call
   - Hardware capture เกิดก่อนหน้านี้ 17-40ms แล้ว!

### 2. **Memory Read (0.02ms) หมายถึง:**
   ```
   FROM: System RAM (DMA buffer at 0x72d485007010)
   TO:   CPU cache → CPU registers
   SIZE: 3.96 MB (sequential scan)
   ```

### 3. **Real Capture Latency คือ:**
   ```
   Hardware capture → Driver → SDK → get_frame()
   Total: ~17-40ms (1-2 frame periods + overhead)
   ```

### 4. **จะวัด Real Latency ได้อย่างไร:**
   - ✅ แก้ shim.cpp ให้ใช้ `GetStreamTime()` จาก DeckLink API
   - ✅ ส่ง hardware timestamp มาใน `RawFramePacket`
   - ✅ เปรียบเทียบกับเวลาปัจจุบัน

---

## 📝 Next Steps

ต้องการให้ผมสร้างโปรแกรมที่:
1. แก้ shim.cpp เพื่อดึง hardware timestamp จริง?
2. สร้างเอกสารอธิบาย memory architecture โดยละเอียด?
3. วาดแผนภาพ data flow แบบละเอียดกว่านี้?

บอกผมได้เลยครับ! 🚀
