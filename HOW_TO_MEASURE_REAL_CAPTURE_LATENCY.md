# วิธีวัด Real Capture Latency

## ปัญหาที่พบ

การเรียก `get_frame()` ได้ ~0.0003 ms (0.3 microseconds) ซึ่ง**ไม่ใช่** latency จริงของการ capture เพราะ:
- มันแค่อ่าน pointer ที่ DeckLink driver เตรียมไว้แล้ว
- Hardware capture เกิดขึ้นก่อนหน้านี้แล้วใน background

## วิธีวัด Real Capture Latency

### 1. **ใช้ Hardware Timestamp จาก DeckLink API** ⭐ (แนะนำมากที่สุด)

DeckLink API มี methods สำหรับดึง hardware timestamp:
- `GetHardwareTime()` - เวลาจาก hardware clock
- `GetStreamTime()` - เวลาของ frame ใน stream
- `GetHardwareReferenceTimestamp()` - reference time

**วิธีทำ:**
```cpp
// ใน shim.cpp - VideoInputFrameArrived callback
HRESULT VideoInputFrameArrived(IDeckLinkVideoInputFrame* videoFrame, ...) {
    if (!videoFrame) return S_OK;
    
    // Get hardware timestamp (when frame was actually captured)
    BMDTimeValue frameTime, frameDuration;
    BMDTimeScale timeScale;
    videoFrame->GetStreamTime(&frameTime, &frameDuration, timeScale);
    
    // Convert to nanoseconds
    uint64_t frame_capture_ns = (frameTime * 1000000000ULL) / timeScale;
    
    // Get current time when we process it
    uint64_t now_ns = ...; // current time
    
    // Real latency = now - hardware_capture_time
    uint64_t latency_ns = now_ns - frame_capture_ns;
}
```

**ข้อดี:**
- ✅ วัดจาก hardware จริง
- ✅ แม่นยำที่สุด
- ✅ รวม buffering latency ของ DeckLink

**ข้อเสีย:**
- ❌ ต้องแก้ไข shim.cpp
- ❌ ต้อง rebuild

---

### 2. **ใช้ `t_capture_ns` ที่มีอยู่แล้ว** ⭐ (ใช้งานง่าย)

ใน `RawFramePacket.meta.t_capture_ns` มีค่าเวลาที่เราได้ frame มาแล้ว

**วิธีทำ:**
```rust
let packet = session.get_frame()?.unwrap();

// เวลาที่ได้ frame จาก DeckLink (บันทึกไว้ใน capture.rs)
let t_capture_ns = packet.meta.t_capture_ns;

// เวลาตอนนี้
let now_ns = std::time::SystemTime::now()
    .duration_since(std::time::UNIX_EPOCH)
    .unwrap()
    .as_nanos() as u64;

// Latency จากเวลาที่ได้ frame ถึงตอนนี้
let latency_ms = (now_ns - t_capture_ns) as f64 / 1_000_000.0;
```

**ข้อดี:**
- ✅ ใช้งานง่าย ไม่ต้องแก้ไข C++ code
- ✅ วัดได้ทันที

**ข้อเสีย:**
- ⚠️ วัดจากเวลาที่ software ได้รับ frame (ไม่ใช่ hardware capture time จริง)
- ⚠️ มี overhead ของ driver และ buffer

---

### 3. **วัด End-to-End Processing Latency** (Practical approach)

วัดเวลาตั้งแต่ get frame จนถึงจบกระบวนการทั้งหมด

**วิธีทำ:**
```rust
// เริ่มต้น
let t_start = telemetry::now_ns();

// 1. Get frame
let t_get = telemetry::now_ns();
let packet = session.get_frame()?.unwrap();
telemetry::record_ms("capture.get_frame", t_get);

// 2. Copy to staging buffer (H2H)
let t_h2h = telemetry::now_ns();
let mut staging = vec![0u8; packet.data.len];
unsafe {
    std::ptr::copy_nonoverlapping(
        packet.data.ptr,
        staging.as_mut_ptr(),
        packet.data.len
    );
}
telemetry::record_ms("capture.h2h_copy", t_h2h);

// 3. Upload to GPU (H2D)
let t_h2d = telemetry::now_ns();
// cudaMemcpy(...);
telemetry::record_ms("capture.h2d_copy", t_h2d);

// Total
telemetry::record_ms("capture.total", t_start);
```

**ข้อดี:**
- ✅ วัด latency ที่เป็นจริงของ pipeline
- ✅ เห็นภาพรวมทั้ง capture + processing
- ✅ ระบุ bottleneck ได้ชัดเจน

**ข้อเสีย:**
- ⚠️ ไม่ใช่ pure capture latency
- ⚠️ รวม processing overhead

---

### 4. **วัดด้วย External Sync Source** (Most accurate)

ใช้ external timestamp เช่น:
- Genlock signal
- Timecode
- Hardware trigger

**วิธีทำ:**
```
1. ส่ง test pattern พร้อม timecode จาก SDI generator
2. บันทึก timestamp ที่ส่ง
3. อ่าน timecode จาก captured frame
4. คำนวณ latency = read_time - timecode_time
```

**ข้อดี:**
- ✅ แม่นยำที่สุด (ground truth)
- ✅ วัด end-to-end latency จริง

**ข้อเสีย:**
- ❌ ต้องมี hardware เพิ่ม (SDI generator, timecode)
- ❌ Setup ซับซ้อน

---

## สรุปและแนะนำ

### สำหรับ Development (ตอนนี้)
**ใช้วิธีที่ 2 + 3**: 
1. ใช้ `t_capture_ns` เพื่อดู latency จากเวลาที่ได้ frame
2. วัด processing latency แต่ละ stage (H2H, H2D, kernels)

```rust
// ตัวอย่างโค้ดที่ใช้งานได้เลย
let packet = session.get_frame()?.unwrap();

// Latency จาก DeckLink buffer
let now = SystemTime::now()
    .duration_since(UNIX_EPOCH).unwrap()
    .as_nanos() as u64;
let buffer_latency = (now - packet.meta.t_capture_ns) as f64 / 1e6;
println!("Buffer latency: {:.3} ms", buffer_latency);

// วัด processing stages
let t_start = now_ns();

// ... copy data ...
record_ms("copy", t_copy);

// ... preprocess ...
record_ms("preprocess", t_pre);

// ... inference ...
record_ms("inference", t_inf);

record_ms("pipeline.total", t_start);
```

### สำหรับ Production (ในอนาคต)
**ใช้วิธีที่ 1**: แก้ไข shim.cpp เพื่อใช้ `GetStreamTime()` และส่ง hardware timestamp มาใน `RawFramePacket`

```rust
// ใน FrameMeta เพิ่ม field
pub struct FrameMeta {
    // ... existing fields ...
    pub t_hardware_ns: u64,  // Hardware capture time
    pub t_receive_ns: u64,   // Software receive time
}
```

---

## Typical Latency Budget (60 FPS = 16.67ms)

```
Hardware capture:        ~0.016 ms  (1 frame @ 60 Hz)
DeckLink buffer:         ~0.1-1 ms
get_frame():             ~0.0003 ms
H2H copy (4MB):          ~0.5-2 ms
H2D transfer:            ~0.5-1 ms
Preprocessing:           ~2-5 ms
Inference:               ~5-15 ms
Postprocessing:          ~0.5-2 ms
Overlay:                 ~1-3 ms
--------------------------------
Total:                   ~10-30 ms
```

## การ Interpret ผลลัพธ์

1. **< 0.1 ms**: แสดงว่าวัดแค่ overhead ของ function call
2. **0.1-1 ms**: Buffer latency จาก DeckLink driver
3. **1-5 ms**: Memory copy/transfer latency
4. **5-20 ms**: Processing latency (GPU kernels, inference)
5. **> 20 ms**: ใหญ่เกิน frame budget (60 FPS)

---

## Next Steps

1. ✅ วัด `t_capture_ns` latency (ทำได้เลย)
2. ✅ วัด per-stage processing (H2H, H2D, kernels)
3. ⏳ แก้ shim.cpp เพื่อใช้ hardware timestamp
4. ⏳ Optimize bottleneck stages
