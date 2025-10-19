# Latency Measurement Explained - test_tensorinputpacket

## Overview

เอกสารนี้อธิบายรายละเอียดของการวัด latency แต่ละประเภทใน `test_tensorinputpacket` ว่าวัดจากอะไรบ้าง และครอบคลุมส่วนไหนของ pipeline

---

## Timeline ของการประมวลผล 1 Frame

```
┌─────────────────────────────────────────────────────────────────────┐
│                     E2E Process Frame Timeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  [t_e2e_start]                                                       │
│       │                                                              │
│       ├─ Print Frame Info                                           │
│       │                                                              │
│       ├─ Check Memory Location (CPU or GPU?)                        │
│       │                                                              │
│       ├─ IF CPU Memory:                                             │
│       │    [t_transfer_start]                                       │
│       │         │                                                    │
│       │         ├─ Allocate GPU buffer                              │
│       │         ├─ Copy CPU → GPU (htod_sync_copy_into)            │
│       │         └─ [transfer_ms calculated]                         │
│       │                                                              │
│       ├─ [t_preprocess_start]                                       │
│       │      │                                                       │
│       │      ├─ preprocessor.process(raw_packet)                    │
│       │      │    ├─ CUDA kernel: YUV422 → RGB conversion          │
│       │      │    ├─ CUDA kernel: Bilinear resize (1920x1080→512x512)│
│       │      │    ├─ CUDA kernel: Normalization (ImageNet)          │
│       │      │    └─ Return TensorInputPacket                       │
│       │      │                                                       │
│       │      └─ [preprocess_ms calculated]                          │
│       │                                                              │
│       └─ [e2e_ms calculated]                                        │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. CPU→GPU Transfer Latency

### วัดจาก:
```rust
let t_transfer_start = now_ns();

// Allocate GPU buffer
let mut gpu_buffer = device.alloc_zeros::<u8>(raw_packet.data.len)?;

// Copy CPU → GPU
let cpu_slice = unsafe {
    std::slice::from_raw_parts(raw_packet.data.ptr, raw_packet.data.len)
};
device.htod_sync_copy_into(cpu_slice, &mut gpu_buffer)?;

let transfer_ms = since_ms(t_transfer_start);
```

### ครอบคลุม:
- ✅ การจองพื้นที่ GPU memory (allocation)
- ✅ การคัดลอกข้อมูลจาก CPU → GPU (host to device)
- ✅ CUDA synchronization overhead

### เกิดขึ้นเมื่อไหร่:
- เฉพาะเมื่อ `raw_packet.data.loc == MemLoc::Cpu`
- มักเกิดกับ frame แรกๆ (frame 0-2) ในช่วง format detection
- Frame ที่มาจาก 720x486 (pre-detection format)

### ผลลัพธ์:
```
Frame #0: 25.866 ms (first allocation + transfer)
Frame #1:  3.978 ms (subsequent transfer, buffer reused)
```

### ทำไมถึงวัด:
- เพื่อเปรียบเทียบกับ GPU Direct (zero-copy)
- แสดงให้เห็น overhead ของการ copy memory
- ช่วย optimize buffer pooling strategies

---

## 2. Preprocessing Latency (All Frames)

### วัดจาก:
```rust
let t_preprocess_start = now_ns();
let tensor = preprocessor.process(raw_packet);
let preprocess_ms = since_ms(t_preprocess_start);
```

### ครอบคลุม:
1. **CUDA Kernel: YUV422 → RGB Conversion**
   - แปลง UYVY → RGB planar
   - ใช้ BT.709 color space matrix

2. **CUDA Kernel: Bilinear Resize**
   - Downscale จาก 1920x1080 → 512x512
   - หรือจาก 720x486 → 512x512

3. **CUDA Kernel: Normalization**
   - ImageNet normalization per-channel
   - mean = [0.485, 0.456, 0.406]
   - std = [0.229, 0.224, 0.225]

4. **Memory Operations**
   - อ่านข้อมูลจาก GPU memory
   - เขียนผลลัพธ์ไปยัง output tensor
   - CUDA stream synchronization

### ไม่ครอบคลุม:
- ❌ CPU→GPU transfer (วัดแยกต่างหาก)
- ❌ Memory allocation overhead
- ❌ Packet metadata copying

### ผลลัพธ์:
```
All Frames (10):
  Min:  0.137 ms (GPU Direct)
  Max:  1.628 ms (with CPU transfer overhead)
  Avg:  0.298 ms
```

### ทำไมถึงวัด:
- วัดประสิทธิภาพของ CUDA kernels
- เปรียบเทียบกับ CPU preprocessing
- Baseline สำหรับ optimization

---

## 3. GPU Direct Preprocessing Latency

### วัดจาก:
```rust
let is_gpu_direct = matches!(raw_packet.data.loc, MemLoc::Gpu { .. });

let t_preprocess_start = now_ns();
let tensor = preprocessor.process(raw_packet);
let preprocess_ms = since_ms(t_preprocess_start);

// Record GPU Direct timing separately
if is_gpu_direct {
    record_ms("gpu_direct_preprocessing", t_preprocess_start);
    latency_stats.add_preprocess(preprocess_ms, is_gpu_direct);
}
```

### ครอบคลุม:
- เหมือนกับ "Preprocessing Latency" ทุกประการ
- **แต่เฉพาะเมื่อ input memory อยู่บน GPU แล้ว** (zero-copy)

### คัดกรองโดย:
- `raw_packet.data.loc == MemLoc::Gpu { device: 0 }`
- Frame ที่มาจาก DeckLink GPUDirect RDMA
- Frame ที่เป็น 1920x1080 @60fps (stable format)

### ผลลัพธ์:
```
GPU Direct Frames (8):
  Min:  0.137 ms
  Max:  0.146 ms
  Avg:  0.141 ms ⚡ (Zero-copy!)
  Throughput: 7,115 FPS
```

### ทำไมถึงวัด:
- แสดงประสิทธิภาพสูงสุดของระบบ (optimal case)
- พิสูจน์ว่า GPU Direct ทำงานได้จริง
- เปรียบเทียบกับ CPU transfer path
- คำนวณ theoretical maximum throughput

### ความแตกต่างจาก All Frames:
| Metric | All Frames | GPU Direct Only |
|--------|-----------|-----------------|
| Frames | 10 (CPU + GPU) | 8 (GPU only) |
| Min | 0.137 ms | 0.137 ms |
| Max | 1.628 ms | 0.146 ms |
| Avg | 0.298 ms | **0.141 ms** |
| Consistency | ❌ Varies | ✅ Stable |

---

## 4. End-to-End (E2E) Latency

### วัดจาก:
```rust
fn process_and_display(...) -> Result<(), Box<dyn Error>> {
    // Start E2E timing
    let t_e2e_start = now_ns();
    
    // ... print info ...
    // ... check memory location ...
    // ... transfer if needed ...
    // ... preprocess ...
    
    // Calculate E2E latency
    let e2e_ms = since_ms(t_e2e_start);
    record_ms("e2e_process_frame", t_e2e_start);
}
```

### ครอบคลุม:
1. **Printing Frame Info**
   - `print_rawframepacket()`
   - I/O to terminal

2. **Memory Location Check**
   - Pattern matching: `MemLoc::Cpu` vs `MemLoc::Gpu`

3. **CPU→GPU Transfer (if needed)**
   - GPU buffer allocation
   - Memory copy
   - Pointer updates

4. **Preprocessing**
   - ทุกขั้นตอนใน preprocessing (YUV→RGB, resize, normalize)

5. **Metadata Operations**
   - TensorInputPacket creation
   - Metadata preservation

### ไม่ครอบคลุม:
- ❌ Tensor info display (หลังจาก return)
- ❌ Verification printing
- ❌ Statistics collection

### ผลลัพธ์:
```
E2E (10 frames):
  Min:  0.172 ms (GPU Direct, minimal overhead)
  Max: 27.546 ms (first frame with allocation)
  Avg:  3.321 ms
  Target FPS: 301.1 fps
```

### ทำไมถึงวัด:
- วัด real-world processing time
- รวม overhead ทั้งหมดที่เกิดขึ้นจริง
- คำนวณ actual throughput ที่ได้
- ใช้วางแผน pipeline capacity

---

## Comparison Table

| Latency Type | Start Point | End Point | Includes Transfer | Includes Print | Only GPU Direct |
|--------------|-------------|-----------|-------------------|----------------|-----------------|
| **CPU→GPU Transfer** | `t_transfer_start` | After `htod_sync_copy` | ✅ Yes (only this) | ❌ No | ❌ No |
| **Preprocessing** | `t_preprocess_start` | After `preprocessor.process()` | ❌ No | ❌ No | ⚠️ Mixed |
| **GPU Direct Preprocessing** | `t_preprocess_start` | After `preprocessor.process()` | ❌ No | ❌ No | ✅ Yes |
| **E2E Process Frame** | `t_e2e_start` | After `preprocessor.process()` | ✅ If needed | ✅ Yes | ⚠️ Mixed |

---

## Example Timeline Breakdown

### Frame #0 (CPU Memory, 720x486):
```
E2E Total: 27.546 ms
├─ Print info:        ~0.05 ms
├─ Check memory:      ~0.001 ms
├─ CPU→GPU transfer: 25.866 ms ◄─ bottleneck
├─ Preprocessing:     1.628 ms
└─ Overhead:          ~0.01 ms
```

### Frame #2 (GPU Direct, 1920x1080):
```
E2E Total: 0.186 ms
├─ Print info:        ~0.03 ms
├─ Check memory:      ~0.001 ms
├─ CPU→GPU transfer:  0.000 ms (skip!)
├─ Preprocessing:     0.146 ms ◄─ pure GPU kernel
└─ Overhead:          ~0.009 ms
```

---

## Statistics Summary Formula

### Average Calculation:
```rust
// Preprocessing - All Frames
preprocess_avg = sum(all 10 frames) / 10
             = (1.628 + 0.223 + 0.146 + ... + 0.139) / 10
             = 0.298 ms

// GPU Direct Only
gpu_direct_avg = sum(frames 2-9) / 8
              = (0.146 + 0.142 + 0.137 + ... + 0.139) / 8
              = 0.141 ms ⚡

// E2E
e2e_avg = sum(all 10 frames) / 10
        = (27.546 + 4.242 + 0.186 + ... + 0.175) / 10
        = 3.321 ms
```

### Throughput Calculation:
```rust
// GPU Direct Throughput (preprocessing only)
fps = 1000 ms / 0.141 ms
    = 7,115 FPS

// E2E Throughput (complete pipeline)
fps = 1000 ms / 3.321 ms
    = 301.1 FPS
```

---

## Key Insights

### 1. GPU Direct เร็วกว่า CPU Transfer มาก
- **GPU Direct**: 0.141 ms (pure kernel execution)
- **With Transfer**: 1.628 ms (11.5x slower)
- **Speedup**: 11.5x ด้วย zero-copy

### 2. E2E Overhead ต่ำมาก
- **E2E**: 0.186 ms (GPU Direct)
- **Preprocessing**: 0.146 ms
- **Overhead**: 0.040 ms (21% of total)
  - Printing: ~30 ms
  - Memory check: ~1 ms
  - Metadata ops: ~9 ms

### 3. First Frame Penalty
- **Frame #0**: 27.546 ms (allocation + transfer)
- **Frame #2**: 0.186 ms (optimized path)
- **Ratio**: 148x slower (first frame)

### 4. Consistency
- **GPU Direct Frames**: 0.137-0.146 ms (±3% variation)
- **With Transfer**: 1.628-25.866 ms (±1500% variation)
- **Conclusion**: GPU Direct แน่นอนและคาดการณ์ได้

---

## Telemetry Logs Explained

```bash
[lat] cpu_to_gpu_transfer=25.87ms    # วัดจาก: allocation + htod_copy
[lat] preprocessing=1.63ms           # วัดจาก: preprocessor.process()
[lat] gpu_direct_preprocessing=0.14ms # วัดจาก: preprocessing (GPU frames only)
[lat] e2e_process_frame=27.55ms      # วัดจาก: function start → preprocessing done
```

---

## Use Cases

### 1. Performance Optimization
- ใช้ **GPU Direct Preprocessing** เป็น baseline
- ถ้า preprocessing ช้ากว่า 0.2 ms → investigate CUDA kernels
- ถ้า E2E ช้ากว่า 0.3 ms → investigate overhead

### 2. Capacity Planning
- ใช้ **E2E avg** คำนวณ maximum FPS
- ปัจจุบัน: 301 FPS (3.3 ms/frame)
- เพิ่ม buffer: 60 FPS → ใช้ 19.8% capacity

### 3. Debugging
- **Preprocessing สูง** → CUDA kernel issue
- **E2E สูงแต่ Preprocessing ต่ำ** → transfer หรือ overhead issue
- **GPU Direct = CPU path** → GPUDirect ไม่ทำงาน

### 4. Profiling
- เปรียบเทียบ latency ก่อน/หลัง optimization
- Track regression across builds
- Benchmark different GPU architectures

---

## Summary Table

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **CPU→GPU Transfer** | Memory copy overhead | Shows benefit of GPU Direct |
| **Preprocessing (All)** | Average kernel time | Overall performance baseline |
| **GPU Direct Preprocessing** | Optimal kernel time | Best-case performance |
| **E2E Process Frame** | Real-world latency | Actual throughput capacity |

---

**Last Updated**: October 13, 2025  
**Measurement Tool**: `test_tensorinputpacket`  
**Telemetry Backend**: `telemetry` crate (human-log)  
**Test Configuration**: 10 frames, 1080p60, FP16, ImageNet normalization
