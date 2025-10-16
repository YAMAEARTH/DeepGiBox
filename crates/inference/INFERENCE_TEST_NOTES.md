# Inference Test Notes

## สรุปผลการทดสอบ

### ✅ ที่ทำสำเร็จ:

1. **สร้าง test program** (`test_inference_pipeline.rs`)
   - ✅ รับ RawFramePacket จาก DeckLink
   - ✅ ส่งไป Preprocessing (FP16)
   - ✅ ส่งต่อไป Inference Engine
   - ✅ แสดงผล RawDetectionsPacket

2. **Inference Engine Updates**
   - ✅ เพิ่ม support FP16 input
   - ✅ แปลง FP16 → FP32 อัตโนมัติ
   - ✅ ใช้ default allocators แทน custom CUDA allocators

3. **Pipeline Integration**
   - ✅ Capture → Preprocessing ทำงานได้
   - ✅ Preprocessing สำเร็จ (3.59 ms GPU Direct)
   - ✅ Tensor พร้อมสำหรับ inference

### ⚠️ ปัญหาที่พบ:

1. **Segmentation Fault**
   - เกิดระหว่างการแปลง FP16 → FP32
   - สาเหตุ: อาจพยายาม access GPU memory pointer โดยตรง
   - FP16 data อยู่บน GPU memory แล้ว ไม่สามารถ dereference โดยตรงจาก CPU

### 🔧 ทางแก้ที่เป็นไปได้:

#### Option 1: ใช้ CUDA kernel แปลง FP16 → FP32 บน GPU
```rust
// ใช้ CUDA kernel แทนการแปลงบน CPU
cudarc::driver::launch_kernel(convert_fp16_to_fp32_kernel, ...);
```

#### Option 2: Copy GPU → CPU ก่อนแปลง
```rust
// Copy FP16 from GPU to CPU
device.dtoh_sync_copy_into(gpu_fp16_slice, &mut cpu_fp16_buffer)?;

// แปลง FP16 → FP32 บน CPU
for (i, &fp16_val) in cpu_fp16_buffer.iter().enumerate() {
    cpu_fp32_buffer[i] = fp16_val.to_f32();
}

// Copy FP32 back to GPU
device.htod_sync_copy_into(&cpu_fp32_buffer, &mut gpu_fp32_buffer)?;
```

#### Option 3: ให้ ORT รับ FP16 โดยตรง (แนะนำ!)
```rust
// สร้าง FP16 tensor แทน FP32
let input_tensor = unsafe {
    Tensor::<half::f16>::new(&self.gpu_allocator, shape)?
};

// ใช้ IO binding กับ FP16 tensor
io_binding.bind_input("images", &input_tensor)?;
```

### 📝 การแก้ไขที่แนะนำ:

**ใช้ Option 3** เพราะ:
1. Model อาจรับ FP16 ได้โดยตรง (TensorRT FP16 mode)
2. ไม่ต้องแปลง → ประหยัด latency
3. ใช้ memory น้อยกว่า (FP16 = 2 bytes, FP32 = 4 bytes)
4. GPU native FP16 operations เร็วกว่า

### 🎯 Next Steps:

1. แก้ inference engine ให้รับ FP16 โดยตรง
2. ตรวจสอบ model input type (ควรเป็น FP16)
3. ทดสอบ inference ใหม่
4. วัด latency ของทั้ง pipeline

### 📊 Performance จนถึงตอนนี้:

```
Capture:       GPU Direct (zero-copy)
Preprocessing: 3.59 ms ⚡
Inference:     (pending fix)
E2E Target:    < 10 ms (100+ FPS)
```

## การใช้งาน:

```bash
# รัน test
cargo run -p playgrounds --bin test_inference_pipeline

# ต้องมี:
# 1. DeckLink device ที่กำลังส่งสัญญาณ
# 2. Model file: crates/inference/YOLOv5.onnx
# 3. GPU พร้อม TensorRT
```

## หมายเหตุสำคัญ:

- ⚠️ **อย่า** dereference GPU pointers โดยตรงจาก CPU code
- ⚠️ การแปลง FP16 ↔ FP32 บน GPU data ต้องใช้ CUDA kernel หรือ copy มา CPU ก่อน
- ✅ ใช้ FP16 native ตลอด pipeline จะให้ performance ดีที่สุด

---

**สถานะ**: กำลังแก้ไข (FP16 conversion issue)  
**วันที่**: October 14, 2025
