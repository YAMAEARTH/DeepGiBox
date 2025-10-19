# Internal Keying Implementation Summary

## สรุปการพัฒนา Internal Keying Module สำหรับ Blackmagic DeckLink

### ภาพรวม
ได้สร้าง module ใหม่ใน `decklink_output` crate สำหรับทำ internal keying ซึ่งเป็นการรวมภาพ (composite) ระหว่าง:
- **Fill Frame**: ภาพหลัก (background) ในรูปแบบ YUV8bit จากการ capture
- **Key Frame**: ภาพ overlay ในรูปแบบ BGRA8 ที่มี alpha channel

### ไฟล์ที่สร้างขึ้น

#### 1. `src/keying.rs` (Core Module)
- **InternalKeyingOutput**: Struct หลักสำหรับจัดการ DeckLink output
- ฟังก์ชันสำคัญ:
  - `new()`: สร้าง output stage
  - `open()`: เปิดอุปกรณ์ DeckLink
  - `submit_keying_frames()`: ส่ง fill และ key frames
  - `start_playback()`: เริ่มการเล่น
  - `stop_playback()`: หยุดการเล่น
  - `buffered_frame_count()`: ตรวจสอบจำนวน frames ที่รอในบัฟเฟอร์

- Helper functions:
  - `bgra_to_argb()`: แปลงรูปแบบสีถ้าจำเป็น

#### 2. `output_shim.cpp` (C++ Wrapper)
- C ABI interface สำหรับเชื่อมต่อ Rust กับ DeckLink SDK
- ฟังก์ชัน extern "C":
  - `decklink_output_open()`: เปิดอุปกรณ์
  - `decklink_output_submit_keying()`: ส่งภาพ
  - `decklink_output_start_playback()`: เริ่มเล่น
  - `decklink_output_stop_playback()`: หยุดเล่น
  - `decklink_output_close()`: ปิดอุปกรณ์
  - `decklink_output_buffered_frame_count()`: นับ frames

- รองรับการหา display mode อัตโนมัติตามความละเอียดและ frame rate

#### 3. `build.rs`
- Compile C++ shim ด้วย `cc` crate
- Link กับ DeckLink SDK headers
- Link กับ system libraries (dl, pthread, stdc++)

#### 4. `src/bin/test_internal_keying.rs` (Test Program)
- โปรแกรมทดสอบที่สามารถรันได้ทันที
- สร้างภาพทดสอบ:
  - **Fill**: SMPTE color bars (8 สี) ในรูปแบบ YUV422
  - **Key**: กล่องสี่เหลี่ยมเคลื่อนที่แบบ semi-transparent ใน BGRA
- แสดงภาพเป็นเวลา 3 วินาที
- รัน: `cargo run --bin test_internal_keying [device_index]`

#### 5. `KEYING_README.md`
- เอกสารแนะนำการใช้งานแบบละเอียด
- API reference
- ตัวอย่างโค้ด
- คำแนะนำการ troubleshooting

#### 6. `src/examples/keying_pipeline.rs`
- ตัวอย่าง integration กับ pipeline หลัก
- แสดงวิธีการเชื่อมต่อกับ stages อื่นๆ

### รูปแบบข้อมูลที่รองรับ

#### Input Formats
```
Fill Frame (ภาพหลัก):
- Format: YUV422_8 (8-bit YUV, UYVY layout)
- Stride: width × 2 bytes
- Memory: CPU (MemLoc::Cpu)
- Colorspace: BT.709

Key Frame (Overlay):
- Format: BGRA8 (8-bit BGRA with alpha)
- Stride: width × 4 bytes
- Memory: CPU (MemLoc::Cpu)
- Alpha: 0=transparent, 255=opaque
```

### ขั้นตอนการใช้งาน

#### 1. การสร้างและเปิดอุปกรณ์
```rust
use decklink_output::keying::InternalKeyingOutput;

let mut output = InternalKeyingOutput::new(
    0,      // device_index (usually 0)
    1920,   // width
    1080,   // height
    60,     // fps_num
    1       // fps_den
);

output.open()?;
```

#### 2. การเริ่มการเล่น
```rust
output.start_playback()?;
```

#### 3. การส่งภาพแต่ละเฟรม
```rust
// สมมติว่ามี fill_frame และ key_frame พร้อมแล้ว
output.submit_keying_frames(&fill_frame, &key_frame)?;
```

#### 4. การตรวจสอบบัฟเฟอร์
```rust
let buffered = output.buffered_frame_count();
if buffered > 10 {
    // มี frames สะสมมากเกินไป
}
```

#### 5. การหยุดและปิด
```rust
output.stop_playback()?;
// output จะถูก close อัตโนมัติเมื่อ drop
```

### การทดสอบ

#### รันโปรแกรมทดสอบ
```bash
# ใช้อุปกรณ์ index 0
cargo run --bin test_internal_keying 0

# หรือระบุ index อื่น
cargo run --bin test_internal_keying 1
```

#### ผลลัพธ์ที่คาดหวัง
- เห็นแถบสี 8 สี (color bars) เป็นพื้นหลัง
- เห็นกล่องสี่เหลี่ยมโปร่งใสเคลื่อนที่ไปทางขวา
- การ composite ที่ smooth ระหว่างพื้นหลังและ overlay

### การ Integration กับ Pipeline

```
Pipeline Flow:
┌──────────────────┐
│ DeckLink Input   │ ──► RawFramePacket (YUV8)
└──────────────────┘         │
                              ├────────────────┐
                              │                │
                              ▼                │ (keep as fill)
┌──────────────────┐          │                │
│ Preprocess CUDA  │          │                │
└──────────────────┘          │                │
         │                    │                │
         ▼                    │                │
┌──────────────────┐          │                │
│ Inference (TRT)  │          │                │
└──────────────────┘          │                │
         │                    │                │
         ▼                    │                │
┌──────────────────┐          │                │
│ Postprocess      │          │                │
└──────────────────┘          │                │
         │                    │                │
         ▼                    │                │
┌──────────────────┐          │                │
│ Overlay Plan     │          │                │
└──────────────────┘          │                │
         │                    │                │
         ▼                    │                │
┌──────────────────┐          │                │
│ Overlay Render   │ ──► OverlayFramePacket   │
│ (BGRA + alpha)   │         (BGRA, key)      │
└──────────────────┘                │          │
                                     │          │
                                     ▼          ▼
                            ┌──────────────────────┐
                            │ Internal Keying      │
                            │ (fill + key)         │
                            └──────────────────────┘
                                        │
                                        ▼
                            ┌──────────────────────┐
                            │ DeckLink Output      │
                            │ (composited video)   │
                            └──────────────────────┘
```

### Technical Details

#### YUV422 Layout (UYVY)
```
Pixel 0-1: [U0 Y0 V0 Y1]
Pixel 2-3: [U2 Y2 V2 Y3]
...
```
- 2 pixels ใช้ chroma (U,V) ร่วมกัน
- แต่ละ pixel มี luma (Y) เป็นของตัวเอง

#### BGRA Layout
```
Pixel: [B G R A]
```
- แต่ละ pixel มีค่า Blue, Green, Red และ Alpha แยกกัน
- Alpha: 0 = โปร่งใส, 255 = ทึบแสง

#### Alpha Blending Formula
```
output_rgb = key_rgb × (alpha/255) + fill_rgb × (1 - alpha/255)
```

### ความต้องการระบบ

#### Hardware
- Blackmagic DeckLink card (SDI/HDMI output)
- Cable เชื่อมต่อกับ monitor/recorder

#### Software
- DeckLink SDK 14.2.1 or later
- CUDA 12.x (สำหรับ pipeline อื่นๆ)
- C++ compiler (g++ with C++14 support)
- Rust 1.78+

#### System Libraries
- libdl
- libpthread
- libstdc++

### Performance Considerations

1. **Frame Rate Matching**: ต้องส่ง frames ด้วยอัตราที่แน่นอนตาม fps ที่กำหนด
2. **Buffer Management**: ตรวจสอบ `buffered_frame_count()` เพื่อป้องกัน overflow
3. **Memory Location**: Fill และ key frames ต้องอยู่ใน CPU memory
4. **Frame Timing**: ใช้ sleep หรือ timer เพื่อรักษาความสม่ำเสมอของ frame rate

### ข้อจำกัดและข้อควรระวัง

1. **True Hardware Keying**: Implementation นี้ใช้ software compositing เนื่องจาก standard DeckLink cards อาจไม่รองรับ true hardware internal keying
2. **Latency**: การ composite บน CPU อาจเพิ่ม latency เล็กน้อย
3. **Format Conversion**: การแปลง YUV → RGB เพื่อ composite อาจมี overhead
4. **Device Support**: ไม่ใช่ DeckLink card ทุกรุ่นที่รองรับ internal keying

### การแก้ปัญหา (Troubleshooting)

#### "Failed to open DeckLink output device"
- ตรวจสอบ device index (ลอง 0, 1, 2...)
- ตรวจสอบว่าติดตั้ง DeckLink drivers แล้ว
- ปิดโปรแกรมอื่นที่ใช้ DeckLink อยู่

#### "Failed to schedule frame"
- ลด rate การส่ง frames
- ตรวจสอบขนาดภาพตรงกับ resolution ที่กำหนด
- ตรวจสอบว่าข้อมูลอยู่ใน CPU memory

#### ไม่มีภาพออก
- ตรวจสอบสาย output
- ตรวจสอบว่าเรียก `start_playback()` แล้ว
- ตรวจสอบว่ามีการส่ง frames อย่างต่อเนื่อง

### ตัวอย่างการใช้งานใน Main Loop

```rust
// สร้าง output
let mut keying_output = InternalKeyingOutput::new(0, 1920, 1080, 60, 1)?;
keying_output.open()?;
keying_output.start_playback()?;

// Main loop
loop {
    // 1. รับ frame จาก input
    let fill_frame = capture_input()?;
    
    // 2-5. Process pipeline (preprocess, inference, postprocess, overlay)
    let key_frame = process_and_render_overlay(&fill_frame)?;
    
    // 6. Output ด้วย internal keying
    keying_output.submit_keying_frames(&fill_frame, &key_frame)?;
    
    // ตรวจสอบ buffer
    if keying_output.buffered_frame_count() > 10 {
        eprintln!("Warning: too many buffered frames");
    }
}

// ปิดเมื่อเสร็จ
keying_output.stop_playback()?;
```

### สรุป

Module นี้ให้ความสามารถในการ:
1. ✅ เปิด/ปิด DeckLink output device
2. ✅ ส่ง fill (YUV8) และ key (BGRA) frames
3. ✅ Composite ภาพด้วย alpha blending
4. ✅ จัดการ timing และ buffer
5. ✅ รองรับหลาย resolution และ frame rate
6. ✅ มี test program ที่พร้อมใช้งาน
7. ✅ มีเอกสารครบถ้วน

### ขั้นตอนถัดไป (Optional Improvements)

1. **Hardware Acceleration**: ใช้ GPU สำหรับ compositing (CUDA kernel)
2. **Format Conversion**: Optimize YUV→RGB conversion
3. **True Hardware Keying**: Support cards with hardware keying (e.g., DeckLink Duo 2)
4. **Audio Support**: เพิ่มการจัดการ audio output
5. **Multi-output**: รองรับหลาย DeckLink devices พร้อมกัน

---

## Files Created/Modified

### New Files
1. `crates/decklink_output/src/keying.rs` - Main Rust module
2. `crates/decklink_output/output_shim.cpp` - C++ wrapper
3. `crates/decklink_output/build.rs` - Build script
4. `crates/decklink_output/src/bin/test_internal_keying.rs` - Test program
5. `crates/decklink_output/KEYING_README.md` - Documentation
6. `crates/decklink_output/src/examples/keying_pipeline.rs` - Integration example
7. `crates/decklink_output/src/examples/mod.rs` - Examples module

### Modified Files
1. `crates/decklink_output/Cargo.toml` - Added cc build dependency
2. `crates/decklink_output/src/lib.rs` - Added keying and examples modules

---

## คำสั่งที่ใช้

```bash
# Build ทั้งหมด
cargo build

# Build เฉพาะ decklink_output
cargo build -p decklink_output

# รัน test
cargo run --bin test_internal_keying 0

# ดู documentation
cargo doc -p decklink_output --open
```

---

สร้างเสร็จสมบูรณ์! 🎉
