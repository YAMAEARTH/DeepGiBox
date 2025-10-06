# การแก้ไข decklink_input เพื่อส่ง RawFramePacket ตาม I/O ที่กำหนด

## สรุปการเปลี่ยนแปลง

ได้ทำการแก้ไข `decklink_input` crate เพื่อให้ส่ง packet ตามโครงสร้าง I/O ที่กำหนดในไฟล์ `INSTRUCTION.md` โดยมีการเปลี่ยนแปลงดังนี้:

### 1. โครงสร้างข้อมูลใหม่

**เดิม:** ใช้ `RawFrame` struct ที่เก็บข้อมูลดิบ
```rust
pub struct RawFrame {
    pub data_ptr: *const u8,
    pub data_len: usize,
    pub width: u32,
    pub height: u32,
    pub row_bytes: u32,
    pub seq: u64,
}
```

**ใหม่:** ใช้ `RawFramePacket` จาก `common_io` ที่มีโครงสร้างตาม spec
```rust
pub struct RawFramePacket {
    pub meta: FrameMeta,
    pub data: MemRef
}
```

### 2. ข้อมูล FrameMeta ที่ครบถ้วน

`RawFramePacket` ประกอบด้วย `FrameMeta` ที่มีข้อมูลครบถ้วนตาม spec:
- `source_id`: ID ของอุปกรณ์ DeckLink
- `width`, `height`: ขนาดของเฟรม
- `pixfmt`: **PixelFormat::YUV422_8** (รูปแบบ UYVY ที่ DeckLink ใช้)
- `colorspace`: **ColorSpace::BT709** (ตามที่กำหนดสำหรับ 1080p60 และ 4K30)
- `frame_idx`: เลขที่เฟรมที่เพิ่มขึ้นเรื่อยๆ
- `pts_ns`: Presentation timestamp (ใช้ sequence number จาก DeckLink)
- `t_capture_ns`: เวลาที่จับภาพเป็น nanoseconds (system time)
- `stride_bytes`: จำนวน bytes ต่อแถว (row_bytes)

### 3. ข้อมูล MemRef

`MemRef` ระบุตำแหน่งและขนาดของข้อมูล:
- `ptr`: pointer ไปยังข้อมูล
- `len`: ขนาดข้อมูลทั้งหมด (stride × height)
- `stride`: stride ต่อแถว
- `loc`: **MemLoc::Cpu** (ข้อมูลอยู่ใน CPU memory)

### 4. การเปลี่ยนแปลงใน CaptureSession

```rust
pub struct CaptureSession {
    open: bool,
    source_id: u32,      // เพิ่ม: ID ของอุปกรณ์
    frame_count: u64,    // เพิ่ม: นับจำนวนเฟรม
}
```

เมธอด `get_frame()`:
- เปลี่ยนจาก `&self` เป็น `&mut self` เพื่อนับเฟรม
- คืนค่า `Option<RawFramePacket>` แทน `Option<RawFrame>`
- สร้าง `FrameMeta` พร้อมข้อมูลครบถ้วน (YUV422_8, BT709, timestamps)
- สร้าง `MemRef` สำหรับข้อมูลใน CPU

### 5. การ Re-export Types

ในไฟล์ `lib.rs` เพิ่มการ re-export types จาก `common_io`:
```rust
pub use common_io::{ColorSpace, FrameMeta, MemLoc, MemRef, PixelFormat, RawFramePacket};
```

### 6. การอัปเดต Playgrounds

แก้ไข `capture_yuv_dump.rs` ให้ใช้โครงสร้างใหม่:
- เปลี่ยนจาก `session.get_frame()` เป็น `session.get_frame()` (with mut session)
- เข้าถึงข้อมูลผ่าน `packet.meta.*` และ `packet.data.*`

## ความสอดคล้องกับ INSTRUCTION.md

การเปลี่ยนแปลงนี้ทำให้ `decklink_input` ส่งข้อมูลตาม pipeline ที่กำหนดใน INSTRUCTION.md:

```
DeckLinkInput
  → RawFramePacket{
      meta {
        w, h,
        pixfmt(YUV422_8),           ✅ ตั้งค่าเป็น YUV422_8
        colorspace(BT709),          ✅ ตั้งค่าเป็น BT709
        stride_bytes,               ✅ ส่งค่า stride_bytes
        frame_idx,                  ✅ นับเฟรม
        pts_ns,                     ✅ ใช้ sequence จาก DeckLink
        t_capture_ns                ✅ บันทึก timestamp
      },
      data                          ✅ ส่งเป็น MemRef
    }
```

## การทดสอบ

สามารถ build และทดสอบได้ด้วยคำสั่ง:
```bash
# Build decklink_input
cargo build -p decklink_input

# Build และรัน playground
cargo build -p playgrounds --bin capture_yuv_dump
cargo run -p playgrounds --bin capture_yuv_dump
```

## ขั้นตอนถัดไป

Stage ถัดไปในท่อ (PreprocessCUDA) สามารถรับ `RawFramePacket` และประมวลผลได้ทันที โดย:
1. อ่าน `meta.pixfmt` เพื่อเลือก kernel ที่เหมาะสม (YUV422→RGB conversion)
2. ใช้ `meta.colorspace` (BT709) สำหรับค่า coefficient ที่ถูกต้อง
3. อ่านข้อมูลจาก `data.ptr` ด้วยขนาด `data.len` และ `data.stride`
4. ส่งต่อ `meta` (FrameMeta) ไปยัง stage ถัดไปตามที่ระบุใน guideline
