# Internal Keying Quick Reference

## สำหรับ DeckLink Output Module

### การใช้งานพื้นฐาน (3 ขั้นตอน)

```rust
// 1. สร้างและเปิด
use decklink_output::keying::InternalKeyingOutput;
let mut output = InternalKeyingOutput::new(0, 1920, 1080, 60, 1)?;
output.open()?;
output.start_playback()?;

// 2. ส่งภาพ (ในลูป)
output.submit_keying_frames(&fill_yuv, &key_bgra)?;

// 3. ปิด
output.stop_playback()?;
```

### รูปแบบ Frame

| ชื่อ | Format | Stride | Alpha | ตำแหน่ง |
|------|--------|--------|-------|---------|
| Fill | YUV422_8 | width × 2 | - | CPU |
| Key | BGRA8 | width × 4 | 0-255 | CPU |

### ความละเอียดที่รองรับ

- **1080p60**: 1920×1080 @ 60fps
- **1080p30**: 1920×1080 @ 30fps  
- **4K30**: 3840×2160 @ 30fps
- **720p60**: 1280×720 @ 60fps

### คำสั่งทดสอบ

```bash
# Build
cargo build -p decklink_output

# รัน test (จะแสดงภาพเคลื่อนไหว 3 วินาที)
cargo run --bin test_internal_keying 0
```

### API สำคัญ

```rust
// สร้าง
InternalKeyingOutput::new(device, w, h, fps_num, fps_den)

// เปิด/ปิด
.open() -> Result<()>
.start_playback() -> Result<()>
.stop_playback() -> Result<()>

// ส่งภาพ
.submit_keying_frames(&fill, &key) -> Result<()>

// ตรวจสอบ
.buffered_frame_count() -> u32
.is_open() -> bool
.is_playing() -> bool
```

### สูตร Alpha Blending

```
output = key × (α/255) + fill × (1 - α/255)
```

- α = 0: เห็นแค่ fill (background)
- α = 255: เห็นแค่ key (overlay)
- α = 128: blend 50/50

### Pipeline Integration

```
Input (YUV8) ──┬──► Preprocess ──► Inference ──► Postprocess
               │         ▼              ▼              ▼
               │    OverlayPlan ──► OverlayRender (BGRA)
               │                           │
               │                           │ (key)
               └───────────────────────────┴──► InternalKeying ──► Output
                        (fill)
```

### ตรวจสอบ Buffer

```rust
let buffered = output.buffered_frame_count();
if buffered > 10 {
    // ส่ง frames เร็วเกินไป - ลดความเร็วลง
}
```

### Troubleshooting

| ปัญหา | แก้ไข |
|-------|-------|
| Device ไม่เปิด | ตรวจ device index, drivers |
| ไม่มีภาพออก | ตรวจสาย, เรียก start_playback() |
| Frames ล้น | ตรวจ buffered_count, ลดความเร็ว |
| Artifacts | ตรวจ timing, stride ถูกต้อง |

### ไฟล์สำคัญ

```
crates/decklink_output/
├── src/
│   ├── keying.rs              ← Main module
│   ├── lib.rs                 ← Export keying
│   ├── bin/
│   │   └── test_internal_keying.rs  ← Test program
│   └── examples/
│       └── keying_pipeline.rs       ← Integration example
├── output_shim.cpp            ← C++ wrapper
├── build.rs                   ← Build script
├── KEYING_README.md           ← Full documentation
└── IMPLEMENTATION_SUMMARY.md  ← Thai summary
```

### Dependencies

```toml
[build-dependencies]
cc = "1.0"
```

### System Requirements

- Blackmagic DeckLink SDK 14.2.1+
- C++14 compiler
- Rust 1.78+
- Libraries: dl, pthread, stdc++

---

**เอกสารเต็ม**: `KEYING_README.md`  
**สรุปการทำงาน**: `IMPLEMENTATION_SUMMARY.md`
