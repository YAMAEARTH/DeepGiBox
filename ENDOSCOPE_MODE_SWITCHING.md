# การสลับโหมดกล้องแบบเรียลไทม์ (Runtime Endoscope Mode Switching)

## ภาพรวม

ระบบสลับโหมดกล้อง endoscope แบบเรียลไทม์ช่วยให้คุณสามารถสลับระหว่าง Fuji, Olympus และ Pentax ขณะที่ pipeline กำลังทำงานอยู่ โดยไม่ต้องหยุดหรือรีสตาร์ทโปรแกรม

## การกำหนดค่า

### ไฟล์ Config เดียว

ใช้ไฟล์ `configs/runner.toml` เดียวสำหรับทุกโหมด:

```toml
[preprocessing]
initial_endoscope_mode = "pentax"  # โหมดเริ่มต้น: fuji, olympus, หรือ pentax
output_width = 640
output_height = 640
# ... การตั้งค่าอื่นๆ
```

### โหมดกล้อง Endoscope ที่รองรับ

| โหมด | Crop Region (x, y, width, height) | คีย์ลัด |
|------|-----------------------------------|---------|
| **Fuji** | (1032, 326, 848, 848) | กด `1` |
| **Olympus** | (830, 330, 655, 490) | กด `2` |
| **Pentax** | (780, 182, 752, 752) | กด `3` |

## การใช้งาน

### 1. เริ่มต้น Pipeline

```bash
cd /home/earth/Documents/Guptun/6/DeepGiBox
cargo run --release -p runner -- configs/runner.toml
```

### 2. สลับโหมดขณะทำงาน

เมื่อ pipeline กำลังทำงาน:

- กด **`1`** → สลับเป็นโหมด **Fuji** 🔵
- กด **`2`** → สลับเป็นโหมด **Olympus** 🟢
- กด **`3`** → สลับเป็นโหมด **Pentax** 🟡

### 3. ข้อความแจ้งเตือน

เมื่อสลับโหมด จะแสดงข้อความบนหน้าจอ:

```
🔵 Switched to FUJI mode
🟢 Switched to OLYMPUS mode
🟡 Switched to PENTAX mode
```

## สถาปัตยกรรม

### Components หลัก

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Pipeline Thread                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Capture  │→ │Preprocess│→ │Inference │→ │Postproc. │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                      ↑                                       │
│                      │ อ่านโหมดปัจจุบัน                     │
│                      │                                       │
└──────────────────────┼───────────────────────────────────────┘
                       │
              ┌────────┴─────────┐
              │  Shared State    │
              │  (AtomicU8)      │
              │  0=Fuji          │
              │  1=Olympus       │
              │  2=Pentax        │
              └────────┬─────────┘
                       │
                       │ เขียนโหมดใหม่
                       │
┌──────────────────────┼───────────────────────────────────────┐
│              Keyboard Listener Thread                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ device_query::DeviceState                            │   │
│  │ - ตรวจจับคีย์ 1, 2, 3                                │   │
│  │ - อัปเดต AtomicU8 เมื่อกดคีย์                       │   │
│  │ - Poll ทุก 50ms (20Hz)                               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### การไหลของข้อมูล

1. **Keyboard Listener Thread**:
   - ทำงานอิสระจาก main pipeline
   - ตรวจจับการกดคีย์ 1, 2, 3
   - อัปเดต `current_mode: Arc<AtomicU8>` ทันที

2. **Main Pipeline Thread**:
   - อ่าน `current_mode` ก่อนประมวลผลทุกเฟรม
   - แปลง crop region ตามโหมดปัจจุบัน
   - เรียก `preprocessor.update_crop_region()` ทุกเฟรม

3. **Thread Safety**:
   - ใช้ `Arc<AtomicU8>` สำหรับ lock-free synchronization
   - ไม่มี mutex หรือการ block
   - รองรับการสลับโหมดความเร็วสูง

## Code Structure

### 1. EndoscopeMode Enum

```rust
// apps/runner/src/config_loader.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EndoscopeMode {
    Fuji,
    Olympus,
    Pentax,
}

impl EndoscopeMode {
    /// ดึง crop region coordinates
    pub fn get_crop_region(&self) -> (u32, u32, u32, u32) {
        match self {
            EndoscopeMode::Fuji => (1032, 326, 848, 848),
            EndoscopeMode::Olympus => (830, 330, 655, 490),
            EndoscopeMode::Pentax => (780, 182, 752, 752),
        }
    }

    /// ดึง overlay plan name (เตรียมไว้สำหรับอนาคต)
    pub fn get_overlay_plan(&self) -> &'static str {
        "default" // ปัจจุบันใช้ plan เดียวกันทุกโหมด
    }

    /// ดึงชื่อโหมดสำหรับแสดงผล
    pub fn name(&self) -> &'static str {
        match self {
            EndoscopeMode::Fuji => "FUJI",
            EndoscopeMode::Olympus => "OLYMPUS",
            EndoscopeMode::Pentax => "PENTAX",
        }
    }
}
```

### 2. Keyboard Listener

```rust
// apps/runner/src/main.rs

fn spawn_keyboard_listener(
    current_mode: Arc<AtomicU8>,
    running: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let device_state = DeviceState::new();
        let mut last_keys: Vec<Keycode> = Vec::new();

        while running.load(Ordering::SeqCst) {
            let keys = device_state.get_keys();

            for key in &keys {
                if !last_keys.contains(key) {
                    match key {
                        Keycode::Key1 => {
                            current_mode.store(0, Ordering::SeqCst); // Fuji
                            println!("\n🔵 Switched to FUJI mode");
                        }
                        Keycode::Key2 => {
                            current_mode.store(1, Ordering::SeqCst); // Olympus
                            println!("\n🟢 Switched to OLYMPUS mode");
                        }
                        Keycode::Key3 => {
                            current_mode.store(2, Ordering::SeqCst); // Pentax
                            println!("\n🟡 Switched to PENTAX mode");
                        }
                        _ => {}
                    }
                }
            }

            last_keys = keys;
            std::thread::sleep(Duration::from_millis(50));
        }
    })
}
```

### 3. Dynamic Crop Region Update

```rust
// ใน run_keying_pipeline() main loop

// อ่านโหมดปัจจุบันและอัปเดต crop region
let current_mode_val = current_mode.load(Ordering::SeqCst);
let active_mode = mode_from_u8(current_mode_val);
let (new_crop_x, new_crop_y, new_crop_w, new_crop_h) = active_mode.get_crop_region();
let new_crop = CropRegion::Custom { 
    x: new_crop_x, 
    y: new_crop_y, 
    width: new_crop_w, 
    height: new_crop_h 
};
preprocessor.update_crop_region(new_crop)?;
```

### 4. Preprocessor Update Method

```rust
// crates/preprocess_cuda/src/lib.rs

// เพิ่ม Custom variant
pub enum CropRegion {
    Fuji,
    Olympus,
    Pentax,
    None,
    Custom { x: u32, y: u32, width: u32, height: u32 },
}

impl Preprocessor {
    /// อัปเดต crop region แบบ dynamic
    pub fn update_crop_region(&mut self, new_crop: CropRegion) -> Result<()> {
        self.crop_region = new_crop;
        Ok(())
    }
}
```

## ประโยชน์

### ✅ ก่อนหน้า (3 Config Files)

```bash
# ต้องใช้ 3 ไฟล์แยก
configs/
  ├── dev_1080p60_yuv422_fp16_trt_fuji.toml
  ├── dev_1080p60_yuv422_fp16_trt_olympus.toml
  └── dev_1080p60_yuv422_fp16_trt_pentax.toml

# สลับโหมด = รีสตาร์ทโปรแกรม
$ cargo run -- configs/dev_1080p60_yuv422_fp16_trt_fuji.toml
# (Ctrl+C หยุด)
$ cargo run -- configs/dev_1080p60_yuv422_fp16_trt_olympus.toml
```

### ✅ ตอนนี้ (1 Config File)

```bash
# ใช้ไฟล์เดียว
configs/
  └── runner.toml

# สลับโหมดแบบเรียลไทม์
$ cargo run -- configs/runner.toml
# (กด 1, 2, 3 สลับโหมดได้ทันที ไม่ต้องหยุด)
```

## ข้อดี

1. **ไม่ต้องรีสตาร์ท**: สลับโหมดได้ทันทีขณะ pipeline ทำงาน
2. **Config น้อยลง**: ใช้ไฟล์เดียวแทน 3 ไฟล์
3. **ง่ายต่อการทดสอบ**: เปลี่ยนโหมดได้เร็ว เหมาะกับการทดสอบ
4. **Thread-safe**: ใช้ Atomic operations ไม่มี race condition
5. **Low Latency**: อัปเดต crop region ภายใน 1 เฟรม
6. **ขยายได้**: เตรียมโครงสร้างสำหรับ overlay plan แยกตามโหมด

## อนาคต

### Overlay Plan แยกตามโหมด

ปัจจุบันใช้ overlay plan เดียว (`"default"`) สำหรับทุกโหมด แต่โครงสร้างพร้อมแล้วสำหรับแยก plan ตามโหมด:

```rust
impl EndoscopeMode {
    pub fn get_overlay_plan(&self) -> &'static str {
        match self {
            EndoscopeMode::Fuji => "fuji_overlay",      // อนาคต
            EndoscopeMode::Olympus => "olympus_overlay", // อนาคต
            EndoscopeMode::Pentax => "pentax_overlay",   // อนาคต
        }
    }
}
```

### UI Visual Indicator

เพิ่ม visual indicator บน overlay เพื่อแสดงโหมดปัจจุบัน:

```
┌─────────────────────┐
│  🔵 FUJI MODE       │  ← ตัวอย่าง indicator
│                     │
│  [Video Content]    │
│                     │
└─────────────────────┘
```

## Troubleshooting

### ปัญหา: กดคีย์แล้วไม่สลับโหมด

**สาเหตุ**: Keyboard listener ไม่ได้รับ keyboard events

**แก้ไข**:
1. ตรวจสอบว่าหน้าต่าง terminal มี focus
2. ใช้ `sudo` หากจำเป็น (บาง Linux systems)
3. ตรวจสอบ permissions สำหรับ `/dev/input/`

### ปัญหา: Crop region ไม่ถูกต้อง

**สาเหตุ**: Coordinates ไม่ตรงกับกล้องจริง

**แก้ไข**:
1. เปิดไฟล์ `apps/runner/src/config_loader.rs`
2. แก้ไข `get_crop_region()` method:

```rust
EndoscopeMode::Fuji => (x, y, width, height), // ปรับค่าตรงนี้
```

### ปัญหา: Warning ขณะคอมไพล์

**Warning**: `device_query v2.1.0 (available: v4.0.1)`

**อธิบาย**: ใช้ v2.1.0 เพื่อความเสถียร (v4.0 อาจมี breaking changes)

**ไม่ต้องแก้**: Warning นี้ไม่กระทบการทำงาน

## Performance

### Overhead

- **Keyboard polling**: 50ms interval (20Hz) - negligible CPU usage
- **Mode check**: 1 atomic load per frame - ~5ns overhead
- **Crop update**: O(1) operation - ~10ns overhead

**สรุป**: ผลกระทบต่อ performance น้อยมาก (<0.001% latency increase)

### Latency

- **กด key → อัปเดต state**: <50ms (keyboard poll interval)
- **อัปเดต state → เฟรมถัดไป**: 16.67ms (1 frame @ 60fps)
- **Total switching latency**: ~66ms (4 frames)

## References

- Config file: `configs/runner.toml`
- EndoscopeMode enum: `apps/runner/src/config_loader.rs`
- Keyboard listener: `apps/runner/src/main.rs`
- Preprocessor update: `crates/preprocess_cuda/src/lib.rs`

---

**เวอร์ชัน**: 1.0  
**วันที่**: 2024  
**ผู้พัฒนา**: DeepGiBox Team
