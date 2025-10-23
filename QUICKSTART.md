# 🚀 Quick Start - Internal Keying Demo

## วิธีรันโปรแกรม

### วิธีที่ 1: ใช้ Script (ง่ายที่สุด)

```bash
./run_demo.sh foreground.png
```

### วิธีที่ 2: รันโดยตรง

```bash
./target/release/internal_keying_demo foreground.png
```

---

## คำสั่งทั้งหมด

### 1. Build โปรเจ็กต์

```bash
cd /home/earth/Documents/Earth/Internal_Keying/DeepGiBox
CC=gcc-12 CXX=g++-12 cargo build --release
```

### 2. รัน Demo

```bash
# แบบเร็ว (Alpha-Only Mode) - 3x เร็วกว่า
./target/release/internal_keying_demo foreground.png

# หรือใช้ skull.png
./target/release/internal_keying_demo skull.png

# หรือภาพอื่นๆ
./target/release/internal_keying_demo /path/to/your/logo.png
```

### 3. สร้างภาพทดสอบใหม่

```bash
python3 create_test_image.py
```

---

## โหมดการทำงาน

### ⚡ Fast Mode (Alpha-Only) - ปัจจุบัน

**ใช้เมื่อ:** PNG มี transparent background (alpha channel)

**Performance:** ~0.5ms per frame @ 1080p60 (เร็ว 3x)

**การตั้งค่า:** 
```rust
// ใน internal_keying_demo/src/main.rs (line 58)
let use_alpha_only = true;  // ← ค่าปัจจุบัน
```

**เหมาะสำหรับ:**
- Logo overlay
- Graphics overlay
- Text, lower thirds
- ภาพที่มี transparency อยู่แล้ว

### 🎨 Slow Mode (Chroma Key)

**ใช้เมื่อ:** ต้องการ key สีเขียว/น้ำเงินออก

**Performance:** ~1.5ms per frame @ 1080p60

**การตั้งค่า:**
```rust
// ใน internal_keying_demo/src/main.rs (line 58)
let use_alpha_only = false;  // ← เปลี่ยนเป็น false
```

**เหมาะสำหรับ:**
- Live green screen
- Blue screen footage
- ภาพที่ไม่มี alpha channel

**จากนั้น rebuild:**
```bash
CC=gcc-12 CXX=g++-12 cargo build --release
```

---

## Output

โปรแกรมจะ:

1. **รับ input จาก DeckLink** (video capture)
2. **Composite PNG overlay** บน video
3. **ส่ง output ไปยัง DeckLink SDI** (real-time)

### Console Output ตัวอย่าง:

```
🎬 Internal Keying Demo with SDI Output
========================================

📸 Loading PNG image...
   ✓ Loaded: 1920x1080 pixels

🎥 Opening DeckLink capture...
   ✓ DeckLink capture opened on device 0

⏳ Waiting for stable frame dimensions...
   ✓ Got frame: 1920x1080

📡 Opening DeckLink output...
   ✓ DeckLink output opened: 1920x1080@60fps

🔧 Setting up output session...
   ✓ Output session ready: 1920x1080

🚀 Mode: ALPHA ONLY (2x faster!)
   Using PNG's alpha channel directly - no chroma keying

▶️  Processing frames → GPU→SDI Direct (Ctrl+C to stop)...
──────────────────────────────────────────────────────────

Frame #    60 | FPS: 60.00 | Mode: ALPHA | GPU→SDI direct (device: 0)
Frame #   120 | FPS: 60.01 | Mode: ALPHA | GPU→SDI direct (device: 0)
Frame #   180 | FPS: 59.98 | Mode: ALPHA | GPU→SDI direct (device: 0)
...
```

---

## ตรวจสอบ Hardware

### DeckLink Card

```bash
# ตรวจสอบว่ามี DeckLink หรือไม่
lspci | grep -i "Blackmagic\|DeckLink"

# ควรเห็นผลลัพธ์แบบนี้:
# 01:00.0 Multimedia video controller: Blackmagic Design DeckLink ...
```

### NVIDIA GPU

```bash
# ตรวจสอบ GPU
nvidia-smi

# ดู CUDA version
nvcc --version
```

---

## Troubleshooting

### ❌ ไม่มี Binary

```bash
# Build ใหม่
CC=gcc-12 CXX=g++-12 cargo build --release
```

### ❌ ไม่พบ DeckLink

```bash
# ตรวจสอบ driver
lsmod | grep blackmagic

# ติดตั้ง DeckLink driver (ถ้าจำเป็น)
# ดาวน์โหลดจาก: https://www.blackmagicdesign.com/support/
```

### ❌ CUDA Error

```bash
# ตรวจสอบ CUDA
nvidia-smi
nvcc --version

# ใช้ GCC 12 (ไม่ใช่ GCC 13)
export CC=gcc-12
export CXX=g++-12
```

### ❌ ภาพไม่มี Alpha Channel

```bash
# ตรวจสอบว่า PNG มี alpha หรือไม่
file foreground.png
# ควรเห็น: PNG image data, ... RGBA, ...

# หรือใช้ identify (ImageMagick)
identify -verbose foreground.png | grep -i alpha
```

ถ้าไม่มี alpha channel:
1. เปลี่ยนเป็น Chroma Key mode (ตั้ง `use_alpha_only = false`)
2. หรือแปลงภาพให้มี alpha channel

---

## ไฟล์ที่สำคัญ

```
DeepGiBox/
├── run_demo.sh                    ← รันโปรแกรมง่ายๆ
├── target/release/
│   └── internal_keying_demo       ← Binary หลัก
├── foreground.png                 ← ภาพ overlay
├── skull.png                      ← ภาพ overlay
├── create_test_image.py           ← สร้างภาพทดสอบ
└── internal_keying_demo/src/
    └── main.rs                    ← เปลี่ยน mode ที่นี่
```

---

## Performance Metrics

| Resolution | Fast Mode | Slow Mode | FPS Capability |
|------------|-----------|-----------|----------------|
| 1080p | 0.5ms | 1.5ms | 2000+ fps |
| 4K | 2.0ms | 6.0ms | 500+ fps |

---

## เอกสารเพิ่มเติม

- **การใช้งาน:** [USAGE_COMPARISON.md](USAGE_COMPARISON.md)
- **Performance:** [PERFORMANCE.md](PERFORMANCE.md)
- **Test Report:** [TEST_REPORT.md](TEST_REPORT.md)
- **API Reference:** [README.md](README.md)

---

## สรุป

**คำสั่งรันหลัก:**
```bash
./target/release/internal_keying_demo foreground.png
```

**หรือใช้ script:**
```bash
./run_demo.sh foreground.png
```

**Output:** Real-time composite video ส่งออกทาง DeckLink SDI 🎬
