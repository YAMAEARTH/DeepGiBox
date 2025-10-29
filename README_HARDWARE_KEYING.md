# 🎬 Hardware Internal Keying with DeckLink

## ภาพรวม (Overview)

โปรแกรมนี้ใช้ **Hardware Internal Keying** ของ DeckLink แทน GPU compositing
- DeckLink จะผสมภาพ overlay (PNG) กับ SDI input ในตัวการ์ดเอง
- ไม่ต้องใช้ GPU compositing หรือ CUDA
- เหมาะสำหรับการ์ดที่มี Hardware Keyer เช่น:
  - DeckLink 4K Extreme
  - DeckLink 8K Pro
  - DeckLink Duo 2
  - DeckLink Quad 2

---

## วิธีการทำงาน

```
┌─────────────────┐
│  SDI Input      │ (Background video)
│  (YUV422)       │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  DeckLink Hardware Keyer        │
│  - รับ overlay BGRA (PNG)       │
│  - ผสมกับ SDI input             │
│  - ใช้ alpha channel สำหรับ    │
│    transparency                 │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│  SDI Output     │ (Composited result)
└─────────────────┘
```

---

## การติดตั้ง

### ข้อกำหนด
- DeckLink card ที่รองรับ Internal Keying
- DeckLink SDK 12.9+
- Rust toolchain
- GCC 12
- SDI input source (camera/video player)

### Build
```bash
cd internal_keying_demo
cargo build --release
```

---

## การใช้งาน

### 1. เตรียมภาพ PNG Overlay

สร้างภาพ PNG ที่มี alpha channel:
```python
from PIL import Image

# สร้างภาพ 1920x1080 โปร่งใส
img = Image.new('RGBA', (1920, 1080), (0, 0, 0, 0))

# วาดเนื้อหา (ส่วนที่ไม่โปร่งใส)
# ... draw your content ...

img.save('overlay.png')
```

**หมายเหตุ:** 
- พื้นที่ที่มี alpha=0 จะโปร่งใส (เห็น SDI input ด้านล่าง)
- พื้นที่ที่มี alpha=255 จะทึบ (เห็นแค่ overlay)

### 2. เชื่อมต่อ DeckLink

**Input:** ต่อสัญญาณ SDI จาก source
**Output:** ต่อ SDI ไปยัง monitor/recorder

### 3. รันโปรแกรม

```bash
# แบบที่ 1: ส่ง PNG ไปให้ keyer ผสมเอง (แนะนำ)
./target/release/internal_keying_demo overlay.png --hardware-keying

# แบบที่ 2: ใช้ GPU compositing (เดิม)
./target/release/internal_keying_demo overlay.png
```

---

## พารามิเตอร์

```bash
Usage: internal_keying_demo <overlay.png> [OPTIONS]

Options:
  --hardware-keying    ใช้ DeckLink hardware keyer (แนะนำ)
  --keyer-level <0-255>  ความเข้มของ overlay (default: 255)
  --help               แสดงความช่วยเหลือ
```

---

## ตัวอย่าง Output

```
🎬 Hardware Internal Keying Demo
===================================

📸 Loading PNG overlay...
   ✓ Loaded: 1920x1080 pixels

🔍 Detecting DeckLink devices...
   [0] DeckLink 4K Extreme
   ✓ Using device: DeckLink 4K Extreme

🔧 Checking device capabilities...
   ✓ Internal keying: supported
   ✓ HD keying: yes

🎥 Opening DeckLink input...
   ✓ Capture opened

⏳ Detecting input signal format...
   ✓ Detected: 1920x1080@60.00fps

📡 Configuring output...
   Opening output: 1920x1080@60.00fps
   ✓ Output opened

🔑 Enabling hardware internal keyer...
   ✓ Internal keyer enabled
   ✓ Keyer level: 255 (fully visible)

▶️  Starting hardware keying (Ctrl+C to stop)
──────────────────────────────────────────────────
   Input:   SDI (1920x1080@60fps)
   Overlay: overlay.png (1920x1080)
   Output:  SDI composited (hardware mixed)
──────────────────────────────────────────────────

Frame #    60 | FPS: 60.00 | Hardware Keyer Active
Frame #   120 | FPS: 60.00 | Hardware Keyer Active
```

---

## ข้อดีของ Hardware Keying

| Feature | Hardware Keying | GPU Compositing |
|---------|----------------|-----------------|
| CPU Usage | ⚡ น้อยมาก | 🔥 ปานกลาง |
| GPU Usage | ⚡ ไม่ใช้ | 🔥 สูง |
| Latency | ⚡ ~1-2 frames | 🔶 ~3-5 frames |
| Quality | 🎯 Hardware precision | 🎯 Software precision |
| Compatibility | 🔶 เฉพาะ DeckLink ที่มี Keyer | ✅ ทุกการ์ด |

---

## Troubleshooting

### ❌ "Device does not support internal keying"

**สาเหตุ:** การ์ด DeckLink ไม่มี Hardware Keyer

**แก้ไข:**
- ใช้การ์ดที่รองรับ เช่น DeckLink 4K Extreme, 8K Pro
- หรือใช้โหมด GPU compositing แทน (ไม่ต้องใส่ `--hardware-keying`)

### ❌ "Failed to enable internal keyer"

**สาเหตุ:** Output ยังไม่ได้เปิด หรือ mode ไม่ตรงกับ input

**แก้ไข:**
- ตรวจสอบว่า SDI input มีสัญญาณเข้ามา
- ตรวจสอบว่า output mode ตรงกับ input mode

### ❌ "No matching display mode found"

**สาเหตุ:** Resolution หรือ frame rate ไม่ตรงกัน

**แก้ไข:**
- ตรวจสอบว่า source signal เป็น standard format (720p/1080p/4K)
- ตรวจสอบว่า frame rate เป็น 30/50/60 fps

---

## API Reference

### Rust FFI

```rust
extern "C" {
    // Enable internal keyer (mix overlay + SDI input in hardware)
    fn decklink_keyer_enable_internal() -> bool;
    
    // Set keyer opacity level (0=transparent, 255=opaque)
    fn decklink_keyer_set_level(level: u8) -> bool;
    
    // Disable keyer
    fn decklink_keyer_disable() -> bool;
    
    // Set output connection to SDI
    fn decklink_set_video_output_connection(connection: i64) -> bool;
    
    // Get SDI connection constant
    fn decklink_get_connection_sdi() -> i64;
}
```

### C++ (shim.cpp)

```cpp
// Enable internal keying
bool decklink_keyer_enable_internal();

// Set keyer level (0-255)
bool decklink_keyer_set_level(uint8_t level);

// Disable keyer
bool decklink_keyer_disable();

// Configure output connection
bool decklink_set_video_output_connection(int64_t connection);

// Get connection constants
int64_t decklink_get_connection_sdi();
```

---

## ความแตกต่างจากเวอร์ชัน GPU Compositing

| Aspect | Hardware Keying | GPU Compositing |
|--------|----------------|-----------------|
| **Processing** | ในตัว DeckLink | ใช้ CUDA kernel |
| **Chroma Key** | ❌ ไม่รองรับ | ✅ รองรับ (green/blue screen) |
| **Alpha Channel** | ✅ รองรับ | ✅ รองรับ |
| **Performance** | ⚡ เร็วมาก | 🔶 เร็ว (100+ FPS) |
| **Requirements** | DeckLink Keyer | NVIDIA GPU + CUDA |
| **Use Case** | Static overlay | Dynamic chromakey |

---

## สรุป

✅ **ใช้ Hardware Keying เมื่อ:**
- มีการ์ด DeckLink ที่รองรับ Keyer
- ต้องการ latency ต่ำสุด
- ต้องการประหยัด CPU/GPU
- ใช้ overlay แบบ static (PNG/logo)

❌ **ใช้ GPU Compositing เมื่อ:**
- ต้องการ chroma keying (green screen)
- ต้องการปรับแต่ง real-time
- ไม่มีการ์ดที่รองรับ Hardware Keyer
- ต้องการ effects พิเศษ

---

## เอกสารอ้างอิง

- [DeckLink SDK Documentation](https://www.blackmagicdesign.com/developer/product/capture-and-playback)
- [Internal Keying Guide](https://documents.blackmagicdesign.com/DeveloperManuals/DeckLinkSDK.pdf) (Chapter 5.4)
- [Hardware Keyer Specifications](https://www.blackmagicdesign.com/products/decklink)

---

**License:** MIT  
**Author:** YAMAEARTH  
**Version:** 1.0.0
