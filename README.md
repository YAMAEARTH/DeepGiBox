# Internal Keying Module

Real-time internal keying system สำหรับ composite PNG overlay บน DeckLink video capture

## โครงสร้างระบบ

```
DeckLink Capture (UYVY) → Background Layer (GPU)
                                  ↓
PNG Image (RGBA) → BGRA → Chroma Key → Foreground Layer (GPU)
                                  ↓
                          Composite (PNG over DeckLink)
                                  ↓
                          Output (BGRA on GPU)
```

## Modules

### 1. **decklink_input** - DeckLink Capture
- Capture video จาก DeckLink card
- รองรับ GPU Direct (CUDA/DVP)
- Format: UYVY (YUV 4:2:2)

### 2. **decklink_output** - Keying & Composite
- โหลดภาพ PNG/JPEG เป็น BGRA
- CUDA kernel สำหรับ chroma keying
- Composite PNG over DeckLink video
- Output: BGRA format

### 3. **internal_keying_demo** - Demo Application
- ทดสอบ real-time keying
- แสดง FPS และสถานะ

## การใช้งาน

### 1. เตรียมสภาพแวดล้อม

**ต้องการ:**
- CUDA Toolkit 12.x
- GCC 11 หรือ 12 (GCC 13 มีปัญหา compatibility กับ CUDA 12.0)
- DeckLink SDK
- DeckLink card พร้อม video input

### 2. แก้ปัญหา GCC 13

หาก build ไม่ผ่านเพราะ GCC 13:

```bash
# ติดตั้ง GCC 12
sudo apt install gcc-12 g++-12

# ตั้งเป็น default
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 120
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 120
```

หรือ ระบุ compiler ใน build:

```bash
export CC=gcc-12
export CXX=g++-12
cargo build
```

### 3. Build

```bash
cd /home/earth/Documents/Earth/Internal_Keying/Test_Internal_Keying
cargo build --release
```

### 4. เตรียมภาพ PNG

สร้าง foreground image (เช่น logo, graphics overlay):

```bash
# ตัวอย่าง: สร้างภาพทดสอบด้วย ImageMagick
convert -size 1920x1080 xc:transparent \
    -fill "rgb(0,177,64)" -draw "rectangle 100,100 400,400" \
    -fill white -pointsize 72 -annotate +150+250 "LIVE" \
    foreground.png
```

### 5. รัน Demo

```bash
./target/release/internal_keying_demo foreground.png
```

## Configuration

### Chroma Key Presets

```rust
// Green screen
let key = ChromaKey::green_screen(); // R=0, G=177, B=64

// Blue screen
let key = ChromaKey::blue_screen();  // R=0, G=0, B=255

// Custom
let key = ChromaKey::custom(0, 255, 0, 0.15); // threshold=0.15
```

### Adjust Threshold

```rust
// Less sensitive (more background visible)
let key = ChromaKey::custom(0, 177, 64, 0.25);

// More sensitive (more foreground visible)
let key = ChromaKey::custom(0, 177, 64, 0.10);
```

## ปัญหาที่พบและแก้ไข

### ❌ CUDA + GCC 13 Incompatibility

**อาการ:**
```
error: identifier "_Float32" is undefined
```

**สาเหตุ:** CUDA 12.0 ไม่รองรับ GCC 13 อย่างเป็นทางการ

**วิธีแก้:**
1. ใช้ GCC 12 หรือต่ำกว่า
2. หรือ รอ CUDA 12.1+ ที่รองรับ GCC 13

### ✅ DeckLink GPU Memory

ข้อมูลจาก DeckLink อยู่บน GPU แล้ว (ผ่าน DVP):
- ไม่ต้อง upload/download ระหว่าง CPU-GPU
- Zero-copy performance
- Real-time keying ได้

## Performance Tips

1. **ใช้ Release build** - `cargo build --release`
2. **GPU memory อยู่บน device เดียวกัน** - ตรวจสอบว่า DeckLink และ CUDA ใช้ GPU เดียวกัน
3. **CUDA streams** - ใช้ non-blocking streams (implemented แล้ว)
4. **Frame buffering** - เพิ่ม ring buffer สำหรับ frames

## API Reference

### OutputSession

```rust
// Create session
let session = OutputSession::new(width, height, &png_image)?;

// Composite
session.composite(
    decklink_uyvy_gpu_ptr,  // GPU pointer from DeckLink
    decklink_pitch,          // Row stride
    ChromaKey::green_screen(),
)?;

// Get output (BGRA on GPU)
let output_ptr = session.output_buffer().ptr;

// Or download to CPU
let output_data = session.download_output()?;
```

### BgraImage

```rust
// Load from file
let img = BgraImage::load_from_file("logo.png")?;

// Create solid color
let img = BgraImage::solid_color(1920, 1080, 0, 0, 255, 255); // Blue

// Upload to GPU
let mut buffer = GpuBuffer::new(img.width, img.height)?;
buffer.upload(&img)?;
```

## TODO

- [ ] แก้ปัญหา CUDA/GCC compatibility
- [ ] เพิ่ม DeckLink output (ส่งผลลัพธ์กลับไปยัง DeckLink)
- [ ] รองรับ 10-bit video
- [ ] Advanced keying (spill suppression, edge refinement)
- [ ] Real-time preview (OpenGL/Vulkan)
- [ ] Performance profiling และ optimization

## ข้อมูลเพิ่มเติม

- DeckLink SDK: https://www.blackmagicdesign.com/support/
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- NVIDIA GPUDirect: https://developer.nvidia.com/gpudirect

---

**Note:** ตอนนี้ระบบออกแบบและเขียนโค้ดเสร็จแล้ว แต่ยังต้องแก้ปัญหา CUDA build environment ก่อนจึงจะรันได้
