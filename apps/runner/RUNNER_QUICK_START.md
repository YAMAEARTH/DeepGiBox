# DeepGiBox Runner - Quick Start Guide

เริ่มต้นใช้งาน DeepGiBox Runner อย่างรวดเร็ว

## 📋 ข้อกำหนดเบื้องต้น

1. **Hardware**
   - DeckLink capture card (installed และ recognized)
   - NVIDIA GPU with CUDA support
   - Endoscope เชื่อมต่อกับ DeckLink

2. **Software**
   - Rust toolchain installed
   - CUDA toolkit installed
   - TensorRT engine built (`configs/model/v7_optimized_YOLOv5.engine`)
   - TRT shim library built (`trt-shim/build/libtrt_shim.so`)

## 🚀 เริ่มใช้งาน (3 ขั้นตอน)

### ขั้นที่ 1: Build Runner

```bash
cd /home/earth/Documents/Guptun/6/DeepGiBox
cargo build --release --bin runner
```

### ขั้นที่ 2: เลือก Config File

เลือก config ตามประเภท endoscope ที่ใช้:

- **Olympus** → `configs/runner_olympus.toml`
- **Pentax** → `configs/runner_pentax.toml`
- **Fuji** → `configs/runner_fuji.toml`
- **General (Auto)** → `configs/runner_keying.toml`

### ขั้นที่ 3: Run!

```bash
./target/release/runner configs/runner_olympus.toml
```

หรือ

```bash
cargo run --release --bin runner -- configs/runner_olympus.toml
```

## 🎯 โหมดการใช้งาน

### 1. Production Mode (Hardware Keying)

ใช้สำหรับ deployment จริงในห้องตรวจ:

```bash
./target/release/runner configs/runner_keying.toml
```

**คุณสมบัติ:**
- ✅ Real-time overlay บน SDI output
- ✅ Hardware keying (alpha blending)
- ✅ รันไปเรื่อยๆ จนกว่าจะกด Ctrl+C
- ✅ แสดงสถิติทุก 60 frames

**Output:** Overlay แสดงบน SDI monitor

---

### 2. Benchmark Mode (Inference Only)

ใช้สำหรับทดสอบ performance:

```bash
./target/release/runner configs/runner_inference_only.toml
```

**คุณสมบัติ:**
- ✅ ทดสอบความเร็ว inference สูงสุด
- ✅ ไม่มี overhead จาก overlay/output
- ✅ รัน 60 วินาทีแล้วหยุดอัตโนมัติ
- ✅ บันทึก debug dumps (first 10 frames)

**Output:** สถิติ performance บน console

---

### 3. Endoscope-Specific Mode

ใช้สำหรับ endoscope แต่ละยี่ห้อ:

**Olympus:**
```bash
./target/release/runner configs/runner_olympus.toml
```

**Pentax:**
```bash
./target/release/runner configs/runner_pentax.toml
```

**Fuji:**
```bash
./target/release/runner configs/runner_fuji.toml
```

**ความแตกต่าง:** แต่ละ config ใช้ crop region ที่เหมาะสมกับ endoscope นั้นๆ

## 📊 ตัวอย่าง Output

### ตอน Start Pipeline

```
╔══════════════════════════════════════════════════════════╗
║  DEEPGIBOX - HARDWARE INTERNAL KEYING PIPELINE          ║
║  Production Mode: Real-time Overlay with Hardware Key   ║
╚══════════════════════════════════════════════════════════╝

🚀 Initializing Pipeline Stages...

📹 [1/7] DeckLink Capture
  ✓ Device 0 opened

🔧 [2/7] CUDA Device
  ✓ GPU 0 initialized

⚙️  [3/7] Preprocessor
  ✓ 512x512 output, Olympus crop

🧠 [4/7] TensorRT Inference V2
  ✓ Engine: configs/model/v7_optimized_YOLOv5.engine
  ✓ Output size: 25200 values

🎯 [5/7] Postprocessing
  ✓ Confidence threshold: 0.25
  ✓ Tracking: enabled

🎨 [6/7] Overlay Planning & GPU Rendering
  ✓ Full UI: true
  ✓ GPU rendering initialized

🔧 [7/7] Hardware Internal Keying
  ✓ Output: 1920x1080 (configs/dev_1080p60_yuv422_fp16_trt.toml)
  ✅ Hardware keying enabled (level=255)
  ✓ Frame timing: 30.00 FPS

╔══════════════════════════════════════════════════════════╗
║  PIPELINE RUNNING - Press Ctrl+C to stop               ║
╚══════════════════════════════════════════════════════════╝
```

### ขณะทำงาน

```
📊 Frame 60 | Latency: 34.52ms | FPS: 29.87 | Queue: 2/3
📊 Frame 120 | Latency: 34.18ms | FPS: 29.91 | Queue: 2/3
📊 Frame 180 | Latency: 33.95ms | FPS: 29.94 | Queue: 2/3
```

### ตอนจบ (กด Ctrl+C)

```
🛑 Stopping pipeline gracefully...

╔══════════════════════════════════════════════════════════╗
║  FINAL SUMMARY - HARDWARE KEYING PIPELINE               ║
╚══════════════════════════════════════════════════════════╝

  📈 Performance:
    Total frames:       1800
    Total time:         60.24s
    Average FPS:        29.88

  ⏱️  Average Latency:
    Capture:            12.34ms
    Preprocessing:       4.21ms
    Inference:          14.67ms
    Postprocessing:      1.85ms
    Overlay Planning:    0.73ms
    GPU Rendering:       2.15ms
    Hardware Keying:     0.58ms
    ─────────────────────────────────
    Total (E2E):        36.53ms

✅ Pipeline completed successfully!
```

## 🔧 การปรับแต่ง Config

### เปลี่ยน Confidence Threshold

แก้ไข config file:

```toml
[postprocessing]
confidence_threshold = 0.50  # เพิ่มเป็น 0.50 (แสดงเฉพาะแน่ใจสูง)
```

### ปิด Tracking

```toml
[postprocessing.tracking]
enable = false  # ปิด object tracking
```

### ปิด Full UI (แสดงแค่ bounding box)

```toml
[overlay]
enable_full_ui = false  # ไม่แสดง HUD elements
```

### เปลี่ยนความหนาของเส้น

```toml
[overlay.bbox]
base_thickness = 3       # เส้น bounding box หนาขึ้น
corner_thickness = 4     # มุมหนาขึ้น
```

### เปลี่ยนขนาดตัวอักษร

```toml
[overlay.label]
font_size = 20  # ตัวอักษรใหญ่ขึ้น
```

## 🐛 แก้ปัญหา

### ❌ "No DeckLink devices found"

**สาเหตุ:** DeckLink card ไม่ถูก detect

**วิธีแก้:**
1. ตรวจสอบ card ติดตั้งถูกต้อง
2. Check driver: `lspci | grep -i blackmagic`
3. Restart Desktop Video service

### ❌ "TensorRT engine not found"

**สาเหตุ:** Engine ยังไม่ build

**วิธีแก้:**
```bash
python rebuild_engine_640.py
```

### ❌ "TRT shim library not found"

**สาเหตุ:** TRT shim ยังไม่ compile

**วิธีแก้:**
```bash
cd trt-shim
mkdir -p build && cd build
cmake .. && make
```

### ❌ FPS ต่ำกว่า 30

**สาเหตุ:** GPU load สูงหรือ bottleneck

**วิธีแก้:**
1. ตรวจสอบ GPU: `nvidia-smi`
2. ลด confidence threshold
3. ปิด tracking: `enable = false`
4. ลด max_detections

### ❌ Overlay ไม่เห็น

**สาเหตุ:** Keyer level ต่ำหรือไม่เปิด

**วิธีแก้:**
```toml
[keying]
enable_internal_keying = true
keyer_level = 255  # ค่าสูงสุด = มองเห็นชัดที่สุด
```

## 📚 เอกสารเพิ่มเติม

- [Runner README](apps/runner/README.md) - เอกสารฉบับเต็ม
- [Pipeline Config Guide](PIPELINE_CONFIG_GUIDE.md) - คู่มือ config ละเอียด
- [GPU Overlay Quick Start](GPU_OVERLAY_QUICK_START.md) - GPU rendering guide

## 💡 Tips

1. **ใช้ Release Build เสมอ:** `--release` flag จำเป็นสำหรับ real-time performance
2. **Monitor GPU Usage:** `watch -n 1 nvidia-smi` เพื่อดู GPU load
3. **Save Custom Configs:** Copy config แล้วปรับเป็นของคุณเอง
4. **Test Inference First:** ใช้ `runner_inference_only.toml` ทดสอบก่อนใช้งานจริง
5. **Check Logs:** Debug dumps จะอยู่ที่ `output/runner/`

## 🎓 Best Practices

### สำหรับ Production
- ✅ ใช้ `runner_keying.toml` หรือ endoscope-specific config
- ✅ ปิด debug dumps: `enable_debug_dumps = false`
- ✅ ปิด detailed timings: `print_detailed_timings = false`
- ✅ Enable tracking สำหรับ smooth tracking

### สำหรับ Development
- ✅ ใช้ `runner_inference_only.toml` ทดสอบ performance
- ✅ เปิด debug dumps เพื่อดูผลลัพธ์
- ✅ ลด `test_duration_seconds` สำหรับทดสอบเร็ว
- ✅ Adjust threshold แล้วทดสอบซ้ำ

---

**พร้อมใช้งานแล้ว!** 🎉

หากมีปัญหาหรือคำถาม สามารถดูเอกสารเพิ่มเติมได้ที่ `apps/runner/README.md`
