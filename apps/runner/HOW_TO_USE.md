# 📖 คู่มือการใช้งาน DeepGiBox

## 🎯 ภาพรวม

DeepGiBox เป็นระบบตรวจจับวัตถุแบบเรียลไทม์สำหรับกล้องเอนโดสโคป ที่ทำงานด้วย:
- **DeckLink** - จับภาพจากกล้อง และส่งออกพร้อม overlay
- **CUDA/TensorRT** - ประมวลผล AI แบบเรียลไทม์บน GPU
- **Hardware Keying** - แสดง overlay โดยไม่กระทบภาพต้นฉบับ

---

## 🚀 เริ่มต้นใช้งาน (ง่ายมาก!)

### ขั้นตอนที่ 1: ตรวจสอบระบบ

```bash
# เช็ค GPU
nvidia-smi

# เช็ค DeckLink
ls /dev/blackmagic/

# เช็คว่า binary มีหรือยัง
ls -lh target/release/runner
```

### ขั้นตอนที่ 2: รันโปรแกรม (ใช้ Config เดียวสำหรับทุกกล้อง!)

```bash
./target/release/runner configs/runner.toml
```

### ขั้นตอนที่ 3: สลับโหมดกล้องแบบเรียลไทม์ ⭐ NEW!

ขณะที่โปรแกรมทำงานอยู่:

- กด **`1`** → สลับเป็น **Fuji** 🔵
- กด **`2`** → สลับเป็น **Olympus** 🟢
- กด **`3`** → สลับเป็น **Pentax** 🟡

**ไม่ต้องหยุดโปรแกรม!** สลับได้ทันทีขณะทำงาน

### ขั้นตอนที่ 4: หยุดโปรแกรม

กด **Ctrl+C** เพื่อหยุด (จะมี summary แสดง)

---

## 📝 คำสั่งที่ใช้บ่อย

### 1. รันแบบปกติ (Production Mode)

```bash
# ใช้ config เดียว สลับโหมดด้วยคีย์ 1, 2, 3
./target/release/runner configs/runner.toml
```

### 2. รันแบบทดสอบ (ไม่ส่งออก Output)
```bash
# เหมาะสำหรับวัด Performance
./target/release/runner configs/runner_inference_only.toml
```

### 3. ทดสอบทุก Feature
```bash
# ทดสอบเร็ว (5-10 วินาที)
./test_all_features.sh

# ทดสอบละเอียด
./test_comprehensive.sh
```

### 4. Build ใหม่
```bash
# Build แบบ optimized
cargo build --release --bin runner

# Build และรันทันที
cargo run --release --bin runner -- configs/runner_pentax.toml
```

---

## ⚙️ การตั้งค่า Config

### เปิดไฟล์ Config
```bash
# แก้ไข Pentax config
nano configs/runner_pentax.toml

# หรือใช้ VS Code
code configs/runner_pentax.toml
```

### ตัวอย่าง Config ที่สำคัญ

```toml
[pipeline]
mode = "HardwareKeying"      # หรือ "InferenceOnly"
max_duration_secs = 0         # 0 = รันไม่จำกัดเวลา

[capture]
device_index = 0              # DeckLink device ตัวแรก
video_mode = "1080p60"        # ความละเอียด
pixel_format = "YUV422"       # รูปแบบสี

[preprocessing]
cuda_device = 0               # GPU ตัวแรก
crop_region = [780, 182, 752, 752]  # [x, y, width, height]

[inference]
confidence_threshold = 0.15   # threshold สำหรับ detection
use_fp16 = true              # ใช้ FP16 เพื่อความเร็ว

[output]
device_index = 0              # DeckLink output device
enable_internal_keying = true # เปิด hardware keying
```

---

## 🎨 การปรับแต่งตาม Use Case

### Use Case 1: ใช้งานจริง (Production)
```toml
[pipeline]
mode = "HardwareKeying"
max_duration_secs = 0  # รันต่อเนื่อง

[inference]
confidence_threshold = 0.15  # ไม่ต่ำเกินไป

[debug]
dump_raw_frames = false      # ปิด debug
dump_detections = false
```

### Use Case 2: ทดสอบประสิทธิภาพ
```toml
[pipeline]
mode = "InferenceOnly"       # ไม่มี output overhead
max_duration_secs = 60       # รัน 60 วินาที

[debug]
dump_raw_frames = false
dump_detections = true       # เก็บ log detection
```

### Use Case 3: Debug ปัญหา
```toml
[debug]
dump_raw_frames = true       # บันทึกภาพดิบ
dump_preprocessed = true     # บันทึก preprocessed
dump_inference = true        # บันทึก inference output
dump_detections = true       # บันทึก detection results
```

---

## 🔍 อ่านผลลัพธ์

### ข้อมูลที่แสดงระหว่างรัน

```
📊 Frame 300 | Latency: 29.62ms | FPS: 33.76 | Queue: 1/2
    Postprocess stats: 16128 total anchors → 23 passed confidence threshold (0.1%)
    Temporal smoothing active: 4 frames in history
    First detection: original=0.5462, smoothed=0.3657
    After NMS: 1 detections retained
  ✓ Postprocess time: 0.07ms
  ✓ Detections found: 1
  ✓ SORT tracking: 1 active tracks
```

**อธิบาย:**
- `Frame 300` - ประมวลผลไป 300 เฟรมแล้ว
- `Latency: 29.62ms` - ความหน่วงต่ำมาก! (< 40ms ดี)
- `FPS: 33.76` - 33 เฟรมต่อวินาที
- `Queue: 1/2` - มี 1 เฟรมรอประมวลผล จาก 2 slots
- `1 detections retained` - เจอวัตถุ 1 ชิ้น
- `1 active tracks` - Track วัตถุ 1 ตัว

### สีของ Bounding Box

- 🟢 **เขียว** (สีเขียว) - Confidence สูง (> 0.3)
- 🟡 **เหลือง** (สีเหลือง) - Confidence ปานกลาง (0.2-0.3)
- 🔴 **แดง** (สีแดง) - Confidence ต่ำ (< 0.2)

---

## 🎯 Crop Region คืออะไร?

Crop Region คือบริเวณที่เราจะตัดภาพมาประมวลผล AI

### ตัวอย่าง Crop Region

```
Full Frame (1920x1080)
┌─────────────────────────────────────┐
│                                     │
│     ┌─────────────┐                 │  <- Crop Region
│     │   AI ดูที่นี่  │                 │     (780, 182, 752, 752)
│     │             │                 │
│     └─────────────┘                 │
│                                     │
└─────────────────────────────────────┘
```

### ปรับ Crop Region ยังไง?

1. **เปิด config file**
   ```bash
   nano configs/runner_pentax.toml
   ```

2. **แก้ไขส่วน `crop_region`**
   ```toml
   [preprocessing]
   crop_region = [x, y, width, height]
   #              ↑  ↑    ↑      ↑
   #              │  │    │      └─ ความสูง
   #              │  │    └──────── ความกว้าง
   #              │  └───────────── ตำแหน่ง Y (บน-ล่าง)
   #              └──────────────── ตำแหน่ง X (ซ้าย-ขวา)
   ```

3. **ค่าแนะนำสำหรับแต่ละแบบ:**
   - **Pentax**: `[780, 182, 752, 752]` - มุมขวา
   - **Olympus**: `[830, 330, 655, 490]` - ตรงกลาง แคบ
   - **Fuji**: `[1032, 326, 848, 848]` - มุมขวา ใหญ่

---

## ⚡ Performance Tips

### 1. ลด Latency
```toml
[preprocessing]
use_fp16 = true              # ใช้ FP16

[inference]
use_fp16 = true              # ใช้ FP16
confidence_threshold = 0.2   # เพิ่ม threshold (ลด detection)

[postprocessing]
nms_iou_threshold = 0.5      # ปรับ NMS
```

### 2. เพิ่ม Accuracy
```toml
[inference]
confidence_threshold = 0.10  # ลด threshold (เพิ่ม sensitivity)

[postprocessing]
temporal_smoothing_frames = 8  # เพิ่มจาก 4 เป็น 8
nms_iou_threshold = 0.3      # เข้มงวดกับ overlapping boxes
```

### 3. ลด GPU Memory
```toml
[inference]
use_fp16 = true              # ใช้ FP16 แทน FP32
max_batch_size = 1           # Batch size เล็ก
```

---

## 🐛 แก้ปัญหาที่พบบ่อย

### ปัญหา 1: โปรแกรมไม่เริ่ม

**อาการ:**
```
Error: Failed to open DeckLink device
```

**วิธีแก้:**
```bash
# 1. เช็คว่า DeckLink มีหรือไม่
ls /dev/blackmagic/

# 2. เช็คว่าใช้ device index ถูกหรือไม่
# ถ้ามี card หลายตัว ลอง device_index = 1, 2, 3...
nano configs/runner_pentax.toml
```

### ปัญหา 2: FPS ต่ำ

**อาการ:**
```
FPS: 15.23  # น้อยเกินไป!
```

**วิธีแก้:**
```bash
# 1. เช็ค GPU load
nvidia-smi dmon

# 2. ลดความละเอียด หรือ crop region เล็กลง
nano configs/runner_pentax.toml
# ลดขนาด crop_region

# 3. เพิ่ม confidence threshold
# confidence_threshold = 0.25  # เพิ่มจาก 0.15
```

### ปัญหา 3: ตรวจจับไม่เจอ

**อาการ:**
```
Detections found: 0
```

**วิธีแก้:**
```toml
# ลด confidence threshold
[inference]
confidence_threshold = 0.10  # ลดจาก 0.15

# เช็ค crop region ว่าครอบคลุมวัตถุหรือไม่
[preprocessing]
crop_region = [780, 182, 752, 752]  # ปรับให้เหมาะสม
```

### ปัญหา 4: ตรวจจับผิดพลาดเยอะ (False Positives)

**อาการ:**
```
Detections found: 50  # เยอะเกินไป!
```

**วิธีแก้:**
```toml
# เพิ่ม confidence threshold
[inference]
confidence_threshold = 0.25  # เพิ่มจาก 0.15

# เข้มงวด NMS
[postprocessing]
nms_iou_threshold = 0.3  # ลดจาก 0.5
```

### ปัญหา 5: Overlay ไม่แสดง

**อาการ:**
- ภาพออกมาแต่ไม่มี bounding box

**วิธีแก้:**
```toml
# เช็คว่าเปิด output หรือยัง
[output]
enable_internal_keying = true

# เช็คว่าใช้ mode ถูกหรือไม่
[pipeline]
mode = "HardwareKeying"  # ไม่ใช่ "InferenceOnly"
```

---

## 📊 การอ่าน Performance Metrics

### Output ที่ควรจะเห็น (ดี)

```
✅ Good Performance:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Frame 300 | Latency: 29.62ms | FPS: 33.76
  ✓ Preprocessing: 0.85ms
  ✓ Inference: 4.12ms
  ✓ Postprocess: 0.07ms
  ✓ Overlay: 1.23ms
```

**หมายความว่า:**
- Latency < 35ms ✓ (ดีมาก!)
- FPS > 30 ✓ (เพียงพอ)
- แต่ละขั้นตอนเร็ว ✓

### Output ที่มีปัญหา (ไม่ดี)

```
❌ Poor Performance:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Frame 100 | Latency: 85.34ms | FPS: 11.72
  ⚠ Preprocessing: 2.45ms
  ⚠ Inference: 15.67ms  <- ช้า!
  ⚠ Postprocess: 0.34ms
  ⚠ Overlay: 3.12ms
```

**ปัญหา:**
- Latency > 80ms (ช้าเกิน)
- FPS < 15 (ไม่เพียงพอ)
- Inference ช้า → ปัญหาที่ GPU

**วิธีแก้:**
1. เช็ค GPU temperature
2. ลด batch size
3. ใช้ FP16

---

## 🎓 Advanced Usage

### 1. Run หลายตัวพร้อมกัน (Multi-GPU)

```bash
# Terminal 1 - GPU 0, DeckLink 0
CUDA_VISIBLE_DEVICES=0 ./target/release/runner configs/runner_pentax.toml

# Terminal 2 - GPU 1, DeckLink 1
CUDA_VISIBLE_DEVICES=1 ./target/release/runner configs/runner_olympus.toml
```

### 2. Custom Crop Region แบบ Interactive

```bash
# 1. Enable debug dump
nano configs/runner_pentax.toml
# dump_raw_frames = true

# 2. Run และดู frame ที่บันทึก
./target/release/runner configs/runner_pentax.toml

# 3. ใช้ Python script วิเคราะห์
python3 visualize_preprocessing.py

# 4. ปรับ crop_region ใน config
```

### 3. Export Statistics

```bash
# Redirect output to file
./target/release/runner configs/runner_pentax.toml 2>&1 | tee stats.log

# ดูสถิติ
grep "Frame" stats.log
grep "FPS:" stats.log | tail -10
```

---

## 📚 เอกสารเพิ่มเติม

### ภาษาไทย
- `RUNNER_QUICK_START.md` - Quick Start 3 ขั้นตอน
- `RUNNER_SUMMARY.md` - สรุปโปรเจค

### ภาษาอังกฤษ
- `apps/runner/README.md` - Runner Documentation
- `FEATURE_TEST_GUIDE.md` - Feature Testing Guide

---

## 🎯 Quick Reference

### คำสั่งที่ใช้บ่อย

```bash
# รัน Pentax
./target/release/runner configs/runner_pentax.toml

# รัน Olympus
./target/release/runner configs/runner_olympus.toml

# รัน Fuji
./target/release/runner configs/runner_fuji.toml

# ทดสอบ Performance
./target/release/runner configs/runner_inference_only.toml

# Build ใหม่
cargo build --release --bin runner

# ทดสอบทุก feature
./test_all_features.sh

# หยุดโปรแกรม
Ctrl+C
```

### Config Files

```
configs/
├── runner_pentax.toml          # Pentax (แนะนำ)
├── runner_olympus.toml         # Olympus
├── runner_fuji.toml            # Fuji
├── runner_keying.toml          # Generic keying
└── runner_inference_only.toml  # Benchmark mode
```

---

## 💡 Tips & Tricks

### 1. ทดสอบก่อนใช้งานจริง
```bash
# รันทดสอบ 60 วินาที
./target/release/runner configs/runner_inference_only.toml
```

### 2. Backup Config ก่อนแก้ไข
```bash
cp configs/runner_pentax.toml configs/runner_pentax.toml.backup
```

### 3. Monitor GPU
```bash
# Terminal แยก
watch -n 1 nvidia-smi
```

### 4. Log Statistics
```bash
./target/release/runner configs/runner_pentax.toml 2>&1 | \
  grep "Frame" | tee frame_stats.log
```

---

## ✅ Checklist ก่อนใช้งาน

- [ ] GPU driver ติดตั้งแล้ว (nvidia-smi ทำงาน)
- [ ] DeckLink driver ติดตั้งแล้ว (มี /dev/blackmagic/)
- [ ] Binary compiled แล้ว (มี target/release/runner)
- [ ] TensorRT cache สร้างแล้ว (มี trt_cache/*.engine)
- [ ] Config เลือกถูกต้องตามกล้อง
- [ ] Crop region เหมาะสม
- [ ] Confidence threshold ปรับแล้ว

---

## 🚀 เริ่มใช้งานเลย!

```bash
# Copy & paste ได้เลย!
cd /home/earth/Documents/Guptun/6/DeepGiBox
./target/release/runner configs/runner_pentax.toml
```

**Happy Detecting! 🎉**

---

## 📞 ติดปัญหาติดต่อ

- Documentation: `apps/runner/README.md`
- Testing: `./test_all_features.sh`
- Troubleshooting: ดูส่วน "🐛 แก้ปัญหาที่พบบ่อย" ด้านบน

---

_Last updated: November 6, 2025_
