# Pipeline Debug Dump Enhancement

## 📋 Overview
เพิ่มความสามารถในการ dump ข้อมูลจากทุก stage ของ pipeline สำหรับการ debug และวิเคราะห์

## 🔧 Changes Made

### 1. เพิ่ม Dump Functions (3 functions ใหม่)

#### `dump_preprocessing()` - Preprocessing Stage
- **Input**: `TensorInputPacket`
- **Output**: `debug_frame_XXXX_preprocessing.txt`
- **ข้อมูล**:
  - Frame metadata (width, height, frame_idx)
  - Tensor descriptor (N, C, H, W)
  - Data type และ device
  - Memory location (CPU/GPU)
  - Tensor bytes และ stride

#### `dump_inference()` - Inference Stage  
- **Input**: `RawDetectionsPacket`
- **Output**: `debug_frame_XXXX_inference.txt`
- **ข้อมูล**:
  - Frame metadata
  - Output shape (e.g., [1, 25200, 85])
  - Raw output size (จำนวน floats)
  - First 50 float values (สำหรับ debug)

#### `dump_overlay_plan()` - Updated
- เพิ่มข้อมูล frame metadata
- แสดง canvas size
- จำนวน operations

### 2. Integration in Main Loop

แต่ละ stage ถูก dump **5 frames แรก** (frame 0-4):

```
Frame 0:
├─ Capture        ─→ debug_frame_0000_raw_yuv422.bin (เดิม - 1 frame only)
├─ Preprocessing  ─→ debug_frame_0000_preprocessing.txt
├─ Inference      ─→ debug_frame_0000_inference.txt
├─ Postprocessing ─→ debug_frame_0000_detections.txt
├─ Overlay Plan   ─→ debug_frame_0000_overlay_plan.txt
└─ BGRA Rendering ─→ debug_frame_0000_overlay_bgra.bin

Frame 1-4: (same pattern)
```

### 3. Console Output Enhancement

เพิ่มข้อมูล packet ที่แสดงใน console:

**Preprocessing:**
```
✓ TensorInputPacket:
    → Shape: [N=1, C=3, H=640, W=640]
    → Location: Gpu { device: 0 }
```

**Inference:**
```
✓ RawDetectionsPacket:
    → Output shape: [1, 25200, 85]
    → Raw output size: 2142000 floats
```

**Postprocessing:**
```
✓ DetectionsPacket:
    → Total detections: 2
```

**Overlay Plan:**
```
✓ OverlayPlanPacket:
    → Operations: 30
    → Canvas: 3840x2160
```

## 📂 Output Files Structure

```
output/test/
├── debug_frame_0000_preprocessing.txt   (Frame 0-4)
├── debug_frame_0000_inference.txt       (Frame 0-4)
├── debug_frame_0000_detections.txt      (Frame 0-4)
├── debug_frame_0000_overlay_plan.txt    (Frame 0-4)
├── debug_frame_0000_overlay_bgra.bin    (Frame 0-4)
├── debug_frame_0001_preprocessing.txt
├── debug_frame_0001_inference.txt
├── ... (pattern repeats for frames 0-4)
└── test_text.png                        (text rendering test)
```

## 🎯 Use Cases

### 1. Debug Pipeline Flow
ตรวจสอบว่าข้อมูลถูกส่งต่อระหว่าง stage อย่างถูกต้อง:
```bash
# Check all stages for frame 0
cat output/test/debug_frame_0000_preprocessing.txt
cat output/test/debug_frame_0000_inference.txt
cat output/test/debug_frame_0000_detections.txt
cat output/test/debug_frame_0000_overlay_plan.txt
```

### 2. Verify Tensor Shapes
ตรวจสอบว่า tensor shape ถูกต้องตามที่คาดหวัง:
```bash
grep "Shape:" output/test/debug_frame_0000_preprocessing.txt
# Output: Shape: [N=1, C=3, H=640, W=640]
```

### 3. Analyze Detection Quality
เปรียบเทียบ raw inference output กับ final detections:
```bash
# Raw inference output
grep "Raw output size" output/test/debug_frame_0000_inference.txt

# After NMS + tracking
grep "Total detections" output/test/debug_frame_0000_detections.txt
```

### 4. Track Object Across Frames
ติดตาม object เดียวกันข้ามหลาย frames:
```bash
for i in {0..4}; do
  echo "Frame $i:"
  grep "Track ID:" output/test/debug_frame_000${i}_detections.txt
done
```

### 5. Verify Overlay Operations
ตรวจสอบว่ามีการวาด overlay ครบถ้วน:
```bash
grep "Total operations:" output/test/debug_frame_0000_overlay_plan.txt
grep "Type:" output/test/debug_frame_0000_overlay_plan.txt | sort | uniq -c
```

## 🧪 Testing

สคริปต์ทดสอบ: `test_pipeline_dumps.sh`

```bash
./test_pipeline_dumps.sh
```

สคริปต์จะ:
1. ✅ ตรวจสอบ DeckLink hardware
2. 🧹 ลบไฟล์ debug เก่า
3. 🎬 รัน pipeline 10 วินาที
4. 📊 แสดงสรุปไฟล์ที่สร้าง

## 🔍 File Format Details

### Preprocessing Output (TXT)
```
Frame: 0
From: 3840x2160 frame #6
Tensor descriptor:
  Shape: [N=1, C=3, H=640, W=640]
  Dtype: F16
  Device: 0
Tensor location: Gpu { device: 0 }
Tensor bytes: 1228800
Tensor stride: 1228800
```

### Inference Output (TXT)
```
Frame: 0
From: 3840x2160 frame #6
Output shape: [1, 25200, 85]
Raw output size: 2142000 floats

First 50 values:
0.0012 0.0034 0.0056 ... (continues)
```

### Detections Output (TXT)
```
Frame: 0
From: 3840x2160 frame #6
Total detections: 1

Detection #0:
  BBox (x,y,w,h): (1941.4, 698.7, 305.7, 344.6)
  Class ID: 1
  Score: 0.5409
  Track ID: Some(1)
```

### Overlay Plan Output (TXT)
```
Frame: 0
From: 3840x2160 frame #6
Canvas: 3840x2160
Total operations: 30

Operation #0:
  Type: Rect
  XYWH: (1941.4, 698.7, 305.7, 344.6)
  Thickness: 2
  Color ARGB: (255, 255, 0, 0)
...
```

### BGRA Buffer Output (BIN)
- Binary file: 32 MB (3840×2160×4 bytes)
- Format: BGRA8 (4 bytes per pixel)
- Can convert to PNG for viewing

## 💡 Tips

1. **ดู inference output แบบละเอียด:**
   ```bash
   head -50 output/test/debug_frame_0000_inference.txt
   ```

2. **เปรียบเทียบ 2 frames:**
   ```bash
   diff output/test/debug_frame_0000_detections.txt \
        output/test/debug_frame_0001_detections.txt
   ```

3. **Count operations by type:**
   ```bash
   grep "Type:" output/test/debug_frame_0000_overlay_plan.txt | \
     sort | uniq -c
   ```

4. **Check tensor memory:**
   ```bash
   grep "Tensor bytes:" output/test/debug_frame_*.txt
   ```

## ⚠️ Notes

- **Dump เฉพาะ 5 frames แรก** เพื่อไม่ให้ใช้ disk มากเกินไป
- Raw YUV422 frame dump เฉพาะ **frame 0 เท่านั้น** (ขนาดใหญ่มาก)
- BGRA buffer แต่ละไฟล์มีขนาด **~32 MB** (4K resolution)
- ข้อมูลทั้งหมดถูกเขียนแบบ synchronous (อาจทำให้ latency เพิ่มขึ้นเล็กน้อยใน 5 frames แรก)

## 🚀 Future Enhancements

- [ ] เพิ่ม option เลือกจำนวน frames ที่จะ dump
- [ ] บีบอัดไฟล์ BGRA เป็น PNG อัตโนมัติ
- [ ] สร้าง visualization script สำหรับแสดงข้อมูล
- [ ] เพิ่ม JSON format option สำหรับ machine-readable output
