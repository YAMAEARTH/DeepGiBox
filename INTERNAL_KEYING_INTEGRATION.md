# Internal Keying Integration Summary

## ✅ Completed: Pipeline with Internal Keying

### 📁 Modified File
- `apps/playgrounds/src/bin/pipeline_capture_to_output_v1.rs`

---

## 🔄 Pipeline Flow (Updated)

```
┌─────────────────────────────────────────────────────────────────┐
│  COMPLETE PIPELINE: Capture → Inference → Overlay → Output     │
└─────────────────────────────────────────────────────────────────┘

1. DeckLink Capture (YUV422)
   └─→ GPU/CPU frame buffer
   
2. CPU → GPU Transfer (if needed)
   └─→ CUDA device memory
   
3. CUDA Preprocessing
   └─→ 512×512 FP16 tensor
   
4. TensorRT Inference V2
   └─→ Raw detections (8400 boxes)
   
5. Postprocessing (NMS + SORT)
   └─→ Filtered detections with tracking IDs
   
6. Overlay Planning
   └─→ Overlay plan (rectangles + labels)
   
7. Overlay Rendering
   └─→ OverlayFramePacket (ARGB format)
   
8. 🆕 Internal Keying (NEW!)
   ├─→ ARGB → BGRA conversion
   ├─→ GPU Composite (CUDA kernel)
   │    Background (YUV422) + Overlay (BGRA)
   └─→ DeckLink Output (BGRA → SDI)
   
9. Save Debug Images
   └─→ original, overlay, comparison PNG files
```

---

## 🆕 New Components Added

### 1. Function: `overlay_to_bgra()`
**Location:** Line ~26-57

```rust
fn overlay_to_bgra(overlay: &common_io::OverlayFramePacket) -> Result<BgraImage>
```

**Purpose:**
- Converts `OverlayFramePacket` (ARGB format) to `BgraImage` (BGRA format)
- Required for GPU compositor which expects BGRA pixel order

**Conversion:**
```
ARGB (from renderer)  →  BGRA (for compositor)
[A, R, G, B]          →  [B, G, R, A]
```

---

### 2. Imports Added
**Location:** Top of file

```rust
use decklink_output::{BgraImage, OutputSession, OutputRequest};
```

**Components:**
- `BgraImage`: Container for BGRA pixel data
- `OutputSession`: GPU compositor (CUDA kernel)
- `OutputRequest`: DeckLink output request structure

---

### 3. Step 5.5: Internal Keying Initialization
**Location:** ~Line 470-505

```rust
// 5.5 Initialize Internal Keying (GPU Compositor + DeckLink Output)
```

**Actions:**
1. Wait for first frame to get dimensions (1920×1080)
2. Create dummy `BgraImage` for initial setup
3. Initialize `OutputSession` (GPU compositor)
4. Initialize DeckLink output device

**Output:**
```
✓ GPU Compositor initialized: 1920x1080
✓ DeckLink output initialized
```

---

### 4. Step 8: Internal Keying Processing
**Location:** ~Line 755-815

```rust
// Step 8: Internal Keying (GPU Composite + DeckLink Output)
```

**Pipeline Steps:**
1. **Convert ARGB → BGRA**
   ```rust
   let bgra_overlay = overlay_to_bgra(&overlay_frame)?;
   ```

2. **Update GPU Compositor**
   ```rust
   gpu_compositor = OutputSession::new(width, height, &bgra_overlay)?;
   ```

3. **GPU Composite (CUDA)**
   ```rust
   gpu_compositor.composite(
       raw_frame_gpu.data.ptr,
       raw_frame_gpu.data.stride,
   )?;
   ```

4. **Create Output Packet**
   ```rust
   let composited_packet = common_io::RawFramePacket {
       meta: ...,  // BGRA8 format
       data: ...,  // GPU pointer
   };
   ```

5. **Submit to DeckLink**
   ```rust
   let output_request = OutputRequest {
       video: Some(&composited_packet),
       overlay: None,
   };
   decklink_out.submit(output_request)?;
   ```

---

### 5. Latency Tracking
**Location:** ~Line 520, ~990

```rust
let mut total_keying_ms = 0.0;  // New accumulator
```

**Summary Output:**
```
║   Internal Keying: X.XX ms                            ║
```

---

## 📊 Performance Metrics

### Expected Latency Breakdown (estimated):
```
Capture:          1-2 ms
H2D Transfer:     0.1-0.5 ms
Preprocess:       2-3 ms
Inference V2:     5-8 ms
Postprocess:      1-2 ms
Overlay Plan:     0.5-1 ms
Overlay Render:   1-2 ms
Internal Keying:  1-2 ms  ← NEW!
─────────────────────────────
TOTAL E2E:        12-20 ms (50-80 FPS)
```

---

## 🔧 Configuration Requirements

### 1. DeckLink Config
**File:** `configs/dev_1080p60_yuv422_fp16_trt.toml`

This config file must be present and configured for:
- Output resolution: 1920×1080
- Frame rate: 60 FPS
- Pixel format: YUV422 (will receive BGRA and convert internally)

### 2. CUDA Kernel
**Required:** CUDA keying kernel must be built

The GPU compositor uses a CUDA kernel located in:
- `keying/keying.cu`

This must be compiled before running the pipeline.

---

## 🚀 How to Run

### Build the project:
```bash
cd /home/earth/Documents/Guptun/6/DeepGiBox
cargo build --release --bin pipeline_capture_to_output_v1
```

### Run the pipeline:
```bash
./target/release/pipeline_capture_to_output_v1
```

### Expected Output:
```
╔══════════════════════════════════════════════════════════╗
║  PIPELINE: CAPTURE → OVERLAY (INFERENCE V2)             ║
║  DeckLink → Preprocess → Inference V2 → Post → Overlay  ║
╚══════════════════════════════════════════════════════════╝

📁 Created output/test directory

📹 Available DeckLink Devices:
  Available DeckLink devices: 1
  Device 0: DeckLink SDI

📹 Step 1: Initialize DeckLink Capture
  ✓ Opened DeckLink device 0

⚙️  Step 2: Initialize Preprocessor
  ✓ Preprocessor ready (512x512, FP16, GPU 0)

🔧 Step 2.5: Initialize CUDA Device
  ✓ CUDA device initialized for CPU->GPU transfers

🧠 Step 3: Initialize TensorRT Inference V2
  ✓ TensorRT Inference V2 loaded
  ✓ Engine: configs/model/v7_optimized_YOLOv5.engine
  ✓ Output size: 176400 values

🎯 Step 4: Initialize Postprocessing
  ✓ Postprocessing ready
  ✓ Classes: 2 (Hyper, Neo)
  ✓ Confidence threshold: 0.25
  ✓ Temporal smoothing: enabled (window=4)
  ✓ SORT tracking: enabled (max_age=30)

🎨 Step 5: Initialize Overlay Stages
  ✓ Overlay planning ready
  ✓ Overlay rendering ready

🔧 Step 5.5: Initialize Internal Keying
  ⏳ Waiting for first frame to determine dimensions...
  ✓ Got frame dimensions: 1920x1080
  ✓ GPU Compositor initialized: 1920x1080
  ✓ DeckLink output initialized

🎬 Step 6: Processing Frames...
════════════════════════════════════════════════════════════

[Frame processing with internal keying output to DeckLink SDI]
```

---

## 🎯 Key Differences from `test_internal_keying_gpu.rs`

| Aspect | test_internal_keying_gpu.rs | pipeline_capture_to_output_v1.rs |
|--------|----------------------------|----------------------------------|
| **Input** | PNG file (static overlay) | Dynamic overlay from inference |
| **Overlay Source** | `BgraImage::load_from_file()` | `overlay_to_bgra()` from renderer |
| **Pipeline** | Capture → Composite → Output | Full AI pipeline + compositing |
| **Overlay Updates** | Static (once at startup) | Dynamic (every frame) |
| **Use Case** | Simple keying demo | Production inference pipeline |

---

## ✅ Verification Checklist

- [x] Import `decklink_output` types
- [x] Add `overlay_to_bgra()` conversion function
- [x] Initialize GPU compositor in step 5.5
- [x] Initialize DeckLink output device
- [x] Add internal keying step 8 in main loop
- [x] Track internal keying latency
- [x] Update pipeline summary with internal keying
- [x] No compilation errors
- [x] Pipeline flow: Capture → Inference → Overlay → **Internal Keying** → Output

---

## 📝 Notes

### Memory Management
- GPU buffers are pooled to avoid repeated allocations
- `gpu_buffers` vector keeps CUDA allocations alive
- Overlay BGRA data is converted on CPU then uploaded to GPU

### Frame Synchronization
- Pipeline processes first 10 frames by default
- Each frame goes through all stages sequentially
- Internal keying happens **after** overlay rendering
- DeckLink output receives composited BGRA frames on GPU

### Output Files
For each frame, the following files are created:
1. `output/test/original_frame_XXXX.png` - Original captured frame
2. `output/test/overlay_frame_XXXX.png` - Rendered overlay with bounding boxes
3. `output/test/comparison_frame_XXXX.png` - Side-by-side comparison
4. `output/test/preprocessing_dump_frame_XXXX_*.bin/txt` - Preprocessing debug data
5. `output/test/inference_output_XXXX.txt` - Raw inference detections
6. `output/test/postprocess_output_XXXX.txt` - Filtered detections after NMS

Plus **live video output** to DeckLink SDI! 🎥

---

## 🎉 Success!

The pipeline now includes **complete internal keying** support:
- ✅ Real-time AI inference (YOLOv5)
- ✅ Dynamic overlay rendering (bounding boxes + labels)
- ✅ GPU-accelerated compositing (CUDA kernel)
- ✅ Live DeckLink SDI output (1080p60)
- ✅ All in one pipeline! 🚀
