# Pipeline: Capture to Overlay using Inference V2

## Overview

This document describes the complete pipeline implementation from DeckLink capture to overlay rendering using the optimized `inference_v2` module. The pipeline demonstrates the OverlayFramePacket output **without** connecting to DeckLink output (internal keying stage not included yet).

## Implementation

**File**: `apps/playgrounds/src/bin/pipeline_capture_to_overlay_v2.rs`

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE PIPELINE FLOW                                │
└─────────────────────────────────────────────────────────────────────────┘

1. DeckLink Capture (YUV422_8, BT709)
   ↓
2. CPU → GPU Transfer (htod_sync_copy)
   ↓
3. CUDA Preprocessing (640×640, FP16)
   ↓
4. TensorRT Inference V2 (Optimized)
   ↓
5. Postprocessing (NMS + SORT Tracking)
   ↓
6. Overlay Planning (Bounding boxes + Labels)
   ↓
7. Overlay Rendering (ARGB output)
   ↓
8. OverlayFramePacket (Ready for DeckLink Output)
```

## Key Components

### 1. DeckLink Capture
- Device: DeckLink 8K Pro
- Format: 1920×1080, 60fps, YUV422_8
- Color Space: BT.709
- Output: `RawFramePacket` (CPU memory)

### 2. CPU to GPU Transfer
- Uses `cudarc` for host-to-device memory copy
- Converts `MemLoc::Cpu` → `MemLoc::Gpu`
- Average latency: ~2.35ms
- GPU buffers are pooled to avoid repeated allocations

```rust
let mut gpu_buffer = cuda_device.htod_sync_copy(cpu_data)?;
let gpu_ptr = *gpu_buffer.device_ptr() as *mut u8;

let gpu_packet = common_io::RawFramePacket {
    meta: raw_frame.meta.clone(),
    data: MemRef {
        ptr: gpu_ptr,
        len: raw_frame.data.len,
        stride: raw_frame.data.stride,
        loc: MemLoc::Gpu { device: 0 },
    },
};
```

### 3. CUDA Preprocessing
- Input: 1920×1080 YUV422_8
- Output: 640×640 RGB FP16 (NCHW)
- Operations: YUV→RGB, Resize, Normalize
- Average latency: ~0.48ms

```rust
let mut preprocessor = Preprocessor::new(
    (640, 640),      // Output size
    true,            // FP16
    0,               // GPU device
)?;
```

### 4. TensorRT Inference V2
- Engine: `v7_optimized_YOLOv5.engine`
- Input: 1×3×512×512 (Note: engine expects 512×512, not 640×640)
- Output: 112,896 values (1×16128×7)
- Average latency: ~13.61ms
- Features:
  - Cached function pointers (no repeated symbol lookups)
  - Pre-allocated output buffer
  - Minimal branching in hot path

```rust
let mut inference_stage = TrtInferenceStage::new(engine_path, lib_path)
    .map_err(|e| anyhow!(e))?;
```

### 5. Postprocessing
- Classes: 2 (Hyper, Neo)
- Confidence threshold: 0.25
- NMS threshold: 0.45
- Features:
  - Temporal smoothing (window=4)
  - SORT tracking (max_age=30)
- Average latency: ~54.66ms (high due to many detections)

```rust
let mut post_stage = postprocess::from_path("")?
    .with_sort_tracking(30, 0.3, 0.3);
```

### 6. Overlay Planning
- Converts `DetectionsPacket` → `OverlayPlanPacket`
- Generates `DrawOp` instructions:
  - Bounding box rectangles
  - Class labels with confidence scores
  - Track IDs (if available)
- Average latency: ~0.03ms

```rust
let mut plan_stage = PlanStage {};
let overlay_plan = plan_stage.process(detections);
```

### 7. Overlay Rendering
- Converts `OverlayPlanPacket` → `OverlayFramePacket`
- Output: ARGB8 with alpha channel
- **Note**: Current implementation is a stub (returns empty buffers)
- Average latency: ~0.00ms (stub)

```rust
let mut render_stage = RenderStage {};
let overlay_frame = render_stage.process(overlay_plan);
```

## OverlayFramePacket Output

The pipeline successfully generates `OverlayFramePacket` with the following structure:

```rust
pub struct OverlayFramePacket {
    pub from: FrameMeta,      // Source frame metadata
    pub argb: MemRef,         // ARGB8 buffer (currently stub)
    pub stride: usize,        // Row stride in bytes
}
```

**Example Output**:
```
✅ OverlayFramePacket:
    → Frame: 1920×1080
    → ARGB buffer: 0 bytes (Cpu)  // Stub implementation
    → Stride: 0 bytes
    → Frame idx: 9
```

## Performance Metrics

**Test Configuration**:
- Input: 1920×1080@60fps YUV422_8
- GPU: NVIDIA RTX/A6000 class
- Frames processed: 10

**Average Latency Breakdown**:
| Stage            | Latency (ms) | % of Total |
|------------------|--------------|------------|
| Capture          | 0.00         | 0.0%       |
| H2D Transfer     | 2.35         | 3.3%       |
| Preprocess       | 0.48         | 0.7%       |
| Inference V2     | 13.61        | 19.1%      |
| Postprocess      | 54.66        | 76.6%      |
| Overlay Plan     | 0.03         | 0.0%       |
| Overlay Render   | 0.00         | 0.0%       |
| **TOTAL E2E**    | **71.34**    | **100%**   |

**Throughput**: 14.0 FPS

**Bottleneck Analysis**:
- 🔴 **Postprocessing (76.6%)**: High latency due to many detections (100+ per frame)
- 🟡 **Inference V2 (19.1%)**: Expected for YOLOv5
- 🟢 **Preprocessing (0.7%)**: Excellent performance
- 🟢 **H2D Transfer (3.3%)**: Acceptable (can be eliminated with GPU Direct)

## Running the Pipeline

```bash
# Build
cargo build -p playgrounds --bin pipeline_capture_to_overlay_v2 --features full

# Run
cargo run -p playgrounds --bin pipeline_capture_to_overlay_v2 --features full
```

**Requirements**:
- DeckLink device connected with video input
- CUDA 12.x
- TensorRT 10.x
- TensorRT engine file: `configs/model/v7_optimized_YOLOv5.engine`
- TRT shim library: `trt-shim/build/libtrt_shim.so`

## Next Steps

### 1. Implement Real Overlay Rendering
Currently `overlay_render` is a stub. Implement actual ARGB rendering:
- Allocate ARGB8 buffer (1920×1080×4 bytes)
- Render bounding boxes from `DrawOp::Rect`
- Render text labels from `DrawOp::Label`
- Output on GPU for zero-copy to DeckLink output

### 2. Connect to DeckLink Output
Add final stage to send overlay to DeckLink output:
```rust
use decklink_output::OutputSession;

let mut output = OutputSession::open(0)?;
output.send_frame(&overlay_frame)?;
```

### 3. Enable Internal Keying
Configure DeckLink output for internal keying mode:
- Fill key: Overlay graphics
- Background: Pass-through or black
- Alpha blending in hardware

### 4. Optimize Postprocessing
Reduce postprocessing latency (currently 54.66ms):
- Lower confidence threshold to reduce false positives
- Limit max detections per frame
- Optimize NMS implementation
- Consider per-class NMS parallelization

### 5. Enable GPU Direct RDMA
Eliminate H2D transfer (2.35ms):
- Configure DeckLink for ancillary GPU output
- Capture directly to GPU memory
- See `GPUDIRECT_RDMA_GUIDE.md` for setup

## Troubleshooting

### Issue: "Expected GPU input on device 0, got Cpu"
**Solution**: The pipeline now includes automatic CPU→GPU transfer. This error should not occur.

### Issue: High postprocessing latency
**Cause**: Too many detections (100+ per frame)
**Solutions**:
1. Increase confidence threshold
2. Reduce max_detections limit
3. Check model output - may be producing invalid detections

### Issue: Engine input size mismatch
**Symptom**: Engine expects 512×512 but preprocessor outputs 640×640
**Solution**: Match preprocessor output to engine input:
```rust
let mut preprocessor = Preprocessor::new(
    (512, 512),  // Match engine input size
    true,
    0,
)?;
```

### Issue: Missing dependencies
**Symptom**: Compile errors about missing crates
**Solution**: Always build with `--features full`:
```bash
cargo build -p playgrounds --bin pipeline_capture_to_overlay_v2 --features full
```

## Code Example

```rust
// Initialize pipeline
let mut capture = CaptureSession::open(0)?;
let mut preprocessor = Preprocessor::new((640, 640), true, 0)?;
let cuda_device = CudaDevice::new(0)?;
let mut inference_stage = TrtInferenceStage::new(engine_path, lib_path)?;
let mut post_stage = postprocess::from_path("")?.with_sort_tracking(30, 0.3, 0.3);
let mut plan_stage = PlanStage {};
let mut render_stage = RenderStage {};

// Process loop
loop {
    // Capture
    let raw_frame = capture.get_frame()?.unwrap();
    
    // H2D transfer
    let cpu_data = unsafe { std::slice::from_raw_parts(raw_frame.data.ptr, raw_frame.data.len) };
    let gpu_buffer = cuda_device.htod_sync_copy(cpu_data)?;
    let gpu_ptr = *gpu_buffer.device_ptr() as *mut u8;
    let raw_frame_gpu = /* ... create GPU packet ... */;
    
    // Process pipeline
    let tensor_packet = preprocessor.process(raw_frame_gpu);
    let raw_detections = inference_stage.process(tensor_packet);
    let detections = post_stage.process(raw_detections);
    let overlay_plan = plan_stage.process(detections);
    let overlay_frame = render_stage.process(overlay_plan);
    
    // Display or output overlay_frame
    println!("OverlayFramePacket: {}×{}", 
        overlay_frame.from.width, 
        overlay_frame.from.height
    );
}
```

## See Also

- [INSTRUCTION.md](INSTRUCTION.md) - Project overview
- [INFERENCE_V2_GUIDE.md](INFERENCE_V2_GUIDE.md) - Inference V2 documentation
- [GPUDIRECT_RDMA_GUIDE.md](GPUDIRECT_RDMA_GUIDE.md) - GPU Direct setup
- [postprocessing_guideline.md](postprocessing_guideline.md) - Postprocessing spec
- [DECKLINK_OUTPUT_README](crates/decklink_output/README_FINAL.md) - Output implementation
