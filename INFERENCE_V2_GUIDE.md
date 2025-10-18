# Inference V2 - TensorRT GPU Inference Guide

## Overview

`inference_v2` is a high-performance TensorRT inference crate that processes GPU tensors with **zero-copy** design.

- **Input**: `TensorInputPacket` (GPU tensor)
- **Output**: `RawDetectionsPacket` (raw model output)
- **Performance**: ~4ms per inference (238 FPS on RTX 3070)

## Architecture

```
TensorInputPacket (GPU) → TensorRT Inference (GPU) → RawDetectionsPacket (CPU)
                           ↑
                    Zero-Copy (no D2D transfer)
```

## Quick Start

### 1. Build TensorRT Engine

```bash
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --fp16 \
        --memPoolSize=workspace:2048M
```

### 2. Basic Usage

```rust
use inference_v2::TrtInferenceStage;
use common_io::Stage;

// Initialize (one-time setup)
let mut inference = TrtInferenceStage::new(
    "path/to/model.engine",
    "TRT_SHIM/libtrt_shim.so"
)?;

// Run inference (every frame)
let detections = inference.process(tensor_input_packet);
```

## Input Requirements

`TensorInputPacket` must have:

```rust
TensorInputPacket {
    from: FrameMeta,           // Frame metadata
    desc: TensorDesc {
        n: 1,                  // Batch size
        c: 3,                  // Channels (RGB)
        h: 512,                // Height
        w: 512,                // Width
        dtype: DType::Fp32,    // FP32 (TensorRT input format)
        device: 0,             // GPU device ID
    },
    data: MemRef {
        ptr: gpu_pointer,      // ⚠️ MUST be GPU pointer
        len: bytes,
        stride: bytes_per_row,
        loc: MemLoc::Gpu { device: 0 },  // ⚠️ MUST be GPU
    },
}
```

**Critical**: 
- ✅ Data **MUST** be on GPU (`MemLoc::Gpu`)
- ✅ DType **MUST** be `Fp32` (no conversion overhead)
- ✅ Shape **MUST** match engine input dimensions

## Output Format

`RawDetectionsPacket`:

```rust
RawDetectionsPacket {
    from: FrameMeta,                    // Same as input
    raw_output: Vec<f32>,               // Flattened model output
    output_shape: Vec<usize>,           // [batch, detections, values]
}
```

For YOLOv5 2-class model:
- `output_shape`: `[1, 16128, 7]`
- Each detection: `[x, y, w, h, objectness, class0_score, class1_score]`

## Performance

### Cold Start vs Steady-State

**First run includes GPU warmup:**
```
First run:        45ms  (CUDA kernel compilation)
Steady-state:      4ms  (actual inference time)
Cold start cost:  41ms  (one-time overhead)
```

**Benchmark results (100 runs):**
```
Average:    4.20ms (238 FPS)
Min:        3.93ms (254 FPS)
Max:        4.78ms
Std dev:    0.28ms (very stable!)
```

### Optimization Features

✅ **Zero-copy GPU inference** - input pointer used directly  
✅ **Cached function pointers** - no symbol lookups per frame  
✅ **Pre-allocated buffers** - no memory allocation per frame  
✅ **Minimal branching** - hot path is streamlined  
✅ **FP32 I/O** - no FP16↔FP32 conversion overhead  

## Complete Example

See [`apps/playgrounds/src/bin/inference_v2.rs`](apps/playgrounds/src/bin/inference_v2.rs):

```rust
// 1. Initialize CUDA
let cuda_device = CudaDevice::new(0)?;

// 2. Load and preprocess image
let img = ImageReader::open("image.jpg")?.decode()?.to_rgb8();
let preprocessed = preprocess_image_cpu(&img, 512, 512)?;

// 3. Upload to GPU
let gpu_buffer = cuda_device.htod_sync_copy(&preprocessed)?;
let gpu_ptr = *gpu_buffer.device_ptr() as *mut u8;

// 4. Create TensorInputPacket
let tensor_packet = TensorInputPacket {
    from: frame_meta,
    desc: TensorDesc { n: 1, c: 3, h: 512, w: 512, dtype: DType::Fp32, device: 0 },
    data: MemRef {
        ptr: gpu_ptr,
        len: preprocessed.len() * 4,
        stride: 512 * 4,
        loc: MemLoc::Gpu { device: 0 },
    },
};

// 5. Initialize inference stage
let mut inference = TrtInferenceStage::new(
    "apps/playgrounds/optimized_YOLOv5.engine",
    "TRT_SHIM/libtrt_shim.so"
)?;

// 6. Run inference (zero-copy GPU!)
let detections = inference.process(tensor_packet);

// 7. Parse detections
for det_idx in 0..detections.output_shape[1] {
    let offset = det_idx * 7;
    let x = detections.raw_output[offset];
    let y = detections.raw_output[offset + 1];
    let w = detections.raw_output[offset + 2];
    let h = detections.raw_output[offset + 3];
    let objectness = detections.raw_output[offset + 4];
    let class0 = detections.raw_output[offset + 5];
    let class1 = detections.raw_output[offset + 6];
    
    let confidence = objectness * class0.max(class1);
    if confidence > 0.5 {
        println!("Detection: bbox=({},{},{},{}), conf={}", x, y, w, h, confidence);
    }
}
```

## Benchmarking

Run multi-iteration benchmark to measure cold start vs steady-state:

```bash
cargo run --release --bin inference_v2_benchmark
```

This runs:
- 5 warmup iterations (includes cold start)
- 100 benchmark iterations (steady-state)
- Statistical analysis (mean, median, p95, p99, std dev)

## Common Issues

### ❌ "Input must be on GPU"
- **Cause**: Input data is on CPU (`MemLoc::Cpu`)
- **Fix**: Upload to GPU first using `cuda_device.htod_sync_copy()`

### ❌ Wrong output shape
- **Cause**: Engine input size mismatch
- **Fix**: Check engine dimensions with `trtexec --loadEngine=model.engine`

### ❌ Slow first inference
- **Normal**: First run includes CUDA kernel compilation (~40ms overhead)
- **Solution**: Run warmup iterations before benchmarking

## Pipeline Integration

```
RawFramePacket (DeckLink GPU capture)
    ↓
PreprocessCUDA (GPU preprocessing)
    ↓
TensorInputPacket (GPU)
    ↓
InferenceV2 (TensorRT GPU)
    ↓
RawDetectionsPacket
    ↓
Postprocess (NMS, tracking)
    ↓
Detection results
```

## See Also

- [`crates/inference_v2/src/lib.rs`](crates/inference_v2/src/lib.rs) - Implementation
- [`apps/playgrounds/src/bin/inference_v2_benchmark.rs`](apps/playgrounds/src/bin/inference_v2_benchmark.rs) - Benchmark tool
- [`PREPROCESSING_QUICK_REFERENCE.md`](PREPROCESSING_QUICK_REFERENCE.md) - GPU preprocessing
- [`common_io`](crates/common_io/src/lib.rs) - I/O packet types
