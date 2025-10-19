# Inference Microservice

This crate provides the inference stage in the DeepGIBox pipeline. It receives preprocessed tensor data and runs ONNX Runtime with TensorRT acceleration.

## Overview

The inference crate implements the `Stage<TensorInputPacket, RawDetectionsPacket>` trait, making it a microservice that:

1. **Receives**: `TensorInputPacket` with preprocessed tensor data (1×3×H×W, FP16/FP32, on GPU)
2. **Processes**: Runs ONNX model inference using TensorRT
3. **Outputs**: `RawDetectionsPacket` with frame metadata (raw detections handled by postprocess stage)

## Architecture

```
TensorInputPacket (from preprocess_cuda)
  ↓
InferenceEngine
  • Validates tensor shape/dtype
  • Runs TensorRT-accelerated inference
  • Manages GPU memory efficiently
  ↓
RawDetectionsPacket (to postprocess)
```

## Key Components

### `InferenceEngine`

The main struct implementing the `Stage` trait:

```rust
pub struct InferenceEngine {
    session: Session,           // ORT session with TensorRT EP
    gpu_allocator: Option<Allocator>,
    cpu_allocator: Allocator,
    model_input_shape: (u32, u32, u32, u32), // (N, C, H, W)
}
```

### Usage

```rust
use inference::InferenceEngine;
use common_io::{Stage, TensorInputPacket};

// Create engine with model path
let mut engine = InferenceEngine::new("./model.onnx")?
    .with_input_shape(1, 3, 512, 512);

// Process tensor packet
let output = engine.process(input_packet);
```

### Helper Function

```rust
// Create from config path
let engine = inference::from_path("./model.onnx")?;
```

## Features

- **TensorRT Acceleration**: FP16/FP32 optimized inference
- **GPU Memory Management**: Direct GPU-to-GPU tensor operations
- **Engine Caching**: TensorRT engines cached in `./trt_cache/`
- **Zero-Copy**: Minimal data movement between stages

## Configuration

The inference engine is initialized with:
- TensorRT execution provider
- FP16 precision enabled
- Engine caching enabled (in `./trt_cache/`)
- Device ID 0 (configurable)

## Input Requirements

- **Shape**: Must match model input (default: 1×3×512×512)
- **DType**: FP32 supported, FP16 requires conversion
- **Memory**: GPU memory (CUDA device)

## Pipeline Integration

In the full pipeline:

```
PreprocessCUDA → TensorInputPacket
                      ↓
                 InferenceEngine
                      ↓
                 RawDetectionsPacket → Postprocess
```

## Files

- `lib.rs` - Main inference engine implementation
- `inference.rs` - Legacy utilities and init functions
- `basic.rs` - Simple test/benchmark program

## Dependencies

- `ort` - ONNX Runtime with TensorRT support
- `common_io` - Shared pipeline types
- `ndarray` - Array operations (optional)

## Notes

- Video processing, visualization, and postprocessing have been moved to their respective crates
- This crate focuses solely on running inference efficiently
- Raw detection decoding happens in the `postprocess` crate
