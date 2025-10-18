# preprocess_cuda

GPU-accelerated preprocessing module for DeepGIBox real-time video AI pipeline.

## Overview

This crate implements a high-performance CUDA-based preprocessing stage that converts raw video frames from various pixel formats (YUV422_8, NV12, BGRA8) to normalized RGB tensors in NCHW layout, ready for inference with TensorRT or ONNX Runtime.

## Features

- **Zero-copy GPU pipeline**: Input and output stay on GPU memory
- **Fused single-pass kernel**: Sample → Convert → Resize → Normalize → Pack in one kernel
- **BT.709 color space**: Proper limited-range YUV to RGB conversion
- **Multiple pixel formats**:
  - YUV422_8 (interleaved 4:2:2, UYVY or YUY2 byte order)
  - NV12 (planar 4:2:0)
  - BGRA8 (interleaved 8-bit)
- **Bilinear resize**: Any input size to any output size
- **Configurable normalization**: Mean/std for different models
- **FP16/FP32 output**: Optimized for TensorRT inference

## Requirements

- CUDA Toolkit 12.x or compatible
- NVIDIA GPU with compute capability >= 7.5 (Turing or later)
- nvcc compiler in PATH

## API

```rust
use preprocess_cuda::{Preprocessor, ChromaOrder};
use common_io::Stage;

// Create preprocessor
let mut preprocessor = Preprocessor::with_params(
    (512, 512),                    // Output size (W, H)
    true,                          // Use FP16
    0,                             // GPU device ID
    [0.485, 0.456, 0.406],        // RGB mean (ImageNet)
    [0.229, 0.224, 0.225],        // RGB std (ImageNet)
    ChromaOrder::UYVY,            // YUV422 byte order
)?;

// Process a frame (RawFramePacket must already be on GPU)
let raw_frame: RawFramePacket = ...; // From DeckLink capture
let tensor_input = preprocessor.process(raw_frame);

// tensor_input.desc: TensorDesc { n:1, c:3, h:512, w:512, dtype:Fp16, device:0 }
// tensor_input.data: NCHW layout on GPU
```

## Configuration

Load from TOML config:

```rust
let preprocessor = preprocess_cuda::from_path("configs/dev_1080p60_yuv422_fp16_trt.toml")?;
```

Config format:

```toml
[preprocess]
size = [512, 512]         # Output tensor size (W, H)
fp16 = true               # Use FP16 (recommended for TensorRT)
device = 0                # CUDA device ID
chroma = "UYVY"           # YUV422 byte order: "UYVY" or "YUY2"
mean = [0.0, 0.0, 0.0]   # RGB normalization mean
std = [1.0, 1.0, 1.0]    # RGB normalization std
```

## Color Conversion

### BT.709 Limited Range (Default)

For HD (1080p) and UHD (4K) content:

```
Y ∈ [16, 235]
U, V ∈ [16, 240]

C = Y - 16
D = U - 128
E = V - 128

R = 1.164 * C + 1.793 * E
G = 1.164 * C - 0.213 * D - 0.534 * E
B = 1.164 * C + 2.112 * D
```

## Chroma Subsampling

### YUV422 (4:2:2)
- U and V shared horizontally across pixel pairs
- Kernel performs horizontal interpolation for accurate color

### NV12 (4:2:0)
- U and V subsampled both horizontally and vertically
- Kernel uses bilinear interpolation in both dimensions

## Performance

Target latency budgets (GPU only, not including H2D/D2H):

- **1080p → 512×512**: ~1-2 ms
- **4K → 512×512**: ~3-5 ms

Actual performance depends on GPU model. Tested on RTX 3090 and A6000.

## Testing

```bash
# Unit tests (some require CUDA)
cargo test -p preprocess_cuda

# Run tests that need GPU
cargo test -p preprocess_cuda -- --ignored

# Playground preview
cargo run -p playgrounds --bin preprocess_gpu_preview
```

## Troubleshooting

### Build fails: "nvcc not found"

Make sure CUDA toolkit is installed and `nvcc` is in PATH:

```bash
which nvcc
# or set environment variable
export CUDA_PATH=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
```

### Runtime error: "Failed to initialize CUDA device"

Check CUDA is working:

```bash
nvidia-smi
```

### Color looks wrong

- Verify `colorspace` is set to `BT709` in input metadata
- Check `chroma` order matches your source (UYVY vs YUY2)
- For SD content, you may need BT.601 (not currently implemented)

### Stride issues

Ensure `stride_bytes` in input metadata accounts for any padding:
- YUV422_8: stride >= width * 2
- NV12: stride >= width
- BGRA8: stride >= width * 4

## Architecture

```
Input: RawFramePacket (GPU)
  ↓
Fused CUDA Kernel:
  1. Sample input with bilinear interpolation
  2. Convert YUV/NV12/BGRA → RGB (BT.709)
  3. Scale to [0, 1]
  4. Normalize: (x - mean) / std
  5. Pack to NCHW layout (FP16 or FP32)
  ↓
Output: TensorInputPacket (GPU)
```

## See Also

- [preprocessing_guideline.md](../../preprocessing_guideline.md) - Full specification
- [INSTRUCTION.md](../../INSTRUCTION.md) - Project overview
- [DATA_FLOW_AND_LATENCY_EXPLAINED.md](../../DATA_FLOW_AND_LATENCY_EXPLAINED.md) - Pipeline architecture
