# Zero-Copy GPU Inference API

## Overview

The zero-copy API (`run_inference_device`) is designed for maximum performance when your data is **already on GPU memory**. This eliminates CPU↔GPU memory transfers entirely.

## When to Use

✅ **Use zero-copy when:**
- Your preprocessing runs on GPU (CUDA kernels)
- Video decoder outputs directly to GPU memory
- Previous pipeline stage has data in GPU
- You're chaining multiple GPU operations

❌ **Don't use zero-copy when:**
- Your input data is on CPU RAM
- You're loading images from disk
- You need simple, straightforward API

## APIs Comparison

### 1. Regular Inference (CPU data)
```rust
run_inference(session, input_cpu, output_cpu, input_size, output_size)
```
- **Input**: CPU RAM pointer
- **Output**: CPU RAM pointer
- **Memory flow**: CPU → GPU → Inference → GPU → CPU
- **Performance**: ~5.35ms (includes 2-4ms PCIe transfer)
- **Use case**: Simple applications, data from files

### 2. Zero-Copy Inference (GPU data)
```rust
run_inference_device(session, d_input_gpu, d_output_gpu, input_size, output_size)
```
- **Input**: GPU device pointer (from cudaMalloc or your CUDA kernel)
- **Output**: GPU device pointer
- **Memory flow**: GPU → Inference → GPU (no PCIe!)
- **Performance**: ~2-3ms (pure inference, no transfers)
- **Use case**: GPU pipelines, real-time video, CUDA preprocessing

## Example Workflow

### Scenario: Real-time video processing

```
┌─────────────┐      ┌──────────────┐      ┌────────────┐      ┌─────────────┐
│ Video       │ GPU  │ Preprocessing│ GPU  │ TensorRT   │ GPU  │ Postprocess │
│ Decoder     │─────▶│ (CUDA)       │─────▶│ Inference  │─────▶│ (CUDA)      │
│ (NVDEC)     │      │ (your kernel)│      │ (zero-copy)│      │ (your kernel│
└─────────────┘      └──────────────┘      └────────────┘      └─────────────┘
       ▲                                                                  │
       │                                                                  ▼
       └──────────────────────── All on GPU! ───────────────────────────┘
```

**Key point**: Data NEVER goes to CPU, everything stays on GPU!

## Code Examples

### Option 1: Using TensorRT's Internal Buffers

```rust
// Get TensorRT's pre-allocated GPU buffers
let buffers = get_device_buffers(session);

// Your CUDA kernel writes directly to TensorRT's input buffer
your_preprocessing_kernel(your_data, buffers.d_input);

// Run inference (zero-copy, buffer is already populated!)
run_inference_device(session, buffers.d_input, buffers.d_output, input_size, output_size);

// Your CUDA kernel reads directly from TensorRT's output buffer
your_postprocessing_kernel(buffers.d_output, your_results);
```

**Advantages**:
- No intermediate buffers needed
- Direct read/write to TensorRT buffers
- Simplest zero-copy approach

**Limitations**:
- Your CUDA code must write to TensorRT's buffer format
- Less flexible if you have complex preprocessing

### Option 2: Using Your Own GPU Buffers

```rust
// Your preprocessing creates its own GPU buffer
let your_gpu_input: *mut f32 = your_preprocessing_stage(); // Returns GPU pointer

// Allocate GPU output buffer
let your_gpu_output: *mut c_void = cuda_malloc(output_size * 4);

// Run inference with YOUR buffers (TensorRT will use them directly)
run_inference_device(session, your_gpu_input, your_gpu_output, input_size, output_size);

// Continue with your GPU pipeline
your_next_gpu_stage(your_gpu_output);
```

**Advantages**:
- Full control over buffer management
- Flexible buffer layouts
- Can integrate into existing CUDA pipelines

**Limitations**:
- You manage buffer lifetimes
- Must ensure buffers are large enough

## Performance Gains

| Method | Time | Memory Copies | Use Case |
|--------|------|---------------|----------|
| Regular `run_inference` | ~5.35ms | 2× PCIe (H2D + D2H) | CPU data, simple apps |
| Zero-copy `run_inference_device` | ~2-3ms | 0× PCIe | GPU pipelines, real-time |
| **Speedup** | **~1.8-2.7x** | **100% reduction** | GPU-accelerated workflows |

## Implementation Notes

1. **Buffer Sizes**: Ensure your GPU buffers match the model's tensor sizes
   - Input: 1×3×640×640 = 786,432 floats = 3,145,728 bytes
   - Output: 25200×85 = 2,142,000 floats = 8,568,000 bytes

2. **CUDA Streams**: TensorRT uses its own CUDA stream internally
   - For synchronization with your kernels, use `cudaStreamSynchronize()`
   - Or use CUDA events for fine-grained control

3. **Memory Layout**: Data must be in correct format
   - NCHW format (batch, channels, height, width)
   - Contiguous memory
   - Float32 precision

## When NOT to Use Zero-Copy

The zero-copy API is NOT beneficial if:
- You load images from disk (CPU) → Use regular `run_inference`
- You don't have CUDA preprocessing → Use regular API
- PCIe transfer time is acceptable for your use case

The regular API is simpler and perfectly fine for many applications!

## Summary

- **Regular API**: Easy to use, works with CPU data, good for most cases
- **Zero-Copy API**: Maximum performance, requires GPU data, for advanced GPU pipelines
- **Performance gain**: ~2x faster when data is already on GPU
- **Key requirement**: Your data must already be on GPU memory (from CUDA kernels, video decoder, etc.)
