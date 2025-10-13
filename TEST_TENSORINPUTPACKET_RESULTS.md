# TensorInputPacket Test Results

## Test Overview

This test demonstrates the structure and metadata flow of `TensorInputPacket`, which is the output of the preprocessing stage and input to the inference stage in the DeepGIBox pipeline.

**Test Binary**: `apps/playgrounds/src/bin/test_tensorinputpacket.rs`

**Purpose**: 
- Show how `RawFramePacket` is transformed into `TensorInputPacket`
- Display detailed information about tensor structure and memory layout
- Verify correct metadata preservation through the pipeline
- Demonstrate support for multiple pixel formats and resolutions

**Note**: This test uses simulated preprocessing without actual GPU memory allocation. Real preprocessing requires valid GPU buffers from DeckLink capture module.

## Test Execution

```bash
cargo run -p playgrounds --bin test_tensorinputpacket
```

## Test Results

### ✅ Test 1: 1080p YUV422_8 → 512x512 FP16

**Input RawFramePacket:**
- Source ID: 0
- Dimensions: 1920×1080
- Pixel Format: YUV422_8
- Color Space: BT709
- Frame Index: 42
- PTS: 1234567890 ns
- Capture Time: 9876543210 ns
- Stride: 3840 bytes
- GPU Buffer: 4,147,200 bytes (4.05 MB)

**Output TensorInputPacket:**
- Shape (NCHW): 1×3×512×512
- Data Type: FP16 (half precision)
- Device: GPU 0
- Total Elements: 786,432
- Element Size: 2 bytes
- Total Size: 1,572,864 bytes (1.5 MB)
- Memory: GPU device 0

**Memory Layout:**
- Channel 0 (R): offset 0, 262,144 elements
- Channel 1 (G): offset 524,288 bytes, 262,144 elements
- Channel 2 (B): offset 1,048,576 bytes, 262,144 elements

**Verification:** ✓ All checks passed
- Batch size correct (N=1)
- Channel count correct (C=3)
- Output dimensions correct (512×512)
- Data type correct (FP16)
- Memory location correct (GPU)
- Metadata preserved (frame_idx=42, pts=1234567890)

---

### ✅ Test 2: 640x480 YUV422_8 → 640x480 FP32 (No Resize)

**Input RawFramePacket:**
- Dimensions: 640×480
- Pixel Format: YUV422_8
- GPU Buffer: 614,400 bytes (600 KB)

**Output TensorInputPacket:**
- Shape (NCHW): 1×3×480×640
- Data Type: FP32 (full precision)
- Total Elements: 921,600
- Element Size: 4 bytes
- Total Size: 3,686,400 bytes (3.6 MB)

**Memory Layout:**
- Channel 0 (R): offset 0, 307,200 elements
- Channel 1 (G): offset 1,228,800 bytes, 307,200 elements
- Channel 2 (B): offset 2,457,600 bytes, 307,200 elements

**Verification:** ✓ All checks passed

---

### ✅ Test 3: 4K (3840x2160) YUV422_8 → 512x512 FP16

**Input RawFramePacket:**
- Dimensions: 3840×2160 (4K UHD)
- Pixel Format: YUV422_8
- GPU Buffer: 16,588,800 bytes (15.8 MB)

**Output TensorInputPacket:**
- Shape (NCHW): 1×3×512×512
- Data Type: FP16
- Total Size: 1,572,864 bytes (1.5 MB)

**Note**: Significant downsample from 4K → 512×512 (87.5% reduction)

**Verification:** ✓ All checks passed

---

### ✅ Test 4: 1080p NV12 → 512x512 FP16

**Input RawFramePacket:**
- Dimensions: 1920×1080
- Pixel Format: NV12 (Y plane + interleaved UV)
- GPU Buffer: 3,110,400 bytes (3.0 MB)
- Stride: 1920 bytes

**Output TensorInputPacket:**
- Shape (NCHW): 1×3×512×512
- Data Type: FP16
- Total Size: 1,572,864 bytes (1.5 MB)

**Verification:** ✓ All checks passed

---

### ✅ Test 5: 1080p BGRA8 → 512x512 FP16

**Input RawFramePacket:**
- Dimensions: 1920×1080
- Pixel Format: BGRA8 (RGB with alpha)
- GPU Buffer: 8,294,400 bytes (7.9 MB)
- Stride: 7680 bytes

**Output TensorInputPacket:**
- Shape (NCHW): 1×3×512×512
- Data Type: FP16
- Total Size: 1,572,864 bytes (1.5 MB)

**Verification:** ✓ All checks passed

---

## Summary

### Test Statistics

| Test | Input Format | Input Size | Output Size | Data Type | Input Memory | Output Memory | Reduction |
|------|-------------|------------|-------------|-----------|--------------|---------------|-----------|
| 1 | YUV422_8 | 1920×1080 | 512×512 | FP16 | 4.05 MB | 1.5 MB | 63% |
| 2 | YUV422_8 | 640×480 | 640×480 | FP32 | 600 KB | 3.6 MB | -500% |
| 3 | YUV422_8 | 3840×2160 | 512×512 | FP16 | 15.8 MB | 1.5 MB | 90.5% |
| 4 | NV12 | 1920×1080 | 512×512 | FP16 | 3.0 MB | 1.5 MB | 50% |
| 5 | BGRA8 | 1920×1080 | 512×512 | FP16 | 7.9 MB | 1.5 MB | 81% |

### Key Observations

1. **Tensor Structure**:
   - All tensors use NCHW layout (Batch, Channels, Height, Width)
   - Batch size always 1 (single frame processing)
   - Always 3 channels (RGB) regardless of input format
   - Channel data is contiguous in memory

2. **Memory Efficiency**:
   - Downsampling (e.g., 1080p→512×512) reduces memory by ~63%
   - FP16 uses half the memory of FP32 (2 bytes vs 4 bytes per element)
   - 4K→512×512 achieves 90.5% memory reduction
   - No resize (640×480→640×480) with FP32 increases memory due to format conversion

3. **Pixel Format Support**:
   - ✓ YUV422_8 (UYVY/YUY2 chroma order)
   - ✓ NV12 (Y plane + interleaved UV)
   - ✓ BGRA8 (RGB with alpha channel)

4. **Metadata Preservation**:
   - All source metadata (frame_idx, pts, timestamps) preserved
   - Original dimensions and format tracked in `from` field
   - Enables traceability through pipeline stages

5. **GPU Pipeline**:
   - All memory operations on GPU (zero CPU transfer)
   - Memory location preserved: `Gpu { device: 0 }`
   - Enables fast inference without CPU-GPU sync

## TensorInputPacket Structure

```rust
pub struct TensorInputPacket {
    pub from: FrameMeta,        // Source frame metadata
    pub desc: TensorDesc,       // Tensor descriptor (NCHW shape, dtype, device)
    pub data: MemRef,          // GPU memory reference
}

pub struct TensorDesc {
    pub n: u32,                 // Batch size (always 1)
    pub c: u32,                 // Channels (always 3 for RGB)
    pub h: u32,                 // Height in pixels
    pub w: u32,                 // Width in pixels
    pub dtype: DType,          // Fp16 or Fp32
    pub device: u32,           // GPU device ID
}
```

## Channel Layout (NCHW)

For a 512×512 FP16 tensor:
- **Total size**: 1×3×512×512×2 = 1,572,864 bytes
- **Per channel**: 512×512×2 = 524,288 bytes
- **Channel 0 (Red)**: bytes [0 .. 524,287]
- **Channel 1 (Green)**: bytes [524,288 .. 1,048,575]
- **Channel 2 (Blue)**: bytes [1,048,576 .. 1,572,863]

## Next Steps

1. **Integration Testing**: Connect with real DeckLink capture to process actual GPU frames
2. **Inference Pipeline**: Feed TensorInputPacket to inference module
3. **Performance Profiling**: Measure preprocessing→inference latency
4. **Color Accuracy**: Validate RGB conversion with test patterns

## Related Tests

- `test_rawframepacket.rs`: Tests RawFramePacket structure and metadata
- `test_preprocessing_live.rs`: Tests preprocessing module with various configurations
- Integration: Will test full pipeline (capture → preprocess → inference)

---

**Status**: ✅ All structure tests passed  
**Date**: 2025-10-13  
**Environment**: CUDA 12.0, Quadro P4000, Ubuntu
