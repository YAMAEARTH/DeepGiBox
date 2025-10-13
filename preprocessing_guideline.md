
# Preprocessing Guideline (GPU-only, No H2D)
**DeepGIBox / `preprocess_cuda` crate – implementable spec**  
Version: v1

This document specifies exactly how to implement the preprocessing stage when `RawFramePacket` already resides on **GPU memory** (`MemLoc::Gpu`). The goal is to output an NCHW tensor (RGB, FP16/FP32) suitable for TensorRT/ORT-TRT with minimal latency, zero unnecessary copies, and correct color handling for **BT.709**.

---

## 1) I/O Contracts

### Input
- Type: `common_io::RawFramePacket`
- Location: **GPU** (`MemLoc::Gpu{ device }`)
- Fields used:
  - `meta.{ width, height, stride_bytes, pixfmt, colorspace, source_id, frame_idx }`
  - `data.{ ptr, stride, loc }` ← device pointer to input frame
- Supported `PixelFormat`:
  - `YUV422_8` (interleaved 4:2:2; runtime variant **UYVY** or **YUY2**)
  - `NV12` (4:2:0, planar Y + interleaved UV)
  - `BGRA8` (interleaved 8-bit)

### Output
- Type: `common_io::TensorInputPacket`
- Location: **GPU** (`MemLoc::Gpu{ device }`)
- Desc: `TensorDesc { n:1, c:3, h:Hout, w:Wout, dtype: DType::Fp16|Fp32, device }`

---

## 2) Public API (Rust)

```rust
/// Create with desired output size, dtype, and device.
pub struct Preprocessor {
    pub size: (u32, u32),   // (Wout, Hout)
    pub fp16: bool,         // true = FP16 (recommended)
    pub device: u32,        // CUDA device id
    pub mean: [f32; 3],     // RGB mean (e.g., [0.0,0.0,0.0] or [0.485,0.456,0.406])
    pub std:  [f32; 3],     // RGB std  (e.g., [1.0,1.0,1.0] or [0.229,0.224,0.225])
    pub chroma: ChromaOrder // For YUV422_8: UYVY or YUY2
}

#[derive(Clone, Copy, Debug)]
pub enum ChromaOrder { UYVY, YUY2 }

impl Preprocessor {
    pub fn new(size: (u32,u32), fp16: bool, device: u32) -> anyhow::Result<Self>;
}

impl common_io::Stage<common_io::RawFramePacket, common_io::TensorInputPacket> for Preprocessor {
    fn process(&mut self, raw: RawFramePacket) -> TensorInputPacket;
}
```

> **Note**: Provide a `from_path(&str) -> anyhow::Result<PreprocessStage>` that reads config and sets `size/fp16/chroma/mean/std/device`.

---

## 3) Color, Range, and Conversions

### 3.1 Color Space
- Assume **BT.709** for 1080p and 4K (validate `raw.meta.colorspace == BT709`).

### 3.2 Video Range (Limited Range)
- Typical DeckLink capture is **video range**: Y ∈ [16,235], U/V ∈ [16,240].  
- Convert to RGB using BT.709 limited-range coefficients:

Let:
```
C = Y - 16
D = U - 128
E = V - 128
```

Then (BT.709, limited range):
```
R = 1.164 * C + 1.793 * E
G = 1.164 * C - 0.213 * D - 0.534 * E
B = 1.164 * C + 2.112 * D
```

- Clamp RGB to [0, 255] before normalization if working in 8-bit domain; for float intermediates, clamp to [0, 1] after scaling (see §3.4).

### 3.3 Chroma Siting (Crucial for 4:2:2 / 4:2:0)
- 4:2:2 (YUV422): U and V are shared across **pairs** of pixels horizontally. Interpolate U,V to the exact x-position.
- 4:2:0 (NV12): U,V subsampled both horizontally and vertically. Use bilinear interpolation (texture fetch recommended).

### 3.4 Normalize (Model-Ready)
Two common modes:
1. **[0,1] only**: `rgb = rgb / 255.0`
2. **Mean/Std**: `rgb = (rgb/255.0 - mean) / std`  
   - e.g., ImageNet: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]

> Implement both; choose via config.

---

## 4) Kernel Strategy (GPU-only, Fused)

### 4.1 Fused Pipeline
**Single pass per output pixel**:
1. Sample input (with stride, subsampling rules)
2. Convert YUV/NV12/BGRA → RGB (BT.709)
3. Resize to `(Wout, Hout)` via **bilinear**
4. Normalize (scale/mean/std)
5. Pack to **NCHW** (FP16 or FP32)

This minimizes global memory traffic and kernel launches.

### 4.2 Threading & Memory
- Blocks: 16×16 or 32×8 depending on GPU.
- Global reads must be **coalesced**.
- Consider **texture objects** for NV12/Y planes (built-in bilinear + clamping).
- Use constant memory for BT.709 coefficients and mean/std.

### 4.3 Pseudocode (CUDA-like)
```cpp
// each thread produces 1 output pixel (x_out, y_out)
__kernel fused_convert_resize_normalize_pack(..., 
    in_ptr, in_w, in_h, in_stride, pixfmt, chroma_order,
    out_ptr, out_w, out_h, fp16, mean[3], std[3]) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;

    // 1) map output -> input coords (bilinear)
    float sx = (x + 0.5f) * (in_w / (float)out_w) - 0.5f;
    float sy = (y + 0.5f) * (in_h / (float)out_h) - 0.5f;

    // 2) sample Y,U,V as float (handle subsampling/siting)
    float Y, U, V;
    if (pixfmt == YUV422_8) {
        // read 4:2:2 with chroma_order (UYVY or YUY2)
        // interpolate U,V horizontally for sx
        // Y at full res; careful with in_stride and boundaries
    } else if (pixfmt == NV12) {
        // read Y plane (full), UV plane (half res both axes)
        // bilinear sample both
    } else if (pixfmt == BGRA8) {
        // direct read, convert BGRA -> RGB
    }

    // 3) limited-range BT.709 to RGB
    float C = Y - 16.0f;
    float D = U - 128.0f;
    float E = V - 128.0f;
    float R = 1.164f*C + 1.793f*E;
    float G = 1.164f*C - 0.213f*D - 0.534f*E;
    float B = 1.164f*C + 2.112f*D;

    // 4) scale & normalize to float
    R = clamp(R/255.f, 0.f, 1.f);
    G = clamp(G/255.f, 0.f, 1.f);
    B = clamp(B/255.f, 0.f, 1.f);

    R = (R - mean[0]) / std[0];
    G = (G - mean[1]) / std[1];
    B = (B - mean[2]) / std[2];

    // 5) write NCHW
    size_t idx_r = 0 * (out_w*out_h) + y*out_w + x;
    size_t idx_g = 1 * (out_w*out_h) + y*out_w + x;
    size_t idx_b = 2 * (out_w*out_h) + y*out_w + x;

    if (fp16) {
        ((half*)out_ptr)[idx_r] = __float2half(R);
        ((half*)out_ptr)[idx_g] = __float2half(G);
        ((half*)out_ptr)[idx_b] = __float2half(B);
    } else {
        ((float*)out_ptr)[idx_r] = R;
        ((float*)out_ptr)[idx_g] = G;
        ((float*)out_ptr)[idx_b] = B;
    }
}
```

### 4.4 Launch Config
```
grid  = ((Wout+Bx-1)/Bx, (Hout+By-1)/By)
block = (Bx, By)  // Bx×By = 128..256 typical
stream = user-provided CUDA stream (single stream per stage)
```

---

## 5) Handling Each PixelFormat

### 5.1 `YUV422_8` (Interleaved, 4:2:2)
- Two common byte orders (detect from config):
  - **UYVY**: bytes per pair = `U0 Y0 V0 Y1`
  - **YUY2**: bytes per pair = `Y0 U0 Y1 V0`
- Stride: `meta.stride_bytes` (may be > `width*2`)
- For fractional `sx`, interpolate U,V horizontally between neighbor pairs.
- Y available at every pixel (interpolate if using bilinear resize).

**Bounds**: clamp reads to [0 .. in_w-1], [0 .. in_h-1].

### 5.2 `NV12` (Planar 4:2:0)
- Y plane: full resolution (W×H)
- UV plane: interleaved at half res ((W/2)×(H/2))
- Bilinear in both axes for UV.
- Stride for Y and UV may differ—supply both (or compute from metadata/layout).

### 5.3 `BGRA8`
- Simply reorder (B,G,R) to (R,G,B) and continue with normalization.
- Watch for `stride` padding; pixels per row = `stride / 4`.

---

## 6) Memory, Buffers, Streams
- **No H2D**: input is already on GPU.
- Maintain an **output buffer pool** keyed by `(c,h,w,dtype,device)`.
- Use a **single CUDA stream** for the whole stage to ensure order with neighboring stages.
- Avoid extra intermediate buffers—**fuse** operations.

---

## 7) Error Handling & Validation
- If `raw.data.loc != MemLoc::Gpu{device}`, return error: `"expected GPU input"`.
- If `raw.meta.colorspace != BT709`, log warn (or support mapping if needed).
- Validate `stride_bytes >= bytes_per_pixel * width` with tolerance for padding.
- For `YUV422_8`, if `chroma_order` is unknown → error.

---

## 8) Telemetry Hooks
- Wrap the stage in runner: `telemetry::time_stage("preprocess", &mut pre, raw)`
- Inside module (optional subspans):
  - `preprocess.kernels` – kernel execution time (GPU events)
  - `preprocess.pack` – packing NCHW if split kernel
- Prefer GPU events for accurate kernel timing.

---

## 9) Unit Tests (CPU-hosted checks acceptable with mocks)
- Create synthetic GPU frames using CUDA kernels (or as CPU + quick H2D for test-only path):
  1. **Color bars (BT.709)**: Verify R/G/B averages after conversion.
  2. **Checkerboard**: Verify bilinear scaling at edges.
  3. **Stride stress**: Use padded stride and ensure no wrap/tearing.
  4. **UYVY vs YUY2**: Swap order and assert color differences (smoke test).
- Budget asserts (on CI or local): 
  - 1080p60 → preprocess ≤ ~2 ms
  - 4Kp30   → preprocess ≤ ~5 ms
  (Numbers are guidance and hardware-dependent—treat as soft thresholds.)

**Rust test sketch:**
```rust
#[test]
fn yuv422_to_nchw_fp16_within_budget() {
    let raw = testsupport::make_gpu_dummy_yuv422(1920,1080, ChromaOrder::UYVY);
    let mut pre = Preprocessor::new((512,512), true, 0).unwrap();
    let t0 = telemetry::now_ns();
    let t = pre.process(raw);
    let ms = telemetry::since_ms(t0);
    assert!(t.desc.h == 512 && t.desc.w == 512);
    assert!(matches!(t.desc.dtype, DType::Fp16));
    assert!(ms <= 6.0, "preprocess too slow: {ms:.2}ms"); // relaxed
}
```

---

## 10) Playground Recipe
Create a bin under `apps/playgrounds/src/bin/preprocess_gpu_preview.rs`:

```rust
use anyhow::Result;
use common_io::Stage;

fn main() -> Result<()> {
    // assume input already GPU (mock or real capture module returning GPU)
    let mut pre = preprocess_cuda::Preprocessor::new((512,512), true, 0)?;
    pre.mean = [0.0, 0.0, 0.0];
    pre.std  = [1.0, 1.0, 1.0];
    pre.chroma = preprocess_cuda::ChromaOrder::UYVY;

    let raw = testsupport::make_gpu_dummy_yuv422(1920,1080, pre.chroma);
    let t = telemetry::time_stage("preprocess", &mut pre, raw);
    println!("tensor: {}x{} dtype={:?} dev={}", t.desc.w, t.desc.h, t.desc.dtype, t.desc.device);
    Ok(())
}
```

---

## 11) Acceptance Checklist
- [ ] Supports `YUV422_8` (UYVY & YUY2), `NV12`, `BGRA8`
- [ ] Correct BT.709 limited-range conversion
- [ ] Handles `stride_bytes` for all formats
- [ ] Produces contiguous **NCHW** tensor (RGB) on **GPU**
- [ ] Configurable `size`, `dtype (FP16/FP32)`, `mean/std`, `chroma_order`
- [ ] Fused kernel (sample→convert→resize→normalize→pack) or equivalent performance
- [ ] Telemetry hooks present (`preprocess`, optional subspans)
- [ ] Unit tests & playground examples provided

---

## 12) Performance Notes
- Prefer **texture fetch** for NV12 planes (built-in bilinear).
- Use constant memory for coefficients and mean/std.
- Consider vectorized stores (`float3`/`half2` where appropriate).
- Minimize divergent branches inside kernel (precompute mode flags).

---

## 13) Gotchas
- Wrong chroma order → severe hue shift (verify UYVY vs YUY2).
- Ignoring stride → horizontal tearing.
- Using BT.601 by mistake → tint errors (green/magenta).
- Not clamping before normalization → out-of-range artifacts.
- Splitting into many kernels → memory traffic explosion.

---

**End of spec.**
