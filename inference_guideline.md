
# Inference Guideline — ONNX Runtime (TensorRT EP, FP16)
**DeepGIBox / `inference` crate – implementable spec**  
Version: v1

This document defines how to implement the inference stage using **ONNX Runtime (ORT)** with the **TensorRT Execution Provider (EP)** for **FP16** models. It aligns with `common_io` packet contracts and `telemetry` hooks used in DeepGIBox.

---

## 1) Goals
- Run ONNX models with **TensorRT EP** in **FP16** on a specified GPU.
- **Zero-/Low‑copy** by binding GPU memory directly via **ORT I/O Binding**.
- Support **engine cache**, **timing cache**, **warmup**, and **(optional) dynamic shape**.
- Return results as `RawDetectionsPacket` for downstream postprocess modules.

---

## 2) I/O Contracts

### Input (from `preprocess_cuda`)
- Type: `common_io::TensorInputPacket`
- Location: **GPU** (`MemLoc::Gpu{ device }`)
- Required fields:
  - `desc = { n:1, c:3, h:H, w:W, dtype: Fp16|Fp32, device }`
  - `data = { ptr: CUdeviceptr, len, loc:Gpu }`
- Must match the ONNX model input (shape/layout/dtype). **Prefer FP16 NCHW**.

### Output (to `postprocess`)
- Type: `common_io::RawDetectionsPacket`
- Contains: source `FrameMeta`, plus **output tensor handles** (names, shapes, dtype, device pointers if GPU).  
  You may keep outputs on **GPU** to avoid D2H (recommended), or allow ORT to place them on CPU if desired.

---

## 3) Config (TOML) Example
```toml
[inference]
backend             = "onnxruntime_trt"
model               = "models/yolov5_fp16.onnx"
device              = 0
fp16                = true
engine_cache        = ".cache/trt_engines"
timing_cache        = ".cache/trt_timing"
max_workspace_mb    = 2048
enable_fallback_cuda = true         # fall back to CUDA EP if TRT build fails
warmup_runs         = 5

# Optional dynamic-shape profile (if your model supports it)
[input_profile]
name = "images"     # ONNX input name (override if needed)
min  = [1, 3, 512, 512]
opt  = [1, 3, 640, 640]
max  = [1, 3, 960, 960]
```

> Map this config to ORT/TRT provider options in your Rust wrapper. The exact option keys of ORT may differ by version; keep the mapping centralized.

---

## 4) Public API (Rust)

```rust
pub struct OrtTrtEngine {
    // ORT handles
    env:        *mut OrtEnv,
    session:    *mut OrtSession,
    iobind:     *mut OrtIoBinding,
    meminfo:    *mut OrtMemoryInfo,   // CUDA memory info for binding
    // meta
    device:     u32,
    fp16:       bool,
    input_name: String,
    output_names: Vec<String>,
    // caches for shapes/dtypes etc.
}

impl OrtTrtEngine {
    pub fn new(cfg: &config::InferenceCfg) -> anyhow::Result<Self>;
}

impl common_io::Stage<common_io::TensorInputPacket, common_io::RawDetectionsPacket> for OrtTrtEngine {
    fn process(&mut self, tin: TensorInputPacket) -> RawDetectionsPacket;
}

/// Optional convenience: read TOML and construct engine directly.
pub fn from_path(cfg_path: &str) -> anyhow::Result<OrtTrtEngine>;
```

---

## 5) Session Initialization (TensorRT EP)

1. Create **Env** + **SessionOptions**. Enable highest graph optimization.
2. Append **TensorRT EP** with options (typical mapping):
   - `trt_fp16_enable = true` (if `cfg.fp16`)
   - `trt_engine_cache_enable = true`
   - `trt_engine_cache_path = cfg.engine_cache`
   - `trt_timing_cache_enable = true`
   - `trt_timing_cache_path = cfg.timing_cache`
   - `trt_max_workspace_size = cfg.max_workspace_mb * 1024 * 1024`
   - `device_id = cfg.device`
   - (optional) `trt_force_sequential_engine_build = true` (stabilize build time)
   - (optional) `trt_dla_enable`, `trt_dla_core` if you target DLA
3. (Optional fallback) Append **CUDA EP** next, so ORT can fall back when TRT cannot build.
4. Create **Session** from the ONNX file.
5. Query **input/output names** & **types** (cache strings).
6. Create **IoBinding** and **CUDA OrtMemoryInfo** for reuse.
7. **Warmup** N runs (see §8).

> Keep provider-option mapping isolated (e.g., `trt_provider_options_from(cfg)`), so you can adapt when ORT changes option names.

---

## 6) I/O Binding (Zero-/Low‑copy)

- Reuse **one `OrtIoBinding`** object across frames.
- Create **CUDA `OrtMemoryInfo`** (device id = `cfg.device`).
- **Bind input (GPU)**
  - Provide: input name, shape `[1,3,H,W]`, element type **FLOAT16** (or FLOAT if FP32).
  - Pointer: `tin.data.ptr`.
- **Bind outputs**
  - If shapes are known: pre‑allocate GPU buffers (cudaMalloc) and bind as outputs.
  - If shapes vary: let ORT allocate on CUDA memory by binding with CUDA memory info (avoid CPU fallback).
- Run with `RunWithBinding(session, iobind)` (or the equivalent in your binding).

**Stream sync:** If ORT/TRT uses its own stream, synchronize via **CUDA events** so that:
- preprocess stream → **event** → ORT stream waits
- ORT stream → **event** → postprocess stream waits

Avoid `cudaDeviceSynchronize()` on the hot path.

---

## 7) Processing Flow

```rust
fn process(&mut self, tin: TensorInputPacket) -> RawDetectionsPacket {
    // 1) Validate device/dtype/shape
    assert_eq!(tin.desc.n, 1);
    assert!(matches!(tin.desc.dtype, DType::Fp16 | DType::Fp32));

    // 2) Clear previous bindings & bind current input
    ort::IoBinding::clear(self.iobind)?;
    ort::IoBinding::bind_input_gpu(self.iobind, &self.input_name, &tin, self.meminfo)?;

    // 3) Bind outputs (GPU-preferred). If unknown shapes, let ORT allocate on CUDA.
    ort::IoBinding::bind_output_gpu(self.iobind, &self.output_names, self.meminfo)?;

    // 4) Run
    ort::session_run_with_binding(self.session, self.iobind)?;

    // 5) Collect output tensors (names, shapes, dtype, device pointers)
    let outs = ort::IoBinding::get_outputs(self.iobind, self.meminfo)?;

    // 6) Wrap into RawDetectionsPacket for postprocess
    RawDetectionsPacket::from_ort_outputs(tin.from, outs)
}
```

> The `ort::*` functions above are your thin wrappers over the ORT C API, e.g. `OrtCreateIoBinding`, `OrtBindInput`, `OrtRunWithBinding`, `OrtGetBoundOutputValues`, etc.

---

## 8) Warmup & Caches

- **Warmup**: Run `cfg.warmup_runs` times with a real or synthetic `TensorInputPacket` of the production shape.  
  This forces TensorRT to **build engines** and populate **timing cache**.
- **Engine cache**: Ensure the path exists and is writable; later startups will reuse it.
- **Timing cache**: Reused across engine builds to speed up plan creation, especially with dynamic shapes.

---

## 9) Dynamic Shape (Optional)

If your model supports dynamic shapes:

- Provide `min/opt/max` in config.  
- Ensure preprocessing emits one of the profiled shapes.  
- On init, set TRT EP profile corresponding to your target shapes (ORT option names vary by version; keep the mapping localized).  
- **Avoid switching shapes frequently** at runtime; it may trigger re‑optimization or reallocation.

---

## 10) Telemetry

Emit at least:
- `inference` (overall)
- Optional sub‑spans:
  - `inference.bind`
  - `inference.run`
  - `inference.get_outputs`

Use GPU events (if possible) for accurate device‑side timing and host‑side `telemetry` for wall‑clock.

---

## 11) Unit Tests (suggested)

- **Session init**: verify TRT EP loads and `fp16_enable` is honored.
- **I/O binding**: bind a small GPU buffer and run → check output shape/type is as expected.
- **Performance**: after warmup, a `1x3x512x512 FP16` model on a mid‑range GPU should typically run in a few milliseconds (hardware‑dependent). Use relaxed thresholds.
- **Fallback**: if `enable_fallback_cuda = true`, simulate a TRT build failure and ensure CUDA EP is used (detect via logs or provider query).

Skeleton:
```rust
#[test]
fn ort_trt_session_starts_and_runs() {
    let cfg = test_cfg(); // points to a tiny fp16-capable model
    let mut eng = OrtTrtEngine::new(&cfg).unwrap();
    let tin = testsupport::make_dummy_tensor_gpu((1,3,512,512), DType::Fp16, cfg.device);
    let r = eng.process(tin);
    assert!(r.has_output("boxes"));
}
```

---

## 12) Playground

`apps/playgrounds/src/bin/infer_ort_trt_preview.rs`
```rust
use anyhow::Result;
use common_io::Stage;

fn main() -> Result<()> {
    let cfg_path = "configs/dev_1080p60_yuv422_fp16_trt.toml";
    let mut eng = inference_ort_trt::from_path(cfg_path)?;

    // Warmup is done inside new(); now run once and print metadata
    let tin = testsupport::make_dummy_tensor_gpu((1,3,512,512), DType::Fp16, 0);
    let _ = telemetry::time_stage("inference", &mut eng, tin);

    println!("inputs:  {}", eng.input_name());
    println!("outputs: {:?}", eng.output_names());
    Ok(())
}
```

---

## 13) Best Practices

- Prefer **FP16 NCHW** inputs to avoid hidden conversions.
- Keep a **single ORT env**; create **one session per model**.
- Use **IOBinding** to keep data on **GPU**; avoid implicit D2H/H2D.
- Always **warm up**.
- Enable **engine/timing caches** and store them on a persistent volume.
- Avoid `cudaDeviceSynchronize()` in hot path — use **CUDA events** + stream waits.
- Handle **errors with context** (provider options, device id, shapes) for faster diagnosis.

---

## 14) Common Pitfalls

- Input dtype/layout mismatch (NHWC vs NCHW, FP32 vs FP16) → implicit copies.
- Not using IOBinding → ORT allocates on CPU by default (adds D2H/H2D).
- Engine cache path unwritable → rebuild cost at every start.
- Reading outputs before the run is complete → missing event sync.
- Dynamic shape out of profile → tensor shape errors or slow builds.

---

## 15) Acceptance Checklist

- [ ] ORT session created with **TensorRT EP** (`fp16_enable` active).
- [ ] **IOBinding** binds GPU input pointer and GPU outputs (or explicit CPU outputs by design).
- [ ] **Warmup** performed; **engine cache** and **timing cache** enabled.
- [ ] `process()` returns `RawDetectionsPacket` containing output tensors (shape/dtype/device info).
- [ ] Telemetry present for `inference` (and optional sub‑spans).
- [ ] Unit tests & playground run successfully on the target GPU.

---

**End of spec.**
