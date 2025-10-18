# DeepGIBox – Full Instruction (v1)

DeepGIBox is a **Rust workspace** for a real‑time video AI overlay pipeline. It runs **in one process** and supports **Blackmagic DeckLink** capture at **1080p60** and **4Kp30** with **8‑bit YUV422 (YUV422_8)**, plus NV12/BGRA8. CUDA + TensorRT are used for preprocessing/inference; output uses internal keying.

## Quick Start
```bash
cargo build

แ
cargo run -p runner -- --config configs/dev_1080p60_yuv422_fp16_trt.toml
# or
cargo run -p runner -- --config configs/dev_4k30_yuv422_fp16_trt.toml
# playgrounds
cargo run -p playgrounds --bin capture_preview
cargo run -p playgrounds --bin preprocessing_1080p60_test
```
Requirements: Rust 1.78+, DeckLink SDK 15.0, CUDA 12.x, TensorRT 10.x, NVIDIA driver. (Optional) SDL2/OpenCV for previews.

## Structure
```
deepgibox/
  Cargo.toml
  INSTRUCTION.md
  configs/
    dev_1080p60_yuv422_fp16_trt.toml
    dev_4k30_yuv422_fp16_trt.toml
  apps/
    runner/ (binary)
    playgrounds/ (src/bin/*.rs)
  crates/
    common_io/ decklink_input/ preprocess_cuda/ inference/ postprocess/
    overlay_plan/ overlay_render/ decklink_output/ telemetry/ config/
  testsupport/
```
Naming: package=kebab-case, crate/file=snake_case, Types=PascalCase, fn/fields=snake_case, binaries in `src/bin` are snake_case.

## Pipeline & I/O
```
DeckLinkInput
  → RawFramePacket{ meta{w,h,pixfmt(BGRA8/NV12/YUV422_8), colorspace(BT709/...),
                        stride_bytes, frame_idx, pts_ns,t_capture_ns}, data }
  → PreprocessCUDA (YUV/NV12/BGRA → RGB + resize + normalize, FP16/FP32)
      → TensorInputPacket{ 1×3×H×W, dtype(Fp16/Fp32), device=GPU }
  → Inference (TRT / ORT-TRT)
      → RawDetectionsPacket **FIX Raw Detection i added raw_output to use the data in postprocessing
  → Postprocess (decode + threshold + NMS + optional tracking)
      → DetectionsPacket{ boxes, score, class_id, track_id? }
  → OverlayPlan (ops)
      → OverlayPlanPacket{ ops, canvas=(W,H) }
  → OverlayRender (ARGB8+alpha)
      → OverlayFramePacket
  → DeckLinkOutput (internal keying)
```
Guidelines: keep data on GPU; pass `FrameMeta` through; support `YUV422_8` for both 1080p60 and 4Kp30.

## Configs
Two samples in `configs/` (1080p60 and 4Kp30) already filled with `pixfmt="YUV422_8"` and `colorspace="BT709"`.

## Latency
- In-module (CUDA events): h2d/kernels/d2d (implement inside CUDA modules).
- Per-stage: `telemetry::time_stage("stage", &mut stage, input)`.
- End-to-end: record start at frame begin in runner.

Example log:
```
[lat] capture=2.0ms preprocess=2.8ms inference=4.1ms postprocess=0.9ms overlay=0.7ms e2e=10.9ms
```

## Extend / Develop
Implement the real logic in each crate:
- `decklink_input`: open device/mode, read frame → `RawFramePacket` (fill `stride_bytes`, BT709).
- `preprocess_cuda`: YUV422/NV12/BGRA → RGB (BT.709), resize, normalize, output NCHW FP16/FP32 on GPU.
- `inference`: TensorRT/ORT-TRT engine → backend raw detections.
- `postprocess`: decode+NMS(+tracking).
- `overlay_plan` / `overlay_render` / `decklink_output`: build/compose and output.

Playgrounds under `apps/playgrounds/src/bin/*.rs` demonstrate wiring with stubs; add your own files for experiments.

