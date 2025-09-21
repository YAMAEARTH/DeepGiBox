```markdown
# DeepGI Pipeline – Standard I/O Packets (Guideline for AI tools)

This document defines the **standard I/O format** for each stage of the DeepGI pipeline.  
AI assistants (e.g., GitHub Copilot) can use this as a reference when generating code in Rust, C, or CUDA.

---

## Pipeline Overview

```mermaid
flowchart LR
  A[DeckLink Capture] --> B[Preprocess (CUDA)]
  B --> C[Inference (ONNXRuntime / TensorRT)]
  C --> D[Postprocess (NMS + Tracking)]
  D --> E[Overlay Plan]
  E --> F[Overlay Render (ARGB + Alpha)]
  F --> G[DeckLink Output (Internal Keying)]
```

---

## Packet Definitions

### 1. RawFramePacket
```rust
struct FrameMeta {
    source_id: u32,
    width: u32,
    height: u32,
    stride: u32,
    pixfmt: PixelFormat,    // BGRA8 | NV12 | P010
    colorspace: ColorSpace, // BT709 | BT2020 | SRGB
    pts_ns: u64,
    timecode: Option<u32>,
    seq_no: u64,
}

struct RawFramePacket {
    mem: MemLoc, // Cpu {ptr} | Cuda {device_ptr}
    meta: FrameMeta,
}
```

### 2. TensorInputPacket
```rust
struct TensorInputPacket {
    mem: TensorMem::Cuda { device_ptr: u64 },
    desc: TensorDesc { shape: [1,3,512,512], dtype: F16|F32, layout: "NCHW", colors: "RGB" },
    meta: FrameMeta,
}
```

### 3. RawDetectionsPacket
```rust
struct RawDetection {
    cx: f32, cy: f32, w: f32, h: f32,
    obj_conf: f32,
    class_conf: f32,
    class_id: i32,
}

struct RawDetectionsPacket {
    dets: Vec<RawDetection>,
    meta: FrameMeta,
}
```

### 4. DetectionsPacket
```rust
struct Detection {
    bbox: BBox { x1: f32, y1: f32, x2: f32, y2: f32 },
    score: f32,
    class_id: i32,
    track_id: Option<i64>,
    label: Option<String>,
}

struct DetectionsPacket {
    dets: Vec<Detection>,
    meta: FrameMeta,
}
```

### 5. OverlayPlanPacket
```rust
enum OverlayOp {
    Rect { bbox: BBox, thickness: i32, alpha: f32 },
    Text { x: i32, y: i32, text: String, font_px: i32, alpha: f32 },
    Line { x1: i32, y1: i32, x2: i32, y2: i32, thickness: i32, alpha: f32 },
}

struct OverlayPlanPacket {
    size: (u32, u32),
    ops: Vec<OverlayOp>,
    meta: FrameMeta,
}
```

### 6. KeyingPacket
```rust
struct OverlayFrame {
    mem: MemLoc, // Cpu or Cuda
    width: u32,
    height: u32,
    stride: u32,
    premultiplied: bool, // true
}

struct KeyingPacket {
    passthrough: RawFramePacket,
    overlay: OverlayFrame,
    meta: FrameMeta,
}
```

---

## Development Guidelines
- **Zero-Copy**: Prefer GPU memory (`Cuda device_ptr`) handoff between Preprocess → Inference.
- **Consistency**: Keep `seq_no` and `pts_ns` consistent across packets for sync.
- **Expandability**: New stages (e.g., calibration, color correction) should respect these I/O contracts.
- **Internal Keying**: Overlay resolution and framerate must match passthrough input.

---

## Example Usage (Rust, Pseudocode)
```rust
let frame: RawFramePacket = decklink.next_frame()?;
let tensor: TensorInputPacket = preproc.process(frame)?;
let raw_dets: RawDetectionsPacket = inference.run(tensor)?;
let dets: DetectionsPacket = postproc.process(raw_dets)?;
let plan: OverlayPlanPacket = planner.build(dets)?;
let overlay: OverlayFrame = renderer.render(plan)?;
let keying: KeyingPacket = KeyingPacket { passthrough: frame, overlay, meta: frame.meta };
decklink_output.send(keying)?;
```
```

