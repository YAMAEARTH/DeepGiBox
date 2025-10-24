# DeckLink Output Integration Guide

## Overview

The `decklink_output` module provides alpha compositing capabilities for the DeepGiBox pipeline. It supports two modes:

1. **Static PNG Overlay** - Load overlay from a PNG file (e.g., `foreground.png`)
2. **Pipeline Overlay** - Use dynamic overlay from `overlay_render` stage

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DECKLINK OUTPUT MODULE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────────────────────────────────────────────┐         │
│  │              PipelineCompositor                         │         │
│  └────────────────────────────────────────────────────────┘         │
│                         │                                            │
│         ┌───────────────┴───────────────┐                           │
│         │                               │                            │
│         ▼                               ▼                            │
│  ┌─────────────────┐           ┌──────────────────┐                │
│  │ Static PNG Mode │           │ Pipeline Mode    │                 │
│  │                 │           │                  │                 │
│  │ • foreground.png│           │ • OverlayFrame   │                │
│  │ • Load once     │           │ • Update each    │                │
│  │ • GPU upload    │           │   frame          │                │
│  └────────┬────────┘           └────────┬─────────┘                │
│           │                              │                           │
│           └───────────┬──────────────────┘                           │
│                       │                                              │
│                       ▼                                              │
│           ┌─────────────────────┐                                   │
│           │  OutputSession      │                                   │
│           │  - CUDA composite   │                                   │
│           │  - Alpha blend      │                                   │
│           └──────────┬──────────┘                                   │
│                      │                                               │
│                      ▼                                               │
│           ┌─────────────────────┐                                   │
│           │  BGRA Output        │                                   │
│           │  (GPU Memory)       │                                   │
│           └─────────────────────┘                                   │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Usage Examples

### Mode 1: Static PNG Overlay

Use this when overlay graphics don't change (e.g., logo, watermark).

```rust
use decklink_output::{CompositorBuilder, PipelineCompositor};
use common_io::RawFramePacket;

// Create compositor with static PNG
let mut compositor = CompositorBuilder::new(1920, 1080)
    .with_png("foreground.png")
    .build()?;

// In processing loop
loop {
    let video_frame: RawFramePacket = capture_from_decklink()?;
    
    // Composite PNG over video
    let output = compositor.composite(&video_frame)?;
    
    // output is BGRA on GPU, ready for display/encoding
    send_to_output(output)?;
}
```

### Mode 2: Pipeline Overlay (Dynamic)

Use this when overlay changes each frame (e.g., bounding boxes, labels).

```rust
use decklink_output::{CompositorBuilder, PipelineCompositor};
use common_io::{RawFramePacket, OverlayFramePacket};

// Create compositor for pipeline overlay
let mut compositor = CompositorBuilder::new(1920, 1080)
    .with_pipeline()
    .build()?;

// In processing loop
loop {
    let video_frame: RawFramePacket = capture_from_decklink()?;
    
    // Get overlay from pipeline (bounding boxes, etc.)
    let overlay: OverlayFramePacket = render_stage.process(detections)?;
    
    // Update overlay
    compositor.update_overlay(&overlay)?;
    
    // Composite overlay over video
    let output = compositor.composite(&video_frame)?;
    
    send_to_output(output)?;
}
```

### Mode 3: Hybrid (Switch between modes)

You can create separate compositors for different use cases:

```rust
// Logo overlay (static)
let mut logo_compositor = CompositorBuilder::new(1920, 1080)
    .with_png("logo.png")
    .build()?;

// Detection overlay (dynamic)
let mut detection_compositor = CompositorBuilder::new(1920, 1080)
    .with_pipeline()
    .build()?;

// Choose compositor based on mode
let output = match mode {
    Mode::LogoOnly => logo_compositor.composite(&video_frame)?,
    Mode::Detections => {
        detection_compositor.update_overlay(&overlay)?;
        detection_compositor.composite(&video_frame)?
    }
};
```

## Integration with DeepGiBox Pipeline

### Full Pipeline Example

```rust
use decklink_input::CaptureSession;
use decklink_output::{CompositorBuilder, PipelineCompositor};
use inference_v2::InferenceStage;
use postprocess::PostprocessStage;
use overlay_render::RenderStage;

// Setup pipeline stages
let mut capture = CaptureSession::new(0)?;
let mut inference = InferenceStage::new("model.trt")?;
let mut postprocess = PostprocessStage::new()?;
let mut overlay_render = RenderStage::new()?;

// Setup compositor (choose mode)
let use_static_png = true;
let mut compositor = if use_static_png {
    CompositorBuilder::new(1920, 1080)
        .with_png("foreground.png")
        .build()?
} else {
    CompositorBuilder::new(1920, 1080)
        .with_pipeline()
        .build()?
};

// Processing loop
loop {
    // 1. Capture video frame (UYVY on GPU)
    let raw_frame = capture.next_frame()?;
    
    // 2. Run inference (optional)
    let tensor = preprocess.process(raw_frame.clone())?;
    let detections = inference.process(tensor)?;
    let objects = postprocess.process(detections)?;
    
    // 3. Render overlay (if using pipeline mode)
    if !use_static_png {
        let overlay = overlay_render.process(objects)?;
        compositor.update_overlay(&overlay)?;
    }
    
    // 4. Composite overlay onto video
    let composited = compositor.composite(&raw_frame)?;
    
    // 5. Output (display, encode, or send to SDI)
    output_display(composited)?;
}
```

## Performance Notes

- **Static PNG Mode**: ~0.5ms per frame @ 1080p60
  - PNG loaded once at startup
  - No CPU overhead during runtime
  
- **Pipeline Mode**: ~0.8ms per frame @ 1080p60
  - Includes ARGB→BGRA conversion if overlay is on CPU
  - Zero overhead if overlay already on GPU

## Configuration

Add to your `Cargo.toml`:

```toml
[dependencies]
decklink_output = { path = "crates/decklink_output" }
common_io = { path = "crates/common_io" }
```

## GPU Requirements

- NVIDIA GPU with CUDA 12.x support
- CUDA Toolkit installed
- DeckLink SDK for video I/O

## Memory Layout

### Input (DeckLink Capture)
```
UYVY 4:2:2 (packed)
[U Y V Y] [U Y V Y] ...
GPU memory via GPUDirect
```

### Overlay (PNG or Pipeline)
```
BGRA 8-bit (packed)
[B G R A] [B G R A] ...
GPU memory
```

### Output (Composited)
```
BGRA 8-bit (packed)
[B G R A] [B G R A] ...
GPU memory
```

## Error Handling

```rust
use decklink_output::OutputError;

match compositor.composite(&video_frame) {
    Ok(output) => {
        // Success
    }
    Err(OutputError::InvalidDimensions) => {
        eprintln!("Dimension mismatch!");
    }
    Err(OutputError::NullPointer) => {
        eprintln!("Invalid frame pointer!");
    }
    Err(e) => {
        eprintln!("Composite error: {}", e);
    }
}
```

## Debugging

### Download Output to CPU

```rust
// Download composited frame for inspection
let bgra_data = compositor.download_output()?;

// Save as PNG
use image::{RgbaImage, ImageBuffer};
let img = ImageBuffer::from_raw(
    compositor.dimensions().0,
    compositor.dimensions().1,
    bgra_data,
).unwrap();
img.save("debug_output.png")?;
```

### Check Source Mode

```rust
use decklink_output::OverlaySource;

match compositor.source() {
    OverlaySource::StaticImage(path) => {
        println!("Using static PNG: {}", path);
    }
    OverlaySource::Pipeline => {
        println!("Using dynamic pipeline overlay");
    }
}
```

## Troubleshooting

### Problem: "CUDA allocation failed"
**Solution**: Check available GPU memory
```bash
nvidia-smi
```

### Problem: "Invalid dimensions"
**Solution**: Ensure all buffers have same resolution
```rust
// Video frame must match compositor dimensions
assert_eq!(video_frame.meta.width, compositor.dimensions().0);
assert_eq!(video_frame.meta.height, compositor.dimensions().1);
```

### Problem: Overlay not visible
**Solution**: Check alpha channel in PNG
```bash
# Verify PNG has alpha channel
file foreground.png
# Should show: "PNG image data, 1920 x 1080, 8-bit/color RGBA"
```

## See Also

- [OUTPUT.md](OUTPUT.md) - Low-level OutputSession API
- [CUDA Kernel](../../keying/keying.cu) - Composite implementation
- [Common IO](../common_io/) - Packet types and memory management
