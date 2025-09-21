# DeepGI Pipeline - Separated Capture and Preview

This project implements a modular pipeline system for DeckLink capture with separated capture and preview stages, following the standard I/O packet specification for the DeepGI pipeline.

## Architecture Overview

The system is now organized into separate modules:

```
┌─────────────┐    ┌─────────────────┐    ┌─────────────┐
│   Capture   │───▶│   Processing    │───▶│   Preview   │
│   Module    │    │     Stage       │    │   Module    │
└─────────────┘    └─────────────────┘    └─────────────┘
     │                       │                     │
     ▼                       ▼                     ▼
RawFramePacket      RawFramePacket        Display/Render
```

### Key Features

- **Standard I/O Packets**: All data flows through standardized packet types (`RawFramePacket`, `DetectionsPacket`, etc.)
- **Modular Design**: Capture, processing, and preview are completely separate and can be used independently
- **Pipeline Orchestration**: Automatic management of data flow between stages
- **Custom Processing**: Easy to add custom processing stages for AI inference, filtering, etc.
- **Zero-Copy Optimization**: GPU memory handoff support for minimal latency
- **Thread Safety**: All stages run in separate threads with proper synchronization

## Modules

### 1. Packets (`src/packets.rs`)
Defines the standard I/O packet format:
- `RawFramePacket`: Raw video frames from capture
- `TensorInputPacket`: Preprocessed data for AI inference
- `DetectionsPacket`: Object detection results
- `OverlayPlanPacket`: Overlay rendering instructions
- `KeyingPacket`: Final composite output

### 2. Capture (`src/capture.rs`)
DeckLink capture functionality:
- `DeckLinkCapture`: Main capture class
- `CaptureStage`: Pipeline-compatible capture stage
- Outputs `RawFramePacket` instances
- Supports multiple DeckLink devices
- Thread-safe frame delivery

### 3. Preview (`src/preview.rs`)
OpenGL preview functionality:
- `DeckLinkPreview`: Hardware-accelerated preview
- `PreviewStage`: Pipeline-compatible preview stage
- Consumes `RawFramePacket` instances
- Real-time performance monitoring
- Optional CPU-based rendering

### 4. Pipeline (`src/pipeline.rs`)
Pipeline orchestration:
- `Pipeline`: Main pipeline coordinator
- `PipelineBuilder`: Fluent API for pipeline construction
- `ProcessingStage`: Trait for custom processing stages
- Automatic threading and data flow management

## Usage Examples

### Basic Capture and Preview

```rust
use decklink_rust::{PipelineBuilder, CaptureConfig, PreviewConfig, ColorSpace};

// Configure capture
let capture_config = CaptureConfig {
    device_index: 0,
    source_id: 1,
    expected_colorspace: ColorSpace::BT709,
};

// Configure preview
let preview_config = PreviewConfig {
    enable_stats: true,
    stats_interval: Duration::from_secs(1),
};

// Build and start pipeline
let mut pipeline = PipelineBuilder::new()
    .with_capture_config(capture_config)
    .with_preview_config(preview_config)
    .build();

pipeline.initialize()?; // Must be called from OpenGL context
pipeline.start()?;

// In render loop
if pipeline.render() {
    // Frame was rendered
    gl_context.swap_buffers();
}
```

### Custom Processing Stage

```rust
use decklink_rust::{ProcessingStage, RawFramePacket, PipelineError};

pub struct AIInferenceStage {
    model: YourAIModel,
}

impl ProcessingStage for AIInferenceStage {
    fn process(&mut self, input: RawFramePacket) -> Result<RawFramePacket, PipelineError> {
        // Run AI inference on the frame
        let detections = self.model.infer(&input)?;
        
        // Store results in frame metadata or separate channel
        // Return modified or original frame
        Ok(input)
    }

    fn name(&self) -> &str {
        "ai_inference"
    }
}

// Add to pipeline
let mut pipeline = PipelineBuilder::new()
    .with_processing_stage(Box::new(AIInferenceStage::new()))
    .build();
```

### Headless Processing

```rust
// Create capture-only pipeline for headless processing
let mut capture = CaptureStage::new(capture_config);
capture.start()?;

loop {
    if let Some(frame) = capture.get_next_frame()? {
        // Process frame
        process_frame(frame);
    }
}
```

## Building and Running

### Prerequisites
- Rust toolchain (1.70+)
- DeckLink SDK headers in `include/` directory
- DeckLink drivers installed
- OpenGL development libraries

### Build Commands

```bash
# Build all binaries
cargo build

# Build specific binary
cargo build --bin capture_preview_gl

# Run with preview (requires DeckLink device)
cargo run --bin capture_preview_gl

# Run headless example
cargo run --bin pipeline_example

# List available devices
cargo run --bin devicelist
```

### Binary Overview

1. **`devicelist`**: Lists available DeckLink devices
2. **`capture_preview_gl`**: Full capture and preview with OpenGL rendering
3. **`pipeline_example`**: Demonstrates separated pipeline with custom processing

## Standard I/O Packet Flow

Following the DeepGI pipeline specification:

```rust
// Example packet flow
let frame: RawFramePacket = capture_stage.get_next_frame()?;
let tensor: TensorInputPacket = preprocess_stage.process(frame)?;
let raw_dets: RawDetectionsPacket = inference_stage.run(tensor)?;
let dets: DetectionsPacket = postprocess_stage.process(raw_dets)?;
let plan: OverlayPlanPacket = overlay_planner.build(dets)?;
let overlay: OverlayFrame = renderer.render(plan)?;
let keying: KeyingPacket = KeyingPacket { 
    passthrough: frame, 
    overlay, 
    meta: frame.meta 
};
output_stage.send(keying)?;
```

## Performance Considerations

- **GPU Memory**: Use CUDA device pointers for zero-copy GPU processing
- **Threading**: Each stage runs in its own thread for maximum throughput
- **Buffering**: Configurable buffer sizes to handle processing latency
- **Frame Dropping**: Automatic frame dropping when processing can't keep up

## Next Steps for Pipeline Integration

This separated architecture enables easy integration of additional pipeline stages:

1. **Preprocessing Stage**: Convert frames to tensor format for AI models
2. **Inference Stage**: Run YOLO/other object detection models
3. **Postprocessing Stage**: Apply NMS, tracking, and filtering
4. **Overlay Stage**: Generate visual overlays for detected objects
5. **Output Stage**: Composite and output to DeckLink or streaming

Each stage follows the same `ProcessingStage` trait interface and can be easily swapped or modified.

## Contributing

When adding new processing stages:
1. Implement the `ProcessingStage` trait
2. Follow the standard packet format
3. Add appropriate error handling
4. Include performance monitoring
5. Write tests for your stage

## License

This project is part of the DeepGI pipeline system. See main project for licensing information.
