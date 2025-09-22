# DeepGI Pipeline - Separated Capture and Preview

This project implements a modular pipeline system for DeckLink capture with separated capture and preview stages, following the standard I/O packet specification for the DeepGI pipeline.

## Architecture Overview

The system is now organized into separate modules following the complete DeepGI pipeline flow:

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│  DeckLink   │───▶│Preprocessing │───▶│ Inference   │───▶│Postprocessing│───▶│   Overlay   │───▶│ DeckLink    │
│  Capture    │    │   (CUDA)     │    │(ONNX/TRT)   │    │ (NMS+Track)  │    │   Render    │    │  Output     │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
       │                    │                   │                    │                   │                │
       ▼                    ▼                   ▼                    ▼                   ▼                ▼
RawFramePacket    TensorInputPacket   RawDetectionsPacket   DetectionsPacket   OverlayPlanPacket   KeyingPacket
```

### Complete Pipeline Flow

The DeepGI pipeline processes video frames through these standardized stages:

1. **Capture Stage** → `RawFramePacket`: Raw video frames from DeckLink devices
2. **Preprocessing Stage** → `TensorInputPacket`: GPU-optimized tensor data for ML inference  
3. **Inference Stage** → `RawDetectionsPacket`: Raw AI model outputs (bounding boxes, confidences)
4. **Postprocessing Stage** → `DetectionsPacket`: Filtered detections with tracking IDs
5. **Overlay Planning Stage** → `OverlayPlanPacket`: UI overlay rendering instructions
6. **Output Stage** → `KeyingPacket`: Final composite for DeckLink internal keying

### Key Features

- **Standard I/O Packets**: All data flows through standardized packet types following the DeepGI specification
- **Complete AI Pipeline**: Full chain from capture → preprocessing → inference → postprocessing → overlay → output
- **GPU Optimization**: Zero-copy CUDA memory handoff for maximum performance
- **Modular Design**: Each stage is independent and can be swapped or modified
- **Real-time Processing**: Optimized for low-latency video processing
- **Thread Safety**: All stages run in separate threads with proper synchronization

## Modules

### 1. Packets (`src/packets.rs`)
Defines the complete standard I/O packet format following the DeepGI specification:
- `RawFramePacket`: Raw video frames from capture (BGRA8, NV12, P010 formats)
- `TensorInputPacket`: GPU tensor data for ML inference (NCHW layout, F16/F32)
- `RawDetectionsPacket`: Raw AI model outputs (center-x, center-y, width, height format)
- `DetectionsPacket`: Processed detections with bounding boxes and tracking IDs
- `OverlayPlanPacket`: UI overlay rendering instructions (rectangles, text, lines)
- `KeyingPacket`: Final composite output for DeckLink internal keying

### 2. Capture (`src/capture.rs`)
DeckLink capture functionality:
- `DeckLinkCapture`: Main capture class supporting multiple input formats
- `CaptureStage`: Pipeline-compatible capture stage
- Outputs `RawFramePacket` instances with proper metadata
- Supports BT709/BT2020 color spaces and various pixel formats
- Thread-safe frame delivery with sequence numbering

### 3. Preprocessing (`src/preprocessing.rs`)
CUDA-accelerated preprocessing stage:
- `PreprocessingStage`: Converts `RawFramePacket` → `TensorInputPacket`
- GPU memory optimization for zero-copy processing
- Pan/zoom/resize operations for region-of-interest processing
- Supports multiple input formats (BGRA8, NV12, P010)
- Color space conversion and normalization for ML models

### 4. Inference (`src/inference.rs`) *[Planned]*
AI model inference stage:
- `InferenceStage`: Converts `TensorInputPacket` → `RawDetectionsPacket`
- ONNXRuntime and TensorRT backend support
- GPU memory management for optimal performance
- Batch processing support for multiple frames
- Model-specific preprocessing and output formatting

### 5. Postprocessing (`src/postprocessing.rs`) *[Planned]*
Detection processing and tracking:
- `PostprocessingStage`: Converts `RawDetectionsPacket` → `DetectionsPacket`
- Non-Maximum Suppression (NMS) for duplicate removal
- Multi-object tracking with persistent IDs
- Confidence threshold filtering
- Bounding box format conversion and validation

### 6. Overlay (`src/overlay.rs`) *[Planned]*
UI overlay generation:
- `OverlayStage`: Converts `DetectionsPacket` → `OverlayPlanPacket`
- Configurable visualization styles (colors, fonts, thickness)
- Dynamic label generation with confidence scores
- Track ID display and trail visualization
- Performance metrics overlay (FPS, latency)

### 7. Output (`src/output.rs`) *[Planned]*
DeckLink output with internal keying:
- `OutputStage`: Converts `OverlayPlanPacket` → `KeyingPacket`
- ARGB overlay rendering with alpha blending
- DeckLink internal keying support
- Frame synchronization and timing control
- Multiple output format support

### 8. Preview (`src/preview.rs`)
**Real-time Video Visualization and Testing Framework**

The preview module provides comprehensive video packet visualization capabilities for testing, debugging, and real-time monitoring of pipeline data flow. It serves as a critical tool for development and production quality assurance.

#### Core Components

##### `DeckLinkPreview` - Main Preview Engine
The primary preview system that handles real-time video frame display with OpenGL acceleration:

```rust
pub struct DeckLinkPreview {
    config: PreviewConfig,
    is_initialized: bool,
    is_enabled: bool,
    frame_receiver: Option<mpsc::Receiver<RawFramePacket>>,
    preview_thread: Option<thread::JoinHandle<()>>,
    shutdown_flag: Arc<Mutex<bool>>,
    stats: Arc<Mutex<PreviewStats>>,
    current_frame: Arc<Mutex<Option<RawFramePacket>>>,
}
```

**Key Features:**
- **OpenGL Hardware Acceleration**: Native GPU-accelerated frame rendering
- **Multi-threaded Architecture**: Separate threads for frame consumption and rendering
- **Real-time Statistics**: FPS, latency, and frame sequence monitoring
- **Thread-safe Operation**: Safe multi-threaded access to frame data

##### `PreviewStage` - Pipeline Integration
Simplified wrapper for seamless integration into the DeepGI pipeline:

```rust
pub struct PreviewStage {
    preview: DeckLinkPreview,
    frame_sender: Option<mpsc::Sender<RawFramePacket>>,
}
```

**Pipeline Integration:**
- **Standard I/O Compliance**: Consumes `RawFramePacket` instances
- **Channel-based Communication**: Thread-safe frame delivery via mpsc channels
- **Automatic Resource Management**: Handles initialization and cleanup

##### `CpuFrameRenderer` - Software Rendering
CPU-based frame renderer for systems without OpenGL or for specialized processing:

```rust
pub struct CpuFrameRenderer {
    target_width: u32,
    target_height: u32,
}
```

**Capabilities:**
- **Format Conversion**: BGRA8 → RGBA conversion for display systems
- **Software Rendering**: No GPU dependencies for headless systems
- **Memory Management**: Direct memory access to frame data

#### Video Packet Testing and Validation

##### Frame Property Analysis
The preview system provides comprehensive frame analysis for testing and quality assurance:

```rust
// Example usage for packet testing
let preview_config = PreviewConfig {
    enable_stats: true,
    stats_interval: Duration::from_secs(1),
};

let mut preview = DeckLinkPreview::new(preview_config);
preview.initialize_gl()?;

// Test frame packet flow
let (sender, receiver) = mpsc::channel();
preview.start(receiver)?;

// Send test frames for analysis
for raw_frame in test_frames {
    sender.send(raw_frame)?;
    
    // Monitor preview statistics
    let stats = preview.get_stats();
    println!("Preview FPS: {:.1}, Latency: {:.2}ms", stats.fps, stats.latency_ms);
    
    // Check frame sequence integrity
    assert_eq!(stats.last_seq, expected_sequence_number);
}
```

##### Real-time Video Quality Monitoring
Monitor video packet properties in real-time during capture or processing:

```rust
// Quality monitoring setup
impl DeckLinkPreview {
    fn frame_consumer_loop(
        receiver: mpsc::Receiver<RawFramePacket>,
        shutdown_flag: Arc<Mutex<bool>>,
        stats: Arc<Mutex<PreviewStats>>,
        current_frame: Arc<Mutex<Option<RawFramePacket>>>,
        config: PreviewConfig,
    ) {
        let mut last_stats_update = Instant::now();
        let mut frames_since_stats = 0u32;
        let mut last_seq = 0u64;

        loop {
            match receiver.recv_timeout(Duration::from_millis(10)) {
                Ok(packet) => {
                    // Validate packet properties
                    assert_eq!(packet.meta.pixfmt, PixelFormat::BGRA8);
                    assert!(packet.meta.width > 0 && packet.meta.height > 0);
                    
                    // Check for frame drops (sequence gaps)
                    if packet.meta.seq_no != last_seq + 1 && last_seq != 0 {
                        println!("Warning: Frame drop detected! Expected {}, got {}", 
                                last_seq + 1, packet.meta.seq_no);
                    }
                    
                    // Update current frame for rendering
                    {
                        let mut current = current_frame.lock().unwrap();
                        *current = Some(packet.clone());
                    }
                    
                    // Calculate real-time statistics
                    frames_since_stats += 1;
                    last_seq = packet.meta.seq_no;
                    
                    // Periodic performance reporting
                    if config.enable_stats && last_stats_update.elapsed() >= config.stats_interval {
                        let elapsed = last_stats_update.elapsed();
                        let fps = frames_since_stats as f64 / elapsed.as_secs_f64();
                        let latency_ns = unsafe { decklink_preview_gl_last_latency_ns() };
                        let latency_ms = latency_ns as f64 / 1_000_000.0;

                        println!("Preview: fps: {:.1}, latency: {:.2} ms", fps, latency_ms);
                        
                        frames_since_stats = 0;
                        last_stats_update = Instant::now();
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                _ => continue,
            }
        }
    }
}
```

#### Testing Applications

##### 1. Capture Pipeline Validation
Test video capture quality and timing:

```rust
// Test capture pipeline with preview
let capture_config = CaptureConfig {
    device_index: 0,
    source_id: 1,
    expected_colorspace: ColorSpace::BT709,
};

let preview_config = PreviewConfig {
    enable_stats: true,
    stats_interval: Duration::from_millis(500),
};

let mut pipeline = PipelineBuilder::new()
    .with_capture_config(capture_config)
    .with_preview_config(preview_config)
    .build();

pipeline.initialize()?;
pipeline.start()?;

// Monitor capture quality
loop {
    let preview_stats = pipeline.get_preview_stats();
    let pipeline_stats = pipeline.get_stats();
    
    // Validate capture performance
    assert!(preview_stats.fps > 25.0, "Frame rate too low: {}", preview_stats.fps);
    assert!(preview_stats.latency_ms < 50.0, "Latency too high: {}ms", preview_stats.latency_ms);
    
    // Check for frame drops
    if pipeline_stats.frames_dropped > 0 {
        println!("Warning: {} frames dropped", pipeline_stats.frames_dropped);
    }
    
    std::thread::sleep(Duration::from_secs(1));
}
```

##### 2. Preprocessing Quality Validation
Verify preprocessing stage output with visual confirmation:

```rust
// Test preprocessing with preview feedback
let preprocessing_config = PreprocessingStageConfig {
    pan_x: 100,
    pan_y: 50,
    zoom: 1.5,
    target_size: (512, 512),
    debug: true,
};

let mut preprocessing_stage = PreprocessingStage::new(preprocessing_config);
let mut preview_stage = PreviewStage::new(PreviewConfig::default());

// Initialize preview for visual feedback
preview_stage.initialize_gl()?;
let preview_sender = preview_stage.start()?;

// Process frames and preview results
for raw_frame in capture_frames {
    // Process frame through preprocessing
    let tensor_input = preprocessing_stage.process(raw_frame.clone())?;
    
    // Convert back to frame for preview (if needed)
    let preview_frame = convert_tensor_to_frame(tensor_input)?;
    preview_sender.send(preview_frame)?;
    
    // Render preview window
    preview_stage.render();
    
    println!("Processed frame: {}x{} → tensor shape: {:?}", 
             raw_frame.meta.width, raw_frame.meta.height,
             tensor_input.desc.shape);
}
```

##### 3. End-to-End Pipeline Testing
Complete pipeline validation with visual monitoring:

```rust
// Full pipeline test with preview at multiple stages
let mut pipeline = PipelineBuilder::new()
    .with_capture_config(capture_config)
    .with_preprocessing_config(preprocessing_config)
    .with_processing_stage(Box::new(FrameInfoStage::new()))
    .with_preview_config(preview_config)
    .build();

pipeline.initialize()?;
pipeline.start()?;

// Test different scenarios
struct TestScenario {
    name: &'static str,
    duration: Duration,
    expected_fps: f64,
    max_latency_ms: f64,
}

let test_scenarios = vec![
    TestScenario {
        name: "Normal Operation",
        duration: Duration::from_secs(10),
        expected_fps: 30.0,
        max_latency_ms: 33.0,
    },
    TestScenario {
        name: "High Load",
        duration: Duration::from_secs(5),
        expected_fps: 25.0,
        max_latency_ms: 50.0,
    },
];

for scenario in test_scenarios {
    println!("Testing scenario: {}", scenario.name);
    
    let start_time = Instant::now();
    while start_time.elapsed() < scenario.duration {
        // Render preview window
        let rendered = pipeline.render();
        if rendered {
            // Window system buffer swap would happen here
        }
        
        // Collect statistics
        let preview_stats = pipeline.get_preview_stats();
        let pipeline_stats = pipeline.get_stats();
        
        // Validate performance against scenario expectations
        if preview_stats.fps < scenario.expected_fps {
            println!("Warning: FPS below expected: {:.1} < {:.1}", 
                    preview_stats.fps, scenario.expected_fps);
        }
        
        if preview_stats.latency_ms > scenario.max_latency_ms {
            println!("Warning: Latency above threshold: {:.1}ms > {:.1}ms", 
                    preview_stats.latency_ms, scenario.max_latency_ms);
        }
        
        std::thread::sleep(Duration::from_millis(16)); // ~60fps update rate
    }
    
    println!("Scenario {} completed successfully", scenario.name);
}
```

#### Integration with Existing Binaries

##### `capture_preview_gl.rs` Integration
The `capture_preview_gl` binary demonstrates complete preview integration:

```rust
// From capture_preview_gl.rs
fn main() -> Result<()> {
    // Initialize OpenGL context
    let event_loop = EventLoop::new()?;
    let gl_state = GlState::new(&event_loop)?;
    
    // Build pipeline with preview
    let mut pipeline = PipelineBuilder::new()
        .with_capture_config(capture_config)
        .with_preview_config(preview_config)
        .with_processing_stage(Box::new(FrameInfoStage::new()))
        .build();

    // Initialize pipeline (must be done in OpenGL context)
    pipeline.initialize()?;
    pipeline.start()?;

    // Main event loop with preview rendering
    event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
            // Render pipeline preview
            let rendered = pipeline.render();
            if rendered {
                gl_state.swap_buffers();
            }

            // Print performance statistics
            let pipeline_stats = pipeline.get_stats();
            let preview_stats = pipeline.get_preview_stats();
            
            println!(
                "Pipeline - Processed: {}, Dropped: {}, Preview FPS: {:.1}, Latency: {:.2}ms",
                pipeline_stats.frames_processed,
                pipeline_stats.frames_dropped,
                preview_stats.fps,
                preview_stats.latency_ms
            );
        }
        _ => {}
    })
}
```

#### Advanced Preview Features

##### Multi-Format Support
The preview system handles various pixel formats and color spaces:

```rust
impl CpuFrameRenderer {
    fn convert_frame_data(&self, data: &[u8], meta: &FrameMeta) -> Result<Vec<u8>, PipelineError> {
        match meta.pixfmt {
            PixelFormat::BGRA8 => {
                // Convert BGRA to RGBA for display
                let mut rgba_data = Vec::with_capacity(data.len());
                for chunk in data.chunks_exact(4) {
                    rgba_data.push(chunk[2]); // R
                    rgba_data.push(chunk[1]); // G  
                    rgba_data.push(chunk[0]); // B
                    rgba_data.push(chunk[3]); // A
                }
                Ok(rgba_data)
            }
            PixelFormat::NV12 => {
                // YUV to RGB conversion
                self.convert_nv12_to_rgba(data, meta)
            }
            PixelFormat::P010 => {
                // 10-bit YUV to RGB conversion
                self.convert_p010_to_rgba(data, meta)
            }
            _ => Err(PipelineError::Preview(
                format!("Unsupported pixel format: {:?}", meta.pixfmt)
            ))
        }
    }
}
```

##### Performance Monitoring Integration
Built-in performance metrics for comprehensive system monitoring:

```rust
#[derive(Debug, Clone)]
pub struct PreviewStats {
    pub frames_rendered: u32,    // Total frames displayed
    pub fps: f64,               // Real-time frame rate
    pub latency_ms: f64,        // Display latency
    pub last_seq: u64,          // Last frame sequence number
}

// Extended statistics for production monitoring
impl DeckLinkPreview {
    pub fn get_extended_stats(&self) -> ExtendedPreviewStats {
        ExtendedPreviewStats {
            basic: self.get_stats(),
            frame_drops: self.calculate_frame_drops(),
            memory_usage: self.get_memory_usage(),
            gpu_utilization: self.get_gpu_utilization(),
            thermal_state: self.get_thermal_state(),
        }
    }
}
```

#### Use Cases Summary

The preview system enables:

1. **Development Testing**: Visual confirmation of capture and processing pipeline functionality
2. **Quality Assurance**: Real-time monitoring of video quality, frame rates, and latency
3. **Debugging**: Frame-by-frame analysis of processing stages and data flow
4. **Performance Validation**: Comprehensive statistics for optimization and tuning
5. **Production Monitoring**: Real-time system health monitoring with visual feedback
6. **Integration Testing**: End-to-end pipeline validation with visual confirmation
7. **Hardware Validation**: DeckLink device testing and configuration verification

#### Native OpenGL Integration

The preview system uses native C bindings for optimal performance:

```rust
// Native C interface for hardware-accelerated preview
extern "C" {
    fn decklink_preview_gl_create() -> bool;
    fn decklink_preview_gl_initialize_gl() -> bool;
    fn decklink_preview_gl_enable() -> bool;
    fn decklink_preview_gl_render() -> bool;
    fn decklink_preview_gl_disable();
    fn decklink_preview_gl_destroy();
    fn decklink_preview_gl_seq() -> u64;
    fn decklink_preview_gl_last_timestamp_ns() -> u64;
    fn decklink_preview_gl_last_latency_ns() -> u64;
}
```

**Hardware Acceleration Benefits:**
- **Zero-Copy Rendering**: Direct GPU texture upload from video memory
- **High Performance**: Native OpenGL rendering without CPU overhead
- **Low Latency**: Hardware-accelerated display pipeline
- **Format Support**: Native support for multiple pixel formats (BGRA8, NV12, P010)

**Integration Requirements:**
- **OpenGL Context**: Must be initialized from main OpenGL thread
- **Hardware Support**: Requires OpenGL 3.3+ compatible graphics
- **Platform Specific**: Optimized for Linux with DeckLink SDK

#### Thread Architecture

The preview system uses a sophisticated multi-threaded architecture for optimal performance:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Capture       │    │   Preview        │    │   Render        │
│   Thread        │───▶│   Consumer       │───▶│   Thread        │
│                 │    │   Thread         │    │   (OpenGL)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
  RawFramePacket         Frame Buffer           OpenGL Render
   Production            Management              Commands
```

**Thread Responsibilities:**
1. **Capture Thread**: Produces `RawFramePacket` instances from DeckLink hardware
2. **Preview Consumer Thread**: Manages frame buffering and statistics collection
3. **Render Thread**: Handles OpenGL rendering and window system integration

#### Advanced Preview Features

##### Performance Monitoring
Real-time performance metrics collection and analysis:

```rust
impl DeckLinkPreview {
    // Get comprehensive performance statistics
    pub fn get_detailed_stats(&self) -> DetailedPreviewStats {
        DetailedPreviewStats {
            // Basic statistics
            frames_rendered: self.get_frames_rendered(),
            current_fps: self.calculate_fps(),
            average_latency_ms: self.get_average_latency(),
            
            // Advanced metrics
            frame_drops: self.count_frame_drops(),
            memory_usage_mb: self.get_memory_usage(),
            gpu_utilization: self.get_gpu_utilization(),
            
            // Timing analysis
            min_frame_time_ms: self.get_min_frame_time(),
            max_frame_time_ms: self.get_max_frame_time(),
            frame_time_variance: self.calculate_frame_time_variance(),
            
            // Quality metrics
            last_sequence_number: self.get_last_sequence(),
            sequence_gaps: self.count_sequence_gaps(),
            temporal_consistency: self.check_temporal_consistency(),
        }
    }
}
```

##### Multi-Format Visualization
Support for different video formats and color spaces:

```rust
// Format-specific preview optimization
match packet.meta.pixfmt {
    PixelFormat::BGRA8 => {
        // Direct GPU upload for BGRA format
        self.upload_bgra_texture(&packet)?;
    }
    PixelFormat::NV12 => {
        // YUV to RGB conversion on GPU
        self.upload_nv12_texture(&packet)?;
    }
    PixelFormat::P010 => {
        // 10-bit YUV processing
        self.upload_p010_texture(&packet)?;
    }
}

// Color space handling
match packet.meta.colorspace {
    ColorSpace::BT709 => self.apply_bt709_transform(),
    ColorSpace::BT2020 => self.apply_bt2020_transform(),
    ColorSpace::DCI_P3 => self.apply_dci_p3_transform(),
}
```

The preview system is essential for both development and production deployment, providing the visual feedback and monitoring capabilities necessary for reliable video processing pipeline operation.

### 9. Pipeline (`src/pipeline.rs`)
Pipeline orchestration and management:
- `Pipeline`: Main pipeline coordinator with threading
- `PipelineBuilder`: Fluent API for pipeline construction
- `ProcessingStage`: Generic trait for all processing stages
- Automatic data flow management between stages
- Performance monitoring and error handling

## Usage Examples

### Real-time Video Preview and Testing

```rust
use decklink_rust::{
    DeckLinkPreview, PreviewStage, PreviewConfig, CpuFrameRenderer,
    CaptureStage, CaptureConfig, ColorSpace, PixelFormat
};

// Basic preview setup for testing capture
let preview_config = PreviewConfig {
    enable_stats: true,
    stats_interval: Duration::from_secs(1),
};

let mut preview = DeckLinkPreview::new(preview_config);

// Initialize OpenGL preview (must be called from OpenGL context thread)
preview.initialize_gl()?;

// Setup capture for testing
let capture_config = CaptureConfig {
    device_index: 0,
    source_id: 1,
    expected_colorspace: ColorSpace::BT709,
};

let mut capture_stage = CaptureStage::new(capture_config);
let frame_receiver = capture_stage.start()?;

// Start preview with frame data
preview.start(frame_receiver)?;

// Main rendering loop (in OpenGL context)
loop {
    // Render current frame
    let rendered = preview.render();
    
    if rendered {
        // Swap OpenGL buffers (window system dependent)
        // gl_context.swap_buffers();
    }
    
    // Monitor preview statistics
    let stats = preview.get_stats();
    println!("Preview - FPS: {:.1}, Latency: {:.2}ms, Frames: {}", 
             stats.fps, stats.latency_ms, stats.frames_rendered);
    
    // Check for performance issues
    if stats.fps < 25.0 {
        println!("Warning: Low frame rate detected");
    }
    
    if stats.latency_ms > 50.0 {
        println!("Warning: High latency detected");
    }
    
    std::thread::sleep(Duration::from_millis(16)); // ~60fps
}
```

### Testing Video Packet Flow

```rust
// Test complete packet flow with preview validation
use decklink_rust::{RawFramePacket, TensorInputPacket, FrameMeta, PixelFormat};

// Create test frame packet
let test_frame = RawFramePacket::new_cpu(
    vec![0u8; 1920 * 1080 * 4], // BGRA data
    FrameMeta {
        source_id: 1,
        width: 1920,
        height: 1080,
        stride: 1920 * 4,
        pixfmt: PixelFormat::BGRA8,
        colorspace: ColorSpace::BT709,
        pts_ns: 0,
        timecode: None,
        seq_no: 1,
    }
);

// Setup preview for packet validation
let mut preview_stage = PreviewStage::new(PreviewConfig::default());
preview_stage.initialize_gl()?;
let preview_sender = preview_stage.start()?;

// Send test packet and verify rendering
preview_sender.send(test_frame.clone())?;
let rendered = preview_stage.render();

assert!(rendered, "Frame should be rendered successfully");

// Validate preview statistics
let stats = preview_stage.get_stats();
assert_eq!(stats.last_seq, 1, "Sequence number should match");
assert!(stats.frames_rendered > 0, "Should have rendered at least one frame");
```

### CPU Frame Rendering for Testing

```rust
// Test frame rendering without OpenGL dependency
let renderer = CpuFrameRenderer::new(1920, 1080);

// Convert frame packet to RGBA for testing
let rgba_data = renderer.render_to_rgba(&test_frame)?;

// Validate conversion
assert_eq!(rgba_data.len(), 1920 * 1080 * 4);

// Check pixel format conversion (BGRA → RGBA)
for chunk in rgba_data.chunks_exact(4) {
    // Validate RGBA format
    let r = chunk[0];
    let g = chunk[1]; 
    let b = chunk[2];
    let a = chunk[3];
    
    // Basic validation (all values should be valid bytes)
    assert!(r <= 255 && g <= 255 && b <= 255 && a <= 255);
}

// Save test frame for visual inspection
std::fs::write("test_frame_rgba.raw", &rgba_data)?;
println!("Test frame saved: {}x{} RGBA, {} bytes", 1920, 1080, rgba_data.len());
```

### Complete AI Processing Pipeline

```rust
use decklink_rust::{
    PipelineBuilder, CaptureConfig, PreprocessingStageConfig, 
    InferenceConfig, PostprocessingConfig, OverlayConfig, OutputConfig,
    ColorSpace, TensorDataType, TensorLayout
};

// Configure each stage of the pipeline
let capture_config = CaptureConfig {
    device_index: 0,
    source_id: 1,
    expected_colorspace: ColorSpace::BT709,
};

let preprocessing_config = PreprocessingStageConfig {
    pan_x: 0,
    pan_y: 0,
    zoom: 1.0,
    target_size: (512, 512),
    debug: false,
};

let inference_config = InferenceConfig {
    model_path: "yolo_v8_model.onnx",
    confidence_threshold: 0.5,
    use_tensorrt: true,
    batch_size: 1,
};

let postprocessing_config = PostprocessingConfig {
    nms_threshold: 0.4,
    max_detections: 100,
    enable_tracking: true,
    track_max_age: 30,
};

let overlay_config = OverlayConfig {
    show_bboxes: true,
    show_labels: true,
    show_confidence: true,
    show_track_ids: true,
    font_size: 16,
};

let output_config = OutputConfig {
    device_index: 1,
    internal_keying: true,
    overlay_alpha: 0.8,
};

// Build complete pipeline
let mut pipeline = PipelineBuilder::new()
    .with_capture_config(capture_config)
    .with_preprocessing_config(preprocessing_config)
    .with_inference_config(inference_config)
    .with_postprocessing_config(postprocessing_config)
    .with_overlay_config(overlay_config)
    .with_output_config(output_config)
    .build();

pipeline.start()?;

// Pipeline runs automatically in background threads
// Monitor performance
loop {
    let stats = pipeline.get_performance_stats();
    println!("FPS: {:.1}, Latency: {:.1}ms", stats.fps, stats.avg_latency_ms);
    std::thread::sleep(Duration::from_secs(1));
}
```

### Custom Processing Stage Implementation

```rust
use decklink_rust::{ProcessingStage, TensorInputPacket, RawDetectionsPacket, PipelineError};

pub struct CustomInferenceStage {
    model: YourMLModel,
    frame_count: u64,
}

impl ProcessingStage for CustomInferenceStage {
    type Input = TensorInputPacket;
    type Output = RawDetectionsPacket;
    
    fn process(&mut self, input: Self::Input) -> Result<Self::Output, PipelineError> {
        self.frame_count += 1;
        
        // Run inference on GPU tensor data
        let raw_outputs = self.model.infer_gpu(&input.mem)?;
        
        // Convert model outputs to standard detection format
        let detections = raw_outputs.into_iter().map(|output| {
            RawDetection {
                cx: output.center_x,
                cy: output.center_y,
                w: output.width,
                h: output.height,
                obj_conf: output.objectness,
                class_conf: output.class_confidence,
                class_id: output.class_id,
            }
        }).collect();
        
        Ok(RawDetectionsPacket {
            dets: detections,
            meta: input.meta,
        })
    }

    fn name(&self) -> &str {
        "custom_inference"
    }
}

// Add to pipeline
let mut pipeline = PipelineBuilder::new()
    .with_custom_inference_stage(Box::new(CustomInferenceStage::new()))
    .build();
```

### Individual Stage Testing

```rust
// Test preprocessing stage independently
let mut preprocessing_stage = PreprocessingStage::new(preprocessing_config);
let raw_frame: RawFramePacket = capture_stage.get_next_frame()?;
let tensor_input: TensorInputPacket = preprocessing_stage.process(raw_frame)?;

// Test inference stage with tensor data
let mut inference_stage = InferenceStage::new(inference_config);
let raw_detections: RawDetectionsPacket = inference_stage.process(tensor_input)?;

// Test postprocessing with raw detections
let mut postprocessing_stage = PostprocessingStage::new(postprocessing_config);
let detections: DetectionsPacket = postprocessing_stage.process(raw_detections)?;

// Test overlay generation
let mut overlay_stage = OverlayStage::new(overlay_config);
let overlay_plan: OverlayPlanPacket = overlay_stage.process(detections)?;

// Test output stage
let mut output_stage = OutputStage::new(output_config);
let keying_packet: KeyingPacket = output_stage.process(overlay_plan)?;
decklink_output.send(keying_packet)?;
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

1. **`devicelist`**: Lists available DeckLink input/output devices
2. **`capture_preview_gl`**: Real-time capture with OpenGL preview (no AI processing)
   - **Purpose**: Visual testing and validation of DeckLink capture functionality
   - **Preview Features**: Real-time video display, FPS monitoring, latency measurement
   - **Use Cases**: Hardware testing, signal validation, capture quality assessment
   
3. **`capture_preprocessing_preview`**: Enhanced capture and preprocessing with frame analysis
   - **Purpose**: Visual debugging of capture and preprocessing pipeline
   - **Preview Features**: Frame property analysis, tensor visualization, sample export
   - **Use Cases**: Preprocessing validation, quality assurance, performance optimization
   
4. **`pipeline_example`**: Demonstrates complete AI processing pipeline
   - **Purpose**: End-to-end pipeline testing with processing stages
   - **Preview Integration**: Optional visual monitoring of pipeline flow
   - **Use Cases**: Integration testing, performance benchmarking
   
5. **`headless_test`**: Pipeline testing without display requirements
   - **Purpose**: Automated testing and validation in server environments
   - **Preview Features**: Statistics-only monitoring (no visual display)
   - **Use Cases**: CI/CD testing, production monitoring, performance validation
   
6. **`preprocessing_test`**: Tests preprocessing stage independently
   - **Purpose**: Isolated preprocessing validation and optimization
   - **Preview Support**: Optional visual confirmation of preprocessing results
   - **Use Cases**: Algorithm testing, parameter tuning, quality validation
   
7. **`preprocessing_v2_test`**: Tests CV-CUDA accelerated preprocessing
   - **Purpose**: GPU-accelerated preprocessing performance testing
   - **Preview Features**: CUDA memory monitoring, tensor analysis
   - **Use Cases**: GPU optimization, memory usage analysis, performance benchmarking

#### Preview System Integration Across Binaries

Each binary can leverage the preview system for different testing and validation purposes:

```rust
// Common preview integration pattern
let preview_config = PreviewConfig {
    enable_stats: true,
    stats_interval: Duration::from_millis(500),
};

// Integration in different binaries:

// 1. capture_preview_gl: Full OpenGL preview
let mut pipeline = PipelineBuilder::new()
    .with_capture_config(capture_config)
    .with_preview_config(preview_config)
    .build();

// 2. capture_preprocessing_preview: Analysis + preview
let mut pipeline = CapturePreprocessingPipeline::new(config)?;
// Built-in frame analysis and visualization

// 3. headless_test: Statistics-only monitoring
let preview_stats = pipeline.get_preview_stats();
println!("Headless monitoring: FPS {:.1}, Latency {:.2}ms", 
         preview_stats.fps, preview_stats.latency_ms);

// 4. preprocessing_test: Optional visual validation
if enable_preview {
    let preview_sender = preview_stage.start()?;
    preview_sender.send(processed_frame)?;
    preview_stage.render();
}
```

## Standard I/O Packet Flow

Following the complete DeepGI pipeline specification with proper packet types:

```rust
// Complete pipeline data flow example
let raw_frame: RawFramePacket = decklink_capture.next_frame()?;

// Stage 1: Preprocessing (Raw → Tensor)
let tensor_input: TensorInputPacket = preprocessing_stage.process(raw_frame)?;
assert_eq!(tensor_input.desc.shape, [1, 3, 512, 512]); // NCHW format
assert_eq!(tensor_input.desc.dtype, TensorDataType::F32);
assert_eq!(tensor_input.desc.layout, TensorLayout::NCHW);

// Stage 2: Inference (Tensor → Raw Detections)
let raw_detections: RawDetectionsPacket = inference_stage.process(tensor_input)?;
println!("Found {} raw detections", raw_detections.dets.len());

// Stage 3: Postprocessing (Raw Detections → Processed Detections)
let detections: DetectionsPacket = postprocessing_stage.process(raw_detections)?;
for det in &detections.dets {
    println!("Object {} at ({:.1}, {:.1}) with confidence {:.2}", 
             det.class_id, det.bbox.x1, det.bbox.y1, det.score);
}

// Stage 4: Overlay Planning (Detections → Overlay Plan)
let overlay_plan: OverlayPlanPacket = overlay_planner.process(detections)?;
println!("Generated {} overlay operations", overlay_plan.ops.len());

// Stage 5: Output Rendering (Overlay Plan → Keying)
let keying_packet: KeyingPacket = output_stage.process(overlay_plan)?;
assert_eq!(keying_packet.passthrough.meta.seq_no, raw_frame.meta.seq_no);

// Stage 6: DeckLink Output
decklink_output.send_keying(keying_packet)?;
```

### Packet Type Validation

Each stage enforces strict input/output type checking:

```rust
// Type-safe pipeline construction
pub trait ProcessingStage<Input, Output> {
    fn process(&mut self, input: Input) -> Result<Output, PipelineError>;
    fn name(&self) -> &str;
}

// Preprocessing: RawFramePacket → TensorInputPacket
impl ProcessingStage<RawFramePacket, TensorInputPacket> for PreprocessingStage { ... }

// Inference: TensorInputPacket → RawDetectionsPacket  
impl ProcessingStage<TensorInputPacket, RawDetectionsPacket> for InferenceStage { ... }

// Postprocessing: RawDetectionsPacket → DetectionsPacket
impl ProcessingStage<RawDetectionsPacket, DetectionsPacket> for PostprocessingStage { ... }

// Overlay: DetectionsPacket → OverlayPlanPacket
impl ProcessingStage<DetectionsPacket, OverlayPlanPacket> for OverlayStage { ... }

// Output: OverlayPlanPacket → KeyingPacket
impl ProcessingStage<OverlayPlanPacket, KeyingPacket> for OutputStage { ... }
```

## Performance Considerations

- **GPU Memory Management**: Zero-copy CUDA device pointer handoff between preprocessing and inference
- **Pipeline Parallelism**: Each stage runs in dedicated threads with optimized buffering
- **Memory Pools**: Pre-allocated GPU memory pools to avoid allocation overhead
- **Batch Processing**: Support for batch inference when latency requirements allow
- **Frame Synchronization**: Precise timing control to maintain A/V sync for broadcast applications
- **Adaptive Quality**: Dynamic resolution/quality adjustment based on processing load

### Memory Layout Optimization

```rust
// Optimal memory flow for GPU processing
struct PipelineMemory {
    // GPU memory pool for tensor data
    gpu_tensor_pool: CudaMemoryPool<TensorInputPacket>,
    
    // CPU memory pool for detection results
    cpu_detection_pool: MemoryPool<DetectionsPacket>,
    
    // Shared overlay framebuffer
    overlay_framebuffer: SharedFramebuffer,
}

// Zero-copy GPU handoff
let cuda_tensor = preprocessing_stage.process_gpu(raw_frame)?; // Stays on GPU
let raw_detections = inference_stage.process_gpu(cuda_tensor)?; // GPU → CPU only for results
```

## Implementation Roadmap

### Phase 1: Core Pipeline (Current)
- ✅ **Standard I/O Packets**: Complete packet type definitions
- ✅ **Capture Stage**: DeckLink input with multiple format support
- ✅ **Preprocessing Stage**: Basic CPU/GPU preprocessing with OpenCV
- 🔄 **Pipeline Framework**: Threading and data flow management

### Phase 2: AI Integration (Next)
- 🔄 **Inference Stage**: ONNXRuntime integration with GPU support
- 🔄 **Postprocessing Stage**: NMS and multi-object tracking
- 🔄 **Performance Optimization**: GPU memory pools and batch processing

### Phase 3: Advanced Features (Future)
- 📋 **TensorRT Backend**: High-performance inference optimization
- 📋 **Advanced Tracking**: DeepSORT and ByteTrack implementations
- 📋 **Overlay Rendering**: Hardware-accelerated overlay composition
- 📋 **DeckLink Output**: Internal keying and multiple output support

### Phase 4: Production Features (Future)
- 📋 **Configuration Management**: Runtime pipeline reconfiguration
- 📋 **Monitoring & Telemetry**: Comprehensive performance metrics
- 📋 **Load Balancing**: Multi-GPU and multi-stream support
- 📋 **Integration APIs**: REST API and streaming protocol support

## Contributing

When adding new processing stages to the pipeline:

1. **Follow Standard I/O Types**: Implement the correct input/output packet types from the specification
2. **GPU Memory Optimization**: Use CUDA device pointers for zero-copy processing where possible
3. **Error Handling**: Provide detailed error messages with context for debugging
4. **Performance Monitoring**: Include timing and throughput metrics in your stage
5. **Type Safety**: Use the generic `ProcessingStage<Input, Output>` trait for compile-time type checking
6. **Documentation**: Document expected input formats, processing behavior, and output guarantees
7. **Testing**: Write unit tests for individual stages and integration tests for stage combinations

### Example Stage Implementation Template

```rust
use decklink_rust::{ProcessingStage, PipelineError};

pub struct YourCustomStage {
    config: YourStageConfig,
    metrics: StageMetrics,
}

impl ProcessingStage<InputPacketType, OutputPacketType> for YourCustomStage {
    fn process(&mut self, input: InputPacketType) -> Result<OutputPacketType, PipelineError> {
        let start_time = std::time::Instant::now();
        
        // Your processing logic here
        let result = your_processing_function(input)?;
        
        // Update metrics
        self.metrics.update_latency(start_time.elapsed());
        self.metrics.increment_processed();
        
        Ok(result)
    }
    
    fn name(&self) -> &str {
        "your_custom_stage"
    }
}
```

## License

This project is part of the DeepGI pipeline system. See main project for licensing information.
