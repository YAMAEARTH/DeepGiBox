use std::time::{Duration, Instant};
use std::collections::HashMap;
use crate::{ProcessingStage, RawFramePacket, PipelineError, CaptureConfig, ColorSpace};
use crate::capture::CaptureStage;

/// Stage performance metrics
#[derive(Debug, Clone, Default)]
pub struct StageMetrics {
    pub frames_processed: u64,
    pub frames_failed: u64,
    pub total_latency_ns: u64,
    pub min_latency_ns: u64,
    pub max_latency_ns: u64,
    pub last_latency_ns: u64,
}

impl StageMetrics {
    pub fn average_latency_ms(&self) -> f64 {
        if self.frames_processed > 0 {
            (self.total_latency_ns as f64 / self.frames_processed as f64) / 1_000_000.0
        } else {
            0.0
        }
    }

    pub fn last_latency_ms(&self) -> f64 {
        self.last_latency_ns as f64 / 1_000_000.0
    }

    pub fn min_latency_ms(&self) -> f64 {
        self.min_latency_ns as f64 / 1_000_000.0
    }

    pub fn max_latency_ms(&self) -> f64 {
        self.max_latency_ns as f64 / 1_000_000.0
    }

    pub fn success_rate(&self) -> f64 {
        let total = self.frames_processed + self.frames_failed;
        if total > 0 {
            (self.frames_processed as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    }

    pub fn update(&mut self, latency_ns: u64, success: bool) {
        if success {
            self.frames_processed += 1;
            self.total_latency_ns += latency_ns;
            self.last_latency_ns = latency_ns;
            
            if self.frames_processed == 1 {
                self.min_latency_ns = latency_ns;
                self.max_latency_ns = latency_ns;
            } else {
                self.min_latency_ns = self.min_latency_ns.min(latency_ns);
                self.max_latency_ns = self.max_latency_ns.max(latency_ns);
            }
        } else {
            self.frames_failed += 1;
        }
    }
}

/// Pipeline stages representing real video processing workflow
#[derive(Debug, Clone)]
pub enum PipelineStage {
    Capture,
    Preprocessing,
    Inference,
    PostProcessing,
    ObjectTracking,
    Overlay,
    Keying,
    Output,
}

impl PipelineStage {
    pub fn name(&self) -> &'static str {
        match self {
            PipelineStage::Capture => "capture",
            PipelineStage::Preprocessing => "preprocessing", 
            PipelineStage::Inference => "inference",
            PipelineStage::PostProcessing => "post_processing",
            PipelineStage::ObjectTracking => "object_tracking",
            PipelineStage::Overlay => "overlay",
            PipelineStage::Keying => "keying",
            PipelineStage::Output => "output",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            PipelineStage::Capture => "Video capture from DeckLink device",
            PipelineStage::Preprocessing => "Frame preprocessing (resize, color conversion, normalization)",
            PipelineStage::Inference => "AI model inference (object detection, classification)",
            PipelineStage::PostProcessing => "Process inference results (NMS, filtering, confidence)",
            PipelineStage::ObjectTracking => "Track objects across frames (Kalman, DeepSORT)",
            PipelineStage::Overlay => "Generate overlay graphics (bounding boxes, labels)",
            PipelineStage::Keying => "Chroma keying and background replacement",
            PipelineStage::Output => "Output encoding and transmission",
        }
    }

    pub fn create_processor(&self) -> Box<dyn ProcessingStage> {
        match self {
            PipelineStage::Capture => Box::new(CaptureProcessor::new()),
            PipelineStage::Preprocessing => Box::new(PreprocessingProcessor::new()),
            PipelineStage::Inference => Box::new(InferenceProcessor::new()),
            PipelineStage::PostProcessing => Box::new(PostProcessingProcessor::new()),
            PipelineStage::ObjectTracking => Box::new(ObjectTrackingProcessor::new()),
            PipelineStage::Overlay => Box::new(OverlayProcessor::new()),
            PipelineStage::Keying => Box::new(KeyingProcessor::new()),
            PipelineStage::Output => Box::new(OutputProcessor::new()),
        }
    }
}

/// Capture processor - handles video frame acquisition
pub struct CaptureProcessor {
    frame_count: u64,
}

impl CaptureProcessor {
    pub fn new() -> Self {
        Self { frame_count: 0 }
    }
}

impl ProcessingStage for CaptureProcessor {
    fn process(&mut self, input: RawFramePacket) -> Result<RawFramePacket, PipelineError> {
        self.frame_count += 1;
        
        // Simulate capture validation and format checks
        std::thread::sleep(Duration::from_micros(50));
        
        Ok(input)
    }

    fn name(&self) -> &str {
        "capture"
    }
}

/// Preprocessing processor - frame preparation for inference
pub struct PreprocessingProcessor {
    frame_count: u64,
}

impl PreprocessingProcessor {
    pub fn new() -> Self {
        Self { frame_count: 0 }
    }
}

impl ProcessingStage for PreprocessingProcessor {
    fn process(&mut self, mut input: RawFramePacket) -> Result<RawFramePacket, PipelineError> {
        self.frame_count += 1;
        
        // Simulate preprocessing operations:
        // - Color space conversion (2ms)
        std::thread::sleep(Duration::from_millis(2));
        
        // - Resize/crop operations (1ms)
        std::thread::sleep(Duration::from_millis(1));
        
        // - Normalization (0.5ms)
        std::thread::sleep(Duration::from_micros(500));
        
        // Update metadata to indicate preprocessing
        input.meta.source_id = (input.meta.source_id & 0x0FFFFFFF) | 0x10000000;
        
        Ok(input)
    }

    fn name(&self) -> &str {
        "preprocessing"
    }
}

/// Inference processor - AI model inference
pub struct InferenceProcessor {
    frame_count: u64,
}

impl InferenceProcessor {
    pub fn new() -> Self {
        Self { frame_count: 0 }
    }
}

impl ProcessingStage for InferenceProcessor {
    fn process(&mut self, input: RawFramePacket) -> Result<RawFramePacket, PipelineError> {
        self.frame_count += 1;
        
        // Simulate AI model inference:
        // - Model forward pass (15-25ms typical for YOLO/ResNet)
        let inference_time = Duration::from_millis(15 + (self.frame_count % 10));
        std::thread::sleep(inference_time);
        
        // Simulate occasional inference failures (model timeout, OOM, etc.)
        if self.frame_count % 200 == 0 {
            return Err(PipelineError::Processing("AI model inference timeout".to_string()));
        }
        
        Ok(input)
    }

    fn name(&self) -> &str {
        "inference"
    }
}

/// Post-processing processor - process inference results
pub struct PostProcessingProcessor {
    frame_count: u64,
}

impl PostProcessingProcessor {
    pub fn new() -> Self {
        Self { frame_count: 0 }
    }
}

impl ProcessingStage for PostProcessingProcessor {
    fn process(&mut self, input: RawFramePacket) -> Result<RawFramePacket, PipelineError> {
        self.frame_count += 1;
        
        // Simulate post-processing operations:
        // - Non-Maximum Suppression (3ms)
        std::thread::sleep(Duration::from_millis(3));
        
        // - Confidence filtering (1ms)
        std::thread::sleep(Duration::from_millis(1));
        
        // - Result validation (0.5ms)
        std::thread::sleep(Duration::from_micros(500));
        
        Ok(input)
    }

    fn name(&self) -> &str {
        "post_processing"
    }
}

/// Object tracking processor - track objects across frames
pub struct ObjectTrackingProcessor {
    frame_count: u64,
}

impl ObjectTrackingProcessor {
    pub fn new() -> Self {
        Self { frame_count: 0 }
    }
}

impl ProcessingStage for ObjectTrackingProcessor {
    fn process(&mut self, input: RawFramePacket) -> Result<RawFramePacket, PipelineError> {
        self.frame_count += 1;
        
        // Simulate tracking operations:
        // - Kalman filter prediction (2ms)
        std::thread::sleep(Duration::from_millis(2));
        
        // - Data association (3ms)
        std::thread::sleep(Duration::from_millis(3));
        
        // - Track state update (1ms)
        std::thread::sleep(Duration::from_millis(1));
        
        // Simulate tracking failures for complex scenes
        if self.frame_count % 300 == 0 {
            return Err(PipelineError::Processing("Tracking lost for complex scene".to_string()));
        }
        
        Ok(input)
    }

    fn name(&self) -> &str {
        "object_tracking"
    }
}

/// Overlay processor - generate visual overlays
pub struct OverlayProcessor {
    frame_count: u64,
}

impl OverlayProcessor {
    pub fn new() -> Self {
        Self { frame_count: 0 }
    }
}

impl ProcessingStage for OverlayProcessor {
    fn process(&mut self, input: RawFramePacket) -> Result<RawFramePacket, PipelineError> {
        self.frame_count += 1;
        
        // Simulate overlay generation:
        // - Bounding box rendering (2ms)
        std::thread::sleep(Duration::from_millis(2));
        
        // - Text label rendering (1ms)
        std::thread::sleep(Duration::from_millis(1));
        
        // - Confidence indicators (0.5ms)
        std::thread::sleep(Duration::from_micros(500));
        
        Ok(input)
    }

    fn name(&self) -> &str {
        "overlay"
    }
}

/// Keying processor - chroma key and background replacement
pub struct KeyingProcessor {
    frame_count: u64,
}

impl KeyingProcessor {
    pub fn new() -> Self {
        Self { frame_count: 0 }
    }
}

impl ProcessingStage for KeyingProcessor {
    fn process(&mut self, input: RawFramePacket) -> Result<RawFramePacket, PipelineError> {
        self.frame_count += 1;
        
        // Simulate keying operations based on frame size
        if let Some(data) = input.as_slice() {
            let pixel_count = data.len() / 4; // BGRA format
            
            // Chroma key processing scales with resolution
            let processing_time_us = pixel_count / 500; // 2μs per 1000 pixels
            std::thread::sleep(Duration::from_micros(processing_time_us as u64));
        }
        
        // Additional fixed operations:
        // - Alpha blending (1ms)
        std::thread::sleep(Duration::from_millis(1));
        
        Ok(input)
    }

    fn name(&self) -> &str {
        "keying"
    }
}

/// Output processor - final encoding and transmission
pub struct OutputProcessor {
    frame_count: u64,
}

impl OutputProcessor {
    pub fn new() -> Self {
        Self { frame_count: 0 }
    }
}

impl ProcessingStage for OutputProcessor {
    fn process(&mut self, input: RawFramePacket) -> Result<RawFramePacket, PipelineError> {
        self.frame_count += 1;
        
        // Simulate output operations:
        // - Format conversion (1ms)
        std::thread::sleep(Duration::from_millis(1));
        
        // - Encoding/compression (3ms)
        std::thread::sleep(Duration::from_millis(3));
        
        // - Network transmission prep (0.5ms)
        std::thread::sleep(Duration::from_micros(500));
        
        // Simulate occasional network issues
        if self.frame_count % 500 == 0 {
            return Err(PipelineError::Processing("Output buffer full - network congestion".to_string()));
        }
        
        Ok(input)
    }

    fn name(&self) -> &str {
        "output"
    }
}
    frame_count: u64,
}

impl AIInferenceProcessor {
    pub fn new() -> Self {
        Self { frame_count: 0 }
    }
}

impl ProcessingStage for AIInferenceProcessor {
    fn process(&mut self, input: RawFramePacket) -> Result<RawFramePacket, PipelineError> {
        self.frame_count += 1;
        
        let inference_time = Duration::from_millis(10 + (self.frame_count % 10));
        std::thread::sleep(inference_time);
        
        if self.frame_count % 100 == 0 {
            return Err(PipelineError::Processing("Simulated AI inference timeout".to_string()));
        }
        
        Ok(input)
    }

    fn name(&self) -> &str {
        "output"
    }
}

/// Enhanced headless processor with stage selection and metrics
pub struct HeadlessProcessor {
pub struct ImageFilterProcessor {
    frame_count: u64,
}

impl ImageFilterProcessor {
    pub fn new() -> Self {
        Self { frame_count: 0 }
    }
}

impl ProcessingStage for ImageFilterProcessor {
    fn process(&mut self, input: RawFramePacket) -> Result<RawFramePacket, PipelineError> {
        self.frame_count += 1;
        
        if let Some(data) = input.as_slice() {
            let pixel_count = data.len() / 4;
            let processing_time_us = pixel_count / 1000;
            std::thread::sleep(Duration::from_micros(processing_time_us as u64));
        }
        
        Ok(input)
    }

    fn name(&self) -> &str {
        "image_filtering"
    }
}

/// Object detection pipeline simulation
pub struct ObjectDetectionProcessor {
    frame_count: u64,
}

impl ObjectDetectionProcessor {
    pub fn new() -> Self {
        Self { frame_count: 0 }
    }
}

impl ProcessingStage for ObjectDetectionProcessor {
    fn process(&mut self, input: RawFramePacket) -> Result<RawFramePacket, PipelineError> {
        self.frame_count += 1;
        
        // Simulate multi-stage object detection pipeline
        std::thread::sleep(Duration::from_millis(2));  // Preprocessing
        std::thread::sleep(Duration::from_millis(15)); // Model inference
        std::thread::sleep(Duration::from_millis(3));  // Post-processing NMS
        std::thread::sleep(Duration::from_millis(1));  // Tracking update
        
        if self.frame_count % 150 == 0 {
            return Err(PipelineError::Processing("Object detection model crashed".to_string()));
        }
        
        Ok(input)
    }

    fn name(&self) -> &str {
        "object_detection"
    }
}

/// Enhanced headless processor configuration
#[derive(Debug, Clone)]
pub struct HeadlessConfig {
    pub capture_config: CaptureConfig,
    pub selected_stages: Vec<PipelineStage>,
    pub stats_interval: Duration,
    pub max_runtime: Option<Duration>,
    pub max_frames: Option<u64>,
    pub enable_detailed_logging: bool,
}

impl Default for HeadlessConfig {
    fn default() -> Self {
        Self {
            capture_config: CaptureConfig {
                device_index: 0,
                source_id: 100,
                expected_colorspace: ColorSpace::BT709,
            },
            selected_stages: vec![PipelineStage::Capture],
            stats_interval: Duration::from_secs(1),
            max_runtime: Some(Duration::from_secs(30)),
            max_frames: None,
            enable_detailed_logging: false,
        }
    }
}

/// Enhanced headless processor with stage selection and metrics
pub struct HeadlessProcessor {
    capture: CaptureStage,
    stages: Vec<(String, Box<dyn ProcessingStage>)>,
    metrics: HashMap<String, StageMetrics>,
    start_time: Instant,
    total_frames: u64,
    config: HeadlessConfig,
}

impl HeadlessProcessor {
    pub fn new(config: HeadlessConfig) -> Self {
        let capture = CaptureStage::new(config.capture_config.clone());
        let mut stages = Vec::new();
        let mut metrics = HashMap::new();
        
        for stage_type in &config.selected_stages {
            let stage_name = format!("{}_{}", stage_type.name(), stages.len());
            let processor = stage_type.create_processor();
            stages.push((stage_name.clone(), processor));
            metrics.insert(stage_name, StageMetrics::default());
        }
        
        Self {
            capture,
            stages,
            metrics,
            start_time: Instant::now(),
            total_frames: 0,
            config,
        }
    }

    pub fn run(&mut self) -> Result<(), anyhow::Error> {
        println!("🚀 Starting Headless Processing Pipeline");
        println!("==========================================");
        self.print_configuration();
        
        self.capture.start().map_err(|e| anyhow::anyhow!("Failed to start capture: {}", e))?;
        println!("✅ Capture started successfully");
        
        let mut last_stats_time = Instant::now();
        
        loop {
            if let Some(max_runtime) = self.config.max_runtime {
                if self.start_time.elapsed() >= max_runtime {
                    println!("⏰ Maximum runtime reached");
                    break;
                }
            }
            
            if let Some(max_frames) = self.config.max_frames {
                if self.total_frames >= max_frames {
                    println!("🎬 Maximum frames processed");
                    break;
                }
            }
            
            match self.capture.get_next_frame() {
                Ok(Some(frame)) => {
                    self.process_frame(frame)?;
                    self.total_frames += 1;
                }
                Ok(None) => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(e) => {
                    eprintln!("❌ Capture error: {}", e);
                    break;
                }
            }
            
            if last_stats_time.elapsed() >= self.config.stats_interval {
                self.print_status();
                last_stats_time = Instant::now();
            }
        }
        
        self.capture.stop().map_err(|e| anyhow::anyhow!("Failed to stop capture: {}", e))?;
        println!("✅ Capture stopped successfully");
        
        self.print_final_summary();
        
        Ok(())
    }

    fn process_frame(&mut self, mut frame: RawFramePacket) -> Result<(), anyhow::Error> {
        let frame_start = Instant::now();
        
        for (stage_name, stage) in &mut self.stages {
            let stage_start = Instant::now();
            
            match stage.process(frame) {
                Ok(processed_frame) => {
                    let latency_ns = stage_start.elapsed().as_nanos() as u64;
                    self.metrics.get_mut(stage_name).unwrap().update(latency_ns, true);
                    frame = processed_frame;
                    
                    if self.config.enable_detailed_logging {
                        println!("  ✓ {} completed in {:.2}ms", 
                            stage_name, 
                            latency_ns as f64 / 1_000_000.0
                        );
                    }
                }
                Err(e) => {
                    let latency_ns = stage_start.elapsed().as_nanos() as u64;
                    self.metrics.get_mut(stage_name).unwrap().update(latency_ns, false);
                    
                    if self.config.enable_detailed_logging {
                        eprintln!("  ❌ {} failed: {} (took {:.2}ms)", 
                            stage_name, 
                            e,
                            latency_ns as f64 / 1_000_000.0
                        );
                    }
                    return Ok(());
                }
            }
        }
        
        let total_latency = frame_start.elapsed();
        if self.config.enable_detailed_logging {
            println!("📦 Frame {} processed in {:.2}ms", 
                self.total_frames + 1,
                total_latency.as_secs_f64() * 1000.0
            );
        }
        
        Ok(())
    }

    fn print_configuration(&self) {
        println!("Configuration:");
        println!("  Device: {}", self.config.capture_config.device_index);
        println!("  Colorspace: {:?}", self.config.capture_config.expected_colorspace);
        println!("  Selected Stages: {}", self.config.selected_stages.len());
        
        for (i, stage) in self.config.selected_stages.iter().enumerate() {
            println!("    {}. {} - {}", i + 1, stage.name(), stage.description());
        }
        
        if let Some(max_runtime) = self.config.max_runtime {
            println!("  Max Runtime: {:.1}s", max_runtime.as_secs_f64());
        }
        if let Some(max_frames) = self.config.max_frames {
            println!("  Max Frames: {}", max_frames);
        }
        println!();
    }

    fn print_status(&self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let fps = self.total_frames as f64 / elapsed;
        
        println!("📊 Pipeline Status - Runtime: {:.1}s, Frames: {}, FPS: {:.1}", 
            elapsed, self.total_frames, fps);
        
        for (stage_name, metrics) in &self.metrics {
            let status_icon = if metrics.success_rate() > 95.0 { "✅" }
                            else if metrics.success_rate() > 80.0 { "⚠️" }
                            else { "❌" };
            
            println!("  {} {} - Processed: {}, Failed: {}, Success: {:.1}%, Avg: {:.2}ms, Last: {:.2}ms",
                status_icon,
                stage_name,
                metrics.frames_processed,
                metrics.frames_failed,
                metrics.success_rate(),
                metrics.average_latency_ms(),
                metrics.last_latency_ms()
            );
        }
        println!();
    }

    fn print_final_summary(&self) {
        println!();
        println!("🏁 Final Pipeline Summary");
        println!("========================");
        
        let total_runtime = self.start_time.elapsed().as_secs_f64();
        let overall_fps = self.total_frames as f64 / total_runtime;
        
        println!("Overall Performance:");
        println!("  Total Runtime: {:.2}s", total_runtime);
        println!("  Total Frames: {}", self.total_frames);
        println!("  Average FPS: {:.2}", overall_fps);
        println!();
        
        println!("Stage Performance:");
        for (stage_name, metrics) in &self.metrics {
            println!("  🔧 {}:", stage_name);
            println!("    Processed: {} frames", metrics.frames_processed);
            println!("    Failed: {} frames", metrics.frames_failed);
            println!("    Success Rate: {:.1}%", metrics.success_rate());
            println!("    Latency - Avg: {:.2}ms, Min: {:.2}ms, Max: {:.2}ms", 
                metrics.average_latency_ms(),
                metrics.min_latency_ms(),
                metrics.max_latency_ms()
            );
            println!();
        }
    }
}
