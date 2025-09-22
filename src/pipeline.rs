use crate::packets::{
    RawFramePacket, TensorInputPacket, RawDetectionsPacket, DetectionsPacket, 
    OverlayPlanPacket, KeyingPacket, PipelineError
};
use crate::capture::{CaptureStage, CaptureConfig};
use crate::preview::{PreviewStage, PreviewConfig};
use crate::preprocessing_v2::{PreprocessingV2Stage, PreprocessingV2Config, ProcessingStageV2};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Stage performance metrics following DeepGI standards
#[derive(Debug, Clone, Default)]
pub struct StageMetrics {
    pub frames_processed: u64,
    pub frames_failed: u64,
    pub total_latency_ns: u64,
    pub min_latency_ns: u64,
    pub max_latency_ns: u64,
    pub last_latency_ns: u64,
    pub throughput_fps: f64,
}

impl StageMetrics {
    pub fn new() -> Self {
        Self::default()
    }

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
        if self.min_latency_ns > 0 {
            self.min_latency_ns as f64 / 1_000_000.0
        } else {
            0.0
        }
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

    pub fn update_throughput(&mut self, elapsed_seconds: f64) {
        if elapsed_seconds > 0.0 {
            self.throughput_fps = self.frames_processed as f64 / elapsed_seconds;
        }
    }
}

/// Pipeline configuration supporting all DeepGI stages
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub capture: CaptureConfig,
    pub preview: PreviewConfig,
    pub preprocessing_v2: Option<PreprocessingV2Config>,
    pub processing_buffer_size: usize,
    pub enable_passthrough: bool,
    pub enable_performance_monitoring: bool,
    pub stats_interval: Duration,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            capture: CaptureConfig::default(),
            preview: PreviewConfig::default(),
            preprocessing_v2: None,
            processing_buffer_size: 16,
            enable_passthrough: false,
            enable_performance_monitoring: true,
            stats_interval: Duration::from_secs(1),
        }
    }
}

/// Comprehensive pipeline statistics following DeepGI standards
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub frames_captured: u64,
    pub frames_processed: u64,
    pub frames_previewed: u64,
    pub frames_dropped: u64,
    pub capture_fps: f64,
    pub processing_fps: f64,
    pub preview_fps: f64,
    pub stage_metrics: std::collections::HashMap<String, StageMetrics>,
    pub end_to_end_latency_ms: f64,
    pub pipeline_efficiency: f64,
}

impl Default for PipelineStats {
    fn default() -> Self {
        Self {
            frames_captured: 0,
            frames_processed: 0,
            frames_previewed: 0,
            frames_dropped: 0,
            capture_fps: 0.0,
            processing_fps: 0.0,
            preview_fps: 0.0,
            stage_metrics: std::collections::HashMap::new(),
            end_to_end_latency_ms: 0.0,
            pipeline_efficiency: 0.0,
        }
    }
}

impl PipelineStats {
    /// Calculate overall pipeline efficiency (success rate)
    pub fn calculate_efficiency(&mut self) {
        let total_frames = self.frames_captured;
        if total_frames > 0 {
            self.pipeline_efficiency = (self.frames_processed as f64 / total_frames as f64) * 100.0;
        }
    }

    /// Get average stage latency
    pub fn average_stage_latency(&self, stage_name: &str) -> f64 {
        self.stage_metrics.get(stage_name)
            .map(|metrics| metrics.average_latency_ms())
            .unwrap_or(0.0)
    }

    /// Get total pipeline latency (sum of all stages)
    pub fn total_pipeline_latency_ms(&self) -> f64 {
        self.stage_metrics.values()
            .map(|metrics| metrics.average_latency_ms())
            .sum()
    }
}

/// Pipeline stage trait for custom processing stages
pub trait ProcessingStage: Send {
    fn process(&mut self, input: RawFramePacket) -> Result<RawFramePacket, PipelineError>;
    fn name(&self) -> &str;
}

/// Passthrough processing stage (no-op)
pub struct PassthroughStage;

impl ProcessingStage for PassthroughStage {
    fn process(&mut self, input: RawFramePacket) -> Result<RawFramePacket, PipelineError> {
        Ok(input)
    }

    fn name(&self) -> &str {
        "passthrough"
    }
}

/// Example processing stage that adds frame information
pub struct FrameInfoStage {
    frame_count: u64,
}

impl FrameInfoStage {
    pub fn new() -> Self {
        Self { frame_count: 0 }
    }
}

impl ProcessingStage for FrameInfoStage {
    fn process(&mut self, mut input: RawFramePacket) -> Result<RawFramePacket, PipelineError> {
        self.frame_count += 1;
        
        // Modify metadata to include processing info
        input.meta.source_id |= 0x80000000; // Mark as processed
        
        println!("Processing frame {} (seq: {})", self.frame_count, input.meta.seq_no);
        
        Ok(input)
    }

    fn name(&self) -> &str {
        "frame_info"
    }
}

/// Main pipeline orchestrator supporting complete DeepGI flow
pub struct Pipeline {
    config: PipelineConfig,
    capture_stage: CaptureStage,
    preview_stage: PreviewStage,
    preprocessing_v2_stage: Option<PreprocessingV2Stage>,
    processing_stage: Option<Box<dyn ProcessingStage>>,
    stats: Arc<Mutex<PipelineStats>>,
    is_running: bool,
    processing_thread: Option<thread::JoinHandle<()>>,
    shutdown_flag: Arc<Mutex<bool>>,
    start_time: Instant,
}

impl Pipeline {
    /// Create a new pipeline with the given configuration
    pub fn new(config: PipelineConfig) -> Self {
        let preprocessing_v2_stage = if let Some(ref preproc_config) = config.preprocessing_v2 {
            PreprocessingV2Stage::new(preproc_config.clone()).ok()
        } else {
            None
        };

        Self {
            capture_stage: CaptureStage::new(config.capture.clone()),
            preview_stage: PreviewStage::new(config.preview.clone()),
            preprocessing_v2_stage,
            processing_stage: None,
            config,
            stats: Arc::new(Mutex::new(PipelineStats::default())),
            is_running: false,
            processing_thread: None,
            shutdown_flag: Arc::new(Mutex::new(false)),
            start_time: Instant::now(),
        }
    }

    /// Get pipeline configuration (read-only access)
    pub fn get_config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Check if performance monitoring is enabled
    pub fn is_performance_monitoring_enabled(&self) -> bool {
        self.config.enable_performance_monitoring
    }

    /// Get stats interval
    pub fn get_stats_interval(&self) -> Duration {
        self.config.stats_interval
    }

    /// Check if passthrough is enabled
    pub fn is_passthrough_enabled(&self) -> bool {
        self.config.enable_passthrough
    }

    /// Add a custom processing stage to the pipeline
    pub fn add_processing_stage(&mut self, stage: Box<dyn ProcessingStage>) {
        self.processing_stage = Some(stage);
    }

    /// Initialize the pipeline (must be called from OpenGL context thread)
    pub fn initialize(&mut self) -> Result<(), PipelineError> {
        self.preview_stage.initialize_gl()?;
        Ok(())
    }

    /// Start the pipeline with enhanced DeepGI processing flow
    pub fn start(&mut self) -> Result<(), PipelineError> {
        if self.is_running {
            return Err(PipelineError::Processing("Pipeline already running".to_string()));
        }

        // Reset stats and timing
        {
            let mut stats = self.stats.lock().unwrap();
            *stats = PipelineStats::default();
        }
        self.start_time = Instant::now();

        // Start capture
        self.capture_stage.start()?;

        // Start preview (this creates the input channel)
        let preview_sender = self.preview_stage.start()?;

        // Reset shutdown flag
        {
            let mut flag = self.shutdown_flag.lock().unwrap();
            *flag = false;
        }

        // Start enhanced processing thread with full DeepGI pipeline
        let stats = Arc::clone(&self.stats);
        let shutdown_flag = Arc::clone(&self.shutdown_flag);
        let config = self.config.clone();
        
        let mut capture_stage = CaptureStage::new(config.capture.clone());
        capture_stage.start()?;
        
        // Move processing stages to thread if available
        let processing_stage = self.processing_stage.take();
        let mut preprocessing_v2_stage = self.preprocessing_v2_stage.take();
        
        let handle = thread::spawn(move || {
            Self::enhanced_processing_loop(
                capture_stage,
                preview_sender,
                processing_stage,
                preprocessing_v2_stage,
                stats,
                shutdown_flag,
                config,
            );
        });

        self.processing_thread = Some(handle);
        self.is_running = true;

        Ok(())
    }

    /// Render current frame (call from main/render thread)
    pub fn render(&self) -> bool {
        if !self.is_running {
            return false;
        }
        self.preview_stage.render()
    }

    /// Stop the pipeline
    pub fn stop(&mut self) -> Result<(), PipelineError> {
        if !self.is_running {
            return Ok(());
        }

        // Signal shutdown
        {
            let mut flag = self.shutdown_flag.lock().unwrap();
            *flag = true;
        }

        // Wait for processing thread
        if let Some(handle) = self.processing_thread.take() {
            if let Err(e) = handle.join() {
                eprintln!("Processing thread panicked: {:?}", e);
            }
        }

        // Stop stages
        self.capture_stage.stop()?;
        self.preview_stage.stop()?;

        self.is_running = false;

        Ok(())
    }

    /// Get comprehensive performance statistics
    pub fn get_performance_stats(&self) -> PipelineStats {
        let stats = self.stats.lock().unwrap();
        stats.clone()
    }

    /// Get performance metrics for a specific stage
    pub fn get_stage_metrics(&self, stage_name: &str) -> Option<StageMetrics> {
        let stats = self.stats.lock().unwrap();
        stats.stage_metrics.get(stage_name).cloned()
    }

    /// Print detailed performance report
    pub fn print_performance_report(&self) {
        let stats = self.stats.lock().unwrap();
        println!("\n🔍 Pipeline Performance Report");
        println!("================================");
        println!("Total processed: {} frames", stats.frames_processed);
        println!("Total dropped: {} frames", stats.frames_dropped);
        println!("Pipeline efficiency: {:.1}%", stats.pipeline_efficiency);
        println!("End-to-end latency: {:.2}ms", stats.end_to_end_latency_ms);
        println!("Total pipeline latency: {:.2}ms", stats.total_pipeline_latency_ms());
        
        println!("\nStage Performance:");
        for (stage_name, metrics) in &stats.stage_metrics {
            println!("  📊 {} Stage:", stage_name);
            println!("    Average latency: {:.2}ms", metrics.average_latency_ms());
            println!("    Min/Max latency: {:.2}ms / {:.2}ms", metrics.min_latency_ms(), metrics.max_latency_ms());
            println!("    Success rate: {:.1}%", metrics.success_rate());
            println!("    Throughput: {:.1} fps", metrics.throughput_fps);
        }
    }

    /// Get preview statistics
    pub fn get_preview_stats(&self) -> crate::preview::PreviewStats {
        self.preview_stage.get_stats()
    }

    /// Check if pipeline is running
    pub fn is_running(&self) -> bool {
        self.is_running
    }

    /// Enhanced processing loop supporting complete DeepGI pipeline flow
    fn enhanced_processing_loop(
        mut capture_stage: CaptureStage,
        preview_sender: mpsc::Sender<RawFramePacket>,
        mut processing_stage: Option<Box<dyn ProcessingStage>>,
        mut preprocessing_v2_stage: Option<PreprocessingV2Stage>,
        stats: Arc<Mutex<PipelineStats>>,
        shutdown_flag: Arc<Mutex<bool>>,
        config: PipelineConfig,
    ) {
        let mut frames_processed = 0u64;
        let mut frames_dropped = 0u64;
        let mut last_stats_time = Instant::now();

        // Initialize stage metrics if performance monitoring is enabled
        if config.enable_performance_monitoring {
            let mut stats_guard = stats.lock().unwrap();
            if preprocessing_v2_stage.is_some() {
                stats_guard.stage_metrics.insert("preprocessing_v2".to_string(), StageMetrics::new());
            }
            if processing_stage.is_some() {
                stats_guard.stage_metrics.insert("processing".to_string(), StageMetrics::new());
            }
        }

        loop {
            // Check shutdown flag
            {
                let flag = shutdown_flag.lock().unwrap();
                if *flag {
                    break;
                }
            }

            // Get next frame from capture
            match capture_stage.get_next_frame() {
                Ok(Some(raw_frame)) => {
                    let frame_start = Instant::now();
                    let mut current_frame = raw_frame.clone();
                    let mut processing_success = true;

                    // Stage 1: Preprocessing V2 (RawFramePacket -> TensorInputPacket)
                    if let Some(ref mut preproc_stage) = preprocessing_v2_stage {
                        let stage_start = Instant::now();
                        
                        match ProcessingStageV2::process(preproc_stage, current_frame.clone()) {
                            Ok(_tensor_packet) => {
                                // For compatibility, continue with original frame
                                // In a real pipeline, you'd pass the tensor to the next stage
                                let latency_ns = stage_start.elapsed().as_nanos() as u64;
                                
                                if config.enable_performance_monitoring {
                                    if let Ok(mut stats_guard) = stats.lock() {
                                        if let Some(metrics) = stats_guard.stage_metrics.get_mut("preprocessing_v2") {
                                            metrics.update(latency_ns, true);
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("Preprocessing V2 error: {:?}", e);
                                processing_success = false;
                                frames_dropped += 1;
                                
                                if config.enable_performance_monitoring {
                                    let latency_ns = stage_start.elapsed().as_nanos() as u64;
                                    if let Ok(mut stats_guard) = stats.lock() {
                                        if let Some(metrics) = stats_guard.stage_metrics.get_mut("preprocessing_v2") {
                                            metrics.update(latency_ns, false);
                                        }
                                    }
                                }
                                continue;
                            }
                        }
                    }

                    // Stage 2: Custom Processing Stage (backward compatibility)
                    if processing_success {
                        if let Some(ref mut stage) = processing_stage {
                            let stage_start = Instant::now();
                            
                            match stage.process(current_frame.clone()) {
                                Ok(processed_frame) => {
                                    current_frame = processed_frame;
                                    let latency_ns = stage_start.elapsed().as_nanos() as u64;
                                    
                                    if config.enable_performance_monitoring {
                                        if let Ok(mut stats_guard) = stats.lock() {
                                            if let Some(metrics) = stats_guard.stage_metrics.get_mut("processing") {
                                                metrics.update(latency_ns, true);
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    eprintln!("Processing error in {}: {:?}", stage.name(), e);
                                    processing_success = false;
                                    frames_dropped += 1;
                                    
                                    if config.enable_performance_monitoring {
                                        let latency_ns = stage_start.elapsed().as_nanos() as u64;
                                        if let Ok(mut stats_guard) = stats.lock() {
                                            if let Some(metrics) = stats_guard.stage_metrics.get_mut("processing") {
                                                metrics.update(latency_ns, false);
                                            }
                                        }
                                    }
                                    continue;
                                }
                            }
                        } else if config.enable_passthrough {
                            // Use passthrough if enabled and no custom stage
                            let mut passthrough = PassthroughStage;
                            match passthrough.process(current_frame.clone()) {
                                Ok(frame) => current_frame = frame,
                                Err(e) => {
                                    eprintln!("Passthrough error: {:?}", e);
                                    processing_success = false;
                                    frames_dropped += 1;
                                    continue;
                                }
                            }
                        }
                    }

                    // Send to preview if processing was successful
                    if processing_success {
                        if let Err(_) = preview_sender.send(current_frame) {
                            // Preview disconnected, exit
                            break;
                        }
                        frames_processed += 1;
                    }

                    // Update comprehensive stats
                    if config.enable_performance_monitoring {
                        let end_to_end_latency = frame_start.elapsed().as_nanos() as u64;
                        if let Ok(mut stats_guard) = stats.lock() {
                            stats_guard.frames_processed = frames_processed;
                            stats_guard.frames_dropped = frames_dropped;
                            stats_guard.end_to_end_latency_ms = end_to_end_latency as f64 / 1_000_000.0;
                            stats_guard.calculate_efficiency();
                        }
                    }

                    // Print periodic stats
                    if config.enable_performance_monitoring && last_stats_time.elapsed() >= config.stats_interval {
                        if let Ok(stats_guard) = stats.lock() {
                            println!("📊 Pipeline Stats - Processed: {}, Dropped: {}, Efficiency: {:.1}%, End-to-End: {:.2}ms",
                                stats_guard.frames_processed,
                                stats_guard.frames_dropped,
                                stats_guard.pipeline_efficiency,
                                stats_guard.end_to_end_latency_ms
                            );
                            
                            // Print stage-specific metrics
                            for (stage_name, metrics) in &stats_guard.stage_metrics {
                                println!("  {} - Avg: {:.2}ms, Last: {:.2}ms, Success: {:.1}%",
                                    stage_name,
                                    metrics.average_latency_ms(),
                                    metrics.last_latency_ms(),
                                    metrics.success_rate()
                                );
                            }
                        }
                        last_stats_time = Instant::now();
                    }
                }
                Ok(None) => {
                    // No frame available, sleep briefly
                    thread::sleep(Duration::from_millis(1));
                }
                Err(e) => {
                    eprintln!("Capture error: {:?}", e);
                    break;
                }
            }
        }

        // Final stats update
        {
            let mut stats_guard = stats.lock().unwrap();
            stats_guard.frames_processed = frames_processed;
            stats_guard.frames_dropped = frames_dropped;
            stats_guard.calculate_efficiency();
            
            // Update throughput for all stages
            let elapsed_seconds = last_stats_time.elapsed().as_secs_f64();
            for metrics in stats_guard.stage_metrics.values_mut() {
                metrics.update_throughput(elapsed_seconds);
            }
        }

        let _ = capture_stage.stop();
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        let _ = self.stop();
        self.preview_stage.cleanup();
    }
}

/// Enhanced builder for creating pipelines with complete DeepGI stage support
pub struct PipelineBuilder {
    config: PipelineConfig,
    processing_stage: Option<Box<dyn ProcessingStage>>,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
            processing_stage: None,
        }
    }

    pub fn with_capture_config(mut self, config: CaptureConfig) -> Self {
        self.config.capture = config;
        self
    }

    pub fn with_preview_config(mut self, config: PreviewConfig) -> Self {
        self.config.preview = config;
        self
    }

    pub fn with_preprocessing_v2_config(mut self, config: PreprocessingV2Config) -> Self {
        self.config.preprocessing_v2 = Some(config);
        self
    }

    pub fn with_processing_stage(mut self, stage: Box<dyn ProcessingStage>) -> Self {
        self.processing_stage = Some(stage);
        self
    }

    pub fn with_passthrough(mut self, enable: bool) -> Self {
        self.config.enable_passthrough = enable;
        self
    }

    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.config.processing_buffer_size = size;
        self
    }

    pub fn with_performance_monitoring(mut self, enable: bool) -> Self {
        self.config.enable_performance_monitoring = enable;
        self
    }

    pub fn with_stats_interval(mut self, interval: Duration) -> Self {
        self.config.stats_interval = interval;
        self
    }

    /// Build pipeline with enhanced DeepGI stage support
    pub fn build(self) -> Pipeline {
        let mut pipeline = Pipeline::new(self.config);
        if let Some(stage) = self.processing_stage {
            pipeline.add_processing_stage(stage);
        }
        pipeline
    }

    /// Quick setup for capture + preprocessing_v2 + preview pipeline
    pub fn build_capture_preprocessing_preview(
        device_index: i32,
        target_size: (u32, u32),
        enable_cuda: bool,
    ) -> Pipeline {
        let capture_config = CaptureConfig {
            device_index,
            source_id: 1,
            expected_colorspace: crate::ColorSpace::BT709,
        };

        let preprocessing_config = PreprocessingV2Config {
            target_size,
            use_cuda: enable_cuda,
            debug: true,
            ..Default::default()
        };

        let preview_config = PreviewConfig {
            enable_stats: true,
            stats_interval: Duration::from_secs(1),
        };

        PipelineBuilder::new()
            .with_capture_config(capture_config)
            .with_preprocessing_v2_config(preprocessing_config)
            .with_preview_config(preview_config)
            .with_performance_monitoring(true)
            .with_stats_interval(Duration::from_secs(1))
            .build()
    }

    /// Quick setup for headless processing pipeline
    pub fn build_headless_processing(
        device_index: i32,
        target_size: (u32, u32),
        enable_cuda: bool,
    ) -> Pipeline {
        let capture_config = CaptureConfig {
            device_index,
            source_id: 1,
            expected_colorspace: crate::ColorSpace::BT709,
        };

        let preprocessing_config = PreprocessingV2Config {
            target_size,
            use_cuda: enable_cuda,
            debug: false, // Less verbose for headless
            ..Default::default()
        };

        let preview_config = PreviewConfig {
            enable_stats: false, // Headless mode
            stats_interval: Duration::from_secs(5),
        };

        PipelineBuilder::new()
            .with_capture_config(capture_config)
            .with_preprocessing_v2_config(preprocessing_config)
            .with_preview_config(preview_config)
            .with_performance_monitoring(true)
            .with_stats_interval(Duration::from_secs(2))
            .build()
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.processing_buffer_size, 16);
        assert!(!config.enable_passthrough);
        assert!(config.enable_performance_monitoring);
        assert_eq!(config.stats_interval, Duration::from_secs(1));
    }

    #[test]
    fn test_enhanced_pipeline_builder() {
        let pipeline = PipelineBuilder::new()
            .with_passthrough(true)
            .with_buffer_size(32)
            .with_performance_monitoring(true)
            .with_stats_interval(Duration::from_millis(500))
            .build();
        
        assert!(pipeline.config.enable_passthrough);
        assert_eq!(pipeline.config.processing_buffer_size, 32);
        assert!(pipeline.config.enable_performance_monitoring);
        assert_eq!(pipeline.config.stats_interval, Duration::from_millis(500));
    }

    #[test]
    fn test_preprocessing_v2_integration() {
        let preprocessing_config = PreprocessingV2Config {
            target_size: (512, 512),
            use_cuda: false, // Disable for testing
            debug: true,
            ..Default::default()
        };

        let pipeline = PipelineBuilder::new()
            .with_preprocessing_v2_config(preprocessing_config)
            .build();
        
        assert!(pipeline.config.preprocessing_v2.is_some());
        if let Some(ref config) = pipeline.config.preprocessing_v2 {
            assert_eq!(config.target_size, (512, 512));
            assert!(!config.use_cuda);
            assert!(config.debug);
        }
    }

    #[test]
    fn test_stage_metrics() {
        let mut metrics = StageMetrics::new();
        
        // Test initial state
        assert_eq!(metrics.frames_processed, 0);
        assert_eq!(metrics.frames_failed, 0);
        assert_eq!(metrics.average_latency_ms(), 0.0);
        assert_eq!(metrics.success_rate(), 0.0);
        
        // Test successful processing
        metrics.update(1_000_000, true); // 1ms
        assert_eq!(metrics.frames_processed, 1);
        assert_eq!(metrics.average_latency_ms(), 1.0);
        assert_eq!(metrics.success_rate(), 100.0);
        
        // Test failed processing
        metrics.update(2_000_000, false); // 2ms
        assert_eq!(metrics.frames_failed, 1);
        assert_eq!(metrics.success_rate(), 50.0);
    }

    #[test]
    fn test_frame_info_stage() {
        let stage = FrameInfoStage::new();
        assert_eq!(stage.name(), "frame_info");
    }

    #[test]
    fn test_passthrough_stage() {
        let stage = PassthroughStage;
        assert_eq!(stage.name(), "passthrough");
    }

    #[test]
    fn test_pipeline_stats_calculation() {
        let mut stats = PipelineStats::default();
        stats.frames_captured = 100;
        stats.frames_processed = 95;
        stats.frames_dropped = 5;
        
        stats.calculate_efficiency();
        assert_eq!(stats.pipeline_efficiency, 95.0);
    }

    #[test]
    fn test_quick_builder_methods() {
        let pipeline1 = PipelineBuilder::build_capture_preprocessing_preview(0, (640, 480), false);
        assert!(pipeline1.config.preprocessing_v2.is_some());
        assert!(pipeline1.config.enable_performance_monitoring);
        
        let pipeline2 = PipelineBuilder::build_headless_processing(1, (512, 512), true);
        assert!(pipeline2.config.preprocessing_v2.is_some());
        if let Some(ref config) = pipeline2.config.preprocessing_v2 {
            assert!(config.use_cuda);
            assert!(!config.debug); // Headless should be less verbose
        }
    }
}
