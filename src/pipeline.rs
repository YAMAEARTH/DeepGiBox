use crate::packets::{RawFramePacket, PipelineError};
use crate::capture::{CaptureStage, CaptureConfig};
use crate::preview::{PreviewStage, PreviewConfig};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub capture: CaptureConfig,
    pub preview: PreviewConfig,
    pub processing_buffer_size: usize,
    pub enable_passthrough: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            capture: CaptureConfig::default(),
            preview: PreviewConfig::default(),
            processing_buffer_size: 16,
            enable_passthrough: false,
        }
    }
}

/// Pipeline statistics
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub frames_captured: u64,
    pub frames_processed: u64,
    pub frames_previewed: u64,
    pub frames_dropped: u64,
    pub capture_fps: f64,
    pub processing_fps: f64,
    pub preview_fps: f64,
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
        }
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

/// Main pipeline orchestrator
pub struct Pipeline {
    config: PipelineConfig,
    capture_stage: CaptureStage,
    preview_stage: PreviewStage,
    processing_stage: Option<Box<dyn ProcessingStage>>,
    stats: Arc<Mutex<PipelineStats>>,
    is_running: bool,
    processing_thread: Option<thread::JoinHandle<()>>,
    shutdown_flag: Arc<Mutex<bool>>,
}

impl Pipeline {
    /// Create a new pipeline with the given configuration
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            capture_stage: CaptureStage::new(config.capture.clone()),
            preview_stage: PreviewStage::new(config.preview.clone()),
            processing_stage: None,
            config,
            stats: Arc::new(Mutex::new(PipelineStats::default())),
            is_running: false,
            processing_thread: None,
            shutdown_flag: Arc::new(Mutex::new(false)),
        }
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

    /// Start the pipeline
    pub fn start(&mut self) -> Result<(), PipelineError> {
        if self.is_running {
            return Err(PipelineError::Processing("Pipeline already running".to_string()));
        }

        // Reset stats
        {
            let mut stats = self.stats.lock().unwrap();
            *stats = PipelineStats::default();
        }

        // Start capture
        self.capture_stage.start()?;

        // Start preview (this creates the input channel)
        let preview_sender = self.preview_stage.start()?;

        // Reset shutdown flag
        {
            let mut flag = self.shutdown_flag.lock().unwrap();
            *flag = false;
        }

        // Start processing thread
        let stats = Arc::clone(&self.stats);
        let shutdown_flag = Arc::clone(&self.shutdown_flag);
        let config = self.config.clone();
        
        let mut capture_stage = CaptureStage::new(config.capture.clone());
        capture_stage.start()?;
        
        // Move processing stage to thread if available
        let processing_stage = self.processing_stage.take();
        
        let handle = thread::spawn(move || {
            Self::processing_loop(
                capture_stage,
                preview_sender,
                processing_stage,
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

    /// Get current pipeline statistics
    pub fn get_stats(&self) -> PipelineStats {
        let stats = self.stats.lock().unwrap();
        stats.clone()
    }

    /// Get preview statistics
    pub fn get_preview_stats(&self) -> crate::preview::PreviewStats {
        self.preview_stage.get_stats()
    }

    /// Check if pipeline is running
    pub fn is_running(&self) -> bool {
        self.is_running
    }

    /// Processing loop running in separate thread
    fn processing_loop(
        mut capture_stage: CaptureStage,
        preview_sender: mpsc::Sender<RawFramePacket>,
        mut processing_stage: Option<Box<dyn ProcessingStage>>,
        stats: Arc<Mutex<PipelineStats>>,
        shutdown_flag: Arc<Mutex<bool>>,
        config: PipelineConfig,
    ) {
        let mut frames_processed = 0u64;
        let mut frames_dropped = 0u64;

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
                    // Process frame through custom stage if available
                    let processed_frame = if let Some(ref mut stage) = processing_stage {
                        match stage.process(raw_frame) {
                            Ok(frame) => frame,
                            Err(e) => {
                                eprintln!("Processing error in {}: {:?}", stage.name(), e);
                                frames_dropped += 1;
                                continue;
                            }
                        }
                    } else if config.enable_passthrough {
                        // Use passthrough if enabled and no custom stage
                        let mut passthrough = PassthroughStage;
                        match passthrough.process(raw_frame) {
                            Ok(frame) => frame,
                            Err(e) => {
                                eprintln!("Passthrough error: {:?}", e);
                                frames_dropped += 1;
                                continue;
                            }
                        }
                    } else {
                        raw_frame
                    };

                    // Send to preview
                    if let Err(_) = preview_sender.send(processed_frame) {
                        // Preview disconnected, exit
                        break;
                    }

                    frames_processed += 1;

                    // Update stats periodically
                    if frames_processed % 30 == 0 {
                        let mut stats_guard = stats.lock().unwrap();
                        stats_guard.frames_processed = frames_processed;
                        stats_guard.frames_dropped = frames_dropped;
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

/// Builder for creating pipelines with fluent interface
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

    pub fn build(self) -> Pipeline {
        let mut pipeline = Pipeline::new(self.config);
        if let Some(stage) = self.processing_stage {
            pipeline.add_processing_stage(stage);
        }
        pipeline
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
    }

    #[test]
    fn test_pipeline_builder() {
        let pipeline = PipelineBuilder::new()
            .with_passthrough(true)
            .with_buffer_size(32)
            .build();
        
        assert!(pipeline.config.enable_passthrough);
        assert_eq!(pipeline.config.processing_buffer_size, 32);
    }

    #[test]
    fn test_frame_info_stage() {
        let mut stage = FrameInfoStage::new();
        assert_eq!(stage.name(), "frame_info");
    }

    #[test]
    fn test_passthrough_stage() {
        let mut stage = PassthroughStage;
        assert_eq!(stage.name(), "passthrough");
    }
}
