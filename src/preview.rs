use crate::packets::{RawFramePacket, MemLoc, PipelineError};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

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

/// Preview configuration
#[derive(Debug, Clone)]
pub struct PreviewConfig {
    pub enable_stats: bool,
    pub stats_interval: Duration,
}

impl Default for PreviewConfig {
    fn default() -> Self {
        Self {
            enable_stats: true,
            stats_interval: Duration::from_secs(1),
        }
    }
}

/// Statistics for preview performance
#[derive(Debug, Clone)]
pub struct PreviewStats {
    pub frames_rendered: u32,
    pub fps: f64,
    pub latency_ms: f64,
    pub last_seq: u64,
}

/// Preview stage that consumes RawFramePacket and displays frames
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

impl DeckLinkPreview {
    /// Create a new DeckLink preview instance
    pub fn new(config: PreviewConfig) -> Self {
        Self {
            config,
            is_initialized: false,
            is_enabled: false,
            frame_receiver: None,
            preview_thread: None,
            shutdown_flag: Arc::new(Mutex::new(false)),
            stats: Arc::new(Mutex::new(PreviewStats {
                frames_rendered: 0,
                fps: 0.0,
                latency_ms: 0.0,
                last_seq: 0,
            })),
            current_frame: Arc::new(Mutex::new(None)),
        }
    }

    /// Initialize the OpenGL preview system
    pub fn initialize_gl(&mut self) -> Result<(), PipelineError> {
        if self.is_initialized {
            return Ok(());
        }

        unsafe {
            if !decklink_preview_gl_create() {
                return Err(PipelineError::Preview("Failed to create GL preview helper".to_string()));
            }

            if !decklink_preview_gl_initialize_gl() {
                decklink_preview_gl_destroy();
                return Err(PipelineError::Preview("Failed to initialize OpenGL".to_string()));
            }
        }

        self.is_initialized = true;
        Ok(())
    }

    /// Start preview with frame packet input channel
    pub fn start(&mut self, frame_receiver: mpsc::Receiver<RawFramePacket>) -> Result<(), PipelineError> {
        if !self.is_initialized {
            return Err(PipelineError::Preview("Preview not initialized".to_string()));
        }

        if self.is_enabled {
            return Err(PipelineError::Preview("Preview already started".to_string()));
        }

        unsafe {
            if !decklink_preview_gl_enable() {
                return Err(PipelineError::Preview("Failed to enable GL preview".to_string()));
            }
        }

        self.frame_receiver = Some(frame_receiver);
        self.is_enabled = true;

        // Reset shutdown flag
        {
            let mut flag = self.shutdown_flag.lock().unwrap();
            *flag = false;
        }

        // Start preview thread for frame consumption
        let shutdown_flag = Arc::clone(&self.shutdown_flag);
        let stats = Arc::clone(&self.stats);
        let current_frame = Arc::clone(&self.current_frame);
        let config = self.config.clone();
        
        if let Some(receiver) = self.frame_receiver.take() {
            let handle = thread::spawn(move || {
                Self::frame_consumer_loop(receiver, shutdown_flag, stats, current_frame, config);
            });
            self.preview_thread = Some(handle);
        }

        Ok(())
    }

    /// Render the current frame (call from main/render thread)
    pub fn render(&self) -> bool {
        if !self.is_enabled {
            return false;
        }

        // The actual rendering is handled by the native DeckLink GL preview
        // This just triggers the render for the current frame
        unsafe { decklink_preview_gl_render() }
    }

    /// Stop the preview
    pub fn stop(&mut self) -> Result<(), PipelineError> {
        if !self.is_enabled {
            return Ok(());
        }

        // Signal shutdown
        {
            let mut flag = self.shutdown_flag.lock().unwrap();
            *flag = true;
        }

        // Wait for frame consumer thread to finish
        if let Some(handle) = self.preview_thread.take() {
            if let Err(e) = handle.join() {
                eprintln!("Preview thread panicked: {:?}", e);
            }
        }

        unsafe {
            decklink_preview_gl_disable();
        }

        self.is_enabled = false;
        self.frame_receiver = None;

        Ok(())
    }

    /// Cleanup preview resources
    pub fn cleanup(&mut self) {
        let _ = self.stop();
        
        if self.is_initialized {
            unsafe {
                decklink_preview_gl_destroy();
            }
            self.is_initialized = false;
        }
    }

    /// Get current preview statistics
    pub fn get_stats(&self) -> PreviewStats {
        let stats = self.stats.lock().unwrap();
        stats.clone()
    }

    /// Get the current sequence number from the native preview
    pub fn get_sequence(&self) -> u64 {
        unsafe { decklink_preview_gl_seq() }
    }

    /// Get the last timestamp from native preview
    pub fn get_last_timestamp_ns(&self) -> u64 {
        unsafe { decklink_preview_gl_last_timestamp_ns() }
    }

    /// Get the last latency from native preview
    pub fn get_last_latency_ns(&self) -> u64 {
        unsafe { decklink_preview_gl_last_latency_ns() }
    }

    /// Frame consumer loop running in separate thread
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
            // Check shutdown flag
            {
                let flag = shutdown_flag.lock().unwrap();
                if *flag {
                    break;
                }
            }

            // Try to receive a frame packet
            match receiver.recv_timeout(Duration::from_millis(10)) {
                Ok(packet) => {
                    // Update current frame for rendering
                    {
                        let mut current = current_frame.lock().unwrap();
                        *current = Some(packet.clone());
                    }

                    // Update statistics
                    if packet.meta.seq_no != last_seq {
                        frames_since_stats += 1;
                        last_seq = packet.meta.seq_no;
                    }

                    // Update stats periodically
                    if config.enable_stats && last_stats_update.elapsed() >= config.stats_interval {
                        let elapsed = last_stats_update.elapsed();
                        let fps = frames_since_stats as f64 / elapsed.as_secs_f64();
                        let latency_ns = unsafe { decklink_preview_gl_last_latency_ns() };
                        let latency_ms = latency_ns as f64 / 1_000_000.0;

                        {
                            let mut stats_guard = stats.lock().unwrap();
                            stats_guard.frames_rendered = frames_since_stats;
                            stats_guard.fps = fps;
                            stats_guard.latency_ms = latency_ms;
                            stats_guard.last_seq = last_seq;
                        }

                        if config.enable_stats {
                            println!("Preview: fps: {:.1}, latency: {:.2} ms", fps, latency_ms);
                        }

                        frames_since_stats = 0;
                        last_stats_update = Instant::now();
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // No frame received, continue
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Sender disconnected, exit
                    break;
                }
            }
        }
    }

    /// Check if preview is active
    pub fn is_active(&self) -> bool {
        self.is_enabled && self.preview_thread.is_some()
    }
}

impl Drop for DeckLinkPreview {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// Simple preview stage that can be used in a pipeline
pub struct PreviewStage {
    preview: DeckLinkPreview,
    frame_sender: Option<mpsc::Sender<RawFramePacket>>,
}

impl PreviewStage {
    pub fn new(config: PreviewConfig) -> Self {
        Self {
            preview: DeckLinkPreview::new(config),
            frame_sender: None,
        }
    }

    pub fn initialize_gl(&mut self) -> Result<(), PipelineError> {
        self.preview.initialize_gl()
    }

    pub fn start(&mut self) -> Result<mpsc::Sender<RawFramePacket>, PipelineError> {
        let (sender, receiver) = mpsc::channel();
        self.preview.start(receiver)?;
        self.frame_sender = Some(sender.clone());
        Ok(sender)
    }

    pub fn render(&self) -> bool {
        self.preview.render()
    }

    pub fn get_stats(&self) -> PreviewStats {
        self.preview.get_stats()
    }

    pub fn stop(&mut self) -> Result<(), PipelineError> {
        self.frame_sender = None;
        self.preview.stop()
    }

    pub fn cleanup(&mut self) {
        self.preview.cleanup()
    }

    pub fn is_active(&self) -> bool {
        self.preview.is_active()
    }
}

/// CPU-based frame renderer that converts RawFramePacket to displayable format
pub struct CpuFrameRenderer {
    target_width: u32,
    target_height: u32,
}

impl CpuFrameRenderer {
    pub fn new(target_width: u32, target_height: u32) -> Self {
        Self {
            target_width,
            target_height,
        }
    }

    /// Convert RawFramePacket to RGBA buffer for display
    pub fn render_to_rgba(&self, packet: &RawFramePacket) -> Result<Vec<u8>, PipelineError> {
        match &packet.mem {
            MemLoc::Cpu { .. } => {
                if let Some(data) = packet.as_slice() {
                    self.convert_frame_data(data, &packet.meta)
                } else {
                    Err(PipelineError::Preview("Failed to access frame data".to_string()))
                }
            }
            MemLoc::Cuda { .. } => {
                Err(PipelineError::Preview("CUDA frame rendering not implemented".to_string()))
            }
        }
    }

    fn convert_frame_data(&self, data: &[u8], meta: &crate::packets::FrameMeta) -> Result<Vec<u8>, PipelineError> {
        // For now, assume BGRA -> RGBA conversion
        if meta.pixfmt != crate::packets::PixelFormat::BGRA8 {
            return Err(PipelineError::Preview(
                format!("Unsupported pixel format: {:?}", meta.pixfmt)
            ));
        }

        let mut rgba_data = Vec::with_capacity(data.len());
        
        // Convert BGRA to RGBA
        for chunk in data.chunks_exact(4) {
            rgba_data.push(chunk[2]); // R
            rgba_data.push(chunk[1]); // G
            rgba_data.push(chunk[0]); // B
            rgba_data.push(chunk[3]); // A
        }

        Ok(rgba_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preview_config_default() {
        let config = PreviewConfig::default();
        assert!(config.enable_stats);
        assert_eq!(config.stats_interval, Duration::from_secs(1));
    }

    #[test]
    fn test_preview_creation() {
        let config = PreviewConfig::default();
        let preview = DeckLinkPreview::new(config);
        assert!(!preview.is_active());
    }

    #[test]
    fn test_cpu_renderer_creation() {
        let renderer = CpuFrameRenderer::new(1920, 1080);
        assert_eq!(renderer.target_width, 1920);
        assert_eq!(renderer.target_height, 1080);
    }
}
