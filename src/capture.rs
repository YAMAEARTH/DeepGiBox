use crate::packets::{FrameMeta, PixelFormat, ColorSpace, RawFramePacket, PipelineError};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

extern "C" {
    fn decklink_capture_open(device_index: i32) -> bool;
    fn decklink_capture_close();
    fn decklink_capture_get_frame(out: *mut CaptureFrame) -> bool;
}

#[repr(C)]
struct CaptureFrame {
    data: *const u8,
    width: i32,
    height: i32,
    row_bytes: i32,
    seq: u64,
}

/// DeckLink capture configuration
#[derive(Debug, Clone)]
pub struct CaptureConfig {
    pub device_index: i32,
    pub source_id: u32,
    pub expected_colorspace: ColorSpace,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            device_index: 0,
            source_id: 0,
            expected_colorspace: ColorSpace::BT709,
        }
    }
}

/// DeckLink capture stage that outputs RawFramePacket
pub struct DeckLinkCapture {
    config: CaptureConfig,
    is_open: bool,
    frame_sender: Option<mpsc::Sender<RawFramePacket>>,
    capture_thread: Option<thread::JoinHandle<()>>,
    shutdown_flag: Arc<Mutex<bool>>,
}

impl DeckLinkCapture {
    /// Create a new DeckLink capture instance
    pub fn new(config: CaptureConfig) -> Self {
        Self {
            config,
            is_open: false,
            frame_sender: None,
            capture_thread: None,
            shutdown_flag: Arc::new(Mutex::new(false)),
        }
    }

    /// Start capturing frames and return a receiver for RawFramePackets
    pub fn start(&mut self) -> Result<mpsc::Receiver<RawFramePacket>, PipelineError> {
        if self.is_open {
            return Err(PipelineError::Capture("Capture already started".to_string()));
        }

        // Open the DeckLink device
        unsafe {
            if !decklink_capture_open(self.config.device_index) {
                return Err(PipelineError::Capture(
                    format!("Failed to open DeckLink device {}", self.config.device_index)
                ));
            }
        }

        self.is_open = true;

        // Create channel for frame packets
        let (sender, receiver) = mpsc::channel();
        self.frame_sender = Some(sender.clone());

        // Reset shutdown flag
        {
            let mut flag = self.shutdown_flag.lock().unwrap();
            *flag = false;
        }

        // Start capture thread
        let config = self.config.clone();
        let shutdown_flag = Arc::clone(&self.shutdown_flag);
        
        let handle = thread::spawn(move || {
            Self::capture_loop(sender, config, shutdown_flag);
        });

        self.capture_thread = Some(handle);

        Ok(receiver)
    }

    /// Stop capturing frames
    pub fn stop(&mut self) -> Result<(), PipelineError> {
        if !self.is_open {
            return Ok(());
        }

        // Signal shutdown
        {
            let mut flag = self.shutdown_flag.lock().unwrap();
            *flag = true;
        }

        // Wait for capture thread to finish
        if let Some(handle) = self.capture_thread.take() {
            if let Err(e) = handle.join() {
                eprintln!("Capture thread panicked: {:?}", e);
            }
        }

        // Close DeckLink device
        unsafe {
            decklink_capture_close();
        }

        self.is_open = false;
        self.frame_sender = None;

        Ok(())
    }

    /// Main capture loop running in separate thread
    fn capture_loop(
        sender: mpsc::Sender<RawFramePacket>,
        config: CaptureConfig,
        shutdown_flag: Arc<Mutex<bool>>,
    ) {
        let mut last_seq = 0u64;

        loop {
            // Check shutdown flag
            {
                let flag = shutdown_flag.lock().unwrap();
                if *flag {
                    break;
                }
            }

            // Try to get a frame
            let mut frame = CaptureFrame {
                data: std::ptr::null(),
                width: 0,
                height: 0,
                row_bytes: 0,
                seq: 0,
            };

            let has_frame = unsafe { decklink_capture_get_frame(&mut frame as *mut CaptureFrame) };

            if !has_frame || frame.data.is_null() {
                // No frame available, sleep briefly
                thread::sleep(std::time::Duration::from_millis(1));
                continue;
            }

            // Skip duplicate frames
            if frame.seq <= last_seq {
                continue;
            }
            last_seq = frame.seq;

            // Convert frame to RawFramePacket
            if let Ok(packet) = Self::convert_frame_to_packet(frame, &config) {
                if sender.send(packet).is_err() {
                    // Receiver disconnected, exit
                    break;
                }
            }
        }
    }

    /// Convert a raw capture frame to RawFramePacket
    fn convert_frame_to_packet(
        frame: CaptureFrame,
        config: &CaptureConfig,
    ) -> Result<RawFramePacket, PipelineError> {
        if frame.data.is_null() || frame.width <= 0 || frame.height <= 0 {
            return Err(PipelineError::Capture("Invalid frame data".to_string()));
        }

        let width = frame.width as u32;
        let height = frame.height as u32;
        let stride = frame.row_bytes as u32;
        let data_size = (stride * height) as usize;

        // Copy frame data to owned buffer
        let buffer = unsafe {
            std::slice::from_raw_parts(frame.data, data_size).to_vec()
        };

        // Generate timestamp
        let pts_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        // Create frame metadata
        let meta = FrameMeta {
            source_id: config.source_id,
            width,
            height,
            stride,
            pixfmt: PixelFormat::BGRA8, // DeckLink shim converts to BGRA
            colorspace: config.expected_colorspace,
            pts_ns,
            timecode: None,
            seq_no: frame.seq,
        };

        Ok(RawFramePacket::new_cpu(buffer, meta))
    }

    /// Get current frame count (if capturing)
    pub fn frame_count(&self) -> u64 {
        // This could be enhanced to track frame count internally
        0
    }

    /// Check if capture is currently active
    pub fn is_active(&self) -> bool {
        self.is_open && self.capture_thread.is_some()
    }
}

impl Drop for DeckLinkCapture {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// Utility function to list available DeckLink devices
pub fn list_decklink_devices() -> Vec<String> {
    crate::devicelist()
}

/// Create a simple capture stage that can be used in a pipeline
pub struct CaptureStage {
    capture: DeckLinkCapture,
    receiver: Option<mpsc::Receiver<RawFramePacket>>,
}

impl CaptureStage {
    pub fn new(config: CaptureConfig) -> Self {
        Self {
            capture: DeckLinkCapture::new(config),
            receiver: None,
        }
    }

    pub fn start(&mut self) -> Result<(), PipelineError> {
        let receiver = self.capture.start()?;
        self.receiver = Some(receiver);
        Ok(())
    }

    pub fn get_next_frame(&mut self) -> Result<Option<RawFramePacket>, PipelineError> {
        if let Some(ref receiver) = self.receiver {
            match receiver.try_recv() {
                Ok(packet) => Ok(Some(packet)),
                Err(mpsc::TryRecvError::Empty) => Ok(None),
                Err(mpsc::TryRecvError::Disconnected) => {
                    Err(PipelineError::Capture("Capture thread disconnected".to_string()))
                }
            }
        } else {
            Err(PipelineError::Capture("Capture not started".to_string()))
        }
    }

    pub fn stop(&mut self) -> Result<(), PipelineError> {
        self.receiver = None;
        self.capture.stop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capture_config_default() {
        let config = CaptureConfig::default();
        assert_eq!(config.device_index, 0);
        assert_eq!(config.source_id, 0);
        assert_eq!(config.expected_colorspace, ColorSpace::BT709);
    }

    #[test]
    fn test_capture_creation() {
        let config = CaptureConfig::default();
        let capture = DeckLinkCapture::new(config);
        assert!(!capture.is_active());
    }
}
