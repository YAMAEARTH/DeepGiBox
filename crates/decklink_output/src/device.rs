// DeckLink output device interface
// This module provides API for sending frames to DeckLink output devices

use common_io::{FrameMeta, MemLoc, PixelFormat, RawFramePacket};
use std::os::raw::{c_double, c_int};
use std::path::Path;

// C API from shim.cpp
extern "C" {
    fn decklink_output_open(device_index: c_int, width: c_int, height: c_int, fps: c_double) -> bool;
    fn decklink_output_send_frame_gpu(gpu_bgra_data: *const u8, gpu_pitch: c_int, width: c_int, height: c_int) -> bool;
    fn decklink_output_close();
}

/// Output device error type
#[derive(Debug)]
pub enum OutputDeviceError {
    /// Configuration error
    ConfigError(String),
    /// Frame size mismatch
    FrameSizeMismatch {
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },
    /// Device not available
    DeviceNotAvailable(String),
}

impl std::fmt::Display for OutputDeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputDeviceError::ConfigError(msg) => write!(f, "Config error: {}", msg),
            OutputDeviceError::FrameSizeMismatch {
                expected_width,
                expected_height,
                actual_width,
                actual_height,
            } => write!(
                f,
                "Frame size {}x{} doesn't match output size {}x{}",
                actual_width, actual_height, expected_width, expected_height
            ),
            OutputDeviceError::DeviceNotAvailable(msg) => {
                write!(f, "Device not available: {}", msg)
            }
        }
    }
}

impl std::error::Error for OutputDeviceError {}

/// Output request containing video and optional overlay frames
#[derive(Debug)]
pub struct OutputRequest<'a> {
    /// Main video frame to output
    pub video: Option<&'a RawFramePacket>,
    /// Optional overlay frame (for compositing)
    pub overlay: Option<&'a RawFramePacket>,
}

/// Video frame information
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Frame metadata
    pub meta: FrameMeta,
    /// Frame stride in bytes
    pub stride: usize,
    /// Frame data (CPU memory)
    pub bytes: Vec<u8>,
}

/// Output frame containing processed video and overlay
#[derive(Debug)]
pub struct OutputFrame {
    /// Processed video frame
    pub video: Option<VideoFrame>,
    /// Processed overlay frame
    pub overlay: Option<VideoFrame>,
}

/// DeckLink output device handle
pub struct OutputDevice {
    width: u32,
    height: u32,
    fps: f64,
    device_index: i32,
    is_open: bool,
}

impl OutputDevice {
    /// Get frame width
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get frame height
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get frame rate
    pub fn fps(&self) -> f64 {
        self.fps
    }

    /// Submit an output request to the device
    pub fn submit(&mut self, request: OutputRequest) -> Result<(), OutputDeviceError> {
        if !self.is_open {
            return Err(OutputDeviceError::DeviceNotAvailable("Device not open".to_string()));
        }
        
        // Get video frame from request
        let video = request.video.ok_or_else(|| {
            OutputDeviceError::ConfigError("No video frame provided".to_string())
        })?;
        
        // Validate frame size
        if video.meta.width != self.width || video.meta.height != self.height {
            return Err(OutputDeviceError::FrameSizeMismatch {
                expected_width: self.width,
                expected_height: self.height,
                actual_width: video.meta.width,
                actual_height: video.meta.height,
            });
        }
        
        // Check if frame is on GPU
        match video.data.loc {
            MemLoc::Gpu { .. } => {
                // Send GPU frame directly to DeckLink output
                let success = unsafe {
                    decklink_output_send_frame_gpu(
                        video.data.ptr,
                        video.data.stride as c_int,
                        self.width as c_int,
                        self.height as c_int,
                    )
                };
                
                if !success {
                    return Err(OutputDeviceError::DeviceNotAvailable(
                        "Failed to send GPU frame to DeckLink output".to_string()
                    ));
                }
            }
            MemLoc::Cpu => {
                return Err(OutputDeviceError::ConfigError(
                    "CPU frames not supported yet - frame must be on GPU".to_string()
                ));
            }
        }
        
        Ok(())
    }

    /// Get the last submitted frame
    pub fn last_frame(&self) -> Option<&OutputFrame> {
        // TODO: Implement frame tracking
        None
    }
}

impl Drop for OutputDevice {
    fn drop(&mut self) {
        if self.is_open {
            println!("🔌 Closing DeckLink output device {}...", self.device_index);
            unsafe {
                decklink_output_close();
            }
            self.is_open = false;
        }
    }
}

/// Initialize DeckLink output from configuration file
pub fn from_path<P: AsRef<Path>>(config_path: P) -> Result<OutputDevice, OutputDeviceError> {
    let config_path = config_path.as_ref();
    
    // Simple config parsing - check if it's 4K or 1080p from filename
    let (width, height, fps) = if config_path.to_string_lossy().contains("4k") {
        (3840, 2160, 30.0)
    } else {
        (1920, 1080, 60.0)
    };
    
    // Default to device 0 (first DeckLink output device)
    let device_index = 0;
    
    println!("📡 Initializing DeckLink output: {}x{}@{}fps", width, height, fps);
    println!("   Opening DeckLink device {}...", device_index);
    
    // Open DeckLink output device
    let success = unsafe {
        decklink_output_open(
            device_index as c_int,
            width as c_int,
            height as c_int,
            fps as c_double,
        )
    };
    
    if !success {
        return Err(OutputDeviceError::DeviceNotAvailable(
            format!("Failed to open DeckLink output device {}", device_index)
        ));
    }
    
    println!("   ✓ DeckLink output device {} opened successfully!", device_index);
    
    Ok(OutputDevice {
        width,
        height,
        fps,
        device_index,
        is_open: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_path_1080p() {
        let device = from_path("configs/dev_1080p60_yuv422_fp16_trt.toml").unwrap();
        assert_eq!(device.width(), 1920);
        assert_eq!(device.height(), 1080);
        assert_eq!(device.fps(), 60.0);
    }

    #[test]
    fn test_from_path_4k() {
        let device = from_path("configs/dev_4k30_yuv422_fp16_trt.toml").unwrap();
        assert_eq!(device.width(), 3840);
        assert_eq!(device.height(), 2160);
        assert_eq!(device.fps(), 30.0);
    }
}
