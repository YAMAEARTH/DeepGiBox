// DeckLink output device interface
// This module provides API for sending frames to DeckLink output devices

use common_io::{FrameMeta, MemLoc, PixelFormat, RawFramePacket};
use std::os::raw::{c_double, c_int};
use std::path::Path;

// C API from shim.cpp
extern "C" {
    fn decklink_output_open(device_index: c_int, width: c_int, height: c_int, fps: c_double) -> bool;
    fn decklink_output_send_frame_gpu(gpu_bgra_data: *const u8, gpu_pitch: c_int, width: c_int, height: c_int) -> bool;
    fn decklink_output_schedule_frame_gpu(gpu_bgra_data: *const u8, gpu_pitch: c_int, width: c_int, height: c_int, display_time: u64, display_duration: u64) -> bool;
    fn decklink_output_start_scheduled_playback(start_time: u64, time_scale: c_double) -> bool;
    fn decklink_output_stop_scheduled_playback() -> bool;
    fn decklink_output_close();
    
    // Hardware keying functions
    fn decklink_keyer_enable_internal() -> bool;
    fn decklink_keyer_set_level(level: u8) -> bool;
    fn decklink_keyer_disable() -> bool;
    fn decklink_set_video_output_connection(connection: i64) -> bool;
    fn decklink_get_connection_sdi() -> i64;
    
    // Get detected input format (for internal keying sync)
    fn decklink_get_detected_input_format(
        out_width: *mut c_int,
        out_height: *mut c_int,
        out_fps: *mut c_double,
        out_mode: *mut u32,
    ) -> bool;
    
    // Get frame timing from display mode (requirement #1)
    fn decklink_output_get_frame_timing(out_frame_duration: *mut i64, out_time_scale: *mut i64) -> bool;
    
    // Get hardware reference clock (requirement #3)
    fn decklink_output_get_hardware_time(out_time: *mut u64, out_timebase: *mut u64) -> bool;
    
    // Get buffered frame count (requirement #4)
    fn decklink_output_get_buffered_frame_count(out_count: *mut u32) -> bool;
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
    scheduled_playback_active: bool,
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
    
    /// Enable hardware internal keying
    pub fn enable_internal_keying(&mut self) -> Result<(), OutputDeviceError> {
        if !self.is_open {
            return Err(OutputDeviceError::DeviceNotAvailable("Device not open".to_string()));
        }
        
        let success = unsafe { decklink_keyer_enable_internal() };
        if !success {
            return Err(OutputDeviceError::ConfigError(
                "Failed to enable internal keyer (device may not support hardware keying)".to_string()
            ));
        }
        
        println!("✓ Hardware internal keying enabled");
        Ok(())
    }
    
    /// Set keyer level (0-255, 255 = fully visible overlay)
    pub fn set_keyer_level(&mut self, level: u8) -> Result<(), OutputDeviceError> {
        if !self.is_open {
            return Err(OutputDeviceError::DeviceNotAvailable("Device not open".to_string()));
        }
        
        let success = unsafe { decklink_keyer_set_level(level) };
        if !success {
            return Err(OutputDeviceError::ConfigError(
                "Failed to set keyer level".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Disable hardware internal keying
    pub fn disable_internal_keying(&mut self) -> Result<(), OutputDeviceError> {
        if !self.is_open {
            return Err(OutputDeviceError::DeviceNotAvailable("Device not open".to_string()));
        }
        
        let success = unsafe { decklink_keyer_disable() };
        if !success {
            return Err(OutputDeviceError::ConfigError(
                "Failed to disable internal keyer".to_string()
            ));
        }
        
        println!("✓ Hardware internal keying disabled");
        Ok(())
    }
    
    /// Set video output connection to SDI
    pub fn set_sdi_output(&mut self) -> Result<(), OutputDeviceError> {
        if !self.is_open {
            return Err(OutputDeviceError::DeviceNotAvailable("Device not open".to_string()));
        }
        
        let sdi_conn = unsafe { decklink_get_connection_sdi() };
        let success = unsafe { decklink_set_video_output_connection(sdi_conn) };
        
        if !success {
            return Err(OutputDeviceError::ConfigError(
                "Failed to set SDI output connection".to_string()
            ));
        }
        
        Ok(())
    }

    /// Submit an output request to the device (synchronous mode)
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
    
    /// Schedule a frame for async playback
    pub fn schedule_frame(&mut self, request: OutputRequest, display_time: u64, display_duration: u64) -> Result<(), OutputDeviceError> {
        if !self.is_open {
            return Err(OutputDeviceError::DeviceNotAvailable("Device not open".to_string()));
        }
        
        // For hardware internal keying: use overlay (key) if provided, otherwise use video (fill)
        // When internal keying is enabled, we send the BGRA overlay as the key signal
        let frame_to_schedule = if let Some(overlay) = request.overlay {
            overlay  // Use overlay (BGRA key signal) for internal keying
        } else if let Some(video) = request.video {
            video    // Fallback to video (fill signal) if no overlay
        } else {
            return Err(OutputDeviceError::ConfigError("No video or overlay frame provided".to_string()));
        };
        
        // Validate frame size
        if frame_to_schedule.meta.width != self.width || frame_to_schedule.meta.height != self.height {
            return Err(OutputDeviceError::FrameSizeMismatch {
                expected_width: self.width,
                expected_height: self.height,
                actual_width: frame_to_schedule.meta.width,
                actual_height: frame_to_schedule.meta.height,
            });
        }
        
        // Check if frame is on GPU
        match frame_to_schedule.data.loc {
            MemLoc::Gpu { .. } => {
                // Schedule GPU frame for async playback
                let success = unsafe {
                    decklink_output_schedule_frame_gpu(
                        frame_to_schedule.data.ptr,
                        frame_to_schedule.data.stride as c_int,
                        self.width as c_int,
                        self.height as c_int,
                        display_time,
                        display_duration,
                    )
                };
                
                if !success {
                    return Err(OutputDeviceError::DeviceNotAvailable(
                        "Failed to schedule GPU frame for DeckLink output".to_string()
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
    
    /// Start scheduled playback (async mode)
    pub fn start_scheduled_playback(&mut self, start_time: u64, time_scale: f64) -> Result<(), OutputDeviceError> {
        if !self.is_open {
            return Err(OutputDeviceError::DeviceNotAvailable("Device not open".to_string()));
        }
        
        let success = unsafe {
            decklink_output_start_scheduled_playback(start_time, time_scale)
        };
        
        if !success {
            return Err(OutputDeviceError::ConfigError(
                "Failed to start scheduled playback".to_string()
            ));
        }
        
        self.scheduled_playback_active = true;
        Ok(())
    }
    
    /// Stop scheduled playback
    pub fn stop_scheduled_playback(&mut self) -> Result<(), OutputDeviceError> {
        if !self.is_open {
            return Err(OutputDeviceError::DeviceNotAvailable("Device not open".to_string()));
        }
        
        if !self.scheduled_playback_active {
            return Ok(()); // Already stopped
        }
        
        let success = unsafe {
            decklink_output_stop_scheduled_playback()
        };
        
        if !success {
            return Err(OutputDeviceError::ConfigError(
                "Failed to stop scheduled playback".to_string()
            ));
        }
        
        self.scheduled_playback_active = false;
        Ok(())
    }
    
    /// Check if scheduled playback is active
    pub fn is_scheduled_playback_active(&self) -> bool {
        self.scheduled_playback_active
    }
    
    /// Get frame timing from display mode (requirement #1)
    pub fn get_frame_timing(&self) -> Result<(i64, i64), OutputDeviceError> {
        if !self.is_open {
            return Err(OutputDeviceError::DeviceNotAvailable("Device not open".to_string()));
        }
        
        let mut frame_duration = 0i64;
        let mut time_scale = 0i64;
        
        let success = unsafe {
            decklink_output_get_frame_timing(&mut frame_duration as *mut i64, &mut time_scale as *mut i64)
        };
        
        if !success {
            return Err(OutputDeviceError::ConfigError(
                "Failed to get frame timing from display mode".to_string()
            ));
        }
        
        Ok((frame_duration, time_scale))
    }
    
    /// Get hardware reference clock time (requirement #3)
    /// Returns (time_in_ticks, timebase_hz)
    /// Use DIFFERENCE between calls to calculate elapsed time
    pub fn get_hardware_time(&self) -> Result<(u64, u64), OutputDeviceError> {
        if !self.is_open {
            return Err(OutputDeviceError::DeviceNotAvailable("Device not open".to_string()));
        }
        
        let mut time = 0u64;
        let mut timebase = 0u64;
        
        let success = unsafe {
            decklink_output_get_hardware_time(&mut time as *mut u64, &mut timebase as *mut u64)
        };
        
        if !success {
            return Err(OutputDeviceError::ConfigError(
                "Failed to get hardware reference clock".to_string()
            ));
        }
        
        Ok((time, timebase))
    }
    
    /// Get buffered video frame count (requirement #4)
    pub fn get_buffered_frame_count(&self) -> Result<u32, OutputDeviceError> {
        if !self.is_open {
            return Err(OutputDeviceError::DeviceNotAvailable("Device not open".to_string()));
        }
        
        let mut count = 0u32;
        
        let success = unsafe {
            decklink_output_get_buffered_frame_count(&mut count as *mut u32)
        };
        
        if !success {
            return Err(OutputDeviceError::ConfigError(
                "Failed to get buffered frame count".to_string()
            ));
        }
        
        Ok(count)
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
            
            // Stop scheduled playback if active
            if self.scheduled_playback_active {
                let _ = self.stop_scheduled_playback();
            }
            
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
        scheduled_playback_active: false,
    })
}

/// Get detected input format for internal keying sync
pub fn get_detected_input_format() -> Option<(u32, u32, f64)> {
    let mut width: c_int = 0;
    let mut height: c_int = 0;
    let mut fps: c_double = 0.0;
    let mut mode: u32 = 0;
    
    let success = unsafe {
        decklink_get_detected_input_format(&mut width, &mut height, &mut fps, &mut mode)
    };
    
    if success && width > 0 && height > 0 && fps > 0.0 {
        Some((width as u32, height as u32, fps))
    } else {
        None
    }
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
