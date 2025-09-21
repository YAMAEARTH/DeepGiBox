use anyhow::{bail, Result};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RawCaptureFrame {
    pub data: *const u8,
    pub width: i32,
    pub height: i32,
    pub row_bytes: i32,
    pub seq: u64,
}

extern "C" {
    fn decklink_capture_open(device_index: i32) -> bool;
    fn decklink_capture_get_frame(out: *mut RawCaptureFrame) -> bool;
    fn decklink_capture_close();
}

/// RAII wrapper that ensures the DeckLink capture pipeline is opened and closed safely.
pub struct CaptureSession {
    device_index: i32,
    is_open: bool,
}

impl CaptureSession {
    pub fn open(device_index: i32) -> Result<Self> {
        unsafe {
            if decklink_capture_open(device_index) {
                Ok(Self {
                    device_index,
                    is_open: true,
                })
            } else {
                bail!("Failed to open DeckLink capture (device {device_index})");
            }
        }
    }

    pub fn device_index(&self) -> i32 {
        self.device_index
    }

    pub fn get_frame(&self) -> Option<RawCaptureFrame> {
        unsafe {
            let mut frame = RawCaptureFrame {
                data: std::ptr::null(),
                width: 0,
                height: 0,
                row_bytes: 0,
                seq: 0,
            };
            if decklink_capture_get_frame(&mut frame as *mut RawCaptureFrame) {
                Some(frame)
            } else {
                None
            }
        }
    }
}

impl Drop for CaptureSession {
    fn drop(&mut self) {
        if self.is_open {
            unsafe {
                decklink_capture_close();
            }
            self.is_open = false;
        }
    }
}
