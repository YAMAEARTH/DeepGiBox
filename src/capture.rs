use std::fmt;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
struct RawCaptureFrame {
    data: *const u8,
    width: i32,
    height: i32,
    row_bytes: i32,
    seq: u64,
}

extern "C" {
    fn decklink_capture_open(device_index: i32) -> bool;
    fn decklink_capture_get_frame(out: *mut RawCaptureFrame) -> bool;
    fn decklink_capture_close();
}

/// Raw DeckLink frame metadata (BGRA interleaved, 8 bits per channel).
///
/// `data_ptr` points at a C++ shim-managed buffer with length `data_len` bytes
/// (`row_bytes * height`). The buffer may be overwritten when a new frame arrives,
/// so callers should finish processing or copy the data before requesting another frame.
#[derive(Debug, Clone, Copy)]
pub struct RawFrame {
    pub data_ptr: *const u8,
    pub data_len: usize,
    pub width: u32,
    pub height: u32,
    pub row_bytes: u32,
    pub seq: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureError {
    OpenFailed,
    SessionClosed,
    InvalidFrameDimension,
    DataOverflow,
}

impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CaptureError::OpenFailed => write!(f, "failed to open capture device"),
            CaptureError::SessionClosed => write!(f, "capture session is closed"),
            CaptureError::InvalidFrameDimension => write!(f, "received invalid frame dimensions"),
            CaptureError::DataOverflow => write!(f, "frame data size overflowed"),
        }
    }
}

impl std::error::Error for CaptureError {}

pub struct CaptureSession {
    open: bool,
}

impl CaptureSession {
    pub fn open(device_index: i32) -> Result<Self, CaptureError> {
        let opened = unsafe { decklink_capture_open(device_index) };
        if opened {
            Ok(Self { open: true })
        } else {
            Err(CaptureError::OpenFailed)
        }
    }

    pub fn get_frame(&self) -> Result<Option<RawFrame>, CaptureError> {
        if !self.open {
            return Err(CaptureError::SessionClosed);
        }

        let mut raw = RawCaptureFrame::default();
        let ok = unsafe { decklink_capture_get_frame(&mut raw) };
        if !ok || raw.data.is_null() {
            return Ok(None);
        }

        if raw.width <= 0 || raw.height <= 0 || raw.row_bytes <= 0 {
            return Err(CaptureError::InvalidFrameDimension);
        }

        let width = raw.width as u32;
        let height = raw.height as u32;
        let row_bytes = raw.row_bytes as usize;
        let len = row_bytes
            .checked_mul(height as usize)
            .ok_or(CaptureError::DataOverflow)?;

        Ok(Some(RawFrame {
            data_ptr: raw.data,
            data_len: len,
            width,
            height,
            row_bytes: raw.row_bytes as u32,
            seq: raw.seq,
        }))
    }

    pub fn close(&mut self) {
        if self.open {
            unsafe { decklink_capture_close() };
            self.open = false;
        }
    }
}

impl Drop for CaptureSession {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_display() {
        let display = format!("{}", CaptureError::OpenFailed);
        assert!(display.contains("open"));
    }
}
