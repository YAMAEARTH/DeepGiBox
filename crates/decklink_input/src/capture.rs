use common_io::{ColorSpace, FrameMeta, MemLoc, MemRef, PixelFormat, RawFramePacket};
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
    source_id: u32,
    frame_count: u64,
}

impl CaptureSession {
    pub fn open(device_index: i32) -> Result<Self, CaptureError> {
        let opened = unsafe { decklink_capture_open(device_index) };
        if opened {
            Ok(Self {
                open: true,
                source_id: device_index as u32,
                frame_count: 0,
            })
        } else {
            Err(CaptureError::OpenFailed)
        }
    }

    pub fn get_frame(&mut self) -> Result<Option<RawFramePacket>, CaptureError> {
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

        // Capture timestamp (nanoseconds)
        let t_capture_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // Build FrameMeta according to INSTRUCTION.md
        let meta = FrameMeta {
            source_id: self.source_id,
            width,
            height,
            pixfmt: PixelFormat::YUV422_8, // DeckLink YUV422 8-bit (UYVY)
            colorspace: ColorSpace::BT709, // BT.709 colorspace
            frame_idx: self.frame_count,
            pts_ns: raw.seq, // Use sequence as PTS
            t_capture_ns,
            stride_bytes: raw.row_bytes as u32,
        };

        // Build MemRef for CPU data
        let data = MemRef {
            ptr: raw.data as *mut u8,
            len,
            stride: row_bytes,
            loc: MemLoc::Cpu,
        };

        self.frame_count += 1;

        Ok(Some(RawFramePacket { meta, data }))
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
