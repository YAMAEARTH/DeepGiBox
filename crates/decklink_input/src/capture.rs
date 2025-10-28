use common_io::{ColorSpace, FrameMeta, MemLoc, MemRef, PixelFormat, RawFramePacket};
use std::{
    env,
    ffi::CStr,
    fmt,
    os::raw::{c_char, c_int, c_void},
    sync::OnceLock,
};

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
struct RawCaptureFrame {
    data: *const u8,
    width: i32,
    height: i32,
    row_bytes: i32,
    seq: u64,
    gpu_data: *const u8,
    gpu_row_bytes: i32,
    gpu_device: u32,
}

extern "C" {
    fn decklink_capture_open(device_index: i32) -> bool;
    fn decklink_capture_get_frame(out: *mut RawCaptureFrame) -> bool;
    fn decklink_capture_close();
    fn decklink_capture_copy_host_region(offset: usize, len: usize, dst: *mut u8) -> bool;
}

fn prefer_gpu_capture() -> bool {
    true
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

#[derive(Debug)]
pub enum FrameCopyError {
    DataPointerNull,
    OffsetOutOfRange,
    CudaError(String),
}

impl fmt::Display for FrameCopyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrameCopyError::DataPointerNull => write!(f, "frame data pointer is null"),
            FrameCopyError::OffsetOutOfRange => write!(f, "requested range is outside frame data"),
            FrameCopyError::CudaError(msg) => write!(f, "cuda error: {}", msg),
        }
    }
}

impl std::error::Error for FrameCopyError {}

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
        let cpu_len = row_bytes
            .checked_mul(height as usize)
            .ok_or(CaptureError::DataOverflow)?;

        let prefer_gpu = prefer_gpu_capture();
        let use_gpu = prefer_gpu && !raw.gpu_data.is_null() && raw.gpu_row_bytes > 0;

        let (ptr, stride, len, loc) = if use_gpu {
            let stride = raw.gpu_row_bytes as usize;
            let len = stride
                .checked_mul(height as usize)
                .ok_or(CaptureError::DataOverflow)?;
            (
                raw.gpu_data as *mut u8,
                stride,
                len,
                MemLoc::Gpu {
                    device: raw.gpu_device,
                },
            )
        } else {
            (raw.data as *mut u8, row_bytes, cpu_len, MemLoc::Cpu)
        };

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
            crop_region: None,
        };

        // Build MemRef for the selected memory location
        let data = MemRef {
            ptr,
            len,
            stride,
            loc,
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

#[repr(C)]
#[derive(Clone, Copy)]
enum CudaMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

extern "C" {
    fn cudaSetDevice(device: c_int) -> c_int;
    fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: CudaMemcpyKind,
    ) -> c_int;
    fn cudaGetErrorString(error: c_int) -> *const c_char;
}

fn format_cuda_error(op: &str, code: c_int) -> String {
    unsafe {
        let ptr = cudaGetErrorString(code);
        if ptr.is_null() {
            return format!("{op} failed with CUDA error code {code}");
        }
        CStr::from_ptr(ptr)
            .to_str()
            .map(|msg| format!("{op} failed: {msg} ({code})"))
            .unwrap_or_else(|_| format!("{op} failed with CUDA error code {code}"))
    }
}

fn copy_device_to_host(device: u32, src: *const u8, dst: &mut [u8]) -> Result<(), FrameCopyError> {
    const CUDA_SUCCESS: c_int = 0;
    if dst.is_empty() {
        return Ok(());
    }
    if src.is_null() {
        return Err(FrameCopyError::DataPointerNull);
    }
    unsafe {
        let set_res = cudaSetDevice(device as c_int);
        if set_res != CUDA_SUCCESS {
            return Err(FrameCopyError::CudaError(format_cuda_error(
                "cudaSetDevice",
                set_res,
            )));
        }
        let copy_res = cudaMemcpy(
            dst.as_mut_ptr() as *mut c_void,
            src as *const c_void,
            dst.len(),
            CudaMemcpyKind::DeviceToHost,
        );
        if copy_res != CUDA_SUCCESS {
            return Err(FrameCopyError::CudaError(format_cuda_error(
                "cudaMemcpy",
                copy_res,
            )));
        }
    }
    Ok(())
}

pub fn copy_frame_region_to_host(
    packet: &RawFramePacket,
    offset: usize,
    dst: &mut [u8],
) -> Result<(), FrameCopyError> {
    if dst.is_empty() {
        return Ok(());
    }
    if packet.data.ptr.is_null() {
        return Err(FrameCopyError::DataPointerNull);
    }
    let end = offset
        .checked_add(dst.len())
        .ok_or(FrameCopyError::OffsetOutOfRange)?;
    if end > packet.data.len {
        return Err(FrameCopyError::OffsetOutOfRange);
    }

    match packet.data.loc {
        MemLoc::Cpu => unsafe {
            std::ptr::copy_nonoverlapping(packet.data.ptr.add(offset), dst.as_mut_ptr(), dst.len());
            Ok(())
        },
        MemLoc::Gpu { device } => {
            let src = unsafe { packet.data.ptr.add(offset) };
            match copy_device_to_host(device, src, dst) {
                Ok(()) => Ok(()),
                Err(FrameCopyError::CudaError(_)) => {
                    let ok = unsafe {
                        decklink_capture_copy_host_region(offset, dst.len(), dst.as_mut_ptr())
                    };
                    if ok {
                        Ok(())
                    } else {
                        Err(FrameCopyError::CudaError(String::from(
                            "cudaMemcpy failed and CPU fallback unavailable",
                        )))
                    }
                }
                Err(err) => Err(err),
            }
        }
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
