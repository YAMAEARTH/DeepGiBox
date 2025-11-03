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
    hw_capture_time_ns: u64, // Hardware timestamp from DeckLink (nanoseconds)
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
    // Reference points for measuring capture latency
    first_hw_time_ns: Option<u64>,    // First STABLE hardware timestamp
    first_system_time_ns: Option<u64>, // System time at first stable frame callback
    last_hw_time_ns: u64,              // Last hardware timestamp (for detecting changes)
}

impl CaptureSession {
    pub fn open(device_index: i32) -> Result<Self, CaptureError> {
        let opened = unsafe { decklink_capture_open(device_index) };
        if opened {
            Ok(Self {
                open: true,
                source_id: device_index as u32,
                frame_count: 0,
                first_hw_time_ns: None,
                first_system_time_ns: None,
                last_hw_time_ns: 0,
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

        // Get current system time when frame becomes available in software
        let system_time_now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // Calculate TRUE capture timestamp using hardware timestamp
        // 
        // The hardware timestamp represents the presentation time (PTS) of the frame
        // in the DeckLink hardware clock domain. By comparing hardware time differences
        // with system time differences, we can measure the processing latency.
        //
        // IMPORTANT: Hardware timestamp may not increment on every frame initially
        // (DeckLink buffering, format changes, etc.). Wait for stable timestamps!
        //
        // Algorithm:
        //   - Wait for hardware timestamp to change (indicates stable operation)
        //   - Record hw_time₀ and system_time₀ when timestamp first changes
        //   - Current frame: 
        //       Δhw = hw_time - hw_time₀ (time since reference frame in hardware clock)
        //       Δsys = system_time - system_time₀ (time since reference callback in system clock)
        //       Processing latency L = Δsys - Δhw
        //       True capture time = system_time_now - L
        let t_capture_ns = if raw.hw_capture_time_ns > 0 {
            // Check if hardware timestamp has changed from last frame
            let hw_changed = raw.hw_capture_time_ns != self.last_hw_time_ns;
            self.last_hw_time_ns = raw.hw_capture_time_ns;
            
            // Initialize reference points when timestamp first becomes stable (changes consistently)
            if self.first_hw_time_ns.is_none() {
                if hw_changed && self.frame_count > 0 {
                    // Hardware timestamp is now changing - use this as reference
                    println!("[capture] Hardware timestamp now stable and incrementing");
                    println!("[capture]   Reference HW timestamp: {} ns", raw.hw_capture_time_ns);
                    println!("[capture]   System time at reference: {} ns", system_time_now);
                    self.first_hw_time_ns = Some(raw.hw_capture_time_ns);
                    self.first_system_time_ns = Some(system_time_now);
                } else if self.frame_count < 5 {
                    println!("[capture] Frame #{}: Waiting for stable HW timestamp (current: {} ns, changed: {})",
                        self.frame_count, raw.hw_capture_time_ns, hw_changed);
                }
                // Use system time until we have stable reference
                system_time_now
            } else {
                // Calculate elapsed time in both clock domains
                let hw_first = self.first_hw_time_ns.unwrap();
                let sys_first = self.first_system_time_ns.unwrap();
                
                let delta_hw_ns = raw.hw_capture_time_ns.saturating_sub(hw_first);
                let delta_sys_ns = system_time_now.saturating_sub(sys_first);
                
                // Processing latency = system elapsed - hardware elapsed
                // (positive means frames are delayed, negative means clock drift)
                let processing_latency_ns = delta_sys_ns as i64 - delta_hw_ns as i64;
                
                // True capture time = current system time - processing latency
                let true_capture_time = if processing_latency_ns >= 0 {
                    system_time_now.saturating_sub(processing_latency_ns as u64)
                } else {
                    // Negative latency indicates clock drift - use system time
                    system_time_now
                };
                
                if self.frame_count < 15 {
                    println!("[capture] Frame #{}: Δhw={:.2}ms, Δsys={:.2}ms, latency={:.2}ms", 
                        self.frame_count,
                        delta_hw_ns as f64 / 1_000_000.0,
                        delta_sys_ns as f64 / 1_000_000.0,
                        processing_latency_ns as f64 / 1_000_000.0);
                }
                
                true_capture_time
            }
        } else {
            // Fallback: no hardware timestamp - use system time directly
            // This will show near-zero capture latency (software overhead only)
            system_time_now
        };

        // Build FrameMeta according to INSTRUCTION.md
        let meta = FrameMeta {
            source_id: self.source_id,
            width,
            height,
            pixfmt: PixelFormat::YUV422_8, // DeckLink YUV422 8-bit (UYVY)
            colorspace: ColorSpace::BT709, // BT.709 colorspace
            frame_idx: self.frame_count,
            pts_ns: raw.seq, // Use sequence as PTS
            t_capture_ns, // Capture timestamp (converted from hardware clock if available)
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
