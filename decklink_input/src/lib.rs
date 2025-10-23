pub mod capture;

// Re-export common I/O types for convenience
pub use capture::{copy_frame_region_to_host, FrameCopyError};
pub use common_io::{ColorSpace, FrameMeta, MemLoc, MemRef, PixelFormat, RawFramePacket};

use libc::{c_char, int32_t};
use std::ffi::CStr;
use std::slice;

#[repr(C)]
struct DLDeviceList {
    names: *mut *mut c_char,
    count: int32_t,
}

extern "C" {
    fn decklink_list_devices() -> DLDeviceList;
    fn decklink_free_device_list(list: DLDeviceList);
}

// Safe Rust wrapper
pub fn devicelist() -> Vec<String> {
    let list = unsafe { decklink_list_devices() };
    if list.names.is_null() || list.count <= 0 {
        return Vec::new();
    }

    let names_ptrs =
        unsafe { slice::from_raw_parts(list.names as *const *const c_char, list.count as usize) };

    let mut out = Vec::with_capacity(names_ptrs.len());
    for &p in names_ptrs {
        if p.is_null() {
            continue;
        }
        let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
        out.push(s);
    }

    unsafe { decklink_free_device_list(list) };
    out
}

// Optional C ABI export if needed by other languages
#[no_mangle]
pub extern "C" fn decklink_devicelist_count() -> i32 {
    devicelist().len() as i32
}

// ---- Output API ----

extern "C" {
    fn decklink_output_open(device_index: int32_t, width: int32_t, height: int32_t, fps: f64) -> bool;
    fn decklink_output_send_frame(bgra_data: *const u8, width: int32_t, height: int32_t) -> bool;
    fn decklink_output_send_frame_gpu(gpu_bgra_data: *const u8, gpu_pitch: int32_t, width: int32_t, height: int32_t) -> bool;
    fn decklink_output_close();
}

pub struct OutputDevice {
    device_index: i32,
    width: i32,
    height: i32,
    fps: f64,
}

impl OutputDevice {
    /// Open DeckLink output device
    pub fn open(device_index: i32, width: i32, height: i32, fps: f64) -> Result<Self, String> {
        let success = unsafe { decklink_output_open(device_index as int32_t, width as int32_t, height as int32_t, fps) };
        if !success {
            return Err(format!("Failed to open DeckLink output device {}", device_index));
        }
        Ok(Self { device_index, width, height, fps })
    }

    /// Send BGRA frame to output (from CPU memory)
    pub fn send_frame(&self, bgra_data: &[u8]) -> Result<(), String> {
        let expected_size = (self.width * self.height * 4) as usize;
        if bgra_data.len() != expected_size {
            return Err(format!("Invalid frame size: expected {}, got {}", expected_size, bgra_data.len()));
        }

        let success = unsafe {
            decklink_output_send_frame(bgra_data.as_ptr(), self.width as int32_t, self.height as int32_t)
        };

        if !success {
            return Err("Failed to send frame to DeckLink output".to_string());
        }

        Ok(())
    }

    /// Send BGRA frame to output directly from GPU (zero-copy)
    pub fn send_frame_gpu(&self, gpu_bgra_ptr: *const u8, gpu_pitch: usize) -> Result<(), String> {
        let success = unsafe {
            decklink_output_send_frame_gpu(
                gpu_bgra_ptr,
                gpu_pitch as int32_t,
                self.width as int32_t,
                self.height as int32_t
            )
        };

        if !success {
            return Err("Failed to send GPU frame to DeckLink output".to_string());
        }

        Ok(())
    }

    pub fn width(&self) -> i32 { self.width }
    pub fn height(&self) -> i32 { self.height }
    pub fn fps(&self) -> f64 { self.fps }
}

impl Drop for OutputDevice {
    fn drop(&mut self) {
        unsafe { decklink_output_close(); }
    }
}
