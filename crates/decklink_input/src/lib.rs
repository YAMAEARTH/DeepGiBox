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
