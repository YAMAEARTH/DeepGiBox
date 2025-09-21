// DeepGI Pipeline - DeckLink Rust Library
use libc::c_char;
use std::ffi::CStr;
use std::slice;

// Export new pipeline modules
pub mod packets;
pub mod capture;
pub mod preview;
pub mod pipeline;
pub mod headless;

// Re-export key types for convenience
pub use packets::*;
pub use capture::{CaptureConfig, CaptureStage, DeckLinkCapture};
pub use preview::{PreviewConfig, PreviewStage, DeckLinkPreview, PreviewStats};
pub use pipeline::{Pipeline, PipelineBuilder, PipelineConfig, ProcessingStage, FrameInfoStage};
pub use headless::{HeadlessProcessor, HeadlessConfig, PipelineStage};

#[repr(C)]
struct DLDeviceList {
    names: *mut *mut c_char,
    count: i32,
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
