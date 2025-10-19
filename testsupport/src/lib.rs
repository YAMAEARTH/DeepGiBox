use common_io::{ColorSpace, FrameMeta, MemLoc, MemRef, PixelFormat, RawFramePacket};

pub fn make_dummy_frame(w: u32, h: u32) -> RawFramePacket {
    let meta = FrameMeta {
        source_id: 0,
        width: w,
        height: h,
        pixfmt: PixelFormat::YUV422_8,
        colorspace: ColorSpace::BT709,
        frame_idx: 0,
        pts_ns: 0,
        t_capture_ns: 0,
        stride_bytes: w * 2,
    };
    let data = MemRef {
        ptr: std::ptr::null_mut(),
        len: 0,
        stride: meta.stride_bytes as usize,
        loc: MemLoc::Cpu,
    };
    RawFramePacket { meta, data }
}

/// Create a dummy GPU frame (requires CUDA device to be available)
/// This is a placeholder that returns a frame with null pointer
/// In real usage, the frame would come from DeckLink capture module
pub fn make_gpu_dummy_yuv422(w: u32, h: u32) -> RawFramePacket {
    let meta = FrameMeta {
        source_id: 0,
        width: w,
        height: h,
        pixfmt: PixelFormat::YUV422_8,
        colorspace: ColorSpace::BT709,
        frame_idx: 0,
        pts_ns: 0,
        t_capture_ns: 0,
        stride_bytes: w * 2,
    };
    let data = MemRef {
        ptr: std::ptr::null_mut(),
        len: (w * h * 2) as usize,
        stride: (w * 2) as usize,
        loc: MemLoc::Gpu { device: 0 },
    };
    RawFramePacket { meta, data }
}

/// Create a dummy GPU frame with NV12 format
pub fn make_gpu_dummy_nv12(w: u32, h: u32) -> RawFramePacket {
    let meta = FrameMeta {
        source_id: 0,
        width: w,
        height: h,
        pixfmt: PixelFormat::NV12,
        colorspace: ColorSpace::BT709,
        frame_idx: 0,
        pts_ns: 0,
        t_capture_ns: 0,
        stride_bytes: w,
    };
    let data = MemRef {
        ptr: std::ptr::null_mut(),
        len: ((w * h * 3) / 2) as usize, // Y plane + UV plane (half size)
        stride: w as usize,
        loc: MemLoc::Gpu { device: 0 },
    };
    RawFramePacket { meta, data }
}

/// Create a dummy GPU frame with BGRA8 format
pub fn make_gpu_dummy_bgra8(w: u32, h: u32) -> RawFramePacket {
    let meta = FrameMeta {
        source_id: 0,
        width: w,
        height: h,
        pixfmt: PixelFormat::BGRA8,
        colorspace: ColorSpace::SRGB,
        frame_idx: 0,
        pts_ns: 0,
        t_capture_ns: 0,
        stride_bytes: w * 4,
    };
    let data = MemRef {
        ptr: std::ptr::null_mut(),
        len: (w * h * 4) as usize,
        stride: (w * 4) as usize,
        loc: MemLoc::Gpu { device: 0 },
    };
    RawFramePacket { meta, data }
}
