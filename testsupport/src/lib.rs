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
