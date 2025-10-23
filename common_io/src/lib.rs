/// Memory location type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemLoc {
    /// CPU system memory
    Cpu,
    /// GPU device memory
    Gpu { device: u32 },
}

/// Memory reference with location info
#[derive(Debug, Clone, Copy)]
pub struct MemRef {
    /// Pointer to memory
    pub ptr: *mut u8,
    /// Total length in bytes
    pub len: usize,
    /// Stride/row bytes
    pub stride: usize,
    /// Memory location (CPU or GPU)
    pub loc: MemLoc,
}

unsafe impl Send for MemRef {}
unsafe impl Sync for MemRef {}

/// Pixel format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// 8-bit YUV 4:2:2 (UYVY)
    YUV422_8,
    /// 10-bit YUV 4:2:2
    YUV422_10,
    /// 8-bit RGB
    RGB8,
    /// 8-bit RGBA
    RGBA8,
}

/// Color space
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// BT.709 (HD)
    BT709,
    /// BT.2020 (UHD)
    BT2020,
    /// sRGB
    SRGB,
}

/// Frame metadata
#[derive(Debug, Clone, Copy)]
pub struct FrameMeta {
    /// Source device/stream ID
    pub source_id: u32,
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// Pixel format
    pub pixfmt: PixelFormat,
    /// Color space
    pub colorspace: ColorSpace,
    /// Frame index/sequence number
    pub frame_idx: u64,
    /// Presentation timestamp (nanoseconds)
    pub pts_ns: u64,
    /// Capture timestamp (nanoseconds)
    pub t_capture_ns: u64,
    /// Stride in bytes
    pub stride_bytes: u32,
}

/// Raw frame packet with metadata and memory reference
#[derive(Debug, Clone, Copy)]
pub struct RawFramePacket {
    /// Frame metadata
    pub meta: FrameMeta,
    /// Memory reference to frame data
    pub data: MemRef,
}
