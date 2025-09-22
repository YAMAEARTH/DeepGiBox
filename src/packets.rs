use std::sync::Arc;

/// Pixel format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    BGRA8,
    NV12,
    P010,
    UYVY,
    YUYV,
    V210,
    ARGB8,
}

/// Color space enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    BT709,
    BT2020,
    SRGB,
}

/// Memory location for frame data
#[derive(Debug, Clone)]
pub enum MemLoc {
    Cpu { ptr: *mut u8, size: usize },
    Cuda { device_ptr: u64 },
}

unsafe impl Send for MemLoc {}
unsafe impl Sync for MemLoc {}

/// Frame metadata containing all timing and format information
#[derive(Debug, Clone)]
pub struct FrameMeta {
    pub source_id: u32,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub pixfmt: PixelFormat,
    pub colorspace: ColorSpace,
    pub pts_ns: u64,
    pub timecode: Option<u32>,
    pub seq_no: u64,
}

/// Raw frame packet from capture stage
#[derive(Debug, Clone)]
pub struct RawFramePacket {
    pub mem: MemLoc,
    pub meta: FrameMeta,
    // Internal buffer management - keeps data alive
    pub(crate) _buffer: Option<Arc<Vec<u8>>>,
}

impl RawFramePacket {
    /// Create a new RawFramePacket from a CPU buffer
    pub fn new_cpu(buffer: Vec<u8>, meta: FrameMeta) -> Self {
        let buffer_arc = Arc::new(buffer);
        let ptr = buffer_arc.as_ptr() as *mut u8;
        let size = buffer_arc.len();
        
        Self {
            mem: MemLoc::Cpu { ptr, size },
            meta,
            _buffer: Some(buffer_arc),
        }
    }

    /// Create a new RawFramePacket from a CUDA device pointer
    pub fn new_cuda(device_ptr: u64, meta: FrameMeta) -> Self {
        Self {
            mem: MemLoc::Cuda { device_ptr },
            meta,
            _buffer: None,
        }
    }

    /// Get the raw data as a slice (CPU only)
    pub fn as_slice(&self) -> Option<&[u8]> {
        match &self.mem {
            MemLoc::Cpu { ptr, size } => {
                if let Some(buffer) = &self._buffer {
                    Some(buffer.as_slice())
                } else {
                    // Unsafe access to external pointer
                    unsafe { Some(std::slice::from_raw_parts(*ptr, *size)) }
                }
            }
            MemLoc::Cuda { .. } => None,
        }
    }
}

/// Tensor description for ML inference
#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub shape: [u32; 4], // [N, C, H, W]
    pub dtype: TensorDataType,
    pub layout: TensorLayout,
    pub colors: ColorFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDataType {
    F16,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLayout {
    NCHW,
    NHWC,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorFormat {
    RGB,
    BGR,
}

/// Memory location for tensor data
#[derive(Debug, Clone)]
pub enum TensorMem {
    Cuda { device_ptr: u64 },
    Cpu { data: Vec<u8> },
}

/// Tensor input packet for inference stage
#[derive(Debug, Clone)]
pub struct TensorInputPacket {
    pub mem: TensorMem,
    pub desc: TensorDesc,
    pub meta: FrameMeta,
}

/// Bounding box
#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }

    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }

    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    }
}

/// Raw detection from inference output
#[derive(Debug, Clone, Copy)]
pub struct RawDetection {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub obj_conf: f32,
    pub class_conf: f32,
    pub class_id: i32,
}

impl RawDetection {
    /// Convert to bounding box format
    pub fn to_bbox(&self) -> BBox {
        BBox {
            x1: self.cx - self.w / 2.0,
            y1: self.cy - self.h / 2.0,
            x2: self.cx + self.w / 2.0,
            y2: self.cy + self.h / 2.0,
        }
    }
}

/// Raw detections packet from inference stage
#[derive(Debug, Clone)]
pub struct RawDetectionsPacket {
    pub dets: Vec<RawDetection>,
    pub meta: FrameMeta,
}

/// Processed detection with tracking info
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: BBox,
    pub score: f32,
    pub class_id: i32,
    pub track_id: Option<i64>,
    pub label: Option<String>,
}

/// Processed detections packet after NMS and tracking
#[derive(Debug, Clone)]
pub struct DetectionsPacket {
    pub dets: Vec<Detection>,
    pub meta: FrameMeta,
}

/// Overlay operation types
#[derive(Debug, Clone)]
pub enum OverlayOp {
    Rect {
        bbox: BBox,
        thickness: i32,
        alpha: f32,
        color: [u8; 3],
    },
    Text {
        x: i32,
        y: i32,
        text: String,
        font_px: i32,
        alpha: f32,
        color: [u8; 3],
    },
    Line {
        x1: i32,
        y1: i32,
        x2: i32,
        y2: i32,
        thickness: i32,
        alpha: f32,
        color: [u8; 3],
    },
}

/// Overlay plan packet for rendering stage
#[derive(Debug, Clone)]
pub struct OverlayPlanPacket {
    pub size: (u32, u32),
    pub ops: Vec<OverlayOp>,
    pub meta: FrameMeta,
}

/// Overlay frame with alpha channel
#[derive(Debug, Clone)]
pub struct OverlayFrame {
    pub mem: MemLoc,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub premultiplied: bool,
    pub(crate) _buffer: Option<Arc<Vec<u8>>>,
}

impl OverlayFrame {
    /// Create a new overlay frame from CPU buffer
    pub fn new_cpu(buffer: Vec<u8>, width: u32, height: u32, stride: u32, premultiplied: bool) -> Self {
        let buffer_arc = Arc::new(buffer);
        let ptr = buffer_arc.as_ptr() as *mut u8;
        let size = buffer_arc.len();
        
        Self {
            mem: MemLoc::Cpu { ptr, size },
            width,
            height,
            stride,
            premultiplied,
            _buffer: Some(buffer_arc),
        }
    }

    /// Get the raw data as a slice (CPU only)
    pub fn as_slice(&self) -> Option<&[u8]> {
        match &self.mem {
            MemLoc::Cpu { ptr, size } => {
                if let Some(buffer) = &self._buffer {
                    Some(buffer.as_slice())
                } else {
                    // Unsafe access to external pointer
                    unsafe { Some(std::slice::from_raw_parts(*ptr, *size)) }
                }
            }
            MemLoc::Cuda { .. } => None,
        }
    }
}

/// Final keying packet for output stage
#[derive(Debug, Clone)]
pub struct KeyingPacket {
    pub passthrough: RawFramePacket,
    pub overlay: OverlayFrame,
    pub meta: FrameMeta,
}

/// Pipeline stage trait for processing packets
pub trait PipelineStage<Input, Output> {
    type Error;
    
    fn process(&mut self, input: Input) -> Result<Output, Self::Error>;
}

/// Generic pipeline error
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("Capture error: {0}")]
    Capture(String),
    #[error("Preview error: {0}")]
    Preview(String),
    #[error("Processing error: {0}")]
    Processing(String),
    #[error("Format conversion error: {0}")]
    Format(String),
}
