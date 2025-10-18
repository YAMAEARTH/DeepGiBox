#[derive(Clone, Copy, Debug)]
pub enum PixelFormat {
    BGRA8,
    RGB8,
    NV12,
    YUV422_8,
    P010,
    ARGB8,
}

#[derive(Clone, Copy, Debug)]
pub enum ColorSpace {
    BT601,
    BT709,
    BT2020,
    SRGB,
}

#[derive(Clone, Copy, Debug)]
pub enum DType {
    Fp32,
    Fp16,
}

#[derive(Clone, Debug)]
pub enum MemLoc {
    Cpu,
    Gpu { device: u32 },
}

#[derive(Clone, Debug)]
pub struct MemRef {
    pub ptr: *mut u8,
    pub len: usize,
    pub stride: usize,
    pub loc: MemLoc,
}

#[derive(Clone, Debug)]
pub struct FrameMeta {
    pub source_id: u32,
    pub width: u32,
    pub height: u32,
    pub pixfmt: PixelFormat,
    pub colorspace: ColorSpace,
    pub frame_idx: u64,
    pub pts_ns: u64,
    pub t_capture_ns: u64,
    pub stride_bytes: u32,
}

#[derive(Clone, Debug)]
pub struct RawFramePacket {
    pub meta: FrameMeta,
    pub data: MemRef,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RawCaptureFrame {
    pub data: *const u8,
    pub width: i32,
    pub height: i32,
    pub row_bytes: i32,
    pub seq: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct TensorDesc {
    pub n: u32,
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub dtype: DType,
    pub device: u32,
}

#[derive(Clone, Debug)]
pub struct TensorInputPacket {
    pub from: FrameMeta,
    pub desc: TensorDesc,
    pub data: MemRef,
}

#[derive(Clone, Debug)]
pub struct RawDetectionsPacket {
    pub from: FrameMeta,
    pub raw_output: Vec<f32>, // Pun:: added raw model output (e.g., YOLO predictions)
    pub output_shape: Vec<usize>, // Shape of the model output tensor (e.g., [1, 25200, 85] for YOLOv5)
}

#[derive(Clone, Copy, Debug)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Clone, Debug)]
pub struct Detection {
    pub bbox: BBox,
    pub score: f32,
    pub class_id: i32,
    pub track_id: Option<i32>,
}

pub struct DetectionsPacket {
    pub from: FrameMeta,
    pub items: Vec<Detection>,
}

pub enum DrawOp {
    Rect {
        xywh: (f32, f32, f32, f32),
        thickness: u8,
    },
    Label {
        anchor: (f32, f32),
        text: String,
        font_px: u16,
    },
    Poly {
        pts: Vec<(f32, f32)>,
        thickness: u8,
    },
}

pub struct OverlayPlanPacket {
    pub from: FrameMeta,
    pub ops: Vec<DrawOp>,
    pub canvas: (u32, u32),
}

pub struct OverlayFramePacket {
    pub from: FrameMeta,
    pub argb: MemRef,
    pub stride: usize,
}

pub trait Stage<I, O> {
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
    fn process(&mut self, input: I) -> O;
}
