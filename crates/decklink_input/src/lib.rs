use anyhow::Result;
use common_io::{Stage, RawFramePacket, FrameMeta, PixelFormat, ColorSpace, MemRef, MemLoc};

pub struct DeckLinkSource {}
impl DeckLinkSource {
    pub fn new(_dev:i32,_mode:&str,_pf:PixelFormat,_cs:ColorSpace)->Result<Self>{ Ok(Self{}) }
    pub fn next_frame(&mut self)->Result<RawFramePacket>{
        let meta = FrameMeta{ source_id:0,width:1920,height:1080,pixfmt:PixelFormat::YUV422_8,
            colorspace:ColorSpace::BT709, frame_idx:0,pts_ns:0,t_capture_ns:0,stride_bytes:3840 };
        let data = MemRef{ ptr:std::ptr::null_mut(), len:0, stride: meta.stride_bytes as usize, loc:MemLoc::Cpu };
        Ok(RawFramePacket{meta,data})
    }
}
pub fn from_path(_cfg:&str)->Result<DeckLinkStage>{ Ok(DeckLinkStage{}) }
pub struct DeckLinkStage {}
impl Stage<(),RawFramePacket> for DeckLinkStage {
    fn process(&mut self,_:())->RawFramePacket{
        let meta = FrameMeta{ source_id:0,width:1920,height:1080,pixfmt:PixelFormat::YUV422_8,
            colorspace:ColorSpace::BT709, frame_idx:0,pts_ns:0,t_capture_ns:0,stride_bytes:3840 };
        let data = MemRef{ ptr:std::ptr::null_mut(), len:0, stride: meta.stride_bytes as usize, loc:MemLoc::Cpu };
        RawFramePacket{meta,data}
    }
}
