use anyhow::Result;
use common_io::{Stage, RawFramePacket, TensorInputPacket, TensorDesc, DType, MemRef, MemLoc};

pub struct Preprocessor { size:(u32,u32), fp16:bool, device:u32 }
impl Preprocessor { pub fn new(size:(u32,u32), fp16:bool, device:u32)->Result<Self>{ Ok(Self{ size, fp16, device }) } }
impl Stage<RawFramePacket, TensorInputPacket> for Preprocessor {
    fn process(&mut self, input: RawFramePacket)->TensorInputPacket{
        let desc = TensorDesc{ n:1,c:3,h:self.size.1,w:self.size.0, dtype: if self.fp16{DType::Fp16}else{DType::Fp32}, device:self.device };
        let data = MemRef{ ptr:std::ptr::null_mut(), len:0, stride:0, loc:MemLoc::Gpu{ device:self.device } };
        TensorInputPacket{ from: input.meta, desc, data }
    }
}
pub fn from_path(_cfg:&str)->Result<PreprocessStage>{ Ok(PreprocessStage{}) }
pub struct PreprocessStage {}
impl Stage<RawFramePacket, TensorInputPacket> for PreprocessStage {
    fn process(&mut self, input: RawFramePacket)->TensorInputPacket{
        let desc = TensorDesc{ n:1,c:3,h:512,w:512, dtype:DType::Fp16, device:0 };
        let data = MemRef{ ptr:std::ptr::null_mut(), len:0, stride:0, loc:MemLoc::Gpu{ device:0 } };
        TensorInputPacket{ from: input.meta, desc, data }
    }
}
