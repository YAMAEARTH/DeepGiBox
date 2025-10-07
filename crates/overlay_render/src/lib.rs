use anyhow::Result;
use common_io::{MemLoc, MemRef, OverlayFramePacket, OverlayPlanPacket, Stage};

pub fn from_path(_cfg: &str) -> Result<RenderStage> {
    Ok(RenderStage {})
}
pub struct RenderStage {}
impl Stage<OverlayPlanPacket, OverlayFramePacket> for RenderStage {
    fn process(&mut self, input: OverlayPlanPacket) -> OverlayFramePacket {
        OverlayFramePacket {
            from: input.from,
            argb: MemRef {
                ptr: std::ptr::null_mut(),
                len: 0,
                stride: 0,
                loc: MemLoc::Cpu,
            },
            stride: 0,
        }
    }
}
