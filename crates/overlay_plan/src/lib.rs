use anyhow::Result;
use common_io::{DetectionsPacket, OverlayPlanPacket, Stage};

pub fn from_path(_cfg: &str) -> Result<PlanStage> {
    Ok(PlanStage {})
}
pub struct PlanStage {}
// impl Stage<DetectionsPacket, OverlayPlanPacket> for PlanStage {
//     fn process(&mut self, input: DetectionsPacket)->OverlayPlanPacket{
//        // OverlayPlanPacket{ from: input.from, ops: Vec::new(), canvas: (input.from.width, input.from.height) }
//     }
// }
