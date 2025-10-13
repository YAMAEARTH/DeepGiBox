use anyhow::Result;
use common_io::OverlayFramePacket;
pub fn from_path(_cfg: &str) -> Result<OutputStage> {
    Ok(OutputStage {})
}
pub struct OutputStage {}
impl OutputStage {
    pub fn submit(&mut self, _frame: OverlayFramePacket) -> Result<()> {
        Ok(())
    }
}
