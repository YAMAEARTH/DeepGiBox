use anyhow::Result;
use common_io::{Stage, RawDetectionsPacket, DetectionsPacket};

pub fn from_path(_cfg:&str)->Result<PostStage>{ Ok(PostStage{}) }
pub struct PostStage {}
impl Stage<RawDetectionsPacket, DetectionsPacket> for PostStage {
    fn process(&mut self, input: RawDetectionsPacket)->DetectionsPacket{
        DetectionsPacket{ from: input.from, items: Vec::new() }
    }
}
