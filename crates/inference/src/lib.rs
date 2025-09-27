use anyhow::Result;
use common_io::{Stage, TensorInputPacket, RawDetectionsPacket};
pub struct InferenceEngine {}
impl InferenceEngine { pub fn new()->Result<Self>{ Ok(Self{}) } }
impl Stage<TensorInputPacket, RawDetectionsPacket> for InferenceEngine {
    fn process(&mut self, input: TensorInputPacket)->RawDetectionsPacket{ RawDetectionsPacket{ from: input.from } }
}
pub fn from_path(_cfg:&str)->Result<InferStage>{ Ok(InferStage{}) }
pub struct InferStage {}
impl Stage<TensorInputPacket, RawDetectionsPacket> for InferStage {
    fn process(&mut self, input: TensorInputPacket)->RawDetectionsPacket{ RawDetectionsPacket{ from: input.from } }
}
