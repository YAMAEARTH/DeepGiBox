right now we have to copy

The problem is: MemRef is just a raw pointer, but ONNX Runtime needs an ort::Tensor object to run inference.
Current Flow (Slow):

TensorInputPacket (raw pointer)
  ↓
Create new ort::Tensor (allocate GPU memory)
  ↓
Copy data: MemRef → ort::Tensor (2-5ms overhead!)
  ↓
Run inference\

What You're Suggesting (Fast):
TensorInputPacket (already contains ort::Tensor)
  ↓
Run inference directly (no copy!)

pub struct TensorInputPacket {
    pub from: FrameMeta,
    pub desc: TensorDesc,
    pub tensor: ort::value::Tensor<f32>,  // ✅ Pass actual tensor, not just pointer!
}


------------------------------------------

Preprocess → Vec<f32> in CPU memory → MemRef (pointer)
   ↓
Inference → Create new ORT Tensor → Copy data → Run

------------------------------------------

Preprocess → Create ORT Tensor directly → Pass Tensor object
   ↓
Inference → Use existing Tensor → Run (no copy!)