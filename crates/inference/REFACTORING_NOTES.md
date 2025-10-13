# Inference Crate Refactoring Summary

## Date: October 7, 2025

## Objective
Refactor the inference crate from a standalone video processing application into a microservice that receives `TensorInputPacket` and performs only inference, following the DeepGIBox pipeline architecture.

## Changes Made

### Files Removed
- ✅ `src/main.rs` - Removed standalone application
- ✅ `src/video.rs` - Video I/O moved to appropriate modules
- ✅ `src/post.rs` - Postprocessing moved to `postprocess` crate
- ✅ `src/visualize.rs` - Visualization moved to `overlay_render` crate

### Files Modified

#### `src/lib.rs` (Complete Rewrite)
**Before**: Stub implementation with empty `process()` method
**After**: Full inference engine implementation

Key features:
- Implements `Stage<TensorInputPacket, RawDetectionsPacket>` trait
- TensorRT initialization with FP16 support
- GPU memory management with allocators
- IO binding for efficient inference
- Shape validation and dtype checking
- Error handling with anyhow

#### `src/inference.rs` (Simplified)
**Before**: Full `InferenceContext` implementation with image preprocessing
**After**: Legacy utilities kept for reference
- Removed image processing dependencies
- Kept `init_environment()` function
- Removed `PreprocessedImage`, `InferenceContext`, `InferenceOutput` structs

#### `Cargo.toml` (Dependencies Cleaned)
**Removed**:
- `image` - No longer needed (preprocessing done by `preprocess_cuda`)
- `opencv` - Video I/O not needed
- `similari` - Tracking moved to `postprocess`

**Kept**:
- `ort` - ONNX Runtime with TensorRT
- `anyhow` - Error handling
- `common_io` - Pipeline data types
- `ndarray` - Array operations
- `telemetry` - Performance monitoring
- `config` - Configuration management

#### Workspace `Cargo.toml`
**Added**: `exclude = ["lib/*"]` to prevent workspace conflicts with local ort library

### New Files Created

#### `README.md`
Comprehensive documentation covering:
- Architecture overview
- Usage examples
- Pipeline integration
- Configuration details
- Input requirements

#### `REFACTORING_NOTES.md` (this file)
Documentation of refactoring process and decisions

## Architecture Changes

### Before
```
Standalone Application
  ├─ Video Input (OpenCV)
  ├─ Frame Extraction
  ├─ Preprocessing (image library)
  ├─ Inference (ONNX/TensorRT)
  ├─ Postprocessing (NMS, tracking)
  ├─ Visualization (drawing on images)
  └─ Video Output (OpenCV)
```

### After
```
Microservice (Stage Implementation)
  │
  ├─ Input: TensorInputPacket
  │    └─ from: FrameMeta
  │    └─ desc: TensorDesc (N×C×H×W, dtype, device)
  │    └─ data: MemRef (GPU memory)
  │
  ├─ Process: InferenceEngine
  │    ├─ Validate tensor shape/dtype
  │    ├─ Create ORT tensors from GPU memory
  │    ├─ Run TensorRT inference
  │    └─ Extract raw predictions
  │
  └─ Output: RawDetectionsPacket
       └─ from: FrameMeta
       └─ [raw predictions passed to postprocess]
```

## Pipeline Integration

The inference crate now fits into the complete pipeline:

```
DeckLinkInput → RawFramePacket
                     ↓
              PreprocessCUDA (YUV/NV12/BGRA → RGB, normalize)
                     ↓
              TensorInputPacket (1×3×H×W, FP16/FP32, GPU)
                     ↓
              InferenceEngine ← [THIS CRATE]
                     ↓
              RawDetectionsPacket
                     ↓
              Postprocess (decode, NMS, tracking)
                     ↓
              DetectionsPacket
                     ↓
              OverlayPlan → OverlayRender → DeckLinkOutput
```

## API Changes

### Public Interface
```rust
// Main struct
pub struct InferenceEngine { ... }

// Constructor
impl InferenceEngine {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self>
    pub fn with_input_shape(self, n: u32, c: u32, h: u32, w: u32) -> Self
}

// Stage implementation
impl Stage<TensorInputPacket, RawDetectionsPacket> for InferenceEngine {
    fn name(&self) -> &'static str
    fn process(&mut self, input: TensorInputPacket) -> RawDetectionsPacket
}

// Helper function
pub fn from_path(model_path: &str) -> Result<InferenceEngine>

// Type alias for compatibility
pub type InferStage = InferenceEngine;
```

## Design Decisions

1. **GPU-First**: All tensors kept on GPU, minimal CPU transfers
2. **Zero-Copy**: Direct pointer passing via `MemRef`
3. **Stage Pattern**: Consistent with other pipeline components
4. **Error Recovery**: Errors logged but don't crash pipeline
5. **TensorRT Caching**: Engines cached to avoid rebuild overhead
6. **Shape Validation**: Explicit checking prevents runtime errors

## Testing

### Build Verification
```bash
cargo build -p inference  # ✅ Success
cargo check -p inference  # ✅ Success
```

### Usage Example
```rust
use inference::InferenceEngine;
use common_io::{Stage, TensorInputPacket, TensorDesc, MemRef, DType, MemLoc};

// Initialize
let mut engine = InferenceEngine::new("./model.onnx")?;

// Process input
let input = TensorInputPacket {
    from: frame_meta,
    desc: TensorDesc { n: 1, c: 3, h: 512, w: 512, dtype: DType::Fp32, device: 0 },
    data: MemRef { ptr, len, stride, loc: MemLoc::Gpu { device: 0 } },
};

let output = engine.process(input);
```

## Future Enhancements

1. **FP16 Support**: Add native FP16 tensor handling
2. **Dynamic Shapes**: Support variable input sizes
3. **Batching**: Process multiple frames simultaneously
4. **Multi-GPU**: Distribute inference across devices
5. **Model Hot-Swap**: Switch models without restarting
6. **Latency Tracking**: Integrate telemetry for per-frame timing

## Breaking Changes

- `main.rs` removed - no longer a binary crate
- Video processing APIs removed
- Image preprocessing removed (moved to `preprocess_cuda`)
- Postprocessing removed (moved to `postprocess` crate)
- Visualization removed (moved to `overlay_render`)

## Migration Guide

### For Users of Old API
```rust
// OLD - Don't use anymore
let mut context = InferenceContext::new(model_path)?;
let output = context.run(&input_tensor)?;

// NEW - Use Stage pattern
let mut engine = InferenceEngine::new(model_path)?;
let output = engine.process(input_packet);
```

### For Pipeline Integration
```rust
// In runner/main pipeline
use inference::InferenceEngine;
use common_io::Stage;

// Initialize stages
let mut inference_stage = InferenceEngine::new("./model.onnx")?
    .with_input_shape(1, 3, 512, 512);

// In processing loop
let tensor_packet = preprocess_stage.process(raw_frame);
let detections = inference_stage.process(tensor_packet);
let processed = postprocess_stage.process(detections);
```

## Conclusion

The refactoring successfully transforms the inference crate from a monolithic application into a focused microservice that:
- ✅ Follows the DeepGIBox pipeline architecture
- ✅ Implements the `Stage` trait for consistency
- ✅ Focuses solely on inference (single responsibility)
- ✅ Maintains GPU efficiency with zero-copy operations
- ✅ Integrates cleanly with preprocessing and postprocessing stages
- ✅ Provides clear documentation and examples

The crate is now ready for integration into the full `runner` application.
