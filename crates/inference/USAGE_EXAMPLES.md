# Inference Crate - Usage Examples

## Quick Start

### Building the Crate

```bash
# Development build
cargo build -p inference

# Release build (optimized)
cargo build -p inference --release

# With verbose output
cargo build -p inference -v
```

### Running Tests

```bash
# Run all unit tests
cargo test -p inference

# Run with output
cargo test -p inference -- --nocapture

# Run specific test
cargo test -p inference test_mock_tensor_packet_creation

# Run integration tests (requires ONNX model)
cargo test -p inference -- --ignored
```

## Basic Usage

### 1. Initialize the Inference Engine

```rust
use inference::InferenceEngine;
use anyhow::Result;

fn main() -> Result<()> {
    // Create engine with model path
    let mut engine = InferenceEngine::new("./models/yolov8.onnx")?;
    
    println!("Inference engine initialized");
    Ok(())
}
```

### 2. Using the Helper Function

```rust
use inference::from_path;

fn main() -> Result<()> {
    // Simplified initialization
    let mut engine = from_path("./models/yolov8.onnx")?;
    
    Ok(())
}
```

### 3. Process Tensor Input (Stage Pattern)

```rust
use inference::InferenceEngine;
use common_io::{Stage, TensorInputPacket};

fn run_inference(mut engine: InferenceEngine, input: TensorInputPacket) {
    // Process using the Stage trait
    let output = engine.process(input);
    
    // Access results
    println!("Frame: {}", output.from.frame_idx);
    println!("Predictions: {} values", output.raw_output.len());
    
    if !output.raw_output.is_empty() {
        println!("First prediction: {}", output.raw_output[0]);
    }
}
```

## Advanced Usage

### Pipeline Integration

```rust
use inference::InferenceEngine;
use common_io::{Stage, TensorInputPacket, RawDetectionsPacket};

struct InferencePipeline {
    engine: InferenceEngine,
}

impl InferencePipeline {
    fn new(model_path: &str) -> Result<Self> {
        Ok(Self {
            engine: InferenceEngine::new(model_path)?,
        })
    }
    
    fn process_frame(&mut self, tensor: TensorInputPacket) -> RawDetectionsPacket {
        // Use Stage trait for processing
        self.engine.process(tensor)
    }
    
    fn name(&self) -> &'static str {
        self.engine.name()
    }
}

fn main() -> Result<()> {
    let mut pipeline = InferencePipeline::new("./models/model.onnx")?;
    
    println!("Pipeline stage: {}", pipeline.name());
    
    // Process frames...
    // let output = pipeline.process_frame(input_tensor);
    
    Ok(())
}
```

### Error Handling

```rust
use inference::InferenceEngine;
use anyhow::{Context, Result};

fn safe_inference_init(model_path: &str) -> Result<InferenceEngine> {
    InferenceEngine::new(model_path)
        .context("Failed to initialize inference engine")?
}

fn main() {
    match safe_inference_init("./models/model.onnx") {
        Ok(engine) => {
            println!("✓ Engine initialized successfully");
            // Use engine...
        }
        Err(e) => {
            eprintln!("✗ Failed to initialize: {:?}", e);
            eprintln!("Make sure:");
            eprintln!("  1. Model file exists");
            eprintln!("  2. TensorRT is properly installed");
            eprintln!("  3. CUDA is available");
        }
    }
}
```

## Configuration

### TensorRT Settings

The inference engine is automatically configured with:

```rust
// Configured in ORT_ENVIRONMENT (automatic)
// - TensorRT execution provider
// - FP16 precision enabled
// - Timing cache enabled
// - Engine cache enabled
// - Cache path: ./trt_cache/
```

### Custom Cache Directory

To use a different cache directory, modify the environment variable:

```bash
# Set custom TensorRT cache location
export TENSORRT_CACHE_DIR=/path/to/cache

# Or modify in code before first engine creation
```

## Testing Your Implementation

### Unit Test Template

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_custom_model() {
        let model_path = "./test_models/custom.onnx";
        
        if !std::path::Path::new(model_path).exists() {
            println!("Skipping: model not found");
            return;
        }
        
        let engine = InferenceEngine::new(model_path);
        assert!(engine.is_ok(), "Engine creation failed");
    }
}
```

### Benchmark Template

```rust
use std::time::Instant;

fn benchmark_inference(engine: &mut InferenceEngine, input: TensorInputPacket, iterations: usize) {
    let start = Instant::now();
    
    for i in 0..iterations {
        let output = engine.process(input.clone());
        
        if i == 0 {
            println!("First inference: {} predictions", output.raw_output.len());
        }
    }
    
    let duration = start.elapsed();
    let avg_ms = duration.as_secs_f64() * 1000.0 / iterations as f64;
    
    println!("Average inference time: {:.2}ms over {} iterations", avg_ms, iterations);
    println!("Throughput: {:.2} FPS", 1000.0 / avg_ms);
}
```

## Common Issues and Solutions

### Issue 1: Model File Not Found

```rust
// ✗ Wrong
let engine = InferenceEngine::new("model.onnx")?;

// ✓ Correct - use absolute or relative path
let engine = InferenceEngine::new("./models/model.onnx")?;
// or
let engine = InferenceEngine::new("/absolute/path/to/model.onnx")?;
```

### Issue 2: TensorRT Not Available

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Check TensorRT installation
dpkg -l | grep tensorrt
```

### Issue 3: Cache Directory Permissions

```bash
# Ensure cache directory is writable
mkdir -p ./trt_cache
chmod 755 ./trt_cache
```

### Issue 4: GPU Memory Issues

```rust
// If running out of GPU memory, check:
// 1. Input tensor size
// 2. Model complexity
// 3. Other GPU processes

// Monitor GPU usage
// nvidia-smi -l 1
```

## Performance Tips

### 1. First-Run Initialization

```rust
// First run builds TensorRT engine (slow)
let engine = InferenceEngine::new("./model.onnx")?;
// Subsequent runs use cached engine (fast)
```

### 2. Batch Processing

```rust
// For multiple frames, reuse the same engine
let mut engine = InferenceEngine::new("./model.onnx")?;

for frame in frames {
    let output = engine.process(frame);
    // Process output...
}
```

### 3. Warm-up Run

```rust
// Do a warm-up inference to initialize GPU kernels
let mut engine = InferenceEngine::new("./model.onnx")?;

// Warm-up (first run is slower)
let _ = engine.process(dummy_input);

// Now measure actual performance
let start = Instant::now();
let output = engine.process(real_input);
println!("Inference time: {:?}", start.elapsed());
```

## Integration with Other Crates

### With preprocess_cuda

```rust
use preprocess_cuda::PreprocessCUDA;
use inference::InferenceEngine;
use common_io::Stage;

fn full_pipeline(raw_frame: RawFramePacket) -> RawDetectionsPacket {
    // Preprocess
    let mut preprocessor = PreprocessCUDA::new(/* config */)?;
    let tensor = preprocessor.process(raw_frame);
    
    // Inference
    let mut inference = InferenceEngine::new("./model.onnx")?;
    let detections = inference.process(tensor);
    
    detections
}
```

### With postprocess

```rust
use inference::InferenceEngine;
use postprocess::PostprocessStage;
use common_io::Stage;

fn detect_objects(tensor_input: TensorInputPacket) -> DetectionsPacket {
    // Inference
    let mut inference = InferenceEngine::new("./model.onnx")?;
    let raw_detections = inference.process(tensor_input);
    
    // Postprocess
    let mut postprocess = PostprocessStage::new(/* config */)?;
    let detections = postprocess.process(raw_detections);
    
    detections
}
```

## API Reference

### InferenceEngine

```rust
pub struct InferenceEngine {
    session: Session,
    gpu_allocator: Allocator,
    cpu_allocator: Allocator,
}

impl InferenceEngine {
    // Create new inference engine
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self>
}

// Stage trait implementation
impl Stage<TensorInputPacket, RawDetectionsPacket> for InferenceEngine {
    fn name(&self) -> &'static str
    fn process(&mut self, input: TensorInputPacket) -> RawDetectionsPacket
}
```

### Helper Functions

```rust
// Create engine from path (convenience function)
pub fn from_path(model_path: &str) -> Result<InferenceEngine>

// Type alias for backwards compatibility
pub type InferStage = InferenceEngine;
```

## Environment Variables

```bash
# TensorRT cache directory (default: ./trt_cache)
export TRT_CACHE_DIR=/path/to/cache

# CUDA device ID (default: 0)
export CUDA_VISIBLE_DEVICES=0

# Enable verbose logging
export ORT_LOGGING_LEVEL=verbose
```

## Build Requirements

- Rust 2021 edition
- CUDA toolkit (11.x or 12.x)
- TensorRT (8.x or later)
- NVIDIA GPU (compute capability 7.0+)

## Dependencies

```toml
[dependencies]
anyhow = "1"
common_io = { path = "../common_io" }
telemetry = { path = "../telemetry" }
config = { path = "../config" }
ort = { version = "2.0.0-rc.10", features = ["tensorrt", "cuda"] }
ndarray = "0.16"
once_cell = "1.19"
```
