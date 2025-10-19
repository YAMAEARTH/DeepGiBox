// preprocess_gpu_preview.rs - Playground to test preprocessing
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== DeepGIBox Preprocessing Module Preview ===\n");
    
    println!("This module implements GPU-accelerated preprocessing:");
    println!("  - Input: RawFramePacket (YUV422_8/NV12/BGRA8) on GPU");
    println!("  - Output: TensorInputPacket (RGB NCHW FP16/FP32) on GPU");
    println!("  - Features:");
    println!("    • BT.709 limited-range YUV→RGB conversion");
    println!("    • Bilinear resize to configurable output size");
    println!("    • Normalization (mean/std)");
    println!("    • Fused single-pass kernel for minimal latency");
    println!("    • Support for UYVY and YUY2 chroma orders\n");
    
    // Try to create a preprocessor (will fail if no CUDA)
    println!("Attempting to initialize CUDA device 0...");
    match preprocess_cuda::Preprocessor::new((512, 512), true, 0) {
        Ok(pre) => {
            println!("✓ Successfully created preprocessor!");
            println!("  Output size: {}x{}", pre.size.0, pre.size.1);
            println!("  FP16: {}", pre.fp16);
            println!("  Device: {}", pre.device);
            println!("  Chroma order: {:?}", pre.chroma);
            println!("  Mean: {:?}", pre.mean);
            println!("  Std: {:?}", pre.std);
        }
        Err(e) => {
            println!("✗ Failed to initialize: {}", e);
            println!("  Make sure CUDA is installed and a CUDA device is available");
            println!("  Try: nvidia-smi");
        }
    }
    
    println!("\nConfiguration example:");
    println!(r#"
[preprocess]
size  = [512, 512]    # Output tensor size (W, H)
fp16  = true          # Use FP16 (recommended for TensorRT)
device = 0            # CUDA device ID
chroma = "UYVY"       # YUV422 byte order: UYVY or YUY2
mean = [0.0, 0.0, 0.0]    # RGB normalization mean
std = [1.0, 1.0, 1.0]     # RGB normalization std
    "#);
    
    println!("\nTo use in pipeline:");
    println!(r#"
let mut preprocessor = preprocess_cuda::from_path("configs/dev_1080p60_yuv422_fp16_trt.toml")?;
let raw_frame: RawFramePacket = ...; // from DeckLink capture (GPU memory)
let tensor_input = preprocessor.process(raw_frame);
// tensor_input is now ready for inference
    "#);
    
    Ok(())
}
