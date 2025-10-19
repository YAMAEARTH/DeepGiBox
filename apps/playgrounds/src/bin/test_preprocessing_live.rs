// test_preprocessing_live.rs - Live test with actual CUDA device
use anyhow::Result;
use preprocess_cuda::{ChromaOrder, Preprocessor};
use common_io::Stage;

fn main() -> Result<()> {
    println!("=== Live Preprocessing Test ===\n");
    
    // Test 1: Create preprocessor
    println!("Test 1: Creating preprocessor...");
    let mut preprocessor = Preprocessor::with_params(
        (640, 480),
        true,  // FP16
        0,     // device
        [0.485, 0.456, 0.406],  // ImageNet mean
        [0.229, 0.224, 0.225],  // ImageNet std
        ChromaOrder::UYVY,
    )?;
    println!("✓ Preprocessor created successfully!");
    println!("  Config: {}x{} FP16={} Device={}", 
             preprocessor.size.0, preprocessor.size.1, 
             preprocessor.fp16, preprocessor.device);
    
    // Test 2: Load from config file
    println!("\nTest 2: Loading from config file...");
    let preprocessor_from_config = preprocess_cuda::from_path(
        "configs/dev_1080p60_yuv422_fp16_trt.toml"
    )?;
    println!("✓ Config loaded successfully!");
    println!("  Config: {}x{} FP16={} Device={} Chroma={:?}", 
             preprocessor_from_config.size.0, preprocessor_from_config.size.1,
             preprocessor_from_config.fp16, preprocessor_from_config.device,
             preprocessor_from_config.chroma);
    
    // Test 3: Different configurations
    println!("\nTest 3: Testing different configurations...");
    
    println!("  - 512x512 FP16...");
    let _pre1 = Preprocessor::new((512, 512), true, 0)?;
    println!("    ✓ OK");
    
    println!("  - 640x480 FP32...");
    let _pre2 = Preprocessor::new((640, 480), false, 0)?;
    println!("    ✓ OK");
    
    println!("  - 1920x1080 FP16...");
    let _pre3 = Preprocessor::new((1920, 1080), true, 0)?;
    println!("    ✓ OK");
    
    println!("  - YUY2 chroma order...");
    let _pre4 = Preprocessor::with_params(
        (512, 512), true, 0,
        [0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
        ChromaOrder::YUY2
    )?;
    println!("    ✓ OK");
    
    println!("\n=== All Tests Passed! ===");
    println!("\nNote: To test actual frame processing, you need:");
    println!("  1. Real GPU memory with YUV422/NV12/BGRA data");
    println!("  2. DeckLink capture module or similar GPU frame source");
    println!("  3. Run integration tests with: cargo test -p preprocess_cuda -- --ignored");
    
    Ok(())
}
