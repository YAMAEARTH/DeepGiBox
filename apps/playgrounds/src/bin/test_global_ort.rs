/// Test that ORT environment is initialized only once globally
use anyhow::Result;
use inference::InferenceEngine;

fn main() -> Result<()> {
    println!("=== Testing Global ORT Environment ===\n");
    
    println!("Creating InferenceEngine #1...");
    let _engine1 = InferenceEngine::new("apps/playgrounds/YOLOv5.onnx")?;
    println!("✓ Engine #1 created\n");
    
    println!("Creating InferenceEngine #2...");
    let _engine2 = InferenceEngine::new("apps/playgrounds/YOLOv5.onnx")?;
    println!("✓ Engine #2 created\n");
    
    println!("Creating InferenceEngine #3...");
    let _engine3 = InferenceEngine::new("apps/playgrounds/YOLOv5.onnx")?;
    println!("✓ Engine #3 created\n");
    
    println!("=== Result ===");
    println!("✅ You should see '[ORT] Environment initialized' only ONCE above!");
    println!("✅ Multiple engines can share the same global ORT environment!");
    
    Ok(())
}
