// Example: Simple telemetry demonstration
// This shows how to use telemetry for per-stage and end-to-end timing

use std::thread;
use std::time::Duration;

fn main() {
    println!("=== Telemetry Demo ===\n");
    
    // Simulate a video pipeline processing 3 frames
    for frame_num in 1..=3 {
        println!("Processing frame {}...", frame_num);
        
        // Start measuring end-to-end frame time
        let t_frame = telemetry::now_ns();
        
        // Stage 1: Capture (simulate 1ms)
        let t_capture = telemetry::now_ns();
        thread::sleep(Duration::from_micros(1000));
        telemetry::record_ms("capture", t_capture);
        
        // Stage 2: Preprocess (simulate 2.5ms)
        let t_preprocess = telemetry::now_ns();
        thread::sleep(Duration::from_micros(2500));
        telemetry::record_ms("preprocess", t_preprocess);
        
        // Stage 3: Inference (simulate 15ms)
        let t_inference = telemetry::now_ns();
        thread::sleep(Duration::from_micros(15000));
        telemetry::record_ms("inference", t_inference);
        
        // Stage 4: Postprocess (simulate 1ms)
        let t_postprocess = telemetry::now_ns();
        thread::sleep(Duration::from_micros(1000));
        telemetry::record_ms("postprocess", t_postprocess);
        
        // Stage 5: Overlay (simulate 3ms)
        let t_overlay = telemetry::now_ns();
        thread::sleep(Duration::from_micros(3000));
        telemetry::record_ms("overlay", t_overlay);
        
        // Record end-to-end frame time
        telemetry::record_ms("frame.e2e", t_frame);
        
        println!();
    }
    
    println!("=== Demo Complete ===");
    println!("\nNote: Run with --features json for JSON output:");
    println!("  cargo run -p playgrounds --bin telemetry_demo --no-default-features --features json");
}
