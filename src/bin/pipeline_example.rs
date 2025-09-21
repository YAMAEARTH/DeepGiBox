use std::time::Duration;
use anyhow::Result;
use decklink_rust::{
    PipelineBuilder, CaptureConfig, PreviewConfig, ColorSpace, 
    ProcessingStage, RawFramePacket, PipelineError
};

/// Custom processing stage that demonstrates frame modification
pub struct CustomProcessingStage {
    frame_count: u64,
    name: String,
}

impl CustomProcessingStage {
    pub fn new(name: &str) -> Self {
        Self {
            frame_count: 0,
            name: name.to_string(),
        }
    }
}

impl ProcessingStage for CustomProcessingStage {
    fn process(&mut self, mut input: RawFramePacket) -> Result<RawFramePacket, PipelineError> {
        self.frame_count += 1;
        
        // Example: Modify frame metadata to mark it as processed by our custom stage
        input.meta.source_id = (input.meta.source_id & 0x0FFFFFFF) | 0x10000000; // Set custom processing bit
        
        // Log every 60th frame to avoid spam
        if self.frame_count % 60 == 0 {
            println!("[{}] Processed frame #{} (seq: {}, {}x{})", 
                self.name, 
                self.frame_count, 
                input.meta.seq_no,
                input.meta.width,
                input.meta.height
            );
        }
        
        // Here you could:
        // 1. Run AI inference on the frame
        // 2. Apply image filters
        // 3. Extract features
        // 4. Perform color correction
        // 5. Add watermarks
        // etc.
        
        Ok(input)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

fn main() -> Result<()> {
    println!("DeepGI Pipeline Example - Separated Capture and Processing");
    println!("===========================================================");
    
    // List available devices
    let devices = decklink_rust::devicelist();
    if devices.is_empty() {
        println!("No DeckLink devices found. This example requires a DeckLink device.");
        return Ok(());
    }
    
    println!("Available DeckLink devices:");
    for (i, name) in devices.iter().enumerate() {
        println!("  {}: {}", i, name);
    }
    println!();

    // Create capture configuration
    let capture_config = CaptureConfig {
        device_index: 0,
        source_id: 42, // Custom source ID
        expected_colorspace: ColorSpace::BT709,
    };

    println!("Pipeline configuration:");
    println!("  Device: {} ({})", 0, devices.get(0).unwrap_or(&"Unknown".to_string()));
    println!("  Source ID: {}", 42);
    println!("  Processing: Custom AI Processing Stage");
    println!("  Mode: Headless capture + processing");
    println!();

    // Create capture stage directly (headless mode)
    let mut capture_stage = decklink_rust::capture::CaptureStage::new(capture_config);
    
    // Start capture
    match capture_stage.start() {
        Ok(_) => {
            println!("✓ Capture started successfully!");
            println!("Processing frames... (Running for 10 seconds)");
            
            let mut processing_stage = CustomProcessingStage::new("AI_Processor");
            let start_time = std::time::Instant::now();
            let mut total_processed = 0u64;
            let mut total_dropped = 0u64;
            
            // Process frames for 10 seconds
            while start_time.elapsed() < Duration::from_secs(10) {
                match capture_stage.get_next_frame() {
                    Ok(Some(frame)) => {
                        // Process the frame through our custom stage
                        match processing_stage.process(frame) {
                            Ok(_processed_frame) => {
                                total_processed += 1;
                                // Here you would normally pass the processed frame to the next stage
                                // For this example, we just count successful processing
                            }
                            Err(e) => {
                                eprintln!("Processing error: {}", e);
                                total_dropped += 1;
                            }
                        }
                    }
                    Ok(None) => {
                        // No frame available, sleep briefly
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(e) => {
                        eprintln!("Capture error: {}", e);
                        break;
                    }
                }
                
                // Print progress every 2 seconds
                if start_time.elapsed().as_secs() % 2 == 0 && start_time.elapsed().as_millis() % 2000 < 50 {
                    println!("📊 Progress - Processed: {}, Dropped: {}, Elapsed: {:.1}s", 
                        total_processed, 
                        total_dropped,
                        start_time.elapsed().as_secs_f32()
                    );
                }
            }
            
            // Stop capture
            if let Err(e) = capture_stage.stop() {
                eprintln!("Error stopping capture: {}", e);
            } else {
                println!("✓ Capture stopped successfully!");
            }
            
            // Final stats
            println!();
            println!("Final Statistics:");
            println!("  Frames Processed: {}", total_processed);
            println!("  Frames Dropped: {}", total_dropped);
            println!("  Total Runtime: {:.1}s", start_time.elapsed().as_secs_f32());
            if total_processed + total_dropped > 0 {
                println!("  Success Rate: {:.1}%", 
                    (total_processed as f64 / (total_processed + total_dropped) as f64) * 100.0
                );
                println!("  Processing Rate: {:.1} fps", 
                    total_processed as f64 / start_time.elapsed().as_secs_f64()
                );
            }
        }
        Err(e) => {
            eprintln!("Failed to start capture: {}", e);
            return Err(anyhow::anyhow!(e));
        }
    }

    Ok(())
}
