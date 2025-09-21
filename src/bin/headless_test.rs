use std::time::Duration;
use anyhow::Result;
use decklink_rust::{
    HeadlessProcessor, HeadlessConfig, PipelineStage, CaptureConfig, ColorSpace
};

fn main() -> Result<()> {
    println!("DeepGI Headless Processing - Stage Selection & Performance Testing");
    println!("==================================================================");
    
    // List available devices
    let devices = decklink_rust::devicelist();
    if devices.is_empty() {
        println!("❌ No DeckLink devices found");
        return Ok(());
    }
    
    println!("Available devices:");
    for (i, name) in devices.iter().enumerate() {
        println!("  {}: {}", i, name);
    }
    println!();
    
    // Show available stages
    println!("Available processing stages:");
    let all_stages = vec![
        PipelineStage::Capture,
        PipelineStage::Preprocessing,
        PipelineStage::Inference,
        PipelineStage::PostProcessing,
        PipelineStage::ObjectTracking,
        PipelineStage::Overlay,
        PipelineStage::Keying,
        PipelineStage::Output,
    ];
    
    for (i, stage) in all_stages.iter().enumerate() {
        println!("  {}: {} - {}", i + 1, stage.name(), stage.description());
    }
    println!();
    
    // Configure test scenarios
    let test_configs = vec![
        // Capture only test
        HeadlessConfig {
            selected_stages: vec![PipelineStage::Capture],
            max_runtime: Some(Duration::from_secs(5)),
            enable_detailed_logging: false,
            ..Default::default()
        },
        
        // Preprocessing pipeline test
        HeadlessConfig {
            selected_stages: vec![PipelineStage::Capture, PipelineStage::Preprocessing],
            max_runtime: Some(Duration::from_secs(10)),
            enable_detailed_logging: false,
            ..Default::default()
        },
        
        // AI inference pipeline test
        HeadlessConfig {
            selected_stages: vec![PipelineStage::Preprocessing, PipelineStage::Inference, PipelineStage::PostProcessing],
            max_runtime: Some(Duration::from_secs(10)),
            enable_detailed_logging: false,
            ..Default::default()
        },
        
        // Full video processing pipeline test
        HeadlessConfig {
            selected_stages: vec![
                PipelineStage::Capture,
                PipelineStage::Preprocessing,
                PipelineStage::Inference,
                PipelineStage::PostProcessing,
                PipelineStage::ObjectTracking,
                PipelineStage::Overlay,
                PipelineStage::Output
            ],
            max_runtime: Some(Duration::from_secs(15)),
            enable_detailed_logging: true,
            capture_config: CaptureConfig {
                device_index: 0,
                source_id: 300,
                expected_colorspace: ColorSpace::BT709,
            },
            stats_interval: Duration::from_secs(2),
            max_frames: None,
        },
    ];
    
    // Run each test configuration
    for (i, config) in test_configs.iter().enumerate() {
        println!("🧪 Running Test Configuration {} of {}", i + 1, test_configs.len());
        println!("{}", "─".repeat(50));
        
        let mut processor = HeadlessProcessor::new(config.clone());
        
        if let Err(e) = processor.run() {
            eprintln!("❌ Test failed: {}", e);
        }
        
        if i < test_configs.len() - 1 {
            println!("⏸️  Waiting 2 seconds before next test...");
            std::thread::sleep(Duration::from_secs(2));
            println!();
        }
    }
    
    println!("🎉 All tests completed!");
    Ok(())
}
