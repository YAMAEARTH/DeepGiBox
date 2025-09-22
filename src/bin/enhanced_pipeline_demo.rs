use std::time::Duration;
use anyhow::Result;
use decklink_rust::{
    CaptureConfig, ColorSpace,
    PipelineBuilder,
    preprocessing_v2::PreprocessingV2Config,
};

/// Demonstrates the enhanced pipeline with complete DeepGI stage support
fn main() -> Result<()> {
    println!("Enhanced DeepGI Pipeline Demo");
    println!("============================");
    println!("Features:");
    println!("  ✓ Complete DeepGI pipeline flow support");
    println!("  ✓ CV-CUDA preprocessing v.2 integration");
    println!("  ✓ Comprehensive performance monitoring");
    println!("  ✓ Type-safe stage processing");
    println!("  ✓ Enhanced pipeline builder patterns");
    println!();

    // List available devices
    let devices = decklink_rust::devicelist();
    if devices.is_empty() {
        println!("No DeckLink devices found. Running in simulation mode...");
        demonstrate_pipeline_builder();
        return Ok(());
    }
    
    println!("Available DeckLink devices:");
    for (i, name) in devices.iter().enumerate() {
        println!("  {}: {}", i, name);
    }
    println!();

    // Demonstrate different pipeline configurations
    demonstrate_pipeline_builder();
    
    if !devices.is_empty() {
        println!("Testing enhanced pipeline with device 0...");
        test_enhanced_pipeline(0)?;
    }

    Ok(())
}

/// Demonstrates the enhanced pipeline builder patterns
fn demonstrate_pipeline_builder() {
    println!("🔧 Pipeline Builder Demonstrations");
    println!("==================================");

    // Example 1: Basic pipeline with performance monitoring
    println!("1. Basic enhanced pipeline:");
    let basic_pipeline = PipelineBuilder::new()
        .with_performance_monitoring(true)
        .with_stats_interval(Duration::from_millis(500))
        .with_passthrough(true)
        .build();
    
    println!("   ✓ Performance monitoring: {}", basic_pipeline.is_performance_monitoring_enabled());
    println!("   ✓ Stats interval: {:?}", basic_pipeline.get_stats_interval());
    println!("   ✓ Passthrough enabled: {}", basic_pipeline.is_passthrough_enabled());

    // Example 2: Pipeline with CV-CUDA preprocessing v.2
    println!("\n2. CV-CUDA preprocessing v.2 pipeline:");
    let preprocessing_config = PreprocessingV2Config {
        target_size: (640, 480),
        use_cuda: true,
        debug: true,
        zoom: 1.2,
        ..Default::default()
    };

    let preprocessing_pipeline = PipelineBuilder::new()
        .with_preprocessing_v2_config(preprocessing_config)
        .with_performance_monitoring(true)
        .build();
    
    if let Some(ref config) = preprocessing_pipeline.get_config().preprocessing_v2 {
        println!("   ✓ Target size: {}x{}", config.target_size.0, config.target_size.1);
        println!("   ✓ CUDA acceleration: {}", config.use_cuda);
        println!("   ✓ Debug mode: {}", config.debug);
        println!("   ✓ Zoom factor: {:.1}x", config.zoom);
    }

    // Example 3: Quick builder methods
    println!("\n3. Quick builder methods:");
    
    let _capture_preview_pipeline = PipelineBuilder::build_capture_preprocessing_preview(
        0, (512, 512), true
    );
    println!("   ✓ Capture + Preprocessing + Preview pipeline created");
    
    let _headless_pipeline = PipelineBuilder::build_headless_processing(
        0, (416, 416), false
    );
    println!("   ✓ Headless processing pipeline created");

    // Example 4: Advanced configuration
    println!("\n4. Advanced configuration:");
    let capture_config = CaptureConfig {
        device_index: 0,
        source_id: 42,
        expected_colorspace: ColorSpace::BT2020,
    };

    let advanced_pipeline = PipelineBuilder::new()
        .with_capture_config(capture_config)
        .with_buffer_size(64)
        .with_performance_monitoring(true)
        .with_stats_interval(Duration::from_millis(100))
        .build();
    
    println!("   ✓ Custom capture configuration with BT2020 colorspace");
    println!("   ✓ Buffer size: {} frames", advanced_pipeline.get_config().processing_buffer_size);
    println!("   ✓ High-frequency stats: {:?}", advanced_pipeline.get_stats_interval());
    
    println!();
}

/// Tests the enhanced pipeline with real hardware
fn test_enhanced_pipeline(device_index: i32) -> Result<()> {
    println!("🚀 Enhanced Pipeline Test");
    println!("=========================");

    // Create pipeline with CV-CUDA preprocessing
    let pipeline = PipelineBuilder::build_capture_preprocessing_preview(
        device_index,
        (512, 512),
        true, // Enable CUDA
    );

    println!("Pipeline configuration:");
    println!("  Device index: {}", device_index);
    println!("  Target size: 512x512");
    println!("  CUDA acceleration: enabled");
    println!("  Performance monitoring: enabled");
    println!();

    // Note: This would require OpenGL context for full testing
    println!("Note: Full pipeline testing requires OpenGL context initialization");
    println!("For complete testing, run capture_preview_gl or similar binary");
    
    // Demonstrate performance monitoring capabilities
    let stats = pipeline.get_performance_stats();
    println!("Initial pipeline statistics:");
    println!("  Frames processed: {}", stats.frames_processed);
    println!("  Pipeline efficiency: {:.1}%", stats.pipeline_efficiency);
    println!("  End-to-end latency: {:.2}ms", stats.end_to_end_latency_ms);
    
    // Show available stage metrics
    for stage_name in ["preprocessing_v2", "processing", "capture", "preview"] {
        if let Some(_metrics) = pipeline.get_stage_metrics(stage_name) {
            println!("  {} stage metrics available", stage_name);
        }
    }

    Ok(())
}
