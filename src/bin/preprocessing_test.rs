use decklink_rust::{
    PreprocessingStage, PreprocessingStageConfig, ProcessingStage,
    RawFramePacket, FrameMeta, PixelFormat, ColorSpace, MemLoc
};

fn main() {
    println!("Testing Preprocessing Stage Integration");
    println!("=====================================");

    // Create a test configuration
    let config = PreprocessingStageConfig {
        pan_x: 0,
        pan_y: 0,
        zoom: 1.0,
        target_size: (512, 512),
        debug: true,
    };

    // Create preprocessing stage
    let mut preprocessing_stage = PreprocessingStage::new(config);
    println!("✓ Created preprocessing stage: {}", preprocessing_stage.name());

    // Create a test frame
    let test_data = vec![0u8; 1920 * 1080 * 4]; // BGRA data
    let meta = FrameMeta {
        source_id: 1,
        width: 1920,
        height: 1080,
        stride: 1920 * 4,
        pixfmt: PixelFormat::BGRA8,
        colorspace: ColorSpace::BT709,
        pts_ns: 0,
        timecode: None,
        seq_no: 1,
    };

    let test_frame = RawFramePacket::new_cpu(test_data, meta);
    println!("✓ Created test frame: {}x{}", test_frame.meta.width, test_frame.meta.height);

    // Process the frame
    match preprocessing_stage.process(test_frame) {
        Ok(processed_frame) => {
            println!("✓ Successfully processed frame (seq: {})", processed_frame.meta.seq_no);
            println!("  Frame size: {}x{}", processed_frame.meta.width, processed_frame.meta.height);
            println!("  Pixel format: {:?}", processed_frame.meta.pixfmt);
        }
        Err(e) => {
            println!("✗ Failed to process frame: {}", e);
        }
    }

    println!("\nPreprocessing stage integration test completed!");
}
