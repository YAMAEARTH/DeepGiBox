// test_postprocessing.rs - Test postprocessing stage with mock data
use anyhow::Result;
use common_io::{Stage, RawDetectionsPacket, FrameMeta, PixelFormat, ColorSpace};
use postprocess::{PostStage, PostprocessConfig};

fn create_mock_detection_packet() -> RawDetectionsPacket {
    // Mock frame metadata
    let meta = FrameMeta {
        source_id: 0,
        width: 1920,
        height: 1080,
        pixfmt: PixelFormat::RGB8,
        colorspace: ColorSpace::BT709,
        frame_idx: 1,
        pts_ns: 1000000,
        t_capture_ns: 1000000,
        stride_bytes: 1920 * 3,
    };

    // Mock YOLO output for 512x512 input
    // Format: [1, 16128, 7] = [batch, anchors, (x, y, w, h, conf, class_prob1, class_prob2...)]
    // For simplicity, let's create a few detections manually
    
    let anchors = 16128;
    let features = 7; // x, y, w, h, objectness, class0, class1 (simplified 2-class for demo)
    let total_size = anchors * features;
    
    let mut raw_output = vec![0.0f32; total_size];
    
    // Add a few fake detections with high confidence
    // Detection 1: Person at center (class 0)
    let idx1 = 100 * features;
    raw_output[idx1] = 256.0;      // x (center)
    raw_output[idx1 + 1] = 256.0;  // y (center)
    raw_output[idx1 + 2] = 80.0;   // w
    raw_output[idx1 + 3] = 150.0;  // h
    raw_output[idx1 + 4] = 0.85;   // objectness
    raw_output[idx1 + 5] = 0.9;    // class 0 prob
    raw_output[idx1 + 6] = 0.1;    // class 1 prob
    
    // Detection 2: Car on left (class 1)
    let idx2 = 500 * features;
    raw_output[idx2] = 150.0;      // x
    raw_output[idx2 + 1] = 300.0;  // y
    raw_output[idx2 + 2] = 100.0;  // w
    raw_output[idx2 + 3] = 60.0;   // h
    raw_output[idx2 + 4] = 0.75;   // objectness
    raw_output[idx2 + 5] = 0.2;    // class 0 prob
    raw_output[idx2 + 6] = 0.8;    // class 1 prob
    
    // Detection 3: Another person on right (class 0)
    let idx3 = 1000 * features;
    raw_output[idx3] = 400.0;      // x
    raw_output[idx3 + 1] = 200.0;  // y
    raw_output[idx3 + 2] = 70.0;   // w
    raw_output[idx3 + 3] = 140.0;  // h
    raw_output[idx3 + 4] = 0.80;   // objectness
    raw_output[idx3 + 5] = 0.85;   // class 0 prob
    raw_output[idx3 + 6] = 0.15;   // class 1 prob
    
    // Low confidence detection (should be filtered out)
    let idx4 = 2000 * features;
    raw_output[idx4] = 350.0;
    raw_output[idx4 + 1] = 350.0;
    raw_output[idx4 + 2] = 50.0;
    raw_output[idx4 + 3] = 50.0;
    raw_output[idx4 + 4] = 0.15;   // Low objectness - should be filtered
    raw_output[idx4 + 5] = 0.5;
    raw_output[idx4 + 6] = 0.5;

    RawDetectionsPacket {
        from: meta,
        raw_output,
        output_shape: vec![1, anchors, features],
    }
}

fn main() -> Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║           Postprocessing Stage Test                     ║");
    println!("║      YOLO Decoding + NMS + Temporal Smoothing           ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Create postprocessing stage
    println!("⚙️  Creating Postprocessing Stage:");
    
    let config = PostprocessConfig {
        num_classes: 80,  // COCO classes (but our mock has only 2 for simplicity)
        confidence_threshold: 0.25,
        nms_threshold: 0.45,
        max_detections: 100,
        letterbox_scale: 1.0,
        letterbox_pad: (0.0, 0.0),
        original_size: (512, 512),
    };
    
    println!("  Config:");
    println!("    Classes:        {}", config.num_classes);
    println!("    Conf Threshold: {}", config.confidence_threshold);
    println!("    NMS Threshold:  {}", config.nms_threshold);
    println!("    Max Detections: {}", config.max_detections);
    
    let mut post_stage = PostStage::new(config)
        .with_temporal_smoothing(4);  // 4-frame smoothing window
    
    println!("    Temporal Smoothing: Enabled (window=4)");
    println!();

    // Process multiple frames to test temporal smoothing
    println!("🎬 Processing frames with mock detections:\n");
    
    for frame_idx in 1..=5 {
        println!("═══════════════════════════════════════════════════════════");
        println!("Frame #{}", frame_idx);
        println!("═══════════════════════════════════════════════════════════");
        
        let mut mock_input = create_mock_detection_packet();
        mock_input.from.frame_idx = frame_idx;
        
        println!("\n📥 Input RawDetectionsPacket:");
        println!("  Frame Index:    {}", mock_input.from.frame_idx);
        println!("  Raw Output:     {} values", mock_input.raw_output.len());
        println!("  Output Shape:   {:?}", mock_input.output_shape);
        
        // Count non-zero confidence values for info
        let non_zero_count = mock_input.raw_output
            .chunks(7)
            .filter(|chunk| chunk[4] > 0.01) // objectness > 0.01
            .count();
        println!("  Non-zero Objs:  {} anchors", non_zero_count);
        
        // Process through postprocessing stage
        println!("\n⚙️  Processing...");
        let result = post_stage.process(mock_input);
        
        // Display results
        println!("\n📤 Output DetectionsPacket:");
        println!("  Detections:     {}", result.items.len());
        
        if !result.items.is_empty() {
            println!("\n  Detection Details:");
            for (i, det) in result.items.iter().enumerate() {
                println!("  ┌─ Detection #{}", i + 1);
                println!("  │  BBox:       x={:.1}, y={:.1}, w={:.1}, h={:.1}", 
                         det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h);
                println!("  │  Score:      {:.3}", det.score);
                println!("  │  Class ID:   {}", det.class_id);
                println!("  │  Track ID:   {:?}", det.track_id);
                println!("  └─");
            }
        } else {
            println!("  (No detections above threshold)");
        }
        
        println!();
    }
    
    println!("═══════════════════════════════════════════════════════════\n");
    
    // Test with SORT tracking
    println!("🔄 Testing with SORT Tracking:\n");
    
    let mut post_stage_with_tracking = PostStage::new(config)
        .with_temporal_smoothing(4)
        .with_sort_tracking(
            5,     // max_idle_epochs (keep tracks for 5 frames without detection)
            0.25,  // min_confidence
            0.3,   // iou_threshold for matching
        );
    
    println!("  ✓ SORT Tracking Enabled");
    println!("    Max Idle:       5 frames");
    println!("    Min Confidence: 0.25");
    println!("    IOU Threshold:  0.3");
    println!();
    
    for frame_idx in 1..=5 {
        println!("─────────────────────────────────────────────────────────");
        println!("Frame #{} with Tracking", frame_idx);
        println!("─────────────────────────────────────────────────────────");
        
        let mut mock_input = create_mock_detection_packet();
        mock_input.from.frame_idx = frame_idx;
        
        let result = post_stage_with_tracking.process(mock_input);
        
        println!("  Detections: {}", result.items.len());
        for (i, det) in result.items.iter().enumerate() {
            println!("    [{}] Class={}, Score={:.2}, Track={:?}, BBox=({:.0},{:.0},{:.0},{:.0})",
                     i, det.class_id, det.score, det.track_id,
                     det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h);
        }
        println!();
    }
    
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║              Postprocessing Test Complete! ✓            ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    Ok(())
}
