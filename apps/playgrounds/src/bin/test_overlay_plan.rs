// test_overlay_plan.rs - Test OverlayPlan stage with mocked DetectionsPacket
// This playground simulates the conversion from DetectionsPacket to OverlayPlanPacket

use common_io::{
    DetectionsPacket, Detection, BBox, FrameMeta, 
    PixelFormat, ColorSpace, Stage, DrawOp, OverlayPlanPacket
};
use overlay_plan::from_path;
use rand::Rng;

/// Create a mock FrameMeta for testing
fn create_mock_frame_meta(frame_idx: u64) -> FrameMeta {
    FrameMeta {
        source_id: 0,
        width: 1920,
        height: 1080,
        pixfmt: PixelFormat::BGRA8,
        colorspace: ColorSpace::BT709,
        frame_idx,
        pts_ns: frame_idx * 33_333_333, // ~30 FPS
        t_capture_ns: frame_idx * 33_333_333,
        stride_bytes: 1920 * 4, // BGRA = 4 bytes per pixel
    }
}

/// Generate random detections for testing
fn create_mock_detections(frame_idx: u64, num_detections: usize) -> Vec<Detection> {
    let mut rng = rand::thread_rng();
    let mut detections = Vec::new();

    for i in 0..num_detections {
        // Random class (0 = Hyper, 1 = Neo)
        let class_id = rng.gen_range(0..=1);
        
        // Random score between 0.6 and 0.99
        let score = rng.gen_range(0.6..0.99);
        
        // Random bounding box
        let x = rng.gen_range(50.0..1700.0);
        let y = rng.gen_range(50.0..900.0);
        let w = rng.gen_range(80.0..200.0);
        let h = rng.gen_range(80.0..200.0);
        
        // Track ID (some have tracking, some don't)
        let track_id = if rng.gen_bool(0.8) {
            Some(i as i32 + (frame_idx as i32 * 100))
        } else {
            None
        };

        detections.push(Detection {
            bbox: BBox { x, y, w, h },
            score,
            class_id,
            track_id,
        });
    }

    detections
}

/// Create a complete mock DetectionsPacket
fn create_mock_detections_packet(frame_idx: u64, num_detections: usize) -> DetectionsPacket {
    DetectionsPacket {
        from: create_mock_frame_meta(frame_idx),
        items: create_mock_detections(frame_idx, num_detections),
    }
}

/// Print detection information
fn print_detections(packet: &DetectionsPacket) {
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ INPUT: DetectionsPacket                                     │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Frame Index: {:<47}│", packet.from.frame_idx);
    println!("│ Resolution:  {}x{:<43}│", packet.from.width, packet.from.height);
    println!("│ Detections:  {:<47}│", packet.items.len());
    println!("└─────────────────────────────────────────────────────────────┘");
    
    for (idx, det) in packet.items.iter().enumerate() {
        let class_name = match det.class_id {
            0 => "Hyper",
            1 => "Neo",
            _ => "Unknown",
        };
        
        let track_str = if let Some(tid) = det.track_id {
            format!("ID:{}", tid)
        } else {
            "No Track".to_string()
        };
        
        println!(
            "  [{}] {} ({:.2}) | bbox: ({:.1}, {:.1}, {:.1}, {:.1}) | {}",
            idx, class_name, det.score, 
            det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h,
            track_str
        );
    }
}

/// Print overlay plan information
fn print_overlay_plan(packet: &OverlayPlanPacket) {
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ OUTPUT: OverlayPlanPacket                                   │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Frame Index: {:<47}│", packet.from.frame_idx);
    println!("│ Canvas:      {}x{:<43}│", packet.canvas.0, packet.canvas.1);
    println!("│ Draw Ops:    {:<47}│", packet.ops.len());
    println!("└─────────────────────────────────────────────────────────────┘");
    
    for (idx, op) in packet.ops.iter().enumerate() {
        match op {
            DrawOp::Rect { xywh, thickness } => {
                println!(
                    "  [{}] Rect: ({:.1}, {:.1}, {:.1}, {:.1}) thickness={}",
                    idx, xywh.0, xywh.1, xywh.2, xywh.3, thickness
                );
            }
            DrawOp::Label { anchor, text, font_px } => {
                println!(
                    "  [{}] Label: \"{}\" @ ({:.1}, {:.1}) font_px={}",
                    idx, text, anchor.0, anchor.1, font_px
                );
            }
            DrawOp::Poly { pts, thickness } => {
                println!(
                    "  [{}] Poly: {} points, thickness={}",
                    idx, pts.len(), thickness
                );
            }
        }
    }
}

/// Verify that the conversion is correct
fn verify_conversion_simple(
    expected_frame_idx: u64,
    expected_canvas: (u32, u32),
    expected_det_count: usize,
    detections: &[(i32, f32, f32, f32, f32, f32)], // (class_id, score, x, y, w, h)
    output: &OverlayPlanPacket
) -> bool {
    let mut success = true;
    
    // Check frame metadata
    if expected_frame_idx != output.from.frame_idx {
        println!("❌ Frame index mismatch!");
        success = false;
    }
    
    // Check canvas size
    if expected_canvas != output.canvas {
        println!("❌ Canvas size mismatch!");
        success = false;
    }
    
    // Each detection should produce 2 DrawOps (Rect + Label)
    let expected_ops = expected_det_count * 2;
    if output.ops.len() != expected_ops {
        println!("❌ Expected {} ops, got {}", expected_ops, output.ops.len());
        success = false;
    }
    
    // Verify DrawOp pairing (Rect followed by Label)
    for i in 0..expected_det_count {
        let rect_idx = i * 2;
        let label_idx = i * 2 + 1;
        
        if rect_idx >= output.ops.len() || label_idx >= output.ops.len() {
            println!("❌ Missing DrawOps for detection {}", i);
            success = false;
            continue;
        }
        
        match (&output.ops[rect_idx], &output.ops[label_idx]) {
            (DrawOp::Rect { xywh, .. }, DrawOp::Label { text, .. }) => {
                let (class_id, _score, x, y, w, h) = detections[i];
                
                // Verify bbox matches
                if (xywh.0 - x).abs() > 0.01 || 
                   (xywh.1 - y).abs() > 0.01 ||
                   (xywh.2 - w).abs() > 0.01 ||
                   (xywh.3 - h).abs() > 0.01 {
                    println!("❌ Bbox mismatch for detection {}", i);
                    success = false;
                }
                
                // Verify label contains class name
                let class_name = match class_id {
                    0 => "Hyper",
                    1 => "Neo",
                    _ => "Unknown",
                };
                
                if !text.contains(class_name) {
                    println!("❌ Label missing class name for detection {}", i);
                    success = false;
                }
            }
            _ => {
                println!("❌ Wrong DrawOp order for detection {}", i);
                success = false;
            }
        }
    }
    
    if success {
        println!("✅ All verifications passed!");
    }
    
    success
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║            OVERLAY PLAN STAGE TEST (MOCK DATA)                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    
    // Create the stage
    let mut stage = from_path("").expect("Failed to create overlay plan stage");
    
    // Test multiple frames with different detection counts
    let test_cases = vec![
        (1, 0),  // Frame 1: no detections
        (2, 1),  // Frame 2: 1 detection
        (3, 3),  // Frame 3: 3 detections
        (4, 5),  // Frame 4: 5 detections
        (5, 8),  // Frame 5: 8 detections
    ];
    
    let mut all_passed = true;
    
    for (frame_idx, num_detections) in test_cases {
        println!("\n");
        println!("═══════════════════════════════════════════════════════════════");
        println!("  TEST CASE: Frame {} with {} detections", frame_idx, num_detections);
        println!("═══════════════════════════════════════════════════════════════");
        
        // Create mock input
        let input_packet = create_mock_detections_packet(frame_idx, num_detections);
        
        // Print input
        print_detections(&input_packet);
        
        // Store values for verification before consuming input_packet
        let expected_frame_idx = input_packet.from.frame_idx;
        let expected_canvas = (input_packet.from.width, input_packet.from.height);
        let expected_det_count = input_packet.items.len();
        let detections_copy: Vec<_> = input_packet.items.iter().map(|d| {
            (d.class_id, d.score, d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h)
        }).collect();
        
        // Process through the stage (consume input_packet)
        let output_packet = stage.process(input_packet);
        
        // Print output
        print_overlay_plan(&output_packet);
        
        // Verify conversion
        println!("\n┌─────────────────────────────────────────────────────────────┐");
        println!("│ VERIFICATION                                                │");
        println!("└─────────────────────────────────────────────────────────────┘");
        let passed = verify_conversion_simple(
            expected_frame_idx,
            expected_canvas,
            expected_det_count,
            &detections_copy,
            &output_packet
        );
        
        if !passed {
            all_passed = false;
        }
    }
    
    // Final summary
    println!("\n\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                        TEST SUMMARY                              ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    if all_passed {
        println!("║  ✅ ALL TESTS PASSED                                            ║");
    } else {
        println!("║  ❌ SOME TESTS FAILED                                           ║");
    }
    println!("╚══════════════════════════════════════════════════════════════════╝");
}
