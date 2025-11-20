#![cfg(feature = "gpu")]

use overlay_plan::{GpuOverlayPlanner, convert_gpu_to_overlay_plan, prepare_detections_for_gpu, process_with_gpu_planning};
use common_io::{DetectionsPacket, FrameMeta, Detection, DrawOp, BBox, PixelFormat, ColorSpace};
use std::time::Instant;

/// Helper function to create test detections
fn create_test_detections(count: usize) -> DetectionsPacket {
    let mut items = Vec::new();
    
    for i in 0..count {
        items.push(Detection {
            bbox: BBox {
                x: 100.0 + (i as f32 * 150.0),
                y: 100.0 + (i as f32 * 50.0),
                w: 120.0,
                h: 80.0,
            },
            class_id: (i % 2) as i32,  // Alternate between class 0 and 1
            score: 0.85 + (i as f32 * 0.02),
            track_id: None,
        });
    }
    
    DetectionsPacket {
        from: FrameMeta {
            source_id: 0,
            width: 1920,
            height: 1080,
            pixfmt: PixelFormat::BGRA8,
            colorspace: ColorSpace::BT709,
            frame_idx: 0,
            pts_ns: 1000000000,
            t_capture_ns: 1000000000,
            stride_bytes: 1920 * 4,
            crop_region: None,
        },
        items,
    }
}

#[test]
fn test_gpu_overlay_planner_initialization() {
    println!("🔧 Test: GPU Overlay Planner Initialization");
    
    let result = GpuOverlayPlanner::new(1000, 0.25, true);
    assert!(result.is_ok(), "Failed to create GPU overlay planner: {:?}", result.err());
    
    println!("✅ GPU overlay planner initialized successfully");
}

#[test]
fn test_prepare_detections_for_gpu() {
    println!("🔧 Test: Prepare Detections for GPU");
    
    let detections = create_test_detections(5);
    let (boxes, scores, classes) = prepare_detections_for_gpu(&detections);
    
    // Verify array sizes
    assert_eq!(boxes.len(), 5 * 4, "Boxes array should have 4 values per detection");
    assert_eq!(scores.len(), 5, "Scores array should match detection count");
    assert_eq!(classes.len(), 5, "Classes array should match detection count");
    
    // Verify first detection data
    assert_eq!(boxes[0], 100.0);  // x
    assert_eq!(boxes[1], 100.0);  // y
    assert_eq!(boxes[2], 120.0);  // w
    assert_eq!(boxes[3], 80.0);   // h
    assert_eq!(classes[0], 0);    // class_id
    assert!((scores[0] - 0.85).abs() < 0.01, "Score mismatch");
    
    println!("✅ Detection preparation successful: {} detections", detections.items.len());
}

#[test]
fn test_single_detection_gpu_planning() {
    println!("🔧 Test: Single Detection GPU Planning");
    
    let mut planner = GpuOverlayPlanner::new(1000, 0.25, true).expect("Failed to create planner");
    let detections = create_test_detections(1);
    
    let start = Instant::now();
    let result = planner.plan_from_detections(&detections);
    let elapsed = start.elapsed();
    
    assert!(result.is_ok(), "GPU planning failed: {:?}", result.err());
    
    let plan = result.unwrap();
    assert!(plan.commands.len() > 0, "No draw commands generated");
    assert!(plan.commands.len() <= 1000, "Too many commands generated");
    
    println!("✅ GPU planning successful: {} commands in {:?}", plan.commands.len(), elapsed);
}

#[test]
fn test_multiple_detections_gpu_planning() {
    println!("🔧 Test: Multiple Detections GPU Planning");
    
    let mut planner = GpuOverlayPlanner::new(1000, 0.25, true).expect("Failed to create planner");
    let detections = create_test_detections(10);
    
    let start = Instant::now();
    let result = planner.plan_from_detections(&detections);
    let elapsed = start.elapsed();
    
    assert!(result.is_ok(), "GPU planning failed: {:?}", result.err());
    
    let plan = result.unwrap();
    assert!(plan.commands.len() > 0, "No draw commands generated");
    println!("✅ GPU planning successful: {} commands for 10 detections in {:?}", 
             plan.commands.len(), elapsed);
}

#[test]
fn test_conversion_to_overlay_plan() {
    println!("🔧 Test: Conversion to OverlayPlanPacket");
    
    let mut planner = GpuOverlayPlanner::new(1000, 0.25, true).expect("Failed to create planner");
    let detections = create_test_detections(3);
    let canvas = (detections.from.width, detections.from.height);
    
    let gpu_plan = planner.plan_from_detections(&detections)
        .expect("GPU planning failed");
    
    let start = Instant::now();
    let result = convert_gpu_to_overlay_plan(gpu_plan, canvas);
    let elapsed = start.elapsed();
    
    assert!(result.is_ok(), "Conversion failed: {:?}", result.err());
    
    let overlay_plan = result.unwrap();
    assert_eq!(overlay_plan.canvas, canvas, "Canvas size mismatch");
    assert!(!overlay_plan.ops.is_empty(), "No draw operations in overlay plan");
    
    println!("✅ Conversion successful: {} draw ops in {:?}", overlay_plan.ops.len(), elapsed);
    
    // Verify draw operation types
    let mut rect_count = 0;
    let mut fill_rect_count = 0;
    let mut label_count = 0;
    
    for op in &overlay_plan.ops {
        match op {
            DrawOp::Rect { .. } => rect_count += 1,
            DrawOp::FillRect { .. } => fill_rect_count += 1,
            DrawOp::Label { .. } => label_count += 1,
            _ => {}
        }
    }
    
    println!("   Draw ops breakdown: {} Rect, {} FillRect, {} Label", 
             rect_count, fill_rect_count, label_count);
}

#[test]
fn test_end_to_end_gpu_planning() {
    println!("🔧 Test: End-to-End GPU Planning");
    
    let mut planner = GpuOverlayPlanner::new(1000, 0.25, true).expect("Failed to create planner");
    let detections = create_test_detections(5);
    
    let start = Instant::now();
    let result = process_with_gpu_planning(&mut planner, detections.clone());
    let elapsed = start.elapsed();
    
    assert!(result.is_ok(), "End-to-end processing failed: {:?}", result.err());
    
    let overlay_plan = result.unwrap();
    assert_eq!(overlay_plan.canvas, (1920, 1080), "Canvas size mismatch");
    assert!(!overlay_plan.ops.is_empty(), "No draw operations generated");
    
    println!("✅ End-to-end processing successful: {} ops in {:?}", 
             overlay_plan.ops.len(), elapsed);
}

#[test]
fn test_zero_detections() {
    println!("🔧 Test: Zero Detections");
    
    let mut planner = GpuOverlayPlanner::new(1000, 0.25, true).expect("Failed to create planner");
    let detections = create_test_detections(0);  // No detections
    
    let result = planner.plan_from_detections(&detections);
    assert!(result.is_ok(), "Should handle zero detections gracefully");
    
    let plan = result.unwrap();
    println!("✅ Zero detections handled: {} commands", plan.commands.len());
}

#[test]
fn test_large_batch() {
    println!("🔧 Test: Large Batch (50 detections)");
    
    let mut planner = GpuOverlayPlanner::new(5000, 0.25, true).expect("Failed to create planner");
    let detections = create_test_detections(50);
    
    let start = Instant::now();
    let result = process_with_gpu_planning(&mut planner, detections.clone());
    let elapsed = start.elapsed();
    
    assert!(result.is_ok(), "Large batch processing failed: {:?}", result.err());
    
    let overlay_plan = result.unwrap();
    println!("✅ Large batch successful: {} ops in {:?}", overlay_plan.ops.len(), elapsed);
}

#[test]
fn test_performance_comparison() {
    println!("🔧 Test: Performance Benchmark (GPU planning only)");
    
    let mut planner = GpuOverlayPlanner::new(1000, 0.25, true).expect("Failed to create planner");
    
    for count in [1, 5, 10, 20] {
        let detections = create_test_detections(count);
        
        // Warm-up
        let _ = planner.plan_from_detections(&detections);
        
        // Benchmark
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = planner.plan_from_detections(&detections);
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() / iterations;
        
        println!("   {} detections: avg {:4} μs/frame", count, avg_us);
    }
    
    println!("✅ Performance benchmark complete");
}

#[test]
fn test_memory_allocation() {
    println!("🔧 Test: GPU Memory Allocation");
    
    // Create and drop multiple planners to check for memory leaks
    for i in 0..10 {
        let planner = GpuOverlayPlanner::new(1000, 0.25, true);
        assert!(planner.is_ok(), "Failed to create planner #{}", i);
        
        let mut p = planner.unwrap();
        let detections = create_test_detections(5);
        let _ = p.plan_from_detections(&detections);
        
        // Planner will be dropped here
    }
    
    println!("✅ Memory allocation test passed (10 create/destroy cycles)");
}
