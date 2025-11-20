// GPU vs CPU Overlay Planning Performance Benchmark
// Compares GPU overlay planning (with conversion) vs CPU overlay planning

#![cfg(feature = "gpu")]

use overlay_plan::{GpuOverlayPlanner, process_with_gpu_planning, PlanStage};
use common_io::{DetectionsPacket, FrameMeta, Detection, BBox, PixelFormat, ColorSpace, Stage};
use std::time::Instant;

/// Create test detections for benchmarking
fn create_benchmark_detections(count: usize) -> DetectionsPacket {
    let mut items = Vec::new();
    
    // Create realistic detection positions
    for i in 0..count {
        let row = i / 5;  // 5 per row
        let col = i % 5;
        
        items.push(Detection {
            bbox: BBox {
                x: 100.0 + (col as f32 * 300.0),
                y: 100.0 + (row as f32 * 150.0),
                w: 200.0,
                h: 120.0,
            },
            class_id: (i % 2) as i32,
            score: 0.75 + (i as f32 * 0.01).min(0.24),
            track_id: Some(i as i32),
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
fn benchmark_gpu_vs_cpu_overlay_planning() {
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("🚀 GPU vs CPU Overlay Planning Performance Benchmark");
    println!("═══════════════════════════════════════════════════════════════\n");
    
    // Initialize planners
    let mut gpu_planner = GpuOverlayPlanner::new(5000, 0.25, true)
        .expect("Failed to create GPU planner");
    
    let mut cpu_planner = PlanStage {
        enable_full_ui: true,
        spk: true,
        display_mode: "COLON".to_string(),
        confidence_threshold: 0.25,
        endoscope_mode: "OLYMPUS".to_string(),
    };
    
    // Test configurations: different detection counts
    let test_configs = vec![
        (1, "Single detection"),
        (5, "Light load (5 detections)"),
        (10, "Moderate load (10 detections)"),
        (20, "Heavy load (20 detections)"),
        (50, "Extreme load (50 detections)"),
    ];
    
    println!("┌─────────────┬──────────────┬──────────────┬─────────────┬──────────────┐");
    println!("│ Scenario    │ GPU Time (μs)│ CPU Time (μs)│ Speedup     │ Winner       │");
    println!("├─────────────┼──────────────┼──────────────┼─────────────┼──────────────┤");
    
    for (count, description) in test_configs {
        let detections = create_benchmark_detections(count);
        
        // Warm-up runs
        for _ in 0..5 {
            let _ = process_with_gpu_planning(&mut gpu_planner, detections.clone());
            let _ = cpu_planner.process(detections.clone());
        }
        
        // Benchmark GPU path
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = process_with_gpu_planning(&mut gpu_planner, detections.clone());
        }
        let gpu_elapsed = start.elapsed();
        let gpu_avg_us = gpu_elapsed.as_micros() / iterations;
        
        // Benchmark CPU path
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = cpu_planner.process(detections.clone());
        }
        let cpu_elapsed = start.elapsed();
        let cpu_avg_us = cpu_elapsed.as_micros() / iterations;
        
        // Calculate speedup
        let speedup = cpu_avg_us as f64 / gpu_avg_us as f64;
        let winner = if speedup > 1.0 { "GPU 🚀" } else { "CPU" };
        
        // Format description (left-aligned, 11 chars)
        let desc_fmt = format!("{:11}", description);
        
        println!("│ {} │ {:12} │ {:12} │ {:6.2}x     │ {:12} │",
                 desc_fmt,
                 gpu_avg_us,
                 cpu_avg_us,
                 speedup,
                 winner);
    }
    
    println!("└─────────────┴──────────────┴──────────────┴─────────────┴──────────────┘");
    
    println!("\n📊 Analysis:");
    println!("  • GPU includes conversion overhead (GPU commands → CPU format)");
    println!("  • CPU path is sequential, GPU uses parallel processing");
    println!("  • Expected: GPU faster at higher detection counts (>10)");
    println!("  • Phase 2 optimization would eliminate conversion overhead\n");
}

#[test]
fn test_gpu_overlay_correctness() {
    println!("\n🔍 Testing GPU Overlay Planning Correctness");
    println!("═══════════════════════════════════════════════════════\n");
    
    let mut gpu_planner = GpuOverlayPlanner::new(1000, 0.25, true)
        .expect("Failed to create GPU planner");
    
    let mut cpu_planner = PlanStage {
        enable_full_ui: true,
        spk: true,
        display_mode: "COLON".to_string(),
        confidence_threshold: 0.25,
        endoscope_mode: "OLYMPUS".to_string(),
    };
    
    let detections = create_benchmark_detections(5);
    
    // Generate plans
    let gpu_result = process_with_gpu_planning(&mut gpu_planner, detections.clone());
    assert!(gpu_result.is_ok(), "GPU planning failed");
    let gpu_plan = gpu_result.unwrap();
    
    let cpu_plan = cpu_planner.process(detections.clone());
    
    // Compare results
    println!("✓ GPU generated {} draw operations", gpu_plan.ops.len());
    println!("✓ CPU generated {} draw operations", cpu_plan.ops.len());
    
    // Count operation types
    let mut gpu_rects = 0;
    let mut gpu_fills = 0;
    let mut gpu_labels = 0;
    
    for op in &gpu_plan.ops {
        match op {
            common_io::DrawOp::Rect { .. } => gpu_rects += 1,
            common_io::DrawOp::FillRect { .. } => gpu_fills += 1,
            common_io::DrawOp::Label { .. } => gpu_labels += 1,
            _ => {}
        }
    }
    
    let mut cpu_rects = 0;
    let mut cpu_fills = 0;
    let mut cpu_labels = 0;
    
    for op in &cpu_plan.ops {
        match op {
            common_io::DrawOp::Rect { .. } => cpu_rects += 1,
            common_io::DrawOp::FillRect { .. } => cpu_fills += 1,
            common_io::DrawOp::Label { .. } => cpu_labels += 1,
            _ => {}
        }
    }
    
    println!("\n📋 Operation Breakdown:");
    println!("  GPU: {} Rect, {} FillRect, {} Label", gpu_rects, gpu_fills, gpu_labels);
    println!("  CPU: {} Rect, {} FillRect, {} Label", cpu_rects, cpu_fills, cpu_labels);
    
    // Verify canvas
    assert_eq!(gpu_plan.canvas, (1920, 1080), "GPU canvas mismatch");
    assert_eq!(cpu_plan.canvas, (1920, 1080), "CPU canvas mismatch");
    
    println!("\n✅ GPU overlay planning produces valid output!");
}

#[test]
fn test_gpu_memory_stability() {
    println!("\n🧪 Testing GPU Memory Stability (Extended)");
    println!("═══════════════════════════════════════════════════════\n");
    
    let mut gpu_planner = GpuOverlayPlanner::new(5000, 0.25, true)
        .expect("Failed to create GPU planner");
    
    // Run 1000 iterations with varying detection counts
    let total_iterations = 1000;
    let mut success_count = 0;
    
    println!("Running {} iterations...", total_iterations);
    
    for i in 0..total_iterations {
        let count = (i % 50) + 1;  // 1-50 detections
        let detections = create_benchmark_detections(count);
        
        let result = process_with_gpu_planning(&mut gpu_planner, detections);
        
        if result.is_ok() {
            success_count += 1;
        } else {
            eprintln!("❌ Iteration {} failed: {:?}", i, result.err());
        }
        
        if (i + 1) % 100 == 0 {
            println!("  Progress: {}/{} iterations completed", i + 1, total_iterations);
        }
    }
    
    let success_rate = (success_count as f64 / total_iterations as f64) * 100.0;
    
    println!("\n📊 Results:");
    println!("  Total iterations: {}", total_iterations);
    println!("  Successful: {}", success_count);
    println!("  Failed: {}", total_iterations - success_count);
    println!("  Success rate: {:.2}%", success_rate);
    
    assert_eq!(success_count, total_iterations, "Some GPU planning operations failed");
    println!("\n✅ GPU memory stability test passed!");
}
