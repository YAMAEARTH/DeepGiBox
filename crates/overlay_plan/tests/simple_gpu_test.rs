#[cfg(feature = "gpu")]
#[test]
fn test_single_detection() {
    use overlay_plan::gpu_overlay::{GpuOverlayPlanner, DrawCommandType};
    use common_io::{FrameMeta, PixelFormat, ColorSpace};
    
    let mut planner = GpuOverlayPlanner::new(1000, 0.1, false)
        .expect("Failed to init GPU overlay planner");
    
    // Single detection with NORMALIZED coordinates (0-1 range)
    // Kernel multiplies by img_width/img_height to get pixel coordinates
    let boxes = vec![0.5f32, 0.5f32, 0.2f32, 0.3f32];  // center (0.5, 0.5), size (0.2, 0.3)
    let scores = vec![0.95f32];
    let classes = vec![0i32];  // polyp
    
    println!("DEBUG: Using normalized boxes = {:?}", boxes);
    println!("DEBUG: Image size = 1000×1000");
    println!("DEBUG: Expected pixel coords: x=400, y=350, w=200, h=300");
    
    unsafe {
        let mut d_boxes: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut d_scores: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut d_classes: *mut std::ffi::c_void = std::ptr::null_mut();
        
        extern "C" {
            fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> i32;
            fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, count: usize, kind: i32) -> i32;
            fn cudaFree(devPtr: *mut std::ffi::c_void) -> i32;
        }
        
        cudaMalloc(&mut d_boxes, 4 * std::mem::size_of::<f32>());
        cudaMalloc(&mut d_scores, 1 * std::mem::size_of::<f32>());
        cudaMalloc(&mut d_classes, 1 * std::mem::size_of::<i32>());
        
        cudaMemcpy(d_boxes, boxes.as_ptr() as *const _, 4 * std::mem::size_of::<f32>(), 1);
        cudaMemcpy(d_scores, scores.as_ptr() as *const _, 1 * std::mem::size_of::<f32>(), 1);
        cudaMemcpy(d_classes, classes.as_ptr() as *const _, 1 * std::mem::size_of::<i32>(), 1);
        
        let frame_meta = FrameMeta {
            source_id: 0,
            width: 1000,
            height: 1000,
            pixfmt: PixelFormat::RGB8,
            colorspace: ColorSpace::BT709,
            frame_idx: 0,
            pts_ns: 0,
            t_capture_ns: 0,
            stride_bytes: 1000 * 3,
            crop_region: None,
        };
        
        let result = planner.plan_gpu(
            d_boxes as *const f32,
            d_scores as *const f32,
            d_classes as *const i32,
            1,  // single detection
            &frame_meta,
        );
        
        cudaFree(d_boxes);
        cudaFree(d_scores);
        cudaFree(d_classes);
        
        match result {
            Ok(plan) => {
                println!("✅ Single detection test passed");
                println!("   Commands: {}", plan.commands.len());
                
                for (i, cmd) in plan.commands.iter().enumerate() {
                    match cmd.command_type() {
                        DrawCommandType::Rect => {
                            if let Some(rect) = cmd.get_rect() {
                                println!("   Cmd {}: RECT x={:.1} y={:.1} w={:.1} h={:.1}",
                                    i, rect.x, rect.y, rect.w, rect.h);
                                
                                // Expected:
                                // x = 0.5 * 1000 - 0.2 * 1000 * 0.5 = 500 - 100 = 400
                                // y = 0.5 * 1000 - 0.3 * 1000 * 0.5 = 500 - 150 = 350
                                // w = 0.2 * 1000 = 200
                                // h = 0.3 * 1000 = 300
                                
                                assert!((rect.x - 400.0).abs() < 1.0, "X should be ~400, got {}", rect.x);
                                assert!((rect.y - 350.0).abs() < 1.0, "Y should be ~350, got {}", rect.y);
                                assert!((rect.w - 200.0).abs() < 1.0, "W should be ~200, got {}", rect.w);
                                assert!((rect.h - 300.0).abs() < 1.0, "H should be ~300, got {}", rect.h);
                            }
                        },
                        DrawCommandType::Label => {
                            if let Some(label) = cmd.get_label() {
                                println!("   Cmd {}: LABEL text='{}'", i, label.text);
                                assert!(label.text.contains("Polyp") || label.text.contains("95"));
                            }
                        },
                        _ => {}
                    }
                }
                
                assert!(plan.commands.len() >= 2, "Expected at least 2 commands (box + label)");
            },
            Err(e) => {
                panic!("Single detection test failed: {:?}", e);
            }
        }
    }
}
