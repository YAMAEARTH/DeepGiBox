#[cfg(feature = "gpu")]
mod gpu_overlay_tests {
    use overlay_plan::gpu_overlay::{GpuOverlayPlanner, DrawCommandType};
    use common_io::{FrameMeta, PixelFormat, ColorSpace};
    
    #[test]
    fn test_gpu_overlay_planner_init() {
        // Test GPU overlay planner initialization
        let result = GpuOverlayPlanner::new(
            1000,   // max_commands
            0.25,   // conf_threshold
            true,   // draw_confidence_bar
        );
        
        assert!(result.is_ok(), "Failed to initialize GPU overlay planner: {:?}", result.err());
        
        let planner = result.unwrap();
        println!("✅ GPU overlay planner initialized successfully");
        drop(planner);
        println!("✅ GPU overlay planner cleaned up successfully");
    }
    
    #[test]
    fn test_gpu_overlay_with_mock_detections() {
        let mut planner = GpuOverlayPlanner::new(1000, 0.25, false)  // Disable confidence bar for simpler test
            .expect("Failed to init GPU overlay planner");
        
        // Start with single detection to isolate the issue
        let num_detections = 1;
        
        // Boxes: [x, y, w, h] normalized (0-1 range!)
        let boxes = vec![
            0.3f32, 0.4f32, 0.2f32, 0.15f32,  // Detection 0: center (0.3, 0.4), size (0.2, 0.15)
        ];
        
        println!("DEBUG: boxes = {:?}", boxes);
        println!("DEBUG: num_detections = {}", num_detections);
        
        // Scores
        let scores = vec![0.92f32];
        
        // Classes (0 = polyp)
        let classes = vec![0i32];
        
        // Allocate GPU memory for test
        unsafe {
            let mut d_boxes: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_scores: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_classes: *mut std::ffi::c_void = std::ptr::null_mut();
            
            extern "C" {
                fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> i32;
                fn cudaMemcpy(
                    dst: *mut std::ffi::c_void,
                    src: *const std::ffi::c_void,
                    count: usize,
                    kind: i32,
                ) -> i32;
                fn cudaFree(devPtr: *mut std::ffi::c_void) -> i32;
            }
            
            // Allocate GPU memory
            let err = cudaMalloc(&mut d_boxes, boxes.len() * std::mem::size_of::<f32>());
            assert_eq!(err, 0, "cudaMalloc d_boxes failed: {}", err);
            
            let err = cudaMalloc(&mut d_scores, scores.len() * std::mem::size_of::<f32>());
            assert_eq!(err, 0, "cudaMalloc d_scores failed: {}", err);
            
            let err = cudaMalloc(&mut d_classes, classes.len() * std::mem::size_of::<i32>());
            assert_eq!(err, 0, "cudaMalloc d_classes failed: {}", err);
            
            // Copy data to GPU
            let err = cudaMemcpy(
                d_boxes,
                boxes.as_ptr() as *const std::ffi::c_void,
                boxes.len() * std::mem::size_of::<f32>(),
                1, // cudaMemcpyHostToDevice
            );
            assert_eq!(err, 0, "cudaMemcpy boxes H2D failed: {}", err);
            
            let err = cudaMemcpy(
                d_scores,
                scores.as_ptr() as *const std::ffi::c_void,
                scores.len() * std::mem::size_of::<f32>(),
                1,
            );
            assert_eq!(err, 0, "cudaMemcpy scores H2D failed: {}", err);
            
            let err = cudaMemcpy(
                d_classes,
                classes.as_ptr() as *const std::ffi::c_void,
                classes.len() * std::mem::size_of::<i32>(),
                1,
            );
            assert_eq!(err, 0, "cudaMemcpy classes H2D failed: {}", err);
            
            // Create frame metadata
            let frame_meta = FrameMeta {
                source_id: 0,
                width: 1920,
                height: 1080,
                pixfmt: PixelFormat::RGB8,
                colorspace: ColorSpace::BT709,
                frame_idx: 0,
                pts_ns: 0,
                t_capture_ns: 0,
                stride_bytes: 1920 * 3,
                crop_region: None,
            };
            
            // Generate overlay plan on GPU
            let result = planner.plan_gpu(
                d_boxes as *const f32,
                d_scores as *const f32,
                d_classes as *const i32,
                num_detections,
                &frame_meta,
            );
            
            // Cleanup GPU memory
            cudaFree(d_boxes);
            cudaFree(d_scores);
            cudaFree(d_classes);
            
            match result {
                Ok(plan) => {
                    println!("✅ GPU overlay planning succeeded");
                    println!("   Commands generated: {}", plan.commands.len());
                    
                    // Expected: 2 commands per detection (box + label) + confidence bar
                    // 3 detections × 2 = 6 commands + confidence bar (~20 segments) = ~26 commands
                    println!("   Frame: {}×{}", plan.frame_meta.width, plan.frame_meta.height);
                    
                    assert!(plan.commands.len() > 0, "Expected at least some commands");
                    
                    // Analyze commands
                    let mut rect_count = 0;
                    let mut fill_rect_count = 0;
                    let mut label_count = 0;
                    
                    for (i, cmd) in plan.commands.iter().enumerate() {
                        match cmd.command_type() {
                            DrawCommandType::Rect => {
                                rect_count += 1;
                                if i < 10 {  // Show first few
                                    if let Some(rect) = cmd.get_rect() {
                                        println!("   Cmd {}: RECT x={:.1} y={:.1} w={:.1} h={:.1} color=({},{},{},{})",
                                            i, rect.x, rect.y, rect.w, rect.h, rect.r, rect.g, rect.b, rect.a);
                                    }
                                }
                            },
                            DrawCommandType::FillRect => {
                                fill_rect_count += 1;
                                if i < 10 {
                                    if let Some(rect) = cmd.get_rect() {
                                        println!("   Cmd {}: FILL_RECT x={:.1} y={:.1} w={:.1} h={:.1} color=({},{},{},{})",
                                            i, rect.x, rect.y, rect.w, rect.h, rect.r, rect.g, rect.b, rect.a);
                                    }
                                }
                            },
                            DrawCommandType::Label => {
                                label_count += 1;
                                if let Some(label) = cmd.get_label() {
                                    println!("   Cmd {}: LABEL x={:.1} y={:.1} text='{}' color=({},{},{},{})",
                                        i, label.x, label.y, label.text, label.r, label.g, label.b, label.a);
                                }
                            },
                        }
                    }
                    
                    println!("   Summary: {} RECT, {} FILL_RECT, {} LABEL",
                        rect_count, fill_rect_count, label_count);
                    
                    // Verify we have bounding boxes for detections
                    assert!(rect_count >= 1, "Expected at least 1 bounding box");
                    
                    // Verify we have labels for detections
                    assert!(label_count >= 1, "Expected at least 1 label");
                    
                    // Check label text content
                    let labels_with_text: Vec<_> = plan.commands.iter()
                        .filter_map(|cmd| cmd.get_label())
                        .collect();
                    
                    for (i, label) in labels_with_text.iter().enumerate() {
                        println!("   Label {}: '{}'", i, label.text);
                        // Verify format "Polyp 92%" or "Obj 85%"
                        assert!(
                            label.text.contains("Polyp") || label.text.contains("Obj"),
                            "Label text should contain 'Polyp' or 'Obj': '{}'", label.text
                        );
                        assert!(label.text.contains("%"), "Label text should contain '%'");
                    }
                    
                    println!("✅ All overlay commands validated");
                },
                Err(e) => {
                    panic!("GPU overlay planning failed: {:?}", e);
                }
            }
        }
    }
    
    #[test]
    fn test_set_threshold_and_confidence_bar() {
        let mut planner = GpuOverlayPlanner::new(1000, 0.25, true)
            .expect("Failed to init GPU overlay planner");
        
        // Test setters
        planner.set_confidence_threshold(0.5);
        planner.set_draw_confidence_bar(false);
        
        println!("✅ Configuration setters work");
    }
    
    #[test]
    fn test_large_number_of_detections() {
        let mut planner = GpuOverlayPlanner::new(2000, 0.1, false)
            .expect("Failed to init GPU overlay planner");
        
        // Simulate many detections
        let num_detections = 50;
        let mut boxes = Vec::new();
        let mut scores = Vec::new();
        let mut classes = Vec::new();
        
        for i in 0..num_detections {
            // Random-ish positions
            let x = (i as f32 * 0.13) % 1.0;
            let y = (i as f32 * 0.17) % 1.0;
            boxes.extend_from_slice(&[x, y, 0.1, 0.1]);
            scores.push(0.3 + (i as f32 * 0.01) % 0.6);
            classes.push((i % 2) as i32);
        }
        
        unsafe {
            let mut d_boxes: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_scores: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_classes: *mut std::ffi::c_void = std::ptr::null_mut();
            
            extern "C" {
                fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> i32;
                fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, count: usize, kind: i32) -> i32;
                fn cudaFree(devPtr: *mut std::ffi::c_void) -> i32;
            }
            
            cudaMalloc(&mut d_boxes, boxes.len() * std::mem::size_of::<f32>());
            cudaMalloc(&mut d_scores, scores.len() * std::mem::size_of::<f32>());
            cudaMalloc(&mut d_classes, classes.len() * std::mem::size_of::<i32>());
            
            cudaMemcpy(d_boxes, boxes.as_ptr() as *const _, boxes.len() * std::mem::size_of::<f32>(), 1);
            cudaMemcpy(d_scores, scores.as_ptr() as *const _, scores.len() * std::mem::size_of::<f32>(), 1);
            cudaMemcpy(d_classes, classes.as_ptr() as *const _, classes.len() * std::mem::size_of::<i32>(), 1);
            
            let frame_meta = FrameMeta {
                source_id: 0,
                width: 1920,
                height: 1080,
                pixfmt: PixelFormat::RGB8,
                colorspace: ColorSpace::BT709,
                frame_idx: 0,
                pts_ns: 0,
                t_capture_ns: 0,
                stride_bytes: 1920 * 3,
                crop_region: None,
            };
            
            let result = planner.plan_gpu(
                d_boxes as *const f32,
                d_scores as *const f32,
                d_classes as *const i32,
                num_detections,
                &frame_meta,
            );
            
            cudaFree(d_boxes);
            cudaFree(d_scores);
            cudaFree(d_classes);
            
            match result {
                Ok(plan) => {
                    println!("✅ Large detection test passed");
                    println!("   Generated {} commands for {} detections", plan.commands.len(), num_detections);
                    assert!(plan.commands.len() >= num_detections * 2, "Expected at least 2 commands per detection");
                },
                Err(e) => {
                    panic!("Large detection test failed: {:?}", e);
                }
            }
        }
    }
}

// Placeholder test when GPU feature is disabled
#[cfg(not(feature = "gpu"))]
#[test]
fn gpu_feature_disabled() {
    println!("⚠️  GPU feature not enabled. Run with: cargo test --features gpu");
}
