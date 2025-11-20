#[cfg(feature = "gpu")]
mod gpu_tests {
    use postprocess::GpuPostprocessor;
    use common_io::{FrameMeta, PixelFormat, ColorSpace};
    
    #[test]
    fn test_gpu_postprocessor_init() {
        // Test that GPU postprocessor can be initialized
        let result = GpuPostprocessor::new(
            25200,  // num_anchors (YOLOv5)
            80,     // num_classes (COCO)
            300,    // max_detections
            0.25,   // conf_threshold
            0.45,   // nms_threshold
        );
        
        assert!(result.is_ok(), "Failed to initialize GPU postprocessor: {:?}", result.err());
        
        let processor = result.unwrap();
        println!("✅ GPU postprocessor initialized successfully");
        drop(processor);
        println!("✅ GPU postprocessor cleaned up successfully");
    }
    
    #[test]
    fn test_gpu_postprocessor_with_mock_data() {
        let mut processor = GpuPostprocessor::new(
            25200, 80, 300, 0.25, 0.45
        ).expect("Failed to init GPU postprocessor");
        
        // Create mock YOLO output (25200 anchors × 85 values)
        let num_anchors = 25200;
        let num_values = 85; // 4 bbox + 1 obj + 80 classes
        let mut raw_output = vec![0.0f32; num_anchors * num_values];
        
        // Add one synthetic detection
        // Format: [cx, cy, w, h, obj_conf, class_0, class_1, ..., class_79]
        let anchor_idx = 1000;
        let offset = anchor_idx * num_values;
        
        raw_output[offset + 0] = 320.0;  // cx (normalized 0-640)
        raw_output[offset + 1] = 240.0;  // cy (normalized 0-480)
        raw_output[offset + 2] = 100.0;  // w
        raw_output[offset + 3] = 150.0;  // h
        raw_output[offset + 4] = 0.85;   // objectness score
        raw_output[offset + 5] = 0.92;   // class 0 (person) score
        
        // Allocate GPU memory for test data
        unsafe {
            let mut d_raw_output: *mut std::ffi::c_void = std::ptr::null_mut();
            let size = raw_output.len() * std::mem::size_of::<f32>();
            
            extern "C" {
                fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> i32;
                fn cudaMemcpy(
                    dst: *mut std::ffi::c_void,
                    src: *const std::ffi::c_void,
                    count: usize,
                    kind: i32, // cudaMemcpyHostToDevice = 1
                ) -> i32;
                fn cudaFree(devPtr: *mut std::ffi::c_void) -> i32;
            }
            
            // Allocate GPU memory
            let err = cudaMalloc(&mut d_raw_output, size);
            assert_eq!(err, 0, "cudaMalloc failed with error: {}", err);
            
            // Copy data to GPU
            let err = cudaMemcpy(
                d_raw_output,
                raw_output.as_ptr() as *const std::ffi::c_void,
                size,
                1, // cudaMemcpyHostToDevice
            );
            assert_eq!(err, 0, "cudaMemcpy H2D failed with error: {}", err);
            
            // Create frame metadata
            let frame_meta = FrameMeta {
                source_id: 0,
                width: 640,
                height: 480,
                pixfmt: PixelFormat::RGB8,
                colorspace: ColorSpace::BT709,
                frame_idx: 0,
                pts_ns: 0,
                t_capture_ns: 0,
                stride_bytes: 640 * 4,
                crop_region: None,
            };
            
            // Process on GPU
            let result = processor.process_gpu(
                d_raw_output as *const f32,
                &frame_meta,
            );
            
            // Cleanup GPU memory
            cudaFree(d_raw_output);
            
            match result {
                Ok(detections) => {
                    println!("✅ GPU processing succeeded");
                    println!("   Detections found: {}", detections.items.len());
                    
                    for (i, det) in detections.items.iter().enumerate() {
                        println!("   Detection {}: bbox=({:.1}, {:.1}, {:.1}, {:.1}) score={:.3} class={}",
                            i,
                            det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h,
                            det.score,
                            det.class_id
                        );
                    }
                    
                    // We expect at least 1 detection from our synthetic data
                    assert!(detections.items.len() > 0, "Expected at least 1 detection");
                },
                Err(e) => {
                    panic!("GPU processing failed: {:?}", e);
                }
            }
        }
    }
    
    #[test]
    fn test_set_confidence_threshold() {
        let mut processor = GpuPostprocessor::new(
            25200, 80, 300, 0.25, 0.45
        ).expect("Failed to init GPU postprocessor");
        
        // Test threshold setter
        processor.set_confidence_threshold(0.5);
        println!("✅ Confidence threshold updated successfully");
    }
}

// Placeholder test when GPU feature is disabled
#[cfg(not(feature = "gpu"))]
#[test]
fn gpu_feature_disabled() {
    println!("⚠️  GPU feature not enabled. Run with: cargo test --features gpu");
}
