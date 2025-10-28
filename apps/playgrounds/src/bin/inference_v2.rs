use anyhow::Result;
use common_io::{
    ColorSpace, DType, FrameMeta, MemLoc, MemRef, PixelFormat, Stage, TensorDesc,
    TensorInputPacket, RawDetectionsPacket,
};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use image::io::Reader as ImageReader;
use inference_v2::TrtInferenceStage;

const ENGINE_PATH: &str = "trt-shim/test_rust/assets/v7_optimized_YOLOv5.engine";
const LIB_PATH: &str = "trt-shim/build/libtrt_shim.so";
// const IMAGE_PATH: &str = "apps/playgrounds/sample_img.jpg";
// const IMAGE_PATH: &str = "/home/earth/Documents/Guptun/6/DeepGiBox/output/test/preprocess_frame_0000.png";  // Don't use this - already preprocessed!
const IMAGE_PATH: &str = "/home/earth/Documents/Guptun/6/DeepGiBox/output/test/original_frame_0000.png";  // Use original 1920×1080 frame
// YOLOv5 input dimensions (MUST match the TensorRT engine!)
// Check the actual model input size with: [TRT] ACTUAL ENGINE TENSOR DIMENSIONS
const INPUT_WIDTH: u32 = 512;   // Changed from 640 to match engine (1, 3, 512, 512)
const INPUT_HEIGHT: u32 = 512;  // Changed from 640 to match engine

fn print_engine_build_instructions() {
    eprintln!("\n╔═══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  ⚠️  TensorRT Engine File Not Found                               ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════╝\n");
    eprintln!("The TensorRT engine file is required for GPU inference.");
    eprintln!("You need to build it from an ONNX model.\n");
    eprintln!("📝 Instructions:\n");
    eprintln!("1. Export YOLOv5 to ONNX format (640x640 input):");
    eprintln!("   python export.py --weights yolov5s.pt --include onnx --imgsz 640\n");
    eprintln!("2. Build TensorRT engine:");
    eprintln!("   trtexec --onnx=yolov5s.onnx \\");
    eprintln!("           --saveEngine=TRT_SHIM/optimized_YOLOv5.engine \\");
    eprintln!("           --fp16 \\");
    eprintln!("           --workspace=4096\n");
    eprintln!("3. Or use existing ONNX at: crates/inference/YOLOv5.onnx\n");
    eprintln!("Once the engine file is ready, run this program again!\n");
}

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   Inference V2 - GPU-Only Pipeline Test                  ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("📋 Configuration:");
    println!("   Engine:  {}", ENGINE_PATH);
    println!("   Library: {}", LIB_PATH);
    println!("   Image:   {}", IMAGE_PATH);
    println!("   Size:    {}x{}\n", INPUT_WIDTH, INPUT_HEIGHT);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 1: Initialize CUDA Device
    // ═══════════════════════════════════════════════════════════════════════
    println!("🚀 Step 1: Initializing CUDA Device...");
    let cuda_device = CudaDevice::new(0)?;
    println!("   ✓ CUDA device 0 initialized");
    println!("   ✓ GPU: {}\n", cuda_device.name()?);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 2: Load and Preprocess Image on CPU
    // ═══════════════════════════════════════════════════════════════════════
    println!("🖼️  Step 2: Loading image from disk...");
    let img = ImageReader::open(IMAGE_PATH)?.decode()?.to_rgb8();
    let (orig_w, orig_h) = img.dimensions();
    println!("   ✓ Image loaded: {}x{}", orig_w, orig_h);

    println!("   Preprocessing to {}x{}...", INPUT_WIDTH, INPUT_HEIGHT);
    let preprocessed = preprocess_image_cpu(&img, INPUT_WIDTH, INPUT_HEIGHT)?;
    println!("   ✓ Preprocessed to NCHW format (FP32)");
    println!("   ✓ Total elements: {}\n", preprocessed.len());

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 3: Upload to GPU Memory
    // ═══════════════════════════════════════════════════════════════════════
    println!("📤 Step 3: Uploading data to GPU...");
    let start = std::time::Instant::now();

    let gpu_buffer: CudaSlice<f32> = cuda_device.htod_sync_copy(&preprocessed)?;
    let upload_time = start.elapsed();

    let gpu_ptr = *gpu_buffer.device_ptr() as *mut u8;
    let gpu_len = preprocessed.len() * std::mem::size_of::<f32>();

    println!("   ✓ Uploaded {} bytes to GPU", gpu_len);
    println!("   ✓ GPU pointer: {:#x}", gpu_ptr as usize);
    println!("   ✓ Upload time: {:.2}ms\n", upload_time.as_secs_f64() * 1000.0);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 4: Create TensorInputPacket
    // ═══════════════════════════════════════════════════════════════════════
    println!("📦 Step 4: Creating TensorInputPacket...");

    let frame_meta = FrameMeta {
        source_id: 1,
        width: orig_w,
        height: orig_h,
        pixfmt: PixelFormat::RGB8,
        colorspace: ColorSpace::BT709,
        frame_idx: 1,
        pts_ns: 0,
        t_capture_ns: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos() as u64,
        stride_bytes: orig_w * 3,
    };

    let tensor_desc = TensorDesc {
        n: 1,
        c: 3,
        h: INPUT_HEIGHT,
        w: INPUT_WIDTH,
        dtype: DType::Fp32,
        device: 0,
    };

    let mem_ref = MemRef {
        ptr: gpu_ptr,
        len: gpu_len,
        stride: (INPUT_WIDTH * std::mem::size_of::<f32>() as u32) as usize,
        loc: MemLoc::Gpu { device: 0 },
    };

    let tensor_packet = TensorInputPacket {
        from: frame_meta,
        desc: tensor_desc,
        data: mem_ref,
    };

    println!("   ✓ TensorInputPacket created");
    println!("   ✓ Shape: {}x{}x{}x{}", 
             tensor_packet.desc.n,
             tensor_packet.desc.c,
             tensor_packet.desc.h,
             tensor_packet.desc.w);
    println!("   ✓ DType: {:?}", tensor_packet.desc.dtype);
    println!("   ✓ MemLoc: {:?}\n", tensor_packet.data.loc);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 5: Initialize TensorRT Inference Stage
    // ═══════════════════════════════════════════════════════════════════════
    println!("🔧 Step 5: Initializing TensorRT Inference Stage...");
    let start = std::time::Instant::now();

    let mut inference_stage = match TrtInferenceStage::new(ENGINE_PATH, LIB_PATH) {
        Ok(stage) => stage,
        Err(e) => {
            eprintln!("❌ Error: {}", e);
            print_engine_build_instructions();
            return Err(anyhow::anyhow!("Failed to create inference stage"));
        }
    };
    let init_time = start.elapsed();

    println!("   ✓ TrtInferenceStage created");
    println!("   ✓ Engine loaded from: {}", ENGINE_PATH);
    println!("   ✓ Initialization time: {:.2}ms\n", init_time.as_secs_f64() * 1000.0);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 6: Run Inference (Zero-Copy GPU)
    // ═══════════════════════════════════════════════════════════════════════
    println!("⚡ Step 6: Running GPU-Only Inference...");
    println!("   Input: GPU pointer {:#x} (ZERO-COPY)", gpu_ptr as usize);

    let start = std::time::Instant::now();
    let detections: RawDetectionsPacket = inference_stage.process(tensor_packet);
    let inference_time = start.elapsed();

    println!("   ✓ Inference completed!");
    println!("   ✓ Inference time: {:.2}ms", inference_time.as_secs_f64() * 1000.0);
    println!("   ✓ Throughput: {:.1} FPS\n", 1000.0 / (inference_time.as_secs_f64() * 1000.0));

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 6.5: Analyze Raw Inference Output
    // ═══════════════════════════════════════════════════════════════════════
    println!("📊 Step 6.5: Analyzing Raw Inference Output...");
    
    let output_shape = &detections.output_shape;
    let raw_output = &detections.raw_output;
    
    println!("   Output shape: {:?}", output_shape);
    println!("   Total output elements: {}", raw_output.len());
    
    // Statistical analysis
    let non_zero = raw_output.iter().filter(|&&x| x.abs() > 0.001).count();
    let max_val = raw_output.iter().fold(0.0f32, |a, &b| a.max(b));
    let min_val = raw_output.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let sum: f32 = raw_output.iter().sum();
    let mean = sum / raw_output.len() as f32;
    
    println!("   Non-zero values: {}/{} ({:.2}%)", 
             non_zero, raw_output.len(), 
             (non_zero as f64 / raw_output.len() as f64) * 100.0);
    println!("   Value range: [{:.6}, {:.6}]", min_val, max_val);
    println!("   Mean: {:.6}", mean);
    
    // Show first 20 values
    println!("\n   First 20 raw values:");
    for (i, &val) in raw_output.iter().take(20).enumerate() {
        if i % 5 == 0 {
            print!("   ");
        }
        print!("{:>12.6}", val);
        if (i + 1) % 5 == 0 {
            println!();
        }
    }
    if raw_output.len() % 5 != 0 {
        println!();
    }
    
    // Analyze YOLO format: [N, num_detections, (cx, cy, w, h, objectness, class0, class1, ...)]
    if output_shape.len() >= 3 {
        let num_detections = output_shape[1];
        let values_per_detection = output_shape[2];
        
        println!("\n   YOLO Format Analysis:");
        println!("      Detections: {}", num_detections);
        println!("      Values per detection: {}", values_per_detection);
        
        if values_per_detection > 5 {
            let num_classes = values_per_detection - 5;
            println!("      Number of classes: {}", num_classes);
            
            // Analyze objectness distribution
            let mut objectness_values = Vec::new();
            for det_idx in 0..num_detections {
                let offset = det_idx * values_per_detection + 4; // objectness is at index 4
                if offset < raw_output.len() {
                    objectness_values.push(raw_output[offset]);
                }
            }
            
            objectness_values.sort_by(|a, b| b.partial_cmp(a).unwrap());
            
            println!("\n      Objectness Distribution:");
            println!("         Max: {:.6}", objectness_values.first().unwrap_or(&0.0));
            println!("         Top 10: {:.6}", objectness_values.get(9).unwrap_or(&0.0));
            println!("         Top 100: {:.6}", objectness_values.get(99).unwrap_or(&0.0));
            println!("         Top 1000: {:.6}", objectness_values.get(999).unwrap_or(&0.0));
            println!("         Median: {:.6}", objectness_values.get(num_detections / 2).unwrap_or(&0.0));
            
            let high_obj = objectness_values.iter().filter(|&&x| x > 0.5).count();
            let med_obj = objectness_values.iter().filter(|&&x| x > 0.25 && x <= 0.5).count();
            let low_obj = objectness_values.iter().filter(|&&x| x > 0.01 && x <= 0.25).count();
            
            println!("         >0.50: {} detections", high_obj);
            println!("         0.25-0.50: {} detections", med_obj);
            println!("         0.01-0.25: {} detections", low_obj);
            
            // Show first 3 detections
            println!("\n   First 3 raw detections:");
            for det_idx in 0..3.min(num_detections) {
                let offset = det_idx * values_per_detection;
                
                if offset + values_per_detection > raw_output.len() {
                    break;
                }
                
                let cx = raw_output[offset];
                let cy = raw_output[offset + 1];
                let w = raw_output[offset + 2];
                let h = raw_output[offset + 3];
                let objectness = raw_output[offset + 4];
                
                println!("\n      Detection #{}:", det_idx + 1);
                println!("         Box: cx={:.4}, cy={:.4}, w={:.4}, h={:.4}", cx, cy, w, h);
                println!("         Objectness: {:.6}", objectness);
                
                // Show top 3 class scores
                let class_scores = &raw_output[offset + 5..offset + values_per_detection];
                let mut indexed_scores: Vec<_> = class_scores.iter()
                    .enumerate()
                    .map(|(i, &s)| (i, s))
                    .collect();
                indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                
                println!("         Top 3 classes:");
                for (i, (class_id, score)) in indexed_scores.iter().take(3).enumerate() {
                    println!("            {}. Class {}: {:.6}", i + 1, class_id, score);
                }
                
                // Calculate final confidence (objectness * class_score)
                let max_class_score = indexed_scores[0].1;
                let final_confidence = objectness * max_class_score;
                println!("         Final confidence: {:.6} (obj * cls)", final_confidence);
            }
            
            // Show top 5 detections by objectness (RAW VALUES - no sigmoid)
            println!("\n   Top 5 detections by objectness:");
            
            let mut detection_data: Vec<_> = (0..num_detections)
                .filter_map(|det_idx| {
                    let offset = det_idx * values_per_detection;
                    if offset + values_per_detection > raw_output.len() {
                        return None;
                    }
                    
                    let cx = raw_output[offset];
                    let cy = raw_output[offset + 1];
                    let w = raw_output[offset + 2];
                    let h = raw_output[offset + 3];
                    let objectness = raw_output[offset + 4];
                    
                    let class_scores = &raw_output[offset + 5..offset + values_per_detection];
                    let (best_class, &best_score) = class_scores.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;
                    
                    let conf = objectness * best_score;
                    
                    Some((det_idx, cx, cy, w, h, objectness, best_class, best_score, conf))
                })
                .collect();
            
            // Sort by objectness
            detection_data.sort_by(|a, b| b.5.partial_cmp(&a.5).unwrap());
            
            for (i, &(idx, cx, cy, w, h, obj, cls, cls_score, conf)) in detection_data.iter().take(5).enumerate() {
                println!("      #{}: det_idx={}, obj={:.4}, cls={}, cls_score={:.4}, conf={:.4}, box=({:.1},{:.1},{:.1},{:.1})",
                         i + 1, idx, obj, cls, cls_score, conf, cx, cy, w, h);
            }
        }
    }
    
    println!("\n   ✓ Raw output analysis complete\n");

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 6.9: Save raw detections to JSON for Python visualization
    // ═══════════════════════════════════════════════════════════════════════
    println!("💾 Step 6.9: Saving raw detections to JSON...");
    
    let output_path = "output/raw_detections.json";
    std::fs::create_dir_all("output")?;
    
    // Extract all detections with their raw values
    let mut all_detections = Vec::new();
    if output_shape.len() >= 3 {
        let num_detections = output_shape[1];
        let values_per_detection = output_shape[2];
        
        for det_idx in 0..num_detections {
            let offset = det_idx * values_per_detection;
            if offset + values_per_detection > raw_output.len() {
                break;
            }
            
            let cx = raw_output[offset];
            let cy = raw_output[offset + 1];
            let w = raw_output[offset + 2];
            let h = raw_output[offset + 3];
            let obj_raw = raw_output[offset + 4];
            
            let class_scores: Vec<f32> = raw_output[offset + 5..offset + values_per_detection].to_vec();
            
            all_detections.push(serde_json::json!({
                "det_idx": det_idx,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "objectness_raw": obj_raw,
                "class_scores_raw": class_scores,
            }));
        }
    }
    
    // Calculate correct letterbox parameters based on actual image size
    let scale = (INPUT_WIDTH as f32 / orig_w as f32).min(INPUT_HEIGHT as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale) as u32;
    let new_h = (orig_h as f32 * scale) as u32;
    let pad_x = (INPUT_WIDTH - new_w) as f32 / 2.0;
    let pad_y = (INPUT_HEIGHT - new_h) as f32 / 2.0;
    
    let json_data = serde_json::json!({
        "image": {
            "path": IMAGE_PATH,
            "original_size": [orig_w, orig_h],
            "model_size": [INPUT_WIDTH, INPUT_HEIGHT],
        },
        "letterbox": {
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
        },
        "num_classes": 2,
        "detections": all_detections,
    });
    
    std::fs::write(output_path, serde_json::to_string_pretty(&json_data)?)?;
    println!("   ✓ Saved {} raw detections to: {}\n", all_detections.len(), output_path);

    // Calculate and display correct letterbox parameters
    let scale = (INPUT_WIDTH as f32 / orig_w as f32).min(INPUT_HEIGHT as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale) as u32;
    let new_h = (orig_h as f32 * scale) as u32;
    let pad_x = (INPUT_WIDTH - new_w) as f32 / 2.0;
    let pad_y = (INPUT_HEIGHT - new_h) as f32 / 2.0;
    
    println!("   📐 Letterbox parameters for {}x{} → {}x{}:", orig_w, orig_h, INPUT_WIDTH, INPUT_HEIGHT);
    println!("      Scale: {:.6}", scale);
    println!("      Resized: {}x{}", new_w, new_h);
    println!("      Padding: pad_x={:.1}, pad_y={:.1}\n", pad_x, pad_y);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 7: Initialize Postprocessing Stage
    // ═══════════════════════════════════════════════════════════════════════
    println!("🔧 Step 7: Initializing Postprocessing Stage...");
    
    let mut postprocess_stage = postprocess::from_path("")?;
    
    println!("   ✓ PostStage created with temporal smoothing");
    println!("   ✓ Confidence threshold: 0.25");
    println!("   ✓ NMS threshold: 0.45\n");

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 8: Run Postprocessing (NMS + Detection Parsing)
    // ═══════════════════════════════════════════════════════════════════════
    println!("📦 Step 8: Running Postprocessing...");
    let start = std::time::Instant::now();
    
    let detections_packet = postprocess_stage.process(detections);
    let postprocess_time = start.elapsed();
    
    println!("   ✓ Postprocessing completed!");
    println!("   ✓ Postprocessing time: {:.2}ms", postprocess_time.as_secs_f64() * 1000.0);
    println!("   ✓ Final detections: {}\n", detections_packet.items.len());

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 9: Display Detection Results
    // ═══════════════════════════════════════════════════════════════════════
    println!("🎯 Step 9: Detection Results:");
    println!("   Frame: {}x{}", detections_packet.from.width, detections_packet.from.height);
    println!("   Total detections: {}\n", detections_packet.items.len());
    
    for (i, det) in detections_packet.items.iter().enumerate().take(10) {
        println!("   Detection #{}:", i + 1);
        println!("      BBox: (x={:.1}, y={:.1}, w={:.1}, h={:.1})", 
                 det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h);
        println!("      Confidence: {:.4}", det.score);
        println!("      Class ID: {}", det.class_id);
        if let Some(track_id) = det.track_id {
            println!("      Track ID: {}", track_id);
        }
    }
    
    if detections_packet.items.len() > 10 {
        println!("\n   ... and {} more detections", detections_packet.items.len() - 10);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 10: Class Distribution Analysis
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n� Step 10: Detection Analysis:");
    
    // Count detections by class
    let mut class_counts = std::collections::HashMap::new();
    for det in &detections_packet.items {
        *class_counts.entry(det.class_id).or_insert(0) += 1;
    }
    
    println!("   Classes detected: {}", class_counts.len());
    let mut sorted_classes: Vec<_> = class_counts.into_iter().collect();
    sorted_classes.sort_by(|a, b| b.1.cmp(&a.1));
    
    for (class_id, count) in sorted_classes.iter().take(5) {
        println!("      Class {}: {} detections", class_id, count);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 10.5: Save postprocessed detections to JSON
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n💾 Step 10.5: Saving postprocessed detections to JSON...");
    
    let postprocessed_output = "output/postprocessed_detections.json";
    
    let postprocessed_json: Vec<_> = detections_packet.items.iter().map(|det| {
        serde_json::json!({
            "bbox": {
                "x": det.bbox.x,
                "y": det.bbox.y,
                "w": det.bbox.w,
                "h": det.bbox.h,
            },
            "score": det.score,
            "class_id": det.class_id,
            "track_id": det.track_id,
        })
    }).collect();
    
    let postprocessed_data = serde_json::json!({
        "image": {
            "path": IMAGE_PATH,
            "size": [orig_w, orig_h],
        },
        "detections": postprocessed_json,
    });
    
    std::fs::write(postprocessed_output, serde_json::to_string_pretty(&postprocessed_data)?)?;
    println!("   ✓ Saved {} postprocessed detections to: {}\n", detections_packet.items.len(), postprocessed_output);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 11: Performance Summary
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n⏱️  Step 11: Performance Summary");
    println!("   ┌──────────────────────────────┬───────────┐");
    println!("   │ Stage                        │ Time      │");
    println!("   ├──────────────────────────────┼───────────┤");
    println!("   │ CPU Preprocessing            │ (varies)  │");
    println!("   │ CPU → GPU Upload             │ {:>7.2}ms │", upload_time.as_secs_f64() * 1000.0);
    println!("   │ TensorRT Init (one-time)     │ {:>7.2}ms │", init_time.as_secs_f64() * 1000.0);
    println!("   │ GPU Inference (zero-copy)    │ {:>7.2}ms │", inference_time.as_secs_f64() * 1000.0);
    println!("   │ Postprocessing (NMS)         │ {:>7.2}ms │", postprocess_time.as_secs_f64() * 1000.0);
    println!("   ├──────────────────────────────┼───────────┤");
    let total_runtime = upload_time.as_secs_f64() * 1000.0 + inference_time.as_secs_f64() * 1000.0 + postprocess_time.as_secs_f64() * 1000.0;
    println!("   │ Total (per frame, sustained) │ {:>7.2}ms │", total_runtime);
    println!("   └──────────────────────────────┴───────────┘");
    println!("   Sustained Throughput: {:.1} FPS\n", 1000.0 / total_runtime);

    println!("💡 Complete Pipeline Architecture:");
    println!("   Image (CPU) → Preprocess (CPU) → Upload (PCIe)");
    println!("   → TensorInputPacket (GPU) → Inference (GPU, zero-copy)");
    println!("   → RawDetectionsPacket → Postprocessing (CPU, NMS)");
    println!("   → DetectionsPacket (ready for overlay/tracking)\n");

    println!("✨ Zero-Copy Benefits:");
    println!("   • Input data stays on GPU (no extra transfers)");
    println!("   • Direct memory access by TensorRT");
    println!("   • Optimal performance for GPU pipelines\n");

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   ✅ Inference V2 + Postprocessing Test Complete!         ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("💡 Next steps:");
    println!("   1. View raw detections: python3 visualize_raw_detections.py");
    println!("   2. View postprocessed: python3 visualize_postprocessed.py\n");

    // Keep GPU buffer alive until program exits
    std::mem::forget(gpu_buffer);

    Ok(())
}

/// Preprocess image on CPU: resize with letterboxing and convert to NCHW format
fn preprocess_image_cpu(img: &image::RgbImage, target_w: u32, target_h: u32) -> Result<Vec<f32>> {
    let (orig_w, orig_h) = img.dimensions();

    // Calculate letterbox scale (maintain aspect ratio)
    let scale = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale) as u32;
    let new_h = (orig_h as f32 * scale) as u32;

    // Resize image
    let resized = image::imageops::resize(
        img,
        new_w,
        new_h,
        image::imageops::FilterType::Lanczos3,
    );

    // Create letterboxed image with gray padding
    let mut letterboxed = image::RgbImage::from_pixel(
        target_w,
        target_h,
        image::Rgb([114u8, 114u8, 114u8]),
    );

    let pad_x = (target_w - new_w) / 2;
    let pad_y = (target_h - new_h) / 2;

    // Overlay resized image
    image::imageops::overlay(&mut letterboxed, &resized, pad_x as i64, pad_y as i64);

    // Convert to NCHW format and normalize to [0, 1]
    let mut tensor = vec![0.0f32; (target_w * target_h * 3) as usize];
    let hw = (target_w * target_h) as usize;

    for y in 0..target_h {
        for x in 0..target_w {
            let pixel = letterboxed.get_pixel(x, y);
            let idx = (y * target_w + x) as usize;

            // NCHW format: [Channel, Height, Width]
            tensor[idx] = pixel[0] as f32 / 255.0; // R channel
            tensor[hw + idx] = pixel[1] as f32 / 255.0; // G channel
            tensor[2 * hw + idx] = pixel[2] as f32 / 255.0; // B channel
        }
    }

    Ok(tensor)
}
