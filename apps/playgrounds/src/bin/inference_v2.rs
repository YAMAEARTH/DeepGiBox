use anyhow::Result;
use common_io::{
    ColorSpace, DType, FrameMeta, MemLoc, MemRef, PixelFormat, Stage, TensorDesc,
    TensorInputPacket, RawDetectionsPacket,
};
use cudarc::driver::{CudaDevice, CudaSlice};
use image::io::Reader as ImageReader;
use inference_v2::TrtInferenceStage;
use std::sync::Arc;

const ENGINE_PATH: &str = "TRT_SHIM/optimized_YOLOv5.engine";
const LIB_PATH: &str = "TRT_SHIM/libtrt_shim.so";
const IMAGE_PATH: &str = "apps/playgrounds/sample_img.jpg";
const INPUT_WIDTH: u32 = 640;
const INPUT_HEIGHT: u32 = 640;

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
    
    let mut inference_stage = TrtInferenceStage::new(ENGINE_PATH, LIB_PATH)?;
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
    // STEP 7: Analyze Results
    // ═══════════════════════════════════════════════════════════════════════
    println!("📊 Step 7: Analyzing Detection Results...");
    
    let output_shape = &detections.output_shape;
    let raw_output = &detections.raw_output;
    
    println!("   Output shape: {:?}", output_shape);
    println!("   Total output elements: {}", raw_output.len());
    
    let non_zero = raw_output.iter().filter(|&&x| x.abs() > 0.001).count();
    let max_val = raw_output.iter().fold(0.0f32, |a, &b| a.max(b));
    let min_val = raw_output.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    
    println!("   Non-zero values: {}/{} ({:.2}%)", 
             non_zero, raw_output.len(), 
             (non_zero as f64 / raw_output.len() as f64) * 100.0);
    println!("   Value range: [{:.4}, {:.4}]", min_val, max_val);
    
    // Parse YOLOv5 detections (25200 detections × 85 values)
    if output_shape.len() >= 3 && output_shape[1] == 25200 && output_shape[2] == 85 {
        println!("\n   🔍 Parsing YOLOv5 Detections (threshold: 0.5)...");
        
        let mut detection_count = 0;
        let confidence_threshold = 0.5;
        
        for det_idx in 0..25200 {
            let offset = det_idx * 85;
            
            if offset + 84 >= raw_output.len() {
                break;
            }
            
            let x = raw_output[offset];
            let y = raw_output[offset + 1];
            let w = raw_output[offset + 2];
            let h = raw_output[offset + 3];
            let objectness = raw_output[offset + 4];
            
            // Get class scores
            let class_scores = &raw_output[offset + 5..offset + 85];
            let (class_id, &class_score) = class_scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            
            let confidence = objectness * class_score;
            
            if confidence > confidence_threshold {
                detection_count += 1;
                
                if detection_count <= 5 {
                    println!("   Detection #{}: ", detection_count);
                    println!("      bbox: ({:.1}, {:.1}, {:.1}, {:.1})", x, y, w, h);
                    println!("      confidence: {:.4}", confidence);
                    println!("      class: {} (score: {:.4})", class_id, class_score);
                }
            }
        }
        
        if detection_count > 5 {
            println!("   ... and {} more detections", detection_count - 5);
        }
        
        println!("\n   ✅ Total detections found: {}", detection_count);
    } else {
        println!("   ⚠️  Unexpected output shape, skipping detection parsing");
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // STEP 8: Performance Summary
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n⏱️  Step 8: Performance Summary");
    println!("   ┌──────────────────────────────┬───────────┐");
    println!("   │ Stage                        │ Time      │");
    println!("   ├──────────────────────────────┼───────────┤");
    println!("   │ CPU Preprocessing            │ (varies)  │");
    println!("   │ CPU → GPU Upload             │ {:>7.2}ms │", upload_time.as_secs_f64() * 1000.0);
    println!("   │ TensorRT Init (one-time)     │ {:>7.2}ms │", init_time.as_secs_f64() * 1000.0);
    println!("   │ GPU Inference (zero-copy)    │ {:>7.2}ms │", inference_time.as_secs_f64() * 1000.0);
    println!("   ├──────────────────────────────┼───────────┤");
    let total_runtime = upload_time.as_secs_f64() * 1000.0 + inference_time.as_secs_f64() * 1000.0;
    println!("   │ Total (per frame, sustained) │ {:>7.2}ms │", total_runtime);
    println!("   └──────────────────────────────┴───────────┘");
    println!("   Sustained Throughput: {:.1} FPS\n", 1000.0 / total_runtime);

    println!("💡 Pipeline Architecture:");
    println!("   Image (CPU) → Preprocess (CPU) → Upload (PCIe)");
    println!("   → TensorInputPacket (GPU) → Inference (GPU, zero-copy)");
    println!("   → RawDetectionsPacket (ready for postprocessing)\n");

    println!("✨ Zero-Copy Benefits:");
    println!("   • Input data stays on GPU (no extra transfers)");
    println!("   • Direct memory access by TensorRT");
    println!("   • Optimal performance for GPU pipelines\n");

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   ✅ Inference V2 Test Completed Successfully!            ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

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
