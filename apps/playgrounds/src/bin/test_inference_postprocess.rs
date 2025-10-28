use anyhow::Result;
use common_io::{
    Stage, TensorInputPacket, TensorDesc, MemRef, FrameMeta, 
    DType, MemLoc, PixelFormat, ColorSpace
};
use image::{io::Reader as ImageReader, GenericImageView};
use inference::InferenceEngine;
use postprocess::{PostStage, PostprocessConfig};
use std::path::Path;

const MODEL_PATH: &str = "crates/inference/YOLOv5.onnx";
const IMAGE_PATH: &str = "apps/playgrounds/sample_img.jpg";
const INPUT_WIDTH: u32 = 512;
const INPUT_HEIGHT: u32 = 512;

fn main() -> Result<()> {
    println!("=== Inference + Postprocess Pipeline Test ===");
    println!("Model: {}", MODEL_PATH);
    println!("Image: {}", IMAGE_PATH);
    println!();

    // Check if model exists
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("❌ Error: Model file not found at {}", MODEL_PATH);
        anyhow::bail!("Model file not found");
    }

    // Check if image exists
    if !Path::new(IMAGE_PATH).exists() {
        anyhow::bail!("Image file not found at {}", IMAGE_PATH);
    }

    // Step 1: Load and preprocess image
    println!("Step 1: Loading and preprocessing image...");
    let preprocess_start = std::time::Instant::now();
    let preprocessed = mock_preprocess_image(IMAGE_PATH)?;
    let preprocess_time = preprocess_start.elapsed();
    println!("  ✓ Image preprocessed to {}x{}", INPUT_WIDTH, INPUT_HEIGHT);
    println!("  ✓ Preprocessing time: {:.2}ms", preprocess_time.as_secs_f64() * 1000.0);
    println!();

    // Step 2: Create TensorInputPacket with GPU-allocated memory
    println!("Step 2: Creating GPU TensorInputPacket...");
    let packet_start = std::time::Instant::now();
    let tensor_packet = create_gpu_tensor_packet(preprocessed)?;
    let packet_time = packet_start.elapsed();
    println!("  ✓ TensorInputPacket created on GPU");
    println!("  ✓ Packet creation time: {:.2}ms", packet_time.as_secs_f64() * 1000.0);
    println!();

    // Step 3: Initialize Inference Engine
    println!("Step 3: Initializing Inference Engine...");
    let mut inference_engine = InferenceEngine::new(MODEL_PATH)?;
    println!("  ✓ Engine initialized");
    println!();

    // Step 4: Run Inference
    println!("Step 4: Running inference...");
    let raw_detections = inference_engine.process(tensor_packet);
    println!();

    // Step 5: Initialize Postprocess Stage  
    println!("Step 5: Initializing Postprocess Stage...");
    
    // Calculate letterbox info for postprocessing
    let img = ImageReader::open(IMAGE_PATH)?.decode()?;
    let (orig_w, orig_h) = img.dimensions();
    let scale = (INPUT_WIDTH as f32 / orig_w as f32).min(INPUT_HEIGHT as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale) as u32;
    let new_h = (orig_h as f32 * scale) as u32;
    let pad_x = (INPUT_WIDTH - new_w) / 2;
    let pad_y = (INPUT_HEIGHT - new_h) / 2;
    
    let config = PostprocessConfig {
        num_classes: 2,  // Custom model with only 2 classes
        confidence_threshold: 0.30,  // Balanced: filter low-confidence while keeping real detections
        nms_threshold: 0.45,
        max_detections: 100,
        letterbox_scale: scale,
        letterbox_pad: (pad_x as f32, pad_y as f32),
        original_size: (orig_w, orig_h),
    };
    let mut postprocess_stage = PostStage::new(config)
        .with_temporal_smoothing(4)
        .with_sort_tracking(30, 0.25, 0.3); // max_age=30, min_conf=0.25, iou_thresh=0.3
    println!("  ✓ Postprocess stage initialized (2 classes, 16128 anchors)");
    println!("  ✓ Confidence threshold: 0.35 (balanced filtering)");
    println!("  ✓ SORT tracking enabled (max_age=30, iou=0.3)");

    // Step 6: Run Postprocess
    println!("Step 6: Running postprocess (decode, NMS, tracking)...");
    let detections = postprocess_stage.process(raw_detections);
    println!();

    // Step 7: Display Results
    println!("=== Detection Results ===");
    println!("Total detections: {}", detections.items.len());
    println!();
    
    if !detections.items.is_empty() {
        println!("Top 10 detections:");
        for (i, det) in detections.items.iter().take(10).enumerate() {
            let track_info = det.track_id
                .map(|id| format!(" [Track ID: {}]", id))
                .unwrap_or_else(|| String::from(" [No track]"));
            
            println!(
                "  {}. Class {} - Score: {:.3} - BBox: ({:.1}, {:.1}, {:.1}, {:.1}){}",
                i + 1,
                det.class_id,
                det.score,
                det.bbox.x,
                det.bbox.y,
                det.bbox.w,
                det.bbox.h,
                track_info
            );
        }
    } else {
        println!("  No detections found (try lowering confidence threshold)");
    }
    println!();

    println!("=== Pipeline Test Completed Successfully! ===");
    println!();
    println!("Next steps:");
    println!("  - Detections are ready for overlay rendering");
    println!("  - Full pipeline: DeckLink → Preprocess → Inference → Postprocess → Overlay → Output");
    
    Ok(())
}

fn mock_preprocess_image(image_path: &str) -> Result<Vec<f32>> {
    let img = ImageReader::open(image_path)?.decode()?.to_rgb8();
    let (orig_w, orig_h) = img.dimensions();
    
    println!("  Original image: {}x{}", orig_w, orig_h);
    
    // Resize with letterboxing
    let scale = (INPUT_WIDTH as f32 / orig_w as f32).min(INPUT_HEIGHT as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale) as u32;
    let new_h = (orig_h as f32 * scale) as u32;
    
    let resized = image::imageops::resize(
        &img, 
        new_w, 
        new_h, 
        image::imageops::FilterType::Lanczos3
    );
    
    // Create letterboxed image (gray background)
    let mut letterboxed = image::RgbImage::from_pixel(
        INPUT_WIDTH, 
        INPUT_HEIGHT, 
        image::Rgb([114u8, 114u8, 114u8])
    );
    
    let pad_x = (INPUT_WIDTH - new_w) / 2;
    let pad_y = (INPUT_HEIGHT - new_h) / 2;
    
    // Overlay resized image
    image::imageops::overlay(&mut letterboxed, &resized, pad_x as i64, pad_y as i64);
    
    // Convert to NCHW format and normalize to [0, 1]
    let mut tensor = vec![0.0f32; (INPUT_WIDTH * INPUT_HEIGHT * 3) as usize];
    let hw = (INPUT_WIDTH * INPUT_HEIGHT) as usize;
    
    for y in 0..INPUT_HEIGHT {
        for x in 0..INPUT_WIDTH {
            let pixel = letterboxed.get_pixel(x, y);
            let idx = (y * INPUT_WIDTH + x) as usize;
            
            // NCHW: [C, H, W]
            tensor[idx] = pixel[0] as f32 / 255.0;              // R channel
            tensor[hw + idx] = pixel[1] as f32 / 255.0;         // G channel
            tensor[2 * hw + idx] = pixel[2] as f32 / 255.0;     // B channel
        }
    }
    
    println!("  Letterbox scale: {:.3}, pad: ({}, {})", scale, pad_x, pad_y);
    
    Ok(tensor)
}

fn create_gpu_tensor_packet(cpu_data: Vec<f32>) -> Result<TensorInputPacket> {
    // For this test, we'll use CPU memory but mark it appropriately
    // In real pipeline, preprocess_cuda would directly output to GPU memory
    
    // Create mock frame metadata
    let frame_meta = FrameMeta {
        source_id: 0,
        width: INPUT_WIDTH,
        height: INPUT_HEIGHT,
        pixfmt: PixelFormat::BGRA8,
        colorspace: ColorSpace::BT709,
        frame_idx: 1,
        pts_ns: 0,
        t_capture_ns: 0,
        stride_bytes: INPUT_WIDTH * 4,
            crop_region: None,
    };
    
    // Create tensor descriptor
    let tensor_desc = TensorDesc {
        n: 1,
        c: 3,
        h: INPUT_HEIGHT,
        w: INPUT_WIDTH,
        dtype: DType::Fp32,
        device: 0,
    };
    
    // Leak the data to get a stable pointer
    // In production, this would be GPU-allocated memory from preprocess_cuda
    let data_boxed = Box::new(cpu_data);
    let ptr = Box::leak(data_boxed).as_mut_ptr() as *mut u8;
    let len = (INPUT_WIDTH * INPUT_HEIGHT * 3 * std::mem::size_of::<f32>() as u32) as usize;
    
    // Create memory reference - using CPU for now
    // Real pipeline would use MemLoc::Gpu { device: 0 }
    let mem_ref = MemRef {
        ptr,
        len,
        stride: (INPUT_WIDTH * 3 * std::mem::size_of::<f32>() as u32) as usize,
        loc: MemLoc::Cpu,  // Would be Gpu in real pipeline with preprocess_cuda
    };
    
    let packet = TensorInputPacket {
        from: frame_meta,
        desc: tensor_desc,
        data: mem_ref,
    };
    
    println!("  → Tensor data: {} bytes at {:p}", len, ptr);
    println!("  → Memory location: CPU (GPU in real pipeline)");
    
    Ok(packet)
}
