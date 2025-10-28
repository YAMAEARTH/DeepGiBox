//! Complete End-to-End Pipeline Test with Visual Preview
//! 
//! Pipeline flow:
//! DeckLink Capture → Preprocessing → Inference → Postprocessing → Overlay Planning → Overlay Rendering → Save Images
//!
//! This test demonstrates the complete real-time detection pipeline and saves
//! the output frames with overlaid bounding boxes for visual inspection.

use anyhow::{Result, anyhow};
use common_io::{Stage, MemLoc};
use decklink_input::capture::CaptureSession;
use preprocess_cuda::Preprocessor;
use inference_v2::TrtInferenceStage;
use postprocess::{PostStage, YoloPostConfig};
use overlay_plan::PlanStage;
use overlay_render::RenderStage;
use std::time::Instant;

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     COMPLETE END-TO-END PIPELINE WITH PREVIEW           ║");
    println!("║  Capture → Preprocess → Inference → Postprocess → Overlay║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Create output directory
    std::fs::create_dir_all("output")?;
    println!("📁 Created output directory");
    println!();

    // 0. List available DeckLink devices
    println!("📹 Available DeckLink Devices:");
    let devices = decklink_input::devicelist();
    println!("  Available DeckLink devices: {}", devices.len());
    for (i, name) in devices.iter().enumerate() {
        println!("    [{}] {}", i, name);
    }
    println!();

    // 1. Initialize DeckLink capture
    println!("📹 Step 1: Initialize DeckLink Capture");
    let mut capture = CaptureSession::open(0)?;
    println!("  ✓ Opened DeckLink device 0");
    println!();

    // 2. Initialize Preprocessor
    println!("⚙️  Step 2: Initialize Preprocessor");
    let mut preprocessor = Preprocessor::new(
        (512, 512),      // Output size
        true,            // FP16
        0,               // GPU device
    )?;
    println!("  ✓ Preprocessor ready (512x512, FP16, GPU 0)");
    println!();

    // 3. Initialize TensorRT Inference
    println!("🧠 Step 3: Initialize TensorRT Inference");
    let engine_path = "configs/model/v7_optimized_YOLOv5.engine";
    let lib_path = "trt-shim/build/libtrt_shim.so";
    
    if !std::path::Path::new(engine_path).exists() {
        return Err(anyhow!("TensorRT engine not found: {}", engine_path));
    }
    if !std::path::Path::new(lib_path).exists() {
        return Err(anyhow!("TRT shim library not found: {}", lib_path));
    }
    
    let mut inference_stage = TrtInferenceStage::new(engine_path, lib_path)?;
    println!("  ✓ TensorRT engine loaded");
    println!("  ✓ Output size: {} values", inference_stage.output_size());
    println!();

    // 4. Initialize Postprocessing
    println!("🎯 Step 4: Initialize Postprocessing");
    let yolo_cfg = YoloPostConfig {
        num_classes: 80,
        confidence_threshold: 0.25,
        nms_threshold: 0.45,
        max_detections: 100,
        letterbox_scale: 1.0,
        letterbox_pad: (0.0, 0.0),
        original_size: (1920, 1080),
    };
    
    let mut post_stage = PostStage::new(yolo_cfg)
        .with_temporal_smoothing(4)
        .with_sort_tracking(30, 0.3, 0.3);
    println!("  ✓ Postprocessing ready");
    println!("  ✓ Temporal smoothing: enabled (window=4)");
    println!("  ✓ SORT tracking: enabled");
    println!();

    // 5. Initialize Overlay stages
    println!("🎨 Step 5: Initialize Overlay Stages");
    let mut plan_stage = PlanStage {};
    let mut render_stage = RenderStage {};
    println!("  ✓ Overlay planning ready");
    println!("  ✓ Overlay rendering ready");
    println!();

    // 6. Process frames
    println!("🎬 Step 6: Processing Frames...");
    println!("════════════════════════════════════════════════════════════");
    println!();

    let frames_to_process = 5;
    let mut frame_count = 0;
    let mut total_latency_ms = 0.0;

    while frame_count < frames_to_process {
        let pipeline_start = Instant::now();
        
        // Capture frame
        let raw_frame = match capture.get_frame()? {
            Some(frame) => frame,
            None => {
                println!("⚠️  No frame received, continuing...");
                std::thread::sleep(std::time::Duration::from_millis(16));
                continue;
            }
        };

        println!("════════════════════════════════════════════════════════════");
        println!("🎬 Processing Frame #{}", frame_count);
        println!("════════════════════════════════════════════════════════════");
        println!("  Source: {}x{} {:?}", 
            raw_frame.meta.width, 
            raw_frame.meta.height,
            raw_frame.meta.pixfmt
        );

        // Preprocess
        let preprocess_start = Instant::now();
        let tensor_packet = preprocessor.process(raw_frame.clone());
        let preprocess_time = preprocess_start.elapsed();
        println!("  ✓ Preprocess: {:.2}ms", preprocess_time.as_secs_f64() * 1000.0);

        // Inference
        let inference_start = Instant::now();
        let raw_detections = inference_stage.process(tensor_packet);
        let inference_time = inference_start.elapsed();
        println!("  ✓ Inference: {:.2}ms", inference_time.as_secs_f64() * 1000.0);

        // Postprocess
        let postprocess_start = Instant::now();
        let detections = post_stage.process(raw_detections);
        let postprocess_time = postprocess_start.elapsed();
        println!("  ✓ Postprocess: {:.2}ms", postprocess_time.as_secs_f64() * 1000.0);
        println!("  ✓ Detections: {}", detections.items.len());

        // Overlay Planning
        let plan_start = Instant::now();
        let overlay_plan = plan_stage.process(detections);
        let plan_time = plan_start.elapsed();
        println!("  ✓ Overlay Plan: {:.2}ms ({} operations)", 
            plan_time.as_secs_f64() * 1000.0, 
            overlay_plan.ops.len()
        );

        // Overlay Rendering
        let render_start = Instant::now();
        let overlay_frame = render_stage.process(overlay_plan);
        let render_time = render_start.elapsed();
        println!("  ✓ Overlay Render: {:.2}ms", render_time.as_secs_f64() * 1000.0);

        // Save frame with overlay visualization
        let save_start = Instant::now();
        save_frame_with_overlay(&raw_frame, &overlay_plan.from, frame_count)?;
        let save_time = save_start.elapsed();
        println!("  ✓ Saved output: {:.2}ms", save_time.as_secs_f64() * 1000.0);

        let pipeline_time = pipeline_start.elapsed();
        let pipeline_ms = pipeline_time.as_secs_f64() * 1000.0;
        total_latency_ms += pipeline_ms;
        
        println!();
        println!("  ⏱️  E2E Pipeline: {:.2}ms ({:.1} FPS)", 
            pipeline_ms, 
            1000.0 / pipeline_ms
        );
        println!();

        frame_count += 1;
    }

    // Summary
    println!("════════════════════════════════════════════════════════════");
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║            PIPELINE TEST SUMMARY                         ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Frames Processed:  {}                                    ║", frames_to_process);
    println!("║ Average Latency:   {:.2} ms                           ║", total_latency_ms / frames_to_process as f64);
    println!("║ Average FPS:       {:.1}                              ║", 1000.0 * frames_to_process as f64 / total_latency_ms);
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Output Files:                                            ║");
    println!("║   → output/frame_*.png (with detections)                 ║");
    println!("║   → output/detections_*.txt (detection details)          ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("✅ Full pipeline test completed successfully!");
    println!("📁 Check the 'output' directory for saved frames");
    println!();
    println!("💡 You can view the images with:");
    println!("   eog output/frame_*.png");
    println!("   or");
    println!("   feh output/frame_*.png");

    Ok(())
}

/// Save frame with detection visualization
fn save_frame_with_overlay(
    raw_frame: &common_io::RawFramePacket,
    detections: &common_io::DetectionsPacket,
    frame_idx: usize,
) -> Result<()> {
    use std::io::Write;

    // Convert YUV to RGB image
    let img = convert_yuv_to_rgb(raw_frame)?;
    
    // Draw detections on the image
    let img_with_boxes = draw_detections_on_image(img, detections);
    
    // Save image
    let image_path = format!("output/frame_{:04}.png", frame_idx);
    img_with_boxes.save(&image_path)?;
    println!("      💾 Saved: {}", image_path);

    // Save detection details as text
    let txt_path = format!("output/detections_{:04}.txt", frame_idx);
    let mut file = std::fs::File::create(&txt_path)?;
    
    writeln!(file, "Frame #{}", frame_idx)?;
    writeln!(file, "Timestamp: {}", raw_frame.meta.t_capture_ns)?;
    writeln!(file, "Size: {}x{}", raw_frame.meta.width, raw_frame.meta.height)?;
    writeln!(file, "Detections: {}", detections.items.len())?;
    writeln!(file, "=========================================")?;
    writeln!(file)?;
    
    for (i, det) in detections.items.iter().enumerate() {
        let class_name = get_coco_class_name(det.class_id);
        
        writeln!(file, "Detection #{}", i + 1)?;
        writeln!(file, "  Class: {} (ID: {})", class_name, det.class_id)?;
        writeln!(file, "  Score: {:.3}", det.score)?;
        writeln!(file, "  BBox: x={:.1}, y={:.1}, w={:.1}, h={:.1}", 
            det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h)?;
        if let Some(track_id) = det.track_id {
            writeln!(file, "  Track ID: {}", track_id)?;
        }
        writeln!(file)?;
    }

    Ok(())
}

/// Convert YUV422 frame to RGB image
fn convert_yuv_to_rgb(frame: &common_io::RawFramePacket) -> Result<image::RgbImage> {
    use image::RgbImage;
    
    let width = frame.meta.width as u32;
    let height = frame.meta.height as u32;
    let mut img = RgbImage::new(width, height);

    // Check memory location
    if frame.data.loc != MemLoc::Cpu {
        return Err("GPU frames not supported for image conversion yet".into());
    }
    
    // Get pixel data
    let data = unsafe {
        std::slice::from_raw_parts(frame.data.ptr, frame.data.len)
    };
    
    let stride = frame.data.stride;
    
    // Convert YUV422 (YUYV format) to RGB
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = (y as usize * stride) + (x as usize * 2);
            
            if pixel_idx + 1 < data.len() {
                // Get Y value for this pixel
                let y_val = data[pixel_idx] as f32;
                
                // Get U and V values (shared between pairs of pixels)
                let u_idx = (y as usize * stride) + ((x as usize / 2) * 4) + 1;
                let v_idx = (y as usize * stride) + ((x as usize / 2) * 4) + 3;
                
                let u_val = if u_idx < data.len() { data[u_idx] as f32 } else { 128.0 };
                let v_val = if v_idx < data.len() { data[v_idx] as f32 } else { 128.0 };
                
                // YUV to RGB conversion
                let c = y_val - 16.0;
                let d = u_val - 128.0;
                let e = v_val - 128.0;
                
                let r = (1.164 * c + 1.596 * e).clamp(0.0, 255.0) as u8;
                let g = (1.164 * c - 0.392 * d - 0.813 * e).clamp(0.0, 255.0) as u8;
                let b = (1.164 * c + 2.017 * d).clamp(0.0, 255.0) as u8;
                
                img.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }
    }

    Ok(img)
}

/// Draw detection bounding boxes on image
fn draw_detections_on_image(
    mut img: image::RgbImage,
    detections: &common_io::DetectionsPacket,
) -> image::RgbImage {
    use imageproc::drawing::{draw_hollow_rect_mut, draw_filled_rect_mut, draw_text_mut};
    use imageproc::rect::Rect;
    use rusttype::{Font, Scale};

    // Try to load a system font
    let font_data = include_bytes!("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
    let font = Font::try_from_bytes(font_data as &[u8]).unwrap();

    let scale = Scale::uniform(20.0);
    
    // Define colors
    let colors = [
        image::Rgb([0u8, 255u8, 0u8]),    // Green
        image::Rgb([255u8, 0u8, 0u8]),    // Red
        image::Rgb([0u8, 0u8, 255u8]),    // Blue
        image::Rgb([255u8, 255u8, 0u8]),  // Yellow
        image::Rgb([255u8, 0u8, 255u8]),  // Magenta
        image::Rgb([0u8, 255u8, 255u8]),  // Cyan
    ];

    for det in &detections.items {
        // Choose color based on class or track ID
        let color_idx = if let Some(track_id) = det.track_id {
            (track_id as usize) % colors.len()
        } else {
            (det.class_id as usize) % colors.len()
        };
        let color = colors[color_idx];
        
        // Draw bounding box (with thickness)
        let x = det.bbox.x as i32;
        let y = det.bbox.y as i32;
        let w = det.bbox.w as u32;
        let h = det.bbox.h as u32;
        
        let rect = Rect::at(x, y).of_size(w, h);
        
        // Draw multiple rectangles for thickness
        for offset in 0..2 {
            let thick_rect = Rect::at(x - offset, y - offset)
                .of_size(w + (offset * 2) as u32, h + (offset * 2) as u32);
            draw_hollow_rect_mut(&mut img, thick_rect, color);
        }

        // Prepare label
        let class_name = get_coco_class_name(det.class_id);
        let label = if let Some(track_id) = det.track_id {
            format!("{}#{} {:.2}", class_name, track_id, det.score)
        } else {
            format!("{} {:.2}", class_name, det.score)
        };

        // Draw label background
        let text_x = x;
        let text_y = (y - 25).max(0);
        
        // Estimate text size (rough approximation)
        let text_width = (label.len() as i32 * 11).min((w - 4) as i32);
        let text_height = 22;
        
        let bg_rect = Rect::at(text_x, text_y)
            .of_size(text_width as u32, text_height as u32);
        draw_filled_rect_mut(&mut img, bg_rect, image::Rgb([0u8, 0u8, 0u8]));
        
        // Draw text
        draw_text_mut(&mut img, image::Rgb([255u8, 255u8, 255u8]), 
                     text_x + 2, text_y + 2, scale, &font, &label);
    }

    img
}

/// Get COCO class name from ID
fn get_coco_class_name(class_id: i32) -> &'static str {
    let coco_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ];
    
    if class_id >= 0 && (class_id as usize) < coco_classes.len() {
        coco_classes[class_id as usize]
    } else {
        "unknown"
    }
}
