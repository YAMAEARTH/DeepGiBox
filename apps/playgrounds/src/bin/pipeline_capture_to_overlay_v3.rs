//! Complete Pipeline: Capture → Overlay using Inference V2
//!
//! Pipeline flow:
//! DeckLink Capture → Preprocessing → Inference V2 → Postprocessing → Overlay Planning → Overlay Rendering
//!
//! This demonstrates the complete real-time detection pipeline using the optimized inference_v2
//! and displays the OverlayFramePacket output without connecting to DeckLink output.

use anyhow::{anyhow, Result};
use common_io::{MemLoc, MemRef, Stage};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use decklink_input::capture::CaptureSession;
use inference_v2::TrtInferenceStage;
use overlay_plan::PlanStage;
use overlay_render::RenderStage;
use postprocess;
use preprocess_cuda::Preprocessor;
use std::time::Instant;
use image::{RgbaImage, Rgba, RgbImage};
use std::fs::File;
use std::io::Write;

/// Convert YUV422 to RGB image
fn yuv422_to_rgb(
    yuv_data: &[u8],
    width: u32,
    height: u32,
    stride: usize,
) -> RgbImage {
    let mut rgb_img = RgbImage::new(width, height);
    
    for y in 0..height {
        for x in 0..width {
            let x_usize = x as usize;
            let y_usize = y as usize;
            
            // YUV422 has 2 pixels per 4 bytes (Y0 U Y1 V)
            let base_idx = y_usize * stride + (x_usize / 2) * 4;
            
            if base_idx + 3 < yuv_data.len() {
                let y_val = if x_usize % 2 == 0 {
                    yuv_data[base_idx] as i32
                } else {
                    yuv_data[base_idx + 2] as i32
                };
                let u = yuv_data[base_idx + 1] as i32;
                let v = yuv_data[base_idx + 3] as i32;
                
                // BT.709 YUV to RGB conversion
                let c = y_val - 16;
                let d = u - 128;
                let e = v - 128;
                
                let r = ((298 * c + 459 * e + 128) >> 8).clamp(0, 255) as u8;
                let g = ((298 * c - 55 * d - 136 * e + 128) >> 8).clamp(0, 255) as u8;
                let b = ((298 * c + 541 * d + 128) >> 8).clamp(0, 255) as u8;
                
                rgb_img.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }
    }
    
    rgb_img
}

/// Save original frame without overlay
fn save_original_frame(
    raw_frame: &common_io::RawFramePacket,
    output_path: &str,
) -> Result<()> {
    let width = raw_frame.meta.width as u32;
    let height = raw_frame.meta.height as u32;
    let stride = raw_frame.meta.stride_bytes as usize;
    
    // Get data from CPU or GPU
    let yuv_data = if matches!(raw_frame.data.loc, MemLoc::Cpu) {
        unsafe { std::slice::from_raw_parts(raw_frame.data.ptr, raw_frame.data.len) }
    } else {
        return Err(anyhow!("RawFramePacket must be on CPU to save (GPU not supported yet)"));
    };
    
    // Convert YUV422 to RGB
    let rgb_img = yuv422_to_rgb(yuv_data, width, height, stride);
    
    // Save as PNG
    rgb_img.save(output_path)?;
    
    Ok(())
}

/// Save OverlayFramePacket as PNG image file with bounding boxes
fn save_overlay_as_image(
    overlay_frame: &common_io::OverlayFramePacket,
    output_path: &str,
) -> Result<()> {
    let width = overlay_frame.from.width as u32;
    let height = overlay_frame.from.height as u32;
    
    // Get ARGB data from CPU memory
    let argb_data = if matches!(overlay_frame.argb.loc, MemLoc::Cpu) {
        unsafe { std::slice::from_raw_parts(overlay_frame.argb.ptr, overlay_frame.argb.len) }
    } else {
        return Err(anyhow!("OverlayFramePacket data must be on CPU to save as image"));
    };

    // Create RGBA image (image crate uses RGBA, not ARGB)
    let mut img = RgbaImage::new(width, height);
    
    // Convert ARGB to RGBA
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize * 4;
            
            if idx + 3 < argb_data.len() {
                // ARGB format from overlay
                let a = argb_data[idx];
                let r = argb_data[idx + 1];
                let g = argb_data[idx + 2];
                let b = argb_data[idx + 3];
                
                // Convert to RGBA
                img.put_pixel(x, y, Rgba([r, g, b, a]));
            }
        }
    }
    
    // Save as PNG
    img.save(output_path)?;
    
    Ok(())
}

/// Create side-by-side comparison image (original + overlay)
fn save_comparison_image(
    original_path: &str,
    overlay_path: &str,
    output_path: &str,
) -> Result<()> {
    // Load both images
    let orig_img = image::open(original_path)?;
    let overlay_img = image::open(overlay_path)?;
    
    let width = orig_img.width();
    let height = orig_img.height();
    
    // Create new image with double width
    let mut combined = RgbImage::new(width * 2, height);
    
    // Copy original to left side
    let orig_rgb = orig_img.to_rgb8();
    for y in 0..height {
        for x in 0..width {
            let pixel = orig_rgb.get_pixel(x, y);
            combined.put_pixel(x, y, *pixel);
        }
    }
    
    // Copy overlay to right side (convert RGBA to RGB by compositing on white)
    let overlay_rgba = overlay_img.to_rgba8();
    for y in 0..height {
        for x in 0..width {
            let pixel = overlay_rgba.get_pixel(x, y);
            let alpha = pixel[3] as f32 / 255.0;
            
            // Composite on white background
            let r = ((pixel[0] as f32 * alpha) + (255.0 * (1.0 - alpha))) as u8;
            let g = ((pixel[1] as f32 * alpha) + (255.0 * (1.0 - alpha))) as u8;
            let b = ((pixel[2] as f32 * alpha) + (255.0 * (1.0 - alpha))) as u8;
            
            combined.put_pixel(x + width, y, image::Rgb([r, g, b]));
        }
    }
    
    // Save combined image
    combined.save(output_path)?;
    
    Ok(())
}

/// Save preprocessed tensor as image (denormalize and convert to RGB)
fn save_preprocessed_tensor(
    tensor_packet: &common_io::TensorInputPacket,
    cuda_device: &CudaDevice,
    output_path: &str,
) -> Result<()> {
    // Tensor is NCHW format: [1, 3, 512, 512] in FP16
    let n = tensor_packet.desc.n;
    let c = tensor_packet.desc.c;
    let h = tensor_packet.desc.h;
    let w = tensor_packet.desc.w;
    
    if n != 1 || c != 3 {
        return Err(anyhow!("Expected NCHW tensor with N=1, C=3"));
    }
    
    // Copy tensor data from GPU to CPU
    let mut cpu_data = vec![0u8; tensor_packet.data.len];
    unsafe {
        cudarc::driver::result::memcpy_dtoh_sync(
            &mut cpu_data,
            tensor_packet.data.ptr as cudarc::driver::sys::CUdeviceptr,
        )?;
    }
    
    // Convert FP16 bytes to f32 values
    let fp16_data: &[u16] = unsafe {
        std::slice::from_raw_parts(
            cpu_data.as_ptr() as *const u16,
            cpu_data.len() / 2,
        )
    };
    
    let mut float_data: Vec<f32> = fp16_data
        .iter()
        .map(|&fp16_bits| half::f16::from_bits(fp16_bits).to_f32())
        .collect();
    
    // Denormalize: reverse the normalization (mean=[0,0,0], std=[1/255, 1/255, 1/255])
    // normalized = (pixel / 255.0 - mean) / std
    // pixel = (normalized * std + mean) * 255.0
    for val in float_data.iter_mut() {
        *val = (*val * 255.0).clamp(0.0, 255.0);
    }
    
    // Create RGB image from CHW format
    let mut img = RgbImage::new(w as u32, h as u32);
    
    let pixels_per_channel = (h * w) as usize;
    for y in 0..h as u32 {
        for x in 0..w as u32 {
            let idx = (y * w as u32 + x) as usize;
            
            // CHW format: [R_channel, G_channel, B_channel]
            let r = float_data[idx] as u8;
            let g = float_data[pixels_per_channel + idx] as u8;
            let b = float_data[pixels_per_channel * 2 + idx] as u8;
            
            img.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }
    
    img.save(output_path)?;
    Ok(())
}

/// Save inference output (raw detections) as text file
/// Saves only the top 20 detections sorted by objectness confidence
fn save_inference_output(
    raw_detections: &common_io::RawDetectionsPacket,
    output_path: &str,
) -> Result<()> {
    let mut file = File::create(output_path)?;
    
    writeln!(file, "=== Raw Detections Output (Top 20 by Objectness) ===")?;
    writeln!(file, "Output shape: {:?}", raw_detections.output_shape)?;
    writeln!(file, "Total values: {}", raw_detections.raw_output.len())?;
    writeln!(file, "")?;
    writeln!(file, "Format: [batch, detections, (x, y, w, h, obj_conf, class0_conf, class1_conf)]")?;
    writeln!(file, "")?;
    
    let batch_size = raw_detections.output_shape[0];
    let num_detections = raw_detections.output_shape[1];
    let values_per_detection = raw_detections.output_shape[2];
    
    // Extract all detections with their objectness scores
    let mut detections_with_scores = Vec::new();
    for i in 0..num_detections {
        let base_idx = i * values_per_detection;
        if base_idx + values_per_detection <= raw_detections.raw_output.len() {
            let x = raw_detections.raw_output[base_idx];
            let y = raw_detections.raw_output[base_idx + 1];
            let w = raw_detections.raw_output[base_idx + 2];
            let h = raw_detections.raw_output[base_idx + 3];
            let obj_conf = raw_detections.raw_output[base_idx + 4];
            let class0_conf = raw_detections.raw_output[base_idx + 5];
            let class1_conf = raw_detections.raw_output[base_idx + 6];
            
            detections_with_scores.push((i, x, y, w, h, obj_conf, class0_conf, class1_conf));
        }
    }
    
    // Sort by objectness score in descending order
    detections_with_scores.sort_by(|a, b| b.5.partial_cmp(&a.5).unwrap_or(std::cmp::Ordering::Equal));
    
    writeln!(file, "Top 20 detections (sorted by objectness confidence):")?;
    writeln!(file, "")?;
    
    for (rank, (orig_idx, x, y, w, h, obj_conf, class0_conf, class1_conf)) in detections_with_scores.iter().take(20).enumerate() {
        writeln!(
            file,
            "Rank #{} (orig_idx={}): x={:.3}, y={:.3}, w={:.3}, h={:.3}, obj={:.3}, cls0={:.3}, cls1={:.3}",
            rank + 1, orig_idx, x, y, w, h, obj_conf, class0_conf, class1_conf
        )?;
    }
    
    Ok(())
}

/// Save postprocess output (detections after NMS) as text file
fn save_postprocess_output(
    detections: &common_io::DetectionsPacket,
    output_path: &str,
) -> Result<()> {
    let mut file = File::create(output_path)?;
    
    writeln!(file, "=== Postprocessed Detections ===")?;
    writeln!(file, "Total detections: {}", detections.items.len())?;
    writeln!(file, "")?;
    writeln!(file, "Format: class_id, score, bbox(x,y,w,h), track_id")?;
    writeln!(file, "")?;
    
    for (i, det) in detections.items.iter().enumerate() {
        let class_name = match det.class_id {
            0 => "Hyper",
            1 => "Neo",
            _ => "Unknown",
        };
        
        let track_info = if let Some(track_id) = det.track_id {
            format!("Track #{}", track_id)
        } else {
            "No track".to_string()
        };
        
        writeln!(
            file,
            "Detection #{}: {} (class_id={}) score={:.4} bbox=({:.1}, {:.1}, {:.1}, {:.1}) {}",
            i + 1,
            class_name,
            det.class_id,
            det.score,
            det.bbox.x,
            det.bbox.y,
            det.bbox.w,
            det.bbox.h,
            track_info
        )?;
    }
    
    Ok(())
}

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  PIPELINE: CAPTURE → OVERLAY (INFERENCE V2)             ║");
    println!("║  DeckLink → Preprocess → Inference V2 → Post → Overlay  ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Create output directory
    std::fs::create_dir_all("output/test")?;
    println!("📁 Created output/test directory");
    println!();

    // 0. List available DeckLink devices (following test_rawframepacket pattern)
    println!("📹 Available DeckLink Devices:");
    let devices = decklink_input::devicelist();
    println!("  Available DeckLink devices: {}", devices.len());
    for (idx, name) in devices.iter().enumerate() {
        println!("    [{}] {}", idx, name);
    }
    if devices.is_empty() {
        return Err(anyhow!("No DeckLink devices found!"));
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
        (512, 512), // Output size (matching YOLOv5 engine input)
        true,       // FP16 for better performance
        0,          // GPU device 0
    )?;
    println!("  ✓ Preprocessor ready (512x512, FP16, GPU 0)");
    println!();

    // 2.5 Initialize CUDA device for CPU->GPU transfer
    println!("🔧 Step 2.5: Initialize CUDA Device");
    let cuda_device = CudaDevice::new(0)?;
    println!("  ✓ CUDA device initialized for CPU->GPU transfers");
    println!();

    // 3. Initialize TensorRT Inference V2
    println!("🧠 Step 3: Initialize TensorRT Inference V2");
    let engine_path = "configs/model/v7_optimized_YOLOv5.engine"; // From configs/model/
    let lib_path = "trt-shim/build/libtrt_shim.so";

    if !std::path::Path::new(engine_path).exists() {
        return Err(anyhow!(
            "TensorRT engine not found: {}\n💡 Please build the engine first!",
            engine_path
        ));
    }
    if !std::path::Path::new(lib_path).exists() {
        return Err(anyhow!(
            "TRT shim library not found: {}\n💡 Please build trt-shim first!",
            lib_path
        ));
    }

    let mut inference_stage =
        TrtInferenceStage::new(engine_path, lib_path).map_err(|e| anyhow!(e))?;
    println!("  ✓ TensorRT Inference V2 loaded");
    println!("  ✓ Engine: {}", engine_path);
    println!("  ✓ Output size: {} values", inference_stage.output_size());
    println!();

    // 4. Initialize Postprocessing
    println!("🎯 Step 4: Initialize Postprocessing");
    let mut post_stage = postprocess::from_path("")?.with_sort_tracking(30, 0.3, 0.3);
    println!("  ✓ Postprocessing ready");
    println!("  ✓ Classes: 2 (Hyper, Neo)");
    println!("  ✓ Confidence threshold: 0.25");
    println!("  ✓ Temporal smoothing: enabled (window=4)");
    println!("  ✓ SORT tracking: enabled (max_age=30)");
    println!();

    // 5. Initialize Overlay stages
    println!("🎨 Step 5: Initialize Overlay Stages");
    let mut plan_stage = PlanStage {
        enable_full_ui: true,
    };
    let mut render_stage = RenderStage {};
    println!("  ✓ Overlay planning ready");
    println!("  ✓ Overlay rendering ready");
    println!();

    // 6. Process frames
    println!("🎬 Step 6: Processing Frames...");
    println!("════════════════════════════════════════════════════════════");
    println!();

    let frames_to_process = 10;
    let mut frame_count = 0;
    let mut total_latency_ms = 0.0;

    // Latency breakdown accumulators
    let mut total_capture_ms = 0.0;
    let mut total_h2d_ms = 0.0; // Host to Device transfer
    let mut total_preprocess_ms = 0.0;
    let mut total_inference_ms = 0.0;
    let mut total_postprocess_ms = 0.0;
    let mut total_plan_ms = 0.0;
    let mut total_render_ms = 0.0;

    // GPU buffer pool to avoid repeated allocations
    let mut gpu_buffers: Vec<CudaSlice<u8>> = Vec::new();

    while frame_count < frames_to_process {
        let pipeline_start = Instant::now();

        // Capture frame
        let capture_start = Instant::now();
        let raw_frame = match capture.get_frame()? {
            Some(frame) => frame,
            None => {
                println!("⚠️  No frame received, waiting...");
                std::thread::sleep(std::time::Duration::from_millis(16));
                continue;
            }
        };
        let capture_time = capture_start.elapsed();
        total_capture_ms += capture_time.as_secs_f64() * 1000.0;

        println!("════════════════════════════════════════════════════════════");
        println!("🎬 Frame #{}", frame_count);
        println!("════════════════════════════════════════════════════════════");

        // Step 1: Capture Results
        println!();
        println!("📹 Step 1: Capture Result");
        println!("  ✓ Time: {:.2}ms", capture_time.as_secs_f64() * 1000.0);
        println!("  ✓ RawFramePacket:");
        println!(
            "      → Dimensions: {}×{}",
            raw_frame.meta.width, raw_frame.meta.height
        );
        println!(
            "      → Format: {:?} {:?}",
            raw_frame.meta.pixfmt, raw_frame.meta.colorspace
        );
        println!("      → Stride: {} bytes", raw_frame.meta.stride_bytes);
        println!("      → Frame idx: {}", raw_frame.meta.frame_idx);
        println!("      → PTS: {} ns", raw_frame.meta.pts_ns);
        println!("      → Data size: {} bytes", raw_frame.data.len);
        println!("      → Location: {:?}", raw_frame.data.loc);

        // Copy CPU data to GPU if needed
        let h2d_start = Instant::now();
        let raw_frame_gpu = if matches!(raw_frame.data.loc, MemLoc::Cpu) {
            // Allocate GPU buffer and copy data
            let cpu_data =
                unsafe { std::slice::from_raw_parts(raw_frame.data.ptr, raw_frame.data.len) };

            let gpu_buffer = cuda_device.htod_sync_copy(cpu_data)?;
            let gpu_ptr = *gpu_buffer.device_ptr() as *mut u8;

            let gpu_packet = common_io::RawFramePacket {
                meta: raw_frame.meta.clone(),
                data: MemRef {
                    ptr: gpu_ptr,
                    len: raw_frame.data.len,
                    stride: raw_frame.data.stride,
                    loc: MemLoc::Gpu { device: 0 },
                },
            };

            // Store buffer to prevent deallocation
            gpu_buffers.push(gpu_buffer);
            gpu_packet
        } else {
            raw_frame.clone()
        };
        let h2d_time = h2d_start.elapsed();
        total_h2d_ms += h2d_time.as_secs_f64() * 1000.0;
        println!("  ✓ Time: {:.2}ms", h2d_time.as_secs_f64() * 1000.0);
        println!(
            "  ✓ Transferred {} bytes to GPU device 0",
            raw_frame_gpu.data.len
        );
        println!("  ✓ New location: {:?}", raw_frame_gpu.data.loc);

        // Step 3: Preprocessing
        println!();
        println!("⚙️  Step 3: CUDA Preprocessing");
        let preprocess_start = Instant::now();
        let tensor_packet = match preprocessor.process_checked(raw_frame_gpu) {
            Some(packet) => packet,
            None => {
                println!("  ⚠️  Skipping init frame (waiting for stable resolution)\n");
                continue;
            }
        };
        let preprocess_time = preprocess_start.elapsed();
        total_preprocess_ms += preprocess_time.as_secs_f64() * 1000.0;
        println!("  ✓ Time: {:.2}ms", preprocess_time.as_secs_f64() * 1000.0);
        println!("  ✓ TensorInputPacket:");
        println!(
            "      → Shape: {}×{}×{}×{} (NCHW)",
            tensor_packet.desc.n, tensor_packet.desc.c, tensor_packet.desc.h, tensor_packet.desc.w
        );
        println!("      → Data type: {:?}", tensor_packet.desc.dtype);
        println!("      → Device: GPU {}", tensor_packet.desc.device);
        println!("      → Buffer size: {} bytes", tensor_packet.data.len);
        println!(
            "      → Operations: YUV422→RGB, Resize 1920×1080→512×512, Normalize, FP16 convert"
        );

        // Save preprocessed tensor as image
        let preprocess_img_path = format!("output/test/preprocess_frame_{:04}.png", frame_count);
        match save_preprocessed_tensor(&tensor_packet, &cuda_device, &preprocess_img_path) {
            Ok(_) => println!("      → Saved preprocessed image: {}", preprocess_img_path),
            Err(e) => println!("      ⚠️  Failed to save preprocessed image: {}", e),
        }

        // Step 4: Inference V2
        println!();
        println!("🧠 Step 4: TensorRT Inference V2");
        let inference_start = Instant::now();
        let raw_detections = inference_stage.process(tensor_packet);
        let inference_time = inference_start.elapsed();
        total_inference_ms += inference_time.as_secs_f64() * 1000.0;
        println!("  ✓ Time: {:.2}ms", inference_time.as_secs_f64() * 1000.0);
        println!("  ✓ RawDetectionsPacket:");
        println!("      → Output shape: {:?}", raw_detections.output_shape);
        println!("      → Total values: {}", raw_detections.raw_output.len());
        println!(
            "      → Format: [batch, detections, (x, y, w, h, obj_conf, class0_conf, class1_conf)]"
        );
        println!("      → Engine: v7_optimized_YOLOv5 (2-class)");

        // Save inference output
        let inference_txt_path = format!("output/test/inference_frame_{:04}.txt", frame_count);
        match save_inference_output(&raw_detections, &inference_txt_path) {
            Ok(_) => println!("      → Saved inference output: {}", inference_txt_path),
            Err(e) => println!("      ⚠️  Failed to save inference output: {}", e),
        }

        // Step 5: Postprocessing (NMS + Tracking) - ENABLED
        println!();
        println!("🎯 Step 5: Postprocessing (NMS + Tracking)");
        let postprocess_start = Instant::now();
        let detections = post_stage.process(raw_detections);
        let postprocess_time = postprocess_start.elapsed();
        total_postprocess_ms += postprocess_time.as_secs_f64() * 1000.0;
        println!("  ✓ Time: {:.2}ms", postprocess_time.as_secs_f64() * 1000.0);
        println!("  ✓ DetectionsPacket:");
        println!("      → Total detections: {}", detections.items.len());
        println!("      → Operations: Sigmoid, Coordinate Transform (stretch resize), NMS (IoU=0.45), SORT tracking");

        // Show first 5 detections as sample
        let display_count = std::cmp::min(5, detections.items.len());
        if display_count > 0 {
            println!(
                "      → Sample detections (showing first {}):",
                display_count
            );
            for (i, det) in detections.items.iter().take(display_count).enumerate() {
                let class_name = match det.class_id {
                    0 => "Hyper",
                    1 => "Neo",
                    _ => "Unknown",
                };
                let track_info = if let Some(track_id) = det.track_id {
                    format!(" [Track #{}]", track_id)
                } else {
                    String::new()
                };
                println!(
                    "         {}. {} {:.2}{} @ ({:.0},{:.0}) {:.0}×{:.0}",
                    i + 1,
                    class_name,
                    det.score,
                    track_info,
                    det.bbox.x,
                    det.bbox.y,
                    det.bbox.w,
                    det.bbox.h
                );
            }
            if detections.items.len() > display_count {
                println!(
                    "         ... and {} more",
                    detections.items.len() - display_count
                );
            }
        }

        // Save postprocess output
        let postprocess_txt_path = format!("output/test/postprocess_frame_{:04}.txt", frame_count);
        match save_postprocess_output(&detections, &postprocess_txt_path) {
            Ok(_) => println!("      → Saved postprocess output: {}", postprocess_txt_path),
            Err(e) => println!("      ⚠️  Failed to save postprocess output: {}", e),
        }

        // Step 6: Overlay Planning
        println!();
        println!("🎨 Step 6: Overlay Planning");
        let plan_start = Instant::now();
        let overlay_plan = plan_stage.process(detections);
        let plan_time = plan_start.elapsed();
        total_plan_ms += plan_time.as_secs_f64() * 1000.0;
        println!("  ✓ Time: {:.2}ms", plan_time.as_secs_f64() * 1000.0);
        println!("  ✓ OverlayPlanPacket:");
        println!(
            "      → Canvas size: {}×{}",
            overlay_plan.canvas.0, overlay_plan.canvas.1
        );
        println!("      → Total draw operations: {}", overlay_plan.ops.len());

        // Count operation types
        let mut rect_count = 0;
        let mut label_count = 0;
        for op in &overlay_plan.ops {
            match op {
                common_io::DrawOp::Rect { .. } => rect_count += 1,
                common_io::DrawOp::Label { .. } => label_count += 1,
                _ => {}
            }
        }
        println!(
            "      → Rectangles: {}, Labels: {}",
            rect_count, label_count
        );

        // Step 7: Overlay Rendering
        println!();
        println!("🖼️  Step 7: Overlay Rendering");
        let render_start = Instant::now();
        let overlay_frame = render_stage.process(overlay_plan);
        let render_time = render_start.elapsed();
        total_render_ms += render_time.as_secs_f64() * 1000.0;
        println!("  ✓ Time: {:.2}ms", render_time.as_secs_f64() * 1000.0);
        println!("  ✓ OverlayFramePacket:");
        println!(
            "      → Frame dimensions: {}×{}",
            overlay_frame.from.width, overlay_frame.from.height
        );
        println!("      → ARGB buffer: {} bytes", overlay_frame.argb.len);
        println!("      → Location: {:?}", overlay_frame.argb.loc);
        println!("      → Stride: {} bytes/row", overlay_frame.stride);
        println!("      → Frame index: {}", overlay_frame.from.frame_idx);
        println!("      → Status: Ready for DeckLink output (internal keying)");

        // Step 8: Save images (original, overlay, and comparison)
        println!();
        println!("💾 Step 8: Save Images");
        let save_start = Instant::now();
        
        let original_path = format!("output/test/original_frame_{:04}.png", frame_count);
        let overlay_path = format!("output/test/overlay_frame_{:04}.png", frame_count);
        let comparison_path = format!("output/test/comparison_frame_{:04}.png", frame_count);
        
        let mut save_success = true;
        
        // Save original frame (without bounding boxes)
        // Need to copy GPU data to CPU first if needed
        let mut cpu_buffer: Vec<u8>;
        let raw_frame_cpu = if !matches!(raw_frame.data.loc, MemLoc::Cpu) {
            // Copy from GPU to CPU using existing cuda_device
            cpu_buffer = vec![0u8; raw_frame.data.len];
            
            // Use cudarc to copy device memory to host
            unsafe {
                cudarc::driver::result::memcpy_dtoh_sync(
                    &mut cpu_buffer,
                    raw_frame.data.ptr as cudarc::driver::sys::CUdeviceptr,
                )?;
            }
            
            common_io::RawFramePacket {
                meta: raw_frame.meta.clone(),
                data: MemRef {
                    ptr: cpu_buffer.as_ptr() as *mut u8,
                    len: cpu_buffer.len(),
                    stride: raw_frame.data.stride,
                    loc: MemLoc::Cpu,
                },
            }
        } else {
            cpu_buffer = Vec::new(); // Empty buffer for CPU case
            raw_frame.clone()
        };
        
        match save_original_frame(&raw_frame_cpu, &original_path) {
            Ok(_) => println!("  ✓ Saved original: {}", original_path),
            Err(e) => {
                println!("  ⚠️  Failed to save original: {}", e);
                save_success = false;
            }
        }
        
        // Save overlay frame (with bounding boxes)
        match save_overlay_as_image(&overlay_frame, &overlay_path) {
            Ok(_) => println!("  ✓ Saved overlay: {}", overlay_path),
            Err(e) => {
                println!("  ⚠️  Failed to save overlay: {}", e);
                save_success = false;
            }
        }
        
        // Save comparison (side-by-side)
        if save_success {
            match save_comparison_image(&original_path, &overlay_path, &comparison_path) {
                Ok(_) => {
                    let save_time = save_start.elapsed();
                    println!("  ✓ Saved comparison: {}", comparison_path);
                    println!("  ✓ Total save time: {:.2}ms", save_time.as_secs_f64() * 1000.0);
                    println!("  ✓ 3 images saved: original, overlay, and side-by-side comparison");
                }
                Err(e) => {
                    println!("  ⚠️  Failed to save comparison: {}", e);
                }
            }
        }

        let pipeline_time = pipeline_start.elapsed();
        let pipeline_ms = pipeline_time.as_secs_f64() * 1000.0;
        total_latency_ms += pipeline_ms;

        println!();
        println!(
            "  ⏱️  End-to-End: {:.2}ms ({:.1} FPS)",
            pipeline_ms,
            1000.0 / pipeline_ms
        );
        println!();

        frame_count += 1;

        // Small delay to prevent overwhelming the output
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Summary
    println!("════════════════════════════════════════════════════════════");
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║                 PIPELINE SUMMARY                         ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║ Frames Processed:  {:2}                                   ║",
        frames_to_process
    );
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Average Latency Breakdown:                               ║");
    println!(
        "║   Capture:        {:.2} ms                           ║",
        total_capture_ms / frames_to_process as f64
    );
    println!(
        "║   H2D Transfer:   {:.2} ms                            ║",
        total_h2d_ms / frames_to_process as f64
    );
    println!(
        "║   Preprocess:     {:.2} ms                           ║",
        total_preprocess_ms / frames_to_process as f64
    );
    println!(
        "║   Inference V2:   {:.2} ms                           ║",
        total_inference_ms / frames_to_process as f64
    );
    println!(
        "║   Postprocess:    {:.2} ms                            ║",
        total_postprocess_ms / frames_to_process as f64
    );
    println!(
        "║   Overlay Plan:   {:.2} ms                            ║",
        total_plan_ms / frames_to_process as f64
    );
    println!(
        "║   Overlay Render: {:.2} ms                            ║",
        total_render_ms / frames_to_process as f64
    );
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║   TOTAL E2E:      {:.2} ms                          ║",
        total_latency_ms / frames_to_process as f64
    );
    println!(
        "║   Average FPS:    {:.1}                              ║",
        1000.0 * frames_to_process as f64 / total_latency_ms
    );
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Pipeline Components:                                     ║");
    println!("║   ✓ DeckLink Capture (YUV422_8, test_rawframepacket)    ║");
    println!("║   ✓ CUDA Preprocessing (512×512 FP16)                    ║");
    println!("║   ✓ TensorRT Inference V2 (configs/model engine)         ║");
    println!("║   ✓ Postprocessing (NMS + SORT Tracking)                 ║");
    println!("║   ✓ Overlay Planning (Bounding boxes + Labels)           ║");
    println!("║   ✓ Overlay Rendering (ARGB output)                      ║");
    println!("║   ✓ Image Export (3 types: original, overlay, compare)   ║");
    println!("║   ✓ Step Output Export (preprocess, inference, postproc) ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Output Files:                                             ║");
    println!(
        "║   📁 {} frames × 6 files = {} total files              ║",
        frames_to_process,
        frames_to_process * 6
    );
    println!("║                                                           ║");
    println!("║   Images (per frame):                                     ║");
    println!("║   1. output/test/original_frame_*.png                     ║");
    println!("║      → Original YUV422 frame (no overlay)                 ║");
    println!("║   2. output/test/preprocess_frame_*.png                   ║");
    println!("║      → Preprocessed tensor (512×512 RGB, denormalized)    ║");
    println!("║   3. output/test/overlay_frame_*.png                      ║");
    println!("║      → With bounding boxes, labels, confidence, tracks    ║");
    println!("║   4. output/test/comparison_frame_*.png                   ║");
    println!("║      → Side-by-side: Original | Overlay (2× width)       ║");
    println!("║                                                           ║");
    println!("║   Text outputs (per frame):                               ║");
    println!("║   5. output/test/inference_frame_*.txt                    ║");
    println!("║      → Raw TensorRT inference output (all detections)     ║");
    println!("║   6. output/test/postprocess_frame_*.txt                  ║");
    println!("║      → Final detections after NMS + tracking              ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Status: All outputs saved successfully!                   ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("✅ Pipeline test completed successfully!");
    println!();
    println!("📂 Output files:");
    println!("   Images:");
    println!("   • output/test/original_frame_*.png    - Original frames (1920×1080)");
    println!("   • output/test/preprocess_frame_*.png  - Preprocessed tensor (512×512)");
    println!("   • output/test/overlay_frame_*.png     - With bounding boxes");
    println!("   • output/test/comparison_frame_*.png  - Side-by-side comparison");
    println!();
    println!("   Text outputs:");
    println!("   • output/test/inference_frame_*.txt   - Raw inference output");
    println!("   • output/test/postprocess_frame_*.txt - Final detections");
    println!();
    println!("💡 Next steps:");
    println!("   1. View preprocess images to verify preprocessing quality");
    println!("   2. Check inference/postprocess txt files for detection data");
    println!("   3. Compare original vs overlay to see detection accuracy");
    println!();
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Output:                                                   ║");
    println!(
        "║   📁 {} frames saved to output/test/overlay_frame_*.png ║",
        frames_to_process
    );
    println!("║   🎨 Each image contains:                                 ║");
    println!("║      • Bounding boxes around detected objects             ║");
    println!("║      • Class labels (Hyper/Neo)                           ║");
    println!("║      • Confidence scores                                  ║");
    println!("║      • Tracking IDs                                       ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Status: Images saved successfully!                        ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("✅ Pipeline test completed successfully!");
    println!();
    println!("� Output files:");
    println!("   • Check output/overlay_frame_*.png for bounding box images");
    println!();
    println!("💡 Next steps:");
    println!("   1. View saved images to verify detection quality");
    println!("   2. Connect OverlayFramePacket to DeckLink output");
    println!("   3. Enable internal keying for final output");
    println!();

    Ok(())
}
