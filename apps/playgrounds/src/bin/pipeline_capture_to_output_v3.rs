//! Complete Pipeline: Capture → Overlay using Hardware Internal Keying
//!
//! Pipeline flow:
//! DeckLink Capture → Preprocessing → Inference V2 → Postprocessing → Overlay Planning → Overlay Rendering → Hardware Keying
//!
//! This demonstrates the complete real-time detection pipeline using inference_v2
//! with hardware internal keying for compositing overlay with SDI input.

use anyhow::{anyhow, Result};
use common_io::{MemLoc, MemRef, Stage};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use decklink_input::capture::CaptureSession;
use decklink_output::OutputRequest;
use inference_v2::TrtInferenceStage;
use overlay_plan::PlanStage;
use overlay_render::RenderStage;
use postprocess;
use preprocess_cuda::{Preprocessor, CropRegion};
use std::time::{Duration, Instant};
use image::{RgbaImage, Rgba, RgbImage};
use std::fs::File;
use std::io::Write;

/// Simple BGRA image structure for overlay conversion
struct BgraImage {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

/// Convert OverlayFramePacket (ARGB) to BgraImage for internal keying
fn overlay_to_bgra(overlay: &common_io::OverlayFramePacket) -> Result<BgraImage> {
    let width = overlay.from.width;  // Keep as u32
    let height = overlay.from.height;  // Keep as u32
    
    // Get ARGB data from CPU
    let argb_data = if matches!(overlay.argb.loc, MemLoc::Cpu) {
        unsafe { std::slice::from_raw_parts(overlay.argb.ptr, overlay.argb.len) }
    } else {
        return Err(anyhow!("Overlay data must be on CPU for conversion"));
    };
    
    // Convert ARGB → BGRA
    let mut bgra_data = Vec::with_capacity(argb_data.len());
    for chunk in argb_data.chunks_exact(4) {
        let a = chunk[0];  // Alpha
        let r = chunk[1];  // Red
        let g = chunk[2];  // Green
        let b = chunk[3];  // Blue
        
        // BGRA order
        bgra_data.push(b);
        bgra_data.push(g);
        bgra_data.push(r);
        bgra_data.push(a);
    }
    
    Ok(BgraImage {
        width,
        height,
        data: bgra_data,
    })
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
    // Tensor is NCHW format: [1, 3, 512, 512] in FP32 or FP16
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
    
    // Convert to f32 values based on data type
    let float_data: Vec<f32> = match tensor_packet.desc.dtype {
        common_io::DType::Fp16 => {
            // Convert FP16 bytes to f32 values
            let fp16_data: &[u16] = unsafe {
                std::slice::from_raw_parts(
                    cpu_data.as_ptr() as *const u16,
                    cpu_data.len() / 2,
                )
            };
            
            fp16_data
                .iter()
                .map(|&fp16_bits| half::f16::from_bits(fp16_bits).to_f32())
                .collect()
        }
        common_io::DType::Fp32 => {
            // Convert FP32 bytes to f32 values
            let fp32_data: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    cpu_data.as_ptr() as *const f32,
                    cpu_data.len() / 4,
                )
            };
            fp32_data.to_vec()
        }
        _ => return Err(anyhow!("Unsupported data type: {:?}", tensor_packet.desc.dtype)),
    };
    
    // Denormalize: reverse the normalization (mean=[0,0,0], std=[1/255, 1/255, 1/255])
    // normalized = (pixel / 255.0 - mean) / std
    // pixel = (normalized * std + mean) * 255.0
    let mut denorm_data = float_data;
    for val in denorm_data.iter_mut() {
        *val = (*val * 255.0).clamp(0.0, 255.0);
    }
    
    // Create RGB image from CHW format
    let mut img = RgbImage::new(w as u32, h as u32);
    
    let pixels_per_channel = (h * w) as usize;
    for y in 0..h as u32 {
        for x in 0..w as u32 {
            let idx = (y * w as u32 + x) as usize;
            
            // CHW format: [R_channel, G_channel, B_channel]
            let r = denorm_data[idx] as u8;
            let g = denorm_data[pixels_per_channel + idx] as u8;
            let b = denorm_data[pixels_per_channel * 2 + idx] as u8;
            
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
    println!("║  PIPELINE: CAPTURE → OVERLAY (HARDWARE KEYING)          ║");
    println!("║  DeckLink → Preprocess → Inference V2 → Post → Overlay  ║");
    println!("║  → Hardware Internal Keying (30 seconds test)           ║");
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
    let crop_region = CropRegion::Olympus; // Change this to Fuji or Pentax as needed
    
    let mut preprocessor = Preprocessor::with_crop_region(
        (512, 512),                    // Output size (matching YOLOv5 engine input)
        false,                         // FP32 (TensorRT engine expects FP32 input!)
        0,                             // GPU device 0
        [0.0, 0.0, 0.0],              // mean
        [1.0, 1.0, 1.0],              // std
        preprocess_cuda::ChromaOrder::UYVY,  // chroma order
        crop_region,                   // crop region
    )?;
    
    let (cx, cy, cw, ch) = crop_region.get_coords();
    println!("  ✓ Preprocessor ready (512x512, FP32, GPU 0)");
    println!("  ✓ Crop region: {:?} => [{}, {}, {}×{}]", crop_region, cx, cy, cw, ch);
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

    // 5.5 Initialize Hardware Internal Keying (DeckLink Output)
    println!("🔧 Step 5.5: Initialize Hardware Internal Keying");
    
    // Wait for first frame to get dimensions
    println!("  ⏳ Waiting for first frame to determine dimensions...");
    let (width, height) = loop {
        if let Some(frame) = capture.get_frame()? {
            let w = frame.meta.width;
            let h = frame.meta.height;
            
            // Check if frame has stable HD resolution
            if w >= 1920 && h >= 1080 {
                println!("   ✓ Got frame: {}x{}", w, h);
                break (w, h);
            } else {
                println!("   ⏳ Frame {}x{} (waiting for HD resolution)...", w, h);
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    
    // Initialize DeckLink output
    let mut decklink_out = decklink_output::from_path("configs/dev_1080p60_yuv422_fp16_trt.toml")?;
    println!("  ✓ DeckLink output initialized: {}x{}", width, height);
    
    // Configure SDI output connection
    decklink_out.set_sdi_output()?;
    println!("  ✓ SDI output connection configured");
    
    // Enable hardware internal keying
    decklink_out.enable_internal_keying()?;
    println!("  ✓ Hardware internal keying ENABLED");
    
    // Set keyer level to maximum (255 = fully visible overlay)
    decklink_out.set_keyer_level(255)?;
    println!("  ✓ Keyer level set to 255 (fully visible)");
    println!();

    // 6. Process frames
    println!("🎬 Step 6: Processing Frames (30 seconds test)...");
    println!("════════════════════════════════════════════════════════════");
    println!("   Mode: Hardware Internal Keying");
    println!("   Input: SDI capture → Overlay (GPU)");
    println!("   Output: SDI (Hardware Composited)");
    println!("   Duration: 30 seconds");
    println!("════════════════════════════════════════════════════════════");
    println!();

    let mut frame_count = 0u64;
    let mut total_latency_ms = 0.0;

    // Latency breakdown accumulators
    let mut total_capture_ms = 0.0;
    let mut total_preprocess_ms = 0.0;
    let mut total_inference_ms = 0.0;
    let mut total_postprocess_ms = 0.0;
    let mut total_plan_ms = 0.0;
    let mut total_render_ms = 0.0;
    let mut total_keying_ms = 0.0; // Internal keying (GPU composite + DeckLink output)
    let mut total_hardware_latency_ms = 0.0; // Hardware timestamp: capture -> output

    // GPU buffer pool to avoid repeated allocations
    let mut gpu_buffers: Vec<CudaSlice<u8>> = Vec::new();
    
    // Track start time for FPS calculation and 30-second timeout
    let pipeline_start_time = Instant::now();
    let test_duration = Duration::from_secs(30);

    loop {
        // Check if 30 seconds have elapsed
        if pipeline_start_time.elapsed() >= test_duration {
            println!();
            println!("⏱️  30 seconds elapsed - stopping test");
            break;
        }
        
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

        // Step 3: Preprocessing
        println!();
        println!("⚙️  Step 2: CUDA Preprocessing");
        let preprocess_start = Instant::now();
        // Clone raw_frame_gpu because we need it later for internal keying
        let tensor_packet = match preprocessor.process_checked(raw_frame_gpu.clone()) {
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

        // Step 4: Inference V2
        println!();
        println!("🧠 Step 3: TensorRT Inference V2");
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

        // Print top 5 raw detections
        let num_detections = raw_detections.output_shape[1];
        let values_per_detection = raw_detections.output_shape[2];
        
        // Collect all detections with objectness scores
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
        
        // Sort by objectness confidence (descending)
        detections_with_scores.sort_by(|a, b| b.5.partial_cmp(&a.5).unwrap_or(std::cmp::Ordering::Equal));
        
        // Print top 5
        println!("      → Top 5 raw detections (by objectness):");
        for (rank, (idx, x, y, w, h, obj_conf, class0_conf, class1_conf)) in detections_with_scores.iter().take(5).enumerate() {
            println!(
                "         {}. [idx={}] x={:.1}, y={:.1}, w={:.1}, h={:.1}, obj={:.4}, cls0={:.4}, cls1={:.4}",
                rank + 1, idx, x, y, w, h, obj_conf, class0_conf, class1_conf
            );
        }

        // Step 5: Postprocessing (NMS + Tracking) - ENABLED
        println!();
        println!("🎯 Step 4: Postprocessing (NMS + Tracking)");
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

        // Step 6: Overlay Planning
        println!();
        println!("🎨 Step 5: Overlay Planning");
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
        println!("🖼️  Step 6: Overlay Rendering");
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

        // Step 7.5: Internal Keying (GPU Composite)
        println!();
        println!("🎬 Step 7: Internal Keying (GPU Composite + DeckLink Output)");
        let keying_start = Instant::now();
        
        // Step 7: Hardware Internal Keying Output
        println!();
        println!("🎬 Step 7: Hardware Internal Keying Output");
        let keying_start = Instant::now();
        
        // Convert overlay ARGB to BGRA for DeckLink output
        let bgra_overlay = overlay_to_bgra(&overlay_frame)?;
        
        // Create RawFramePacket from overlay BGRA (on CPU)
        // We need to upload it to GPU for DeckLink output
        
        // Allocate or reuse GPU buffer for overlay
        let mut overlay_gpu_buffer = if gpu_buffers.is_empty() {
            let buffer: CudaSlice<u8> = cuda_device.alloc_zeros(bgra_overlay.data.len())?;
            buffer
        } else {
            gpu_buffers.pop().unwrap()
        };
        
        // Upload overlay to GPU
        cuda_device.htod_sync_copy_into(&bgra_overlay.data, &mut overlay_gpu_buffer)?;
        
        // Create GPU packet for overlay
        let overlay_gpu_packet = common_io::RawFramePacket {
            meta: common_io::FrameMeta {
                source_id: raw_frame_gpu.meta.source_id,
                width,
                height,
                pixfmt: common_io::PixelFormat::BGRA8,
                colorspace: common_io::ColorSpace::SRGB,
                frame_idx: raw_frame_gpu.meta.frame_idx,
                pts_ns: raw_frame_gpu.meta.pts_ns,
                t_capture_ns: raw_frame_gpu.meta.t_capture_ns,
                stride_bytes: (width * 4) as u32,
                crop_region: None,
            },
            data: MemRef {
                ptr: *overlay_gpu_buffer.device_ptr() as *mut u8,
                len: bgra_overlay.data.len(),
                stride: (width * 4) as usize,
                loc: common_io::MemLoc::Gpu { device: 0 },
            },
        };
        
        // Submit overlay to DeckLink output (hardware will composite with SDI input)
        let output_request = OutputRequest {
            video: Some(&overlay_gpu_packet),
            overlay: None,
        };
        
        decklink_out.submit(output_request)?;
        
        // Return GPU buffer to pool
        gpu_buffers.push(overlay_gpu_buffer);
        
        let keying_time = keying_start.elapsed();
        total_keying_ms += keying_time.as_secs_f64() * 1000.0;
        println!("  ✓ Time: {:.2}ms", keying_time.as_secs_f64() * 1000.0);
        println!("  ✓ Mode: HARDWARE INTERNAL KEYING");
        println!("  ✓ Pipeline: Overlay ARGB → BGRA → GPU → DeckLink Fill");
        println!("  ✓ Overlay dimensions: {}×{}", bgra_overlay.width, bgra_overlay.height);
        println!("  ✓ Output format: BGRA8 on GPU");
        println!("  ✓ Hardware keyer: ACTIVE (compositing with SDI input)");
        println!("  ✓ Frame submitted to DeckLink output");

        // Calculate hardware-to-hardware latency (using timestamps)
        let output_complete_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let hardware_latency_ns = output_complete_ns - raw_frame_gpu.meta.t_capture_ns;
        let hardware_latency_ms = hardware_latency_ns as f64 / 1_000_000.0;
        total_hardware_latency_ms += hardware_latency_ms;

        let pipeline_time = pipeline_start.elapsed();
        let pipeline_ms = pipeline_time.as_secs_f64() * 1000.0;
        total_latency_ms += pipeline_ms;

        // Print detailed latency breakdown
        println!();
        println!("════════════════════════════════════════════════════════════");
        println!("⏱️  LATENCY BREAKDOWN - Frame #{}", frame_count);
        println!("════════════════════════════════════════════════════════════");
        println!("  1. 📹 Capture:           {:6.2}ms ({:4.1}%)", 
            capture_time.as_secs_f64() * 1000.0,
            (capture_time.as_secs_f64() * 1000.0 / pipeline_ms) * 100.0);
        println!("  2. ⚙️  Preprocessing:     {:6.2}ms ({:4.1}%)", 
            preprocess_time.as_secs_f64() * 1000.0,
            (preprocess_time.as_secs_f64() * 1000.0 / pipeline_ms) * 100.0);
        println!("  3. 🧠 Inference:         {:6.2}ms ({:4.1}%)", 
            inference_time.as_secs_f64() * 1000.0,
            (inference_time.as_secs_f64() * 1000.0 / pipeline_ms) * 100.0);
        println!("  4. 🎯 Postprocessing:    {:6.2}ms ({:4.1}%)", 
            postprocess_time.as_secs_f64() * 1000.0,
            (postprocess_time.as_secs_f64() * 1000.0 / pipeline_ms) * 100.0);
        println!("  5. 🎨 Overlay Planning:  {:6.2}ms ({:4.1}%)", 
            plan_time.as_secs_f64() * 1000.0,
            (plan_time.as_secs_f64() * 1000.0 / pipeline_ms) * 100.0);
        println!("  6. 🖼️  Overlay Rendering: {:6.2}ms ({:4.1}%)", 
            render_time.as_secs_f64() * 1000.0,
            (render_time.as_secs_f64() * 1000.0 / pipeline_ms) * 100.0);
        println!("  7. 🎬 Hardware Keying:   {:6.2}ms ({:4.1}%)", 
            keying_time.as_secs_f64() * 1000.0,
            (keying_time.as_secs_f64() * 1000.0 / pipeline_ms) * 100.0);
        println!("────────────────────────────────────────────────────────────");
        println!("  🔴 END-TO-END (N2N):    {:6.2}ms ({:.1} FPS)", 
            pipeline_ms, 1000.0 / pipeline_ms);
        println!("  🔴 HARDWARE LATENCY:    {:6.2}ms (capture timestamp → output)", 
            hardware_latency_ms);
        println!("  🎯 Mode: HARDWARE INTERNAL KEYING (SDI Input + Overlay Fill)");
        println!("════════════════════════════════════════════════════════════");
        println!();

        frame_count += 1;

        // Print cumulative stats every 60 frames
        if frame_count % 60 == 0 {
            let elapsed = pipeline_start_time.elapsed().as_secs_f64();
            let avg_fps = frame_count as f64 / elapsed;
            
            println!();
            println!("╔══════════════════════════════════════════════════════════╗");
            println!("║  📊 CUMULATIVE STATISTICS - {} FRAMES                    ", frame_count);
            println!("╚══════════════════════════════════════════════════════════╝");
            println!();
            println!("  ⏱️  Average Latency per Stage:");
            println!("  ─────────────────────────────────────────────────────────");
            println!("  1. 📹 Capture:           {:6.2}ms avg", total_capture_ms / frame_count as f64);
            println!("  2. ⚙️  Preprocessing:     {:6.2}ms avg", total_preprocess_ms / frame_count as f64);
            println!("  3. 🧠 Inference:         {:6.2}ms avg", total_inference_ms / frame_count as f64);
            println!("  4. 🎯 Postprocessing:    {:6.2}ms avg", total_postprocess_ms / frame_count as f64);
            println!("  5. 🎨 Overlay Planning:  {:6.2}ms avg", total_plan_ms / frame_count as f64);
            println!("  6. 🖼️  Overlay Rendering: {:6.2}ms avg", total_render_ms / frame_count as f64);
            println!("  7. 🎬 Hardware Keying:   {:6.2}ms avg", total_keying_ms / frame_count as f64);
            println!("  ─────────────────────────────────────────────────────────");
            println!("  🔴 END-TO-END (N2N):    {:6.2}ms avg ({:.2} FPS avg)", 
                total_latency_ms / frame_count as f64,
                1000.0 / (total_latency_ms / frame_count as f64));
            println!("  🔴 HARDWARE LATENCY:    {:6.2}ms avg (capture → output)", 
                total_hardware_latency_ms / frame_count as f64);
            println!("  🎯 Mode: HARDWARE INTERNAL KEYING");
            println!();
            println!("  📈 Performance Metrics:");
            println!("  ─────────────────────────────────────────────────────────");
            println!("  Total frames processed:  {}", frame_count);
            println!("  Total elapsed time:      {:.2}s", elapsed);
            println!("  Real-time FPS:           {:.2} FPS", avg_fps);
            println!("  ─────────────────────────────────────────────────────────");
            println!();
        }
    }

    // Print final summary after 30 seconds
    let total_elapsed = pipeline_start_time.elapsed().as_secs_f64();
    let avg_fps = frame_count as f64 / total_elapsed;
    
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  🏁 FINAL SUMMARY - 30 SECOND TEST COMPLETE             ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("  ⏱️  Average Latency per Stage:");
    println!("  ─────────────────────────────────────────────────────────");
    println!("  1. 📹 Capture:           {:6.2}ms avg", total_capture_ms / frame_count as f64);
    println!("  2. ⚙️  Preprocessing:     {:6.2}ms avg", total_preprocess_ms / frame_count as f64);
    println!("  3. 🧠 Inference:         {:6.2}ms avg", total_inference_ms / frame_count as f64);
    println!("  4. 🎯 Postprocessing:    {:6.2}ms avg", total_postprocess_ms / frame_count as f64);
    println!("  5. 🎨 Overlay Planning:  {:6.2}ms avg", total_plan_ms / frame_count as f64);
    println!("  6. 🖼️  Overlay Rendering: {:6.2}ms avg", total_render_ms / frame_count as f64);
    println!("  7. 🎬 Hardware Keying:   {:6.2}ms avg", total_keying_ms / frame_count as f64);
    println!("  ─────────────────────────────────────────────────────────");
    println!("  🔴 END-TO-END (N2N):    {:6.2}ms avg ({:.2} FPS avg)", 
        total_latency_ms / frame_count as f64,
        1000.0 / (total_latency_ms / frame_count as f64));
    println!("  🔴 HARDWARE LATENCY:    {:6.2}ms avg", 
        total_hardware_latency_ms / frame_count as f64);
    println!();
    println!("  📈 Performance Metrics:");
    println!("  ─────────────────────────────────────────────────────────");
    println!("  Total frames processed:  {}", frame_count);
    println!("  Total elapsed time:      {:.2}s", total_elapsed);
    println!("  Real-time FPS:           {:.2} FPS", avg_fps);
    println!("  Mode:                    HARDWARE INTERNAL KEYING");
    println!("  ─────────────────────────────────────────────────────────");
    println!();
    println!("✅ Test completed successfully!");
    
    Ok(())
}
