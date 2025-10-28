//! Complete Pipeline with DeckLink Output: Capture → Overlay → Output
//!
//! Pipeline flow:
//! DeckLink Capture → Preprocessing → Inference V2 → Postprocessing → Overlay Planning → Overlay Rendering → DeckLink Output
//!
//! This demonstrates the complete real-time detection pipeline with actual video output via DeckLink.

use anyhow::{anyhow, Result};
use common_io::{MemLoc, MemRef, Stage};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use decklink_input::capture::CaptureSession;
use decklink_output::keying::InternalKeyingOutput;
use inference_v2::TrtInferenceStage;
use overlay_plan::PlanStage;
use overlay_render::RenderStage;
use postprocess;
use preprocess_cuda::Preprocessor;
use std::time::Instant;

// Helper functions for drawing on BGRA overlay
fn draw_rect_bgra(buffer: &mut [u8], width: u32, height: u32, x: i32, y: i32, w: i32, h: i32, color: (u8, u8, u8, u8)) {
    let (r, g, b, a) = color;
    let thickness = 2;
    
    for t in 0..thickness {
        // Top and bottom lines
        for dx in 0..w {
            let px = x + dx;
            if px >= 0 && px < width as i32 {
                // Top
                let py_top = y + t;
                if py_top >= 0 && py_top < height as i32 {
                    set_pixel_bgra(buffer, width, px, py_top, b, g, r, a);
                }
                // Bottom
                let py_bottom = y + h - 1 - t;
                if py_bottom >= 0 && py_bottom < height as i32 {
                    set_pixel_bgra(buffer, width, px, py_bottom, b, g, r, a);
                }
            }
        }
        
        // Left and right lines
        for dy in 0..h {
            let py = y + dy;
            if py >= 0 && py < height as i32 {
                // Left
                let px_left = x + t;
                if px_left >= 0 && px_left < width as i32 {
                    set_pixel_bgra(buffer, width, px_left, py, b, g, r, a);
                }
                // Right
                let px_right = x + w - 1 - t;
                if px_right >= 0 && px_right < width as i32 {
                    set_pixel_bgra(buffer, width, px_right, py, b, g, r, a);
                }
            }
        }
    }
}

fn draw_filled_rect_bgra(buffer: &mut [u8], width: u32, height: u32, x: i32, y: i32, w: i32, h: i32, color: (u8, u8, u8, u8)) {
    let (r, g, b, a) = color;
    for dy in 0..h {
        for dx in 0..w {
            let px = x + dx;
            let py = y + dy;
            if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                set_pixel_bgra(buffer, width, px, py, b, g, r, a);
            }
        }
    }
}

fn set_pixel_bgra(buffer: &mut [u8], width: u32, x: i32, y: i32, b: u8, g: u8, r: u8, a: u8) {
    let idx = ((y as u32 * width + x as u32) * 4) as usize;
    if idx + 3 < buffer.len() {
        buffer[idx] = b;
        buffer[idx + 1] = g;
        buffer[idx + 2] = r;
        buffer[idx + 3] = a;
    }
}

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  FULL PIPELINE: CAPTURE → OVERLAY → OUTPUT              ║");
    println!("║  Real-time Detection with DeckLink Output               ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // 0. List available DeckLink devices
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
    println!("📹 Step 1: Initialize DeckLink Capture (Input)");
    let mut capture = CaptureSession::open(0)?;
    println!("  ✓ Opened DeckLink device 0 for capture");
    println!();

    // 2. Initialize DeckLink output
    println!("📺 Step 2: Initialize DeckLink Output");
    let output_device_idx = if devices.len() > 1 { 1 } else { 0 };
    let mut output = InternalKeyingOutput::new(output_device_idx, 1920, 1080, 60, 1);
    output.open()?;
    output.start_playback()?;
    println!("  ✓ Opened DeckLink device {} for output", output_device_idx);
    println!("  ✓ Internal keying enabled (1920x1080@60fps)");
    println!();

    // 3. Initialize Preprocessor
    println!("⚙️  Step 3: Initialize Preprocessor");
    let mut preprocessor = Preprocessor::new(
        (512, 512), // Output size (matching YOLOv5 engine input)
        true,       // FP16 for better performance
        0,          // GPU device 0
    )?;
    println!("  ✓ Preprocessor ready (512x512, FP16, GPU 0)");
    println!();

    // 4. Initialize CUDA device for CPU->GPU transfer
    println!("🔧 Step 4: Initialize CUDA Device");
    let cuda_device = CudaDevice::new(0)?;
    println!("  ✓ CUDA device initialized for CPU->GPU transfers");
    println!();

    // 5. Initialize TensorRT Inference V2
    println!("🧠 Step 5: Initialize TensorRT Inference V2");
    let engine_path = "configs/model/v7_optimized_YOLOv5.engine";
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

    // 6. Initialize Postprocessing
    println!("🎯 Step 6: Initialize Postprocessing");
    let mut post_stage = postprocess::from_path("")?.with_sort_tracking(30, 0.3, 0.3);
    println!("  ✓ Postprocessing ready");
    println!("  ✓ Classes: 2 (Hyper, Neo)");
    println!("  ✓ Confidence threshold: 0.25");
    println!("  ✓ Temporal smoothing: enabled (window=4)");
    println!("  ✓ SORT tracking: enabled (max_age=30)");
    println!();

    // 7. Initialize Overlay stages
    println!("🎨 Step 7: Initialize Overlay Stages");
    let mut plan_stage = PlanStage {
        enable_full_ui: true,
    };
    let mut render_stage = RenderStage {};
    println!("  ✓ Overlay planning ready");
    println!("  ✓ Overlay rendering ready");
    println!();

    // 8. Process frames
    println!("🎬 Step 8: Processing Frames with Live Output...");
    println!("════════════════════════════════════════════════════════════");
    println!("Press Ctrl+C to stop");
    println!();

    let frames_to_process = 100; // Process more frames for continuous output
    let mut frame_count = 0;
    let mut total_latency_ms = 0.0;

    // Latency breakdown accumulators
    let mut total_capture_ms = 0.0;
    let mut total_h2d_ms = 0.0;
    let mut total_preprocess_ms = 0.0;
    let mut total_inference_ms = 0.0;
    let mut total_postprocess_ms = 0.0;
    let mut total_plan_ms = 0.0;
    let mut total_render_ms = 0.0;
    let mut total_output_ms = 0.0;

    // GPU buffer pool
    let mut gpu_buffers: Vec<CudaSlice<u8>> = Vec::new();
    let mut cpu_output_buffers: Vec<Vec<u8>> = Vec::new(); // CPU buffers for output
    let mut overlay_buffers: Vec<Vec<u8>> = Vec::new(); // Overlay buffers

    while frame_count < frames_to_process {
        let pipeline_start = Instant::now();

        // Capture frame
        let capture_start = Instant::now();
        let raw_frame = match capture.get_frame()? {
            Some(frame) => frame,
            None => {
                std::thread::sleep(std::time::Duration::from_millis(16));
                continue;
            }
        };
        let capture_time = capture_start.elapsed();
        total_capture_ms += capture_time.as_secs_f64() * 1000.0;

        // Keep a CPU copy of the original frame for output
        let raw_frame_for_output = if matches!(raw_frame.data.loc, MemLoc::Gpu { device }) {
            // Need to copy GPU data back to CPU for output
            // Use a persistent CPU buffer to avoid reallocations
            let mut cpu_buffer = vec![0u8; raw_frame.data.len];
            
            // Copy from GPU to CPU using CUDA device
            let gpu_slice = unsafe {
                std::slice::from_raw_parts(raw_frame.data.ptr, raw_frame.data.len)
            };
            let temp_gpu_slice = cuda_device.htod_sync_copy(gpu_slice)?;
            cuda_device.dtoh_sync_copy_into(&temp_gpu_slice, &mut cpu_buffer)?;
            
            let packet = common_io::RawFramePacket {
                meta: raw_frame.meta.clone(),
                data: MemRef {
                    ptr: cpu_buffer.as_mut_ptr(),
                    len: cpu_buffer.len(),
                    stride: raw_frame.data.stride,
                    loc: MemLoc::Cpu,
                },
            };
            
            cpu_output_buffers.push(cpu_buffer);
            packet
        } else {
            raw_frame.clone()
        };

        // Print progress every 10 frames
        if frame_count % 10 == 0 {
            println!("🎬 Processing Frame #{}", frame_count);
        }

        // Copy CPU data to GPU if needed
        let h2d_start = Instant::now();
        let raw_frame_gpu = if matches!(raw_frame.data.loc, MemLoc::Cpu) {
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

            gpu_buffers.push(gpu_buffer);
            gpu_packet
        } else {
            raw_frame.clone()
        };
        let h2d_time = h2d_start.elapsed();
        total_h2d_ms += h2d_time.as_secs_f64() * 1000.0;

        // Preprocessing
        let preprocess_start = Instant::now();
        let tensor_packet = match preprocessor.process_checked(raw_frame_gpu) {
            Some(packet) => packet,
            None => continue,
        };
        let preprocess_time = preprocess_start.elapsed();
        total_preprocess_ms += preprocess_time.as_secs_f64() * 1000.0;

        // Inference V2
        let inference_start = Instant::now();
        let raw_detections = inference_stage.process(tensor_packet);
        let inference_time = inference_start.elapsed();
        total_inference_ms += inference_time.as_secs_f64() * 1000.0;

        // Postprocessing
        let postprocess_start = Instant::now();
        let detections = post_stage.process(raw_detections);
        let postprocess_time = postprocess_start.elapsed();
        total_postprocess_ms += postprocess_time.as_secs_f64() * 1000.0;

        // Overlay Planning
        let plan_start = Instant::now();
        let overlay_plan = plan_stage.process(detections);
        let plan_time = plan_start.elapsed();
        total_plan_ms += plan_time.as_secs_f64() * 1000.0;

        // Overlay Rendering (create BGRA8 overlay with graphics)
        let render_start = Instant::now();
        
        // Create BGRA overlay buffer (1920x1080)
        let overlay_width = 1920u32;
        let overlay_height = 1080u32;
        let overlay_stride = (overlay_width * 4) as usize; // 4 bytes per pixel (BGRA)
        let mut overlay_buffer = vec![0u8; (overlay_stride * overlay_height as usize)];
        
        // Draw detection boxes on overlay
        for op in &overlay_plan.ops {
            match op {
                common_io::DrawOp::Rect { xywh, thickness, color } => {
                    // Draw simple rectangle border
                    let (x, y, w, h) = *xywh;
                    draw_rect_bgra(&mut overlay_buffer, overlay_width, overlay_height, 
                                   x as i32, y as i32, w as i32, h as i32, 
                                   *color);
                }
                common_io::DrawOp::Label { anchor, text, font_px, color } => {
                    // Draw label background
                    let (x, y) = *anchor;
                    let label_width = text.len() as i32 * 8 + 10;
                    let label_height = 20;
                    draw_filled_rect_bgra(&mut overlay_buffer, overlay_width, overlay_height,
                                         x as i32, y as i32 - label_height, 
                                         label_width, label_height, 
                                         (0, 0, 0, 200)); // Semi-transparent black
                }
                _ => {}
            }
        }
        
        let overlay_frame = common_io::OverlayFramePacket {
            from: common_io::FrameMeta {
                source_id: raw_frame.meta.source_id,
                width: overlay_width,
                height: overlay_height,
                pixfmt: common_io::PixelFormat::BGRA8,
                colorspace: raw_frame.meta.colorspace,
                stride_bytes: overlay_stride as u32,
            crop_region: None,
                frame_idx: raw_frame.meta.frame_idx,
                pts_ns: raw_frame.meta.pts_ns,
                t_capture_ns: raw_frame.meta.t_capture_ns,
            },
            argb: MemRef {
                ptr: overlay_buffer.as_mut_ptr(),
                len: overlay_buffer.len(),
                stride: overlay_stride,
                loc: MemLoc::Cpu,
            },
            stride: overlay_stride,
        };
        
        let render_time = render_start.elapsed();
        total_render_ms += render_time.as_secs_f64() * 1000.0;
        
        // Keep overlay buffer alive
        overlay_buffers.push(overlay_buffer);

        // Output to DeckLink
        let output_start = Instant::now();
        output.submit_keying_frames(&raw_frame_for_output, &overlay_frame)?;
        let output_time = output_start.elapsed();
        total_output_ms += output_time.as_secs_f64() * 1000.0;

        let pipeline_time = pipeline_start.elapsed();
        let pipeline_ms = pipeline_time.as_secs_f64() * 1000.0;
        total_latency_ms += pipeline_ms;

        frame_count += 1;

        // Print detailed stats every 30 frames
        if frame_count % 30 == 0 {
            let avg_e2e = total_latency_ms / frame_count as f64;
            println!("  ⏱️  Frame {}: {:.2}ms E2E ({:.1} FPS)", 
                     frame_count, pipeline_ms, 1000.0 / avg_e2e);
        }
    }

    // Stop playback
    println!();
    println!("Stopping playback...");
    output.stop_playback()?;

    // Summary
    println!();
    println!("════════════════════════════════════════════════════════════");
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║             FULL PIPELINE SUMMARY                        ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║ Frames Processed:  {:3}                                  ║",
        frames_to_process
    );
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Average Latency Breakdown:                               ║");
    println!(
        "║   Capture:        {:6.2} ms                          ║",
        total_capture_ms / frames_to_process as f64
    );
    println!(
        "║   H2D Transfer:   {:6.2} ms                          ║",
        total_h2d_ms / frames_to_process as f64
    );
    println!(
        "║   Preprocess:     {:6.2} ms                          ║",
        total_preprocess_ms / frames_to_process as f64
    );
    println!(
        "║   Inference V2:   {:6.2} ms                          ║",
        total_inference_ms / frames_to_process as f64
    );
    println!(
        "║   Postprocess:    {:6.2} ms                          ║",
        total_postprocess_ms / frames_to_process as f64
    );
    println!(
        "║   Overlay Plan:   {:6.2} ms                          ║",
        total_plan_ms / frames_to_process as f64
    );
    println!(
        "║   Overlay Render: {:6.2} ms                          ║",
        total_render_ms / frames_to_process as f64
    );
    println!(
        "║   DeckLink Out:   {:6.2} ms                          ║",
        total_output_ms / frames_to_process as f64
    );
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║   TOTAL E2E:      {:6.2} ms                         ║",
        total_latency_ms / frames_to_process as f64
    );
    println!(
        "║   Average FPS:    {:6.1}                             ║",
        1000.0 * frames_to_process as f64 / total_latency_ms
    );
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Pipeline Components:                                     ║");
    println!("║   ✓ DeckLink Capture (YUV422_8)                          ║");
    println!("║   ✓ CUDA Preprocessing (512×512 FP16)                    ║");
    println!("║   ✓ TensorRT Inference V2                                ║");
    println!("║   ✓ Postprocessing (NMS + SORT Tracking)                 ║");
    println!("║   ✓ Overlay Planning & Rendering                         ║");
    println!("║   ✓ DeckLink Output (Internal Keying)                    ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Status: Live output successfully displayed! ✅           ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("✅ Full pipeline test completed successfully!");
    println!();

    Ok(())
}
