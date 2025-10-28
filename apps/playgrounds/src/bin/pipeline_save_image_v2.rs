//! Pipeline: Capture → Overlay → Save as Image with Labels
//!
//! This pipeline captures frames, runs inference, creates overlays with text labels,
//! and saves the result as PNG images.

use anyhow::{anyhow, Result};
use common_io::{MemLoc, MemRef, Stage};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use decklink_input::capture::CaptureSession;
use inference_v2::TrtInferenceStage;
use overlay_plan::PlanStage;
use postprocess;
use preprocess_cuda::Preprocessor;
use std::time::Instant;
use image::{ImageBuffer, Rgba, RgbaImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_filled_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use rusttype::{Font, Scale};

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  PIPELINE: CAPTURE → INFERENCE → OVERLAY → SAVE         ║");
    println!("║  Save detection results with labels as PNG images       ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Create output directory
    std::fs::create_dir_all("output/detections")?;
    println!("📁 Created output directory: output/detections");
    println!();

    // Load font for text rendering
    let font_data = include_bytes!("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf");
    let font = Font::try_from_bytes(font_data as &[u8])
        .ok_or_else(|| anyhow!("Failed to load font"))?;
    println!("🔤 Loaded DejaVu Sans Bold font");
    println!();

    // 1. List available DeckLink devices
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

    // 2. Initialize DeckLink capture
    println!("📹 Step 1: Initialize DeckLink Capture");
    let mut capture = CaptureSession::open(0)?;
    println!("  ✓ Opened DeckLink device 0");
    println!();

    // 3. Initialize Preprocessor
    println!("⚙️  Step 2: Initialize Preprocessor");
    let mut preprocessor = Preprocessor::new(
        (512, 512),
        true,
        0,
    )?;
    println!("  ✓ Preprocessor ready (512x512, FP16, GPU 0)");
    println!();

    // 4. Initialize CUDA device
    println!("🔧 Step 3: Initialize CUDA Device");
    let cuda_device = CudaDevice::new(0)?;
    println!("  ✓ CUDA device initialized");
    println!();

    // 5. Initialize TensorRT Inference V2
    println!("🧠 Step 4: Initialize TensorRT Inference V2");
    let engine_path = "configs/model/v7_optimized_YOLOv5.engine";
    let lib_path = "trt-shim/build/libtrt_shim.so";

    if !std::path::Path::new(engine_path).exists() {
        return Err(anyhow!("TensorRT engine not found: {}", engine_path));
    }
    if !std::path::Path::new(lib_path).exists() {
        return Err(anyhow!("TRT shim library not found: {}", lib_path));
    }

    let mut inference_stage = TrtInferenceStage::new(engine_path, lib_path)
        .map_err(|e| anyhow!(e))?;
    println!("  ✓ TensorRT Inference V2 loaded");
    println!();

    // 6. Initialize Postprocessing
    println!("🎯 Step 5: Initialize Postprocessing");
    let mut post_stage = postprocess::from_path("")?.with_sort_tracking(30, 0.3, 0.3);
    println!("  ✓ Postprocessing ready (NMS + SORT Tracking)");
    println!();

    // 7. Initialize Overlay planning
    println!("🎨 Step 6: Initialize Overlay Planning");
    let mut plan_stage = PlanStage {
        enable_full_ui: true,
    };
    println!("  ✓ Overlay planning ready (Full UI enabled)");
    println!();

    // 8. Process frames and save images
    println!("🎬 Step 7: Processing Frames and Creating Overlays...");
    println!("════════════════════════════════════════════════════════════");
    println!();

    let frames_to_save = 5;
    let mut frame_count = 0;
    let mut saved_count = 0;
    let mut gpu_buffers: Vec<CudaSlice<u8>> = Vec::new();

    while saved_count < frames_to_save {
        // Capture frame
        let raw_frame = match capture.get_frame()? {
            Some(frame) => frame,
            None => {
                std::thread::sleep(std::time::Duration::from_millis(16));
                continue;
            }
        };

        frame_count += 1;
        println!("🎬 Processing Frame #{}", frame_count);

        // Store original frame dimensions
        let orig_width = raw_frame.meta.width;
        let orig_height = raw_frame.meta.height;
        let orig_stride = raw_frame.meta.stride_bytes as usize;

        // Copy frame data to CPU for image conversion
        let cpu_frame_data = if matches!(raw_frame.data.loc, MemLoc::Gpu { .. }) {
            let gpu_slice = unsafe {
                std::slice::from_raw_parts(raw_frame.data.ptr, raw_frame.data.len)
            };
            let temp_gpu_slice = cuda_device.htod_sync_copy(gpu_slice)?;
            let mut cpu_buffer = vec![0u8; raw_frame.data.len];
            cuda_device.dtoh_sync_copy_into(&temp_gpu_slice, &mut cpu_buffer)?;
            cpu_buffer
        } else {
            let cpu_slice = unsafe {
                std::slice::from_raw_parts(raw_frame.data.ptr, raw_frame.data.len)
            };
            cpu_slice.to_vec()
        };

        // Copy to GPU for inference
        let raw_frame_gpu = if matches!(raw_frame.data.loc, MemLoc::Cpu) {
            let cpu_data = unsafe {
                std::slice::from_raw_parts(raw_frame.data.ptr, raw_frame.data.len)
            };
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

        // Preprocessing
        let tensor_packet = match preprocessor.process_checked(raw_frame_gpu) {
            Some(packet) => packet,
            None => {
                println!("  ⚠️  Skipping frame (waiting for stable resolution)");
                continue;
            }
        };

        // Inference
        let inference_start = Instant::now();
        let raw_detections = inference_stage.process(tensor_packet);
        let inference_time = inference_start.elapsed();

        // Postprocessing
        let detections = post_stage.process(raw_detections);
        println!("  ✓ Detected {} objects", detections.items.len());

        // Display detections
        if !detections.items.is_empty() {
            println!("  📊 Detections:");
            for (i, det) in detections.items.iter().take(5).enumerate() {
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
                    "    {}. {} {:.2}{} @ ({:.0},{:.0}) {:.0}×{:.0}",
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
            if detections.items.len() > 5 {
                println!("    ... and {} more", detections.items.len() - 5);
            }
        }

        // Overlay Planning
        let overlay_plan = plan_stage.process(detections);
        println!("  ✓ Created {} overlay operations", overlay_plan.ops.len());

        // Convert YUV422 to RGB
        let rgb_data = yuv422_to_rgb(&cpu_frame_data, orig_width, orig_height, orig_stride);
        
        // Create image from RGB data
        let mut img: RgbaImage = ImageBuffer::new(orig_width, orig_height);
        for y in 0..orig_height {
            for x in 0..orig_width {
                let rgb_idx = ((y * orig_width + x) * 3) as usize;
                if rgb_idx + 2 < rgb_data.len() {
                    img.put_pixel(x, y, Rgba([
                        rgb_data[rgb_idx],
                        rgb_data[rgb_idx + 1],
                        rgb_data[rgb_idx + 2],
                        255,
                    ]));
                }
            }
        }

        // Draw overlays on image using imageproc
        for op in &overlay_plan.ops {
            match op {
                common_io::DrawOp::Rect { xywh, thickness, color } => {
                    let (x, y, w, h) = *xywh;
                    let (a, r, g, b) = *color;
                    
                    if x >= 0.0 && y >= 0.0 && w > 0.0 && h > 0.0 {
                        let rect = Rect::at(x as i32, y as i32)
                            .of_size(w as u32, h as u32);
                        
                        // Draw multiple rectangles for thickness
                        for t in 0..*thickness {
                            if let Some(inner_rect) = rect.shrink(t as u32, t as u32) {
                                draw_hollow_rect_mut(&mut img, inner_rect, Rgba([r, g, b, a]));
                            }
                        }
                    }
                }
                common_io::DrawOp::Label { anchor, text, font_px, color } => {
                    let (x, y) = *anchor;
                    let (a, r, g, b) = *color;
                    
                    if x >= 0.0 && y >= 20.0 {
                        let scale = Scale::uniform(*font_px as f32);
                        
                        // Draw background for label
                        let text_width = text.len() as u32 * (*font_px as u32 / 2) + 10;
                        let text_height = *font_px as u32 + 4;
                        let bg_rect = Rect::at(x as i32, (y - text_height as f32) as i32)
                            .of_size(text_width, text_height);
                        draw_filled_rect_mut(&mut img, bg_rect, Rgba([0, 0, 0, 200]));
                        
                        // Draw text
                        draw_text_mut(
                            &mut img,
                            Rgba([r, g, b, a]),
                            x as i32 + 5,
                            (y - text_height as f32 + 2.0) as i32,
                            scale,
                            &font,
                            text,
                        );
                    }
                }
                common_io::DrawOp::Poly { pts, thickness, color } => {
                    // Draw polylines (for corners)
                    let (a, r, g, b) = *color;
                    if pts.len() >= 2 {
                        for i in 0..pts.len()-1 {
                            let (x1, y1) = pts[i];
                            let (x2, y2) = pts[i+1];
                            draw_line_mut(&mut img, x1, y1, x2, y2, Rgba([r, g, b, a]), *thickness);
                        }
                    }
                }
                _ => {}
            }
        }

        // Save image
        let output_path = format!("output/detections/frame_{:04}.png", saved_count);
        img.save(&output_path)?;
        
        println!("  💾 Saved: {}", output_path);
        println!("  ⏱️  Inference: {:.2}ms", inference_time.as_secs_f64() * 1000.0);
        println!();

        saved_count += 1;
    }

    println!("════════════════════════════════════════════════════════════");
    println!("✅ Successfully saved {} images to output/detections/", saved_count);
    println!();
    println!("💡 You can view the images with:");
    println!("   eog output/detections/*.png");
    println!("   or");
    println!("   feh output/detections/*.png");
    println!();

    Ok(())
}

// Helper function to convert YUV422 (UYVY) to RGB
fn yuv422_to_rgb(yuv_data: &[u8], width: u32, height: u32, stride: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    
    for y in 0..height {
        let row_offset = (y as usize) * stride;
        for x in 0..(width / 2) {
            let base = row_offset + (x as usize) * 4;
            if base + 3 >= yuv_data.len() {
                continue;
            }
            
            let u = yuv_data[base] as i32;
            let y0 = yuv_data[base + 1] as i32;
            let v = yuv_data[base + 2] as i32;
            let y1 = yuv_data[base + 3] as i32;
            
            // Convert first pixel
            let (r0, g0, b0) = yuv_to_rgb(y0, u, v);
            let rgb_idx0 = ((y * width + x * 2) * 3) as usize;
            if rgb_idx0 + 2 < rgb.len() {
                rgb[rgb_idx0] = r0;
                rgb[rgb_idx0 + 1] = g0;
                rgb[rgb_idx0 + 2] = b0;
            }
            
            // Convert second pixel
            let (r1, g1, b1) = yuv_to_rgb(y1, u, v);
            let rgb_idx1 = ((y * width + x * 2 + 1) * 3) as usize;
            if rgb_idx1 + 2 < rgb.len() {
                rgb[rgb_idx1] = r1;
                rgb[rgb_idx1 + 1] = g1;
                rgb[rgb_idx1 + 2] = b1;
            }
        }
    }
    
    rgb
}

fn yuv_to_rgb(y: i32, u: i32, v: i32) -> (u8, u8, u8) {
    let c = y - 16;
    let d = u - 128;
    let e = v - 128;
    
    let r = ((298 * c + 409 * e + 128) >> 8).clamp(0, 255) as u8;
    let g = ((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255) as u8;
    let b = ((298 * c + 516 * d + 128) >> 8).clamp(0, 255) as u8;
    
    (r, g, b)
}

// Simple line drawing using Bresenham's algorithm
fn draw_line_mut(img: &mut RgbaImage, x1: f32, y1: f32, x2: f32, y2: f32, 
                 color: Rgba<u8>, thickness: u8) {
    let (x1, y1, x2, y2) = (x1 as i32, y1 as i32, x2 as i32, y2 as i32);
    let dx = (x2 - x1).abs();
    let dy = (y2 - y1).abs();
    let sx = if x1 < x2 { 1 } else { -1 };
    let sy = if y1 < y2 { 1 } else { -1 };
    let mut err = dx - dy;
    let mut x = x1;
    let mut y = y1;

    loop {
        // Draw thick line by drawing multiple pixels
        for ty in -(thickness as i32 / 2)..=(thickness as i32 / 2) {
            for tx in -(thickness as i32 / 2)..=(thickness as i32 / 2) {
                let px = x + tx;
                let py = y + ty;
                if px >= 0 && px < img.width() as i32 && py >= 0 && py < img.height() as i32 {
                    img.put_pixel(px as u32, py as u32, color);
                }
            }
        }

        if x == x2 && y == y2 {
            break;
        }

        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}
