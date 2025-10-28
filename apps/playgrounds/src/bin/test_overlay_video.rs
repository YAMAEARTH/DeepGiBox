use anyhow::Result;
use common_io::{MemLoc, MemRef, PixelFormat, RawDetectionsPacket, RawFramePacket, Stage};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use decklink_input::capture::CaptureSession;
use image::{imageops::FilterType, ImageBuffer, RgbImage, RgbaImage};
use inference_v2::TrtInferenceStage;
use overlay_plan::PlanStage;
use overlay_render::RenderStage;
use preprocess_cuda::Preprocessor;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::slice;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

const ENGINE_PATH: &str = "trt-shim/test_rust/assets/v7_optimized_YOLOv5.engine";
const LIB_PATH: &str = "trt-shim/build/libtrt_shim.so";
const INPUT_WIDTH: u32 = 512;
const INPUT_HEIGHT: u32 = 512;
const FRAMES_TO_CAPTURE: usize = 10;
const OUTPUT_DIR: &str = "output/video_frames";
const VIDEO_PATH: &str = "output/overlay_sequence.mp4";

#[cfg(not(all(
    feature = "decklink_input",
    feature = "preprocess_cuda",
    feature = "overlay_plan",
    feature = "overlay_render"
)))]
compile_error!(
    "`test_overlay_video` requires features decklink_input, preprocess_cuda, overlay_plan, overlay_render.\n\
     Run with: cargo run -p playgrounds --bin test_overlay_video --features decklink_input,preprocess_cuda,overlay_plan,overlay_render"
);

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     Overlay Video Test (DeckLink → CUDA → TensorRT)     ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    fs::create_dir_all(OUTPUT_DIR)?;

    println!("🚀 Initializing CUDA device...");
    let cuda_device = CudaDevice::new(0)?;
    println!("   ✓ CUDA device 0 ({}) ready\n", cuda_device.name()?);

    println!("📹 Opening DeckLink capture (device 0)...");
    let mut capture = CaptureSession::open(0)?;
    println!("   ✓ Capture session ready\n");

    println!("⚙️  Initializing CUDA preprocessor...");
    let mut preprocessor = Preprocessor::new((INPUT_WIDTH, INPUT_HEIGHT), true, 0)?;
    println!("   ✓ Preprocessor configured (512×512 FP16)\n");

    println!("🔧 Initializing TensorRT inference stage...");
    let mut inference_stage = TrtInferenceStage::new(ENGINE_PATH, LIB_PATH)
        .map_err(|e| anyhow::anyhow!("TensorRT init failed: {}", e))?;
    println!("   ✓ TensorRT ready\n");

    let mut post_stage = postprocess::from_path("")?;
    let mut plan_stage = PlanStage {
        enable_full_ui: true,
    };
    let mut render_stage = RenderStage {};

    let mut saved_frames = Vec::new();

    for frame_idx in 0..FRAMES_TO_CAPTURE {
        println!(
            "\n════════ Frame {:03}/{} ════════",
            frame_idx + 1,
            FRAMES_TO_CAPTURE
        );

        let (base_image, tensor_packet) = loop {
            let raw_frame = capture_frame(&mut capture)?;
            println!(
                "   → Captured frame #{}, {}x{} {:?}",
                raw_frame.meta.frame_idx,
                raw_frame.meta.width,
                raw_frame.meta.height,
                raw_frame.meta.pixfmt
            );

            let base_image = frame_to_letterboxed_image(&raw_frame, INPUT_WIDTH, INPUT_HEIGHT);
            let (gpu_frame, upload_buffer) = move_frame_to_gpu(raw_frame, cuda_device.clone())?;
            match preprocessor.process_checked(gpu_frame) {
                Some(tensor) => {
                    drop(upload_buffer);
                    break (base_image, tensor);
                }
                None => {
                    drop(upload_buffer);
                    println!("   ⚠️  Skipping init frame...");
                    continue;
                }
            }
        };

        let raw_detections: RawDetectionsPacket = inference_stage.process(tensor_packet);
        let detections_packet = post_stage.process(raw_detections);
        println!("   ✓ Detections: {}", detections_packet.items.len());

        let overlay_plan = plan_stage.process(detections_packet.clone());
        let overlay_frame = render_stage.process(overlay_plan);
        let overlay_image = overlay_frame_to_image(overlay_frame);

        let composite = if let Some(base) = &base_image {
            blend_with_base(base, &overlay_image)
        } else {
            overlay_image.clone()
        };

        let frame_path = Path::new(OUTPUT_DIR).join(format!("frame_{:04}.png", frame_idx + 1));
        composite.save(&frame_path)?;
        saved_frames.push(frame_path);
    }

    if saved_frames.is_empty() {
        println!("⚠️  No frames captured; skipping video generation.");
        return Ok(());
    }

    println!("\n🎬 Encoding video with ffmpeg...");
    if ffmpeg_available() {
        let status = Command::new("ffmpeg")
            .args([
                "-y",
                "-framerate",
                "30",
                "-i",
                &format!("{OUTPUT_DIR}/frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                VIDEO_PATH,
            ])
            .status()?;
        if status.success() {
            println!("   ✓ Video saved to {}", VIDEO_PATH);
        } else {
            println!(
                "⚠️  ffmpeg exited with status {} — PNG sequence remains in {}",
                status, OUTPUT_DIR
            );
        }
    } else {
        println!(
            "⚠️  ffmpeg not found. PNG sequence available at {}",
            OUTPUT_DIR
        );
        println!(
            "    Run manually:\n    ffmpeg -framerate 30 -i {}/frame_%04d.png -c:v libx264 {}",
            OUTPUT_DIR, VIDEO_PATH
        );
    }

    println!("\nDone.");
    Ok(())
}

fn capture_frame(capture: &mut CaptureSession) -> Result<RawFramePacket> {
    loop {
        match capture.get_frame()? {
            Some(frame) => return Ok(frame),
            None => thread::sleep(Duration::from_millis(16)),
        }
    }
}

fn move_frame_to_gpu(
    raw_frame: RawFramePacket,
    cuda_device: Arc<CudaDevice>,
) -> Result<(RawFramePacket, Option<CudaSlice<u8>>)> {
    if matches!(raw_frame.data.loc, MemLoc::Gpu { .. }) {
        return Ok((raw_frame, None));
    }

    let cpu_slice = unsafe { slice::from_raw_parts(raw_frame.data.ptr, raw_frame.data.len) };
    let gpu_buffer = cuda_device.htod_sync_copy(cpu_slice)?;
    let gpu_ptr = *gpu_buffer.device_ptr() as *mut u8;
    let gpu_packet = RawFramePacket {
        meta: raw_frame.meta,
        data: MemRef {
            ptr: gpu_ptr,
            len: raw_frame.data.len,
            stride: raw_frame.data.stride,
            loc: MemLoc::Gpu { device: 0 },
        },
    };

    Ok((gpu_packet, Some(gpu_buffer)))
}

fn frame_to_letterboxed_image(
    frame: &RawFramePacket,
    target_w: u32,
    target_h: u32,
) -> Option<RgbImage> {
    if !matches!(frame.data.loc, MemLoc::Cpu) {
        println!("   ⚠️  Frame data on GPU; skipping base image blend");
        return None;
    }

    if !matches!(frame.meta.pixfmt, PixelFormat::BGRA8) {
        println!(
            "   ⚠️  Unsupported pixel format {:?}; skipping base image blend",
            frame.meta.pixfmt
        );
        return None;
    }

    let width = frame.meta.width as usize;
    let height = frame.meta.height as usize;
    let stride = frame.meta.stride_bytes as usize;
    let data = unsafe { slice::from_raw_parts(frame.data.ptr, frame.data.len) };

    let mut rgb = RgbImage::new(frame.meta.width, frame.meta.height);
    for y in 0..height {
        let row = &data[y * stride..y * stride + width * 4];
        for x in 0..width {
            let idx = x * 4;
            let b = row[idx];
            let g = row[idx + 1];
            let r = row[idx + 2];
            rgb.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    let scale =
        (target_w as f32 / frame.meta.width as f32).min(target_h as f32 / frame.meta.height as f32);
    let new_w = (frame.meta.width as f32 * scale) as u32;
    let new_h = (frame.meta.height as f32 * scale) as u32;
    let resized = image::imageops::resize(&rgb, new_w, new_h, FilterType::Lanczos3);

    let pad_x = ((target_w - new_w) / 2) as i64;
    let pad_y = ((target_h - new_h) / 2) as i64;
    let mut letterboxed = ImageBuffer::from_pixel(target_w, target_h, image::Rgb([0u8, 0u8, 0u8]));
    image::imageops::overlay(&mut letterboxed, &resized, pad_x, pad_y);
    Some(letterboxed)
}

fn overlay_frame_to_image(frame: common_io::OverlayFramePacket) -> RgbaImage {
    let width = frame.from.width as usize;
    let height = frame.from.height as usize;
    let stride = frame.stride;

    let argb = unsafe { Vec::from_raw_parts(frame.argb.ptr, frame.argb.len, frame.argb.len) };
    let mut rgba = vec![0u8; width * height * 4];
    for y in 0..height {
        let src_row = &argb[y * stride..y * stride + width * 4];
        let dst_row = &mut rgba[y * width * 4..(y + 1) * width * 4];
        for (dst, src) in dst_row.chunks_exact_mut(4).zip(src_row.chunks_exact(4)) {
            let (a, r, g, b) = (src[0], src[1], src[2], src[3]);
            dst.copy_from_slice(&[r, g, b, a]);
        }
    }

    RgbaImage::from_raw(frame.from.width, frame.from.height, rgba)
        .expect("overlay buffer dimensions mismatch")
}

fn blend_with_base(base: &RgbImage, overlay: &RgbaImage) -> RgbaImage {
    assert_eq!(base.width(), overlay.width());
    assert_eq!(base.height(), overlay.height());
    let mut output = RgbaImage::new(base.width(), base.height());
    for (x, y, pixel) in output.enumerate_pixels_mut() {
        let base_px = base.get_pixel(x, y).0;
        let overlay_px = overlay.get_pixel(x, y).0;
        let alpha = overlay_px[3] as f32 / 255.0;
        let inv_alpha = 1.0 - alpha;
        let blended = [
            (overlay_px[0] as f32 * alpha + base_px[0] as f32 * inv_alpha).round() as u8,
            (overlay_px[1] as f32 * alpha + base_px[1] as f32 * inv_alpha).round() as u8,
            (overlay_px[2] as f32 * alpha + base_px[2] as f32 * inv_alpha).round() as u8,
            255,
        ];
        *pixel = image::Rgba(blended);
    }
    output
}

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
