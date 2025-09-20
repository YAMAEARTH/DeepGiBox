#[cfg(all(
    ocvrs_has_cuda,
    ocvrs_has_module_cudawarping,
    ocvrs_has_module_cudaimgproc
))]
use opencv::{core::GpuMat, cudaimgproc, cudawarping};
use opencv::{
    core::{self, Mat, Rect, Scalar, Size},
    dnn, imgcodecs, imgproc,
    prelude::*,
    videoio::{
        VideoCapture, CAP_ANY, CAP_FFMPEG, CAP_GSTREAMER, CAP_PROP_BUFFERSIZE, CAP_PROP_FPS,
        CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH,
    },
};
use std::time::{Duration, Instant};

/// คืนค่าเป็น (blob สำหรับโมเดล, bgr_512 สำหรับ preview)
fn preprocess_to_nchw_with_dnn(
    bgr_4k: &Mat,
    pan_x: i32,
    pan_y: i32,
    zoom: f32,
) -> opencv::Result<(Mat, Mat)> {
    // ป้องกันค่า zoom แปลก ๆ
    let zoom = if zoom <= 0.0 { 1.0 } else { zoom };

    // เลือก ROI ตาม pan/zoom
    let (w, h) = (bgr_4k.cols(), bgr_4k.rows());
    let tw = (w as f32 / zoom).round() as i32;
    let th = (h as f32 / zoom).round() as i32;
    let x = pan_x.clamp(0, (w - tw).max(0));
    let y = pan_y.clamp(0, (h - th).max(0));
    let roi = Rect::new(x, y, tw.max(1), th.max(1));

    if let Some(result) = preprocess_with_cuda(bgr_4k, roi)? {
        return Ok(result);
    }

    preprocess_with_cpu(bgr_4k, roi)
}

fn preprocess_with_cpu(bgr_4k: &Mat, roi: Rect) -> opencv::Result<(Mat, Mat)> {
    let bgr_roi = Mat::roi(bgr_4k, roi)?;

    // resize → 512x512 (ยังคงเป็น BGR) — ใช้ตัวนี้เป็น preview ได้เลย
    let mut bgr_512 = Mat::default();
    imgproc::resize(
        &bgr_roi,
        &mut bgr_512,
        Size::new(512, 512),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    // ได้ NCHW โดยตรง: normalize(1/255), swapRB=true (BGR→RGB), ไม่ crop, float32
    let blob = dnn::blob_from_image(
        &bgr_512,
        1.0 / 255.0,
        Size::new(512, 512),
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        true,  // swapRB => BGR->RGB
        false, // crop
        core::CV_32F,
    )?;

    Ok((blob, bgr_512))
}

#[cfg(all(
    ocvrs_has_cuda,
    ocvrs_has_module_cudawarping,
    ocvrs_has_module_cudaimgproc
))]
fn preprocess_with_cuda(bgr_4k: &Mat, roi: Rect) -> opencv::Result<Option<(Mat, Mat)>> {
    let device_count = core::get_cuda_enabled_device_count()?;
    if device_count <= 0 {
        return Ok(None);
    }

    let mut gpu_full = GpuMat::new_def()?;
    gpu_full.upload(bgr_4k)?;

    let roi_gpu = gpu_full.roi(roi)?;
    let mut gpu_resized = GpuMat::new_def()?;
    cudawarping::resize_def(&roi_gpu, &mut gpu_resized, Size::new(512, 512))?;

    let mut gpu_rgb = GpuMat::new_def()?;
    cudaimgproc::cvt_color_def(&gpu_resized, &mut gpu_rgb, imgproc::COLOR_BGR2RGB)?;

    let mut gpu_float = GpuMat::new_def()?;
    gpu_rgb.convert_to_def(&mut gpu_float, core::CV_32F, 1.0 / 255.0)?;

    let mut bgr_512 = Mat::default();
    gpu_resized.download(&mut bgr_512)?;

    let blob = dnn::blob_from_image(
        &gpu_float,
        1.0,
        Size::new(512, 512),
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        false,
        false,
        core::CV_32F,
    )?;

    println!("ใช้ CUDA ครบขั้นตอน preprocess (device count = {device_count})");

    Ok(Some((blob, bgr_512)))
}

#[cfg(not(all(
    ocvrs_has_cuda,
    ocvrs_has_module_cudawarping,
    ocvrs_has_module_cudaimgproc
)))]
fn preprocess_with_cuda(_: &Mat, _: Rect) -> opencv::Result<Option<(Mat, Mat)>> {
    Ok(None)
}

fn main() -> opencv::Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("--decklink") => {
            let source = args.next().unwrap_or_else(|| "decklink:0".to_string());
            let width = args.next().and_then(|v| v.parse::<i32>().ok());
            let height = args.next().and_then(|v| v.parse::<i32>().ok());
            let fps = args.next().and_then(|v| v.parse::<f64>().ok());
            run_decklink_capture(&source, width, height, fps)
        }
        Some(path) => preprocess_single_image(path),
        None => preprocess_single_image("frame_0001_4k.png"),
    }
}

fn preprocess_single_image(path: &str) -> opencv::Result<()> {
    let bgr_4k = imgcodecs::imread(path, imgcodecs::IMREAD_COLOR)?;
    if bgr_4k.empty() {
        panic!("ไม่พบภาพ: {}", path);
    }

    let t0 = Instant::now();
    let (blob, bgr_512) = preprocess_to_nchw_with_dnn(&bgr_4k, 0, 0, 1.0)?;
    let dt = t0.elapsed();
    println!(
        "Preprocess OK, time = {:.3} ms, shape ~= (1,3,512,512)",
        dt.as_secs_f64() * 1000.0
    );

    let params = core::Vector::<i32>::new();
    imgcodecs::imwrite("preview_512.png", &bgr_512, &params)?;

    let _ = blob;

    Ok(())
}

fn run_decklink_capture(
    source: &str,
    width: Option<i32>,
    height: Option<i32>,
    fps: Option<f64>,
) -> opencv::Result<()> {
    let lower_source = source.to_ascii_lowercase();
    let is_decklink = source.starts_with("decklink");
    let is_video_file = lower_source.ends_with(".mp4")
        || lower_source.ends_with(".mov")
        || lower_source.ends_with(".mkv")
        || lower_source.ends_with(".avi");

    let normalized_path = if is_decklink {
        source.to_string()
    } else {
        std::fs::canonicalize(source)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| source.to_string())
    };

    let mut backend = if is_decklink || is_video_file {
        "FFmpeg"
    } else {
        "Auto"
    };

    let mut cap = if is_decklink {
        VideoCapture::from_file(source, CAP_FFMPEG)?
    } else if is_video_file {
        VideoCapture::from_file(&normalized_path, CAP_FFMPEG)?
    } else {
        VideoCapture::from_file(&normalized_path, CAP_ANY)?
    };

    if !cap.is_opened()? {
        if is_video_file {
            println!(
                "เปิดด้วย FFMPEG ไม่สำเร็จ → ลอง GStreamer pipeline สำหรับ {}",
                source
            );
            let pipeline = format!(
                "filesrc location=\"{}\" ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink",
                normalized_path.replace('"', "\\\"")
            );
            cap = VideoCapture::from_file(&pipeline, CAP_GSTREAMER)?;
            backend = "GStreamer";
        }
    }

    if !cap.is_opened()? {
        return Err(opencv::Error::new(
            opencv::core::StsError,
            format!("เปิดสตรีมไม่สำเร็จ: {}", source),
        ));
    }

    if let Some(w) = width {
        let _ = cap.set(CAP_PROP_FRAME_WIDTH, w as f64);
    }
    if let Some(h) = height {
        let _ = cap.set(CAP_PROP_FRAME_HEIGHT, h as f64);
    }
    if let Some(fps) = fps {
        let _ = cap.set(CAP_PROP_FPS, fps);
    }
    let _ = cap.set(CAP_PROP_BUFFERSIZE, 1.0);

    println!("เริ่มอ่านจาก backend {}: {}", backend, source);

    let mut frame = Mat::default();
    let mut frame_idx: u64 = 0;
    let mut no_frame_counter = 0u32;
    let mut last_warn = Instant::now();
    loop {
        let grab_start = Instant::now();
        if !cap.read(&mut frame)? || frame.empty() {
            no_frame_counter = no_frame_counter.saturating_add(1);
            if no_frame_counter == 60 || last_warn.elapsed() >= Duration::from_secs(5) {
                println!(
                    "ไม่มีเฟรมจากแหล่งสัญญาณ ({} ครั้ง) — ตรวจสอบไฟล์/สตรีมยังเล่นได้หรือไม่",
                    no_frame_counter
                );
                last_warn = Instant::now();
            }
            if !is_decklink && no_frame_counter >= 120 {
                println!("สิ้นสุดวิดีโอหรือไม่มีเฟรมใหม่แล้ว → จบการประมวลผล");
                break;
            }
            continue;
        }
        no_frame_counter = 0;
        last_warn = Instant::now();
        let capture_ms = grab_start.elapsed().as_secs_f64() * 1000.0;

        let preprocess_start = Instant::now();
        let (blob, _preview) = preprocess_to_nchw_with_dnn(&frame, 0, 0, 1.0)?;
        let preprocess_ms = preprocess_start.elapsed().as_secs_f64() * 1000.0;

        frame_idx += 1;
        if frame_idx == 1 || frame_idx % 30 == 0 {
            println!(
                "frame #{frame_idx}: capture={capture_ms:.2} ms, preprocess={preprocess_ms:.2} ms"
            );
        }

        // ต่อ blob ให้โมเดลได้เลย
        let _ = blob;
    }

    Ok(())
}
