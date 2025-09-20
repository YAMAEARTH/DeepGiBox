#[cfg(all(ocvrs_has_cuda, ocvrs_has_module_cudawarping))]
use opencv::{core::GpuMat, cudawarping};
use opencv::{
    core::{self, Mat, Rect, Scalar, Size},
    dnn, imgcodecs, imgproc,
    prelude::*,
};
use std::time::Instant;

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

#[cfg(all(ocvrs_has_cuda, ocvrs_has_module_cudawarping))]
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

    let mut bgr_512 = Mat::default();
    gpu_resized.download(&mut bgr_512)?;

    println!("ใช้ CUDA สำหรับ resize (device count = {device_count})");

    let blob = dnn::blob_from_image(
        &bgr_512,
        1.0 / 255.0,
        Size::new(512, 512),
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        true,
        false,
        core::CV_32F,
    )?;

    Ok(Some((blob, bgr_512)))
}

#[cfg(not(all(ocvrs_has_cuda, ocvrs_has_module_cudawarping)))]
fn preprocess_with_cuda(_: &Mat, _: Rect) -> opencv::Result<Option<(Mat, Mat)>> {
    Ok(None)
}

fn main() -> opencv::Result<()> {
    let img_path = std::env::args()
        .nth(1)
        .unwrap_or("frame_0001_4k.png".into());
    let bgr_4k = imgcodecs::imread(&img_path, imgcodecs::IMREAD_COLOR)?;
    if bgr_4k.empty() {
        panic!("ไม่พบภาพ: {}", img_path);
    }

    let t0 = Instant::now();
    // เลือก pan/zoom ตามต้องการ เช่น (0,0,1.0)
    let (blob, bgr_512) = preprocess_to_nchw_with_dnn(&bgr_4k, 0, 0, 1.0)?;
    let dt = t0.elapsed();
    println!(
        "Preprocess OK, time = {:.3} ms, shape ~= (1,3,512,512)",
        dt.as_secs_f64() * 1000.0
    );

    // --- Preview สำหรับดูด้วยตา (ส่ง BGR เข้า imwrite → สีตรง) ---
    let params = core::Vector::<i32>::new();
    // ตัวอย่าง: เซฟ JPEG คุณภาพ 95 (ถ้าบันทึกเป็น PNG ก็ไม่ต้องตั้งค่า)
    // params.push(imgcodecs::IMWRITE_JPEG_QUALITY);
    // params.push(95);
    imgcodecs::imwrite("preview_512.png", &bgr_512, &params)?;

    // ใช้ blob ต่อเข้าโมเดลได้เลย
    let _ = blob;

    Ok(())
}
