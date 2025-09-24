mod inference;
mod post;
mod visualize;

use crate::inference::{init_environment, InferenceContext, PreprocessedImage, INPUT_WIDTH, INPUT_HEIGHT};
use crate::post::{postprocess_yolov5, SortTracker, YoloPostConfig};
use crate::visualize::visualize_detections_on_image;
use ort::Result;
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use opencv::{
    core::{Mat, Size},
    imgproc,
    videoio::{self, VideoCapture, VideoWriter, CAP_PROP_FRAME_COUNT, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT},
    prelude::*,
};
use image::{ImageBuffer, RgbImage, Rgb};

fn main() -> Result<()> {
    init_environment()?;

    let model_path = Path::new("./YOLOv5.onnx");
    let video_path = "./demo_yt.mp4";
    let output_path = "./output_detections.mp4";

    // Open input video
    let mut cap = VideoCapture::from_file(video_path, videoio::CAP_ANY)
        .map_err(|e| ort::Error::new(format!("Failed to open video: {}", e)))?;
    
    if !cap.is_opened().unwrap_or(false) {
        return Err(ort::Error::new("Failed to open video file"));
    }

    // Get video properties
    let fps = cap.get(CAP_PROP_FPS).unwrap_or(30.0);
    let frame_count = cap.get(CAP_PROP_FRAME_COUNT).unwrap_or(0.0) as i32;
    let width = cap.get(CAP_PROP_FRAME_WIDTH).unwrap_or(640.0) as i32;
    let height = cap.get(CAP_PROP_FRAME_HEIGHT).unwrap_or(480.0) as i32;
    
    println!("Video: {}x{}, {:.1} fps, {} frames", width, height, fps, frame_count);

    // Setup video writer
    let fourcc = videoio::VideoWriter::fourcc('m', 'p', '4', 'v')
        .map_err(|e| ort::Error::new(format!("Failed to get fourcc: {}", e)))?;
    let mut writer = VideoWriter::new(
        output_path,
        fourcc,
        fps,
        Size::new(width, height),
        true,
    ).map_err(|e| ort::Error::new(format!("Failed to create video writer: {}", e)))?;

    // Initialize inference context
    let mut context = InferenceContext::new(model_path)?;
    let mut tracker = SortTracker::new(3, 1, 0.3);
    
    let post_cfg = YoloPostConfig {
        num_classes: 2,
        confidence_threshold: 0.35,
        nms_threshold: 0.45,
        max_detections: 20,
        original_size: (width as u32, height as u32),
        letterbox_scale: 1.0,
        letterbox_pad: (0.0, 0.0),
    };

    // Process video frames
    let mut frame_num = 0;
    let mut frame = Mat::default();
    
    println!("Processing video frames...");
    
    while cap.read(&mut frame).unwrap_or(false) && !frame.empty() {
        frame_num += 1;
        
        if frame_num % 30 == 0 {
            println!("Processing frame {}/{}", frame_num, frame_count);
        }

        // Convert OpenCV Mat to image
        let rgb_image = opencv_mat_to_rgb_image(&frame)?;
        
        // Preprocess for inference
        let preprocessed = preprocess_frame(&rgb_image)?;
        
        // Run inference
        let input_tensor = context.prepare_input(&preprocessed.normalized)?;
        let inference_output = context.run(&input_tensor)?;
        
        // Postprocess detections
        let post_result = postprocess_yolov5(&inference_output.predictions, &post_cfg, Some(&mut tracker));
        
        // Draw detections on frame
        let output_image = visualize_detections_on_image(&rgb_image, &post_result)?;
        
        // Convert back to OpenCV Mat and write to video
        let output_mat = rgb_image_to_opencv_mat(&output_image)?;
        writer.write(&output_mat)
            .map_err(|e| ort::Error::new(format!("Failed to write frame: {}", e)))?;
    }

    writer.release().unwrap_or(());
    cap.release().unwrap_or(());
    
    println!("Video processing complete! Output saved to {}", output_path);
    println!("Processed {} frames", frame_num);
    
    Ok(())
}

fn opencv_mat_to_rgb_image(mat: &Mat) -> Result<RgbImage> {
    let rows = mat.rows();
    let cols = mat.cols();
    
    let mut rgb_mat = Mat::default();
    imgproc::cvt_color(mat, &mut rgb_mat, imgproc::COLOR_BGR2RGB, 0)
        .map_err(|e| ort::Error::new(format!("Color conversion failed: {}", e)))?;
    
    let data = rgb_mat.data_bytes()
        .map_err(|e| ort::Error::new(format!("Failed to get image data: {}", e)))?;
    
    ImageBuffer::from_raw(cols as u32, rows as u32, data.to_vec())
        .ok_or_else(|| ort::Error::new("Failed to create ImageBuffer"))
}

fn rgb_image_to_opencv_mat(image: &RgbImage) -> Result<Mat> {
    let (width, height) = image.dimensions();
    let data = image.as_raw();
    
    // Create Mat from RGB data using proper OpenCV Rust binding
    let mat = Mat::from_slice(data).map_err(|e| ort::Error::new(format!("Failed to create Mat: {}", e)))?;
    
    // Reshape to proper dimensions (height, width, 3 channels)
    let reshaped = mat.reshape(3, height as i32).map_err(|e| ort::Error::new(format!("Failed to reshape Mat: {}", e)))?;
    
    // Convert RGB to BGR for OpenCV
    let mut output = Mat::default();
    imgproc::cvt_color(&reshaped, &mut output, imgproc::COLOR_RGB2BGR, 0)
        .map_err(|e| ort::Error::new(format!("Color conversion failed: {}", e)))?;
    
    Ok(output)
}

fn preprocess_frame(image: &RgbImage) -> Result<PreprocessedImage> {
    let (orig_width, orig_height) = image.dimensions();
    
    // Resize to model input size with letterboxing
    let scale = (INPUT_WIDTH as f32 / orig_width as f32).min(INPUT_HEIGHT as f32 / orig_height as f32);
    let new_width = (orig_width as f32 * scale) as u32;
    let new_height = (orig_height as f32 * scale) as u32;
    
    let resized = image::imageops::resize(image, new_width, new_height, image::imageops::FilterType::Lanczos3);
    
    // Create letterboxed image
    let mut letterboxed = RgbImage::new(INPUT_WIDTH, INPUT_HEIGHT);
    let pad_x = (INPUT_WIDTH - new_width) / 2;
    let pad_y = (INPUT_HEIGHT - new_height) / 2;
    
    // Fill with gray
    for pixel in letterboxed.pixels_mut() {
        *pixel = Rgb([128, 128, 128]);
    }
    
    // Copy resized image to center
    image::imageops::overlay(&mut letterboxed, &resized, pad_x as i64, pad_y as i64);
    
    // Normalize to [0, 1] and convert to CHW format
    let mut normalized = vec![0.0f32; INPUT_WIDTH as usize * INPUT_HEIGHT as usize * 3];
    
    for (i, pixel) in letterboxed.pixels().enumerate() {
        let x = i % INPUT_WIDTH as usize;
        let y = i / INPUT_WIDTH as usize;
        
        // CHW format: [C, H, W]
        normalized[y * INPUT_WIDTH as usize + x] = pixel[0] as f32 / 255.0; // R
        normalized[INPUT_WIDTH as usize * INPUT_HEIGHT as usize + y * INPUT_WIDTH as usize + x] = pixel[1] as f32 / 255.0; // G  
        normalized[2 * INPUT_WIDTH as usize * INPUT_HEIGHT as usize + y * INPUT_WIDTH as usize + x] = pixel[2] as f32 / 255.0; // B
    }
    
    Ok(PreprocessedImage {
        normalized,
        original_size: (orig_width, orig_height),
        letterbox_scale: scale,
        letterbox_pad: (pad_x as f32, pad_y as f32),
    })
}
