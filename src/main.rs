mod inference;
mod post;
mod visualize;
mod video;

use crate::inference::{init_environment, InferenceContext, PreprocessedImage, INPUT_WIDTH, INPUT_HEIGHT};
use crate::post::{postprocess_yolov5, SortTracker, YoloPostConfig};
use crate::visualize::visualize_detections_on_image;
use crate::video::VideoProcessor;
use ort::Result;
use std::path::Path;
use image::{RgbImage, Rgb};

fn main() -> Result<()> {
    init_environment()?;

    let model_path = Path::new("./YOLOv5.onnx");
    let video_path = "./demo_yt.mp4";
    let output_path = "./output_detections_v1_conf40.mp4";

    // Initialize video processor
    let mut video_processor = VideoProcessor::new(video_path)?;
    video_processor.setup_output(output_path)?;

    // Initialize inference context
    let mut context = InferenceContext::new(model_path)?;
    let mut tracker = SortTracker::new(3, 1, 0.3);
    
    let post_cfg = YoloPostConfig {
        num_classes: 2,
        confidence_threshold: 0.4,
        nms_threshold: 0.45,
        max_detections: 20,
        original_size: (video_processor.info.width as u32, video_processor.info.height as u32),
        letterbox_scale: 1.0,
        letterbox_pad: (0.0, 0.0),
    };

    // Process video frames
    let mut frame_num = 0;
    
    println!("Processing video frames...");
    
    while let Some(rgb_image) = video_processor.read_frame()? {
        frame_num += 1;
        
        if frame_num % 30 == 0 {
            println!("Processing frame {}/{}", frame_num, video_processor.info.frame_count);
        }

        // Preprocess for inference
        let preprocessed = preprocess_frame(&rgb_image)?;
        
        // Run inference
        let input_tensor = context.prepare_input(&preprocessed.normalized)?;
        let inference_output = context.run(&input_tensor)?;
        
        // Postprocess detections
        let post_result = postprocess_yolov5(&inference_output.predictions, &post_cfg, Some(&mut tracker));
        
        // Draw detections on frame
        let output_image = visualize_detections_on_image(&rgb_image, &post_result)?;
        
        // Write frame to output video
        video_processor.write_frame(&output_image)?;
    }

    video_processor.release();
    
    println!("Video processing complete! Output saved to {}", output_path);
    println!("Processed {} frames", frame_num);
    
    Ok(())
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
