use opencv::{
    core::{Mat, Size},
    imgproc,
    videoio::{self, VideoCapture, VideoWriter, CAP_PROP_FRAME_COUNT, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT},
    prelude::*,
};
use image::{ImageBuffer, RgbImage};
use ort::Result;

pub struct VideoInfo {
    pub width: i32,
    pub height: i32,
    pub fps: f64,
    pub frame_count: i32,
}

pub struct VideoProcessor {
    cap: VideoCapture,
    writer: Option<VideoWriter>,
    pub info: VideoInfo,
}

impl VideoProcessor {
    pub fn new(input_path: &str) -> Result<Self> {
        // Open input video
        let cap = VideoCapture::from_file(input_path, videoio::CAP_ANY)
            .map_err(|e| ort::Error::new(format!("Failed to open video: {}", e)))?;
        
        if !cap.is_opened().unwrap_or(false) {
            return Err(ort::Error::new("Failed to open video file"));
        }

        // Get video properties
        let fps = cap.get(CAP_PROP_FPS).unwrap_or(30.0);
        let frame_count = cap.get(CAP_PROP_FRAME_COUNT).unwrap_or(0.0) as i32;
        let width = cap.get(CAP_PROP_FRAME_WIDTH).unwrap_or(640.0) as i32;
        let height = cap.get(CAP_PROP_FRAME_HEIGHT).unwrap_or(480.0) as i32;
        
        let info = VideoInfo {
            width,
            height,
            fps,
            frame_count,
        };

        println!("Video: {}x{}, {:.1} fps, {} frames", width, height, fps, frame_count);

        Ok(Self {
            cap,
            writer: None,
            info,
        })
    }

    pub fn setup_output(&mut self, output_path: &str) -> Result<()> {
        let fourcc = videoio::VideoWriter::fourcc('m', 'p', '4', 'v')
            .map_err(|e| ort::Error::new(format!("Failed to get fourcc: {}", e)))?;
        
        let writer = VideoWriter::new(
            output_path,
            fourcc,
            self.info.fps,
            Size::new(self.info.width, self.info.height),
            true,
        ).map_err(|e| ort::Error::new(format!("Failed to create video writer: {}", e)))?;

        self.writer = Some(writer);
        Ok(())
    }

    pub fn read_frame(&mut self) -> Result<Option<RgbImage>> {
        let mut frame = Mat::default();
        
        if !self.cap.read(&mut frame).unwrap_or(false) || frame.empty() {
            return Ok(None);
        }

        // Convert OpenCV Mat to RGB image
        let rgb_image = opencv_mat_to_rgb_image(&frame)?;
        Ok(Some(rgb_image))
    }

    pub fn write_frame(&mut self, image: &RgbImage) -> Result<()> {
        if let Some(writer) = &mut self.writer {
            let output_mat = rgb_image_to_opencv_mat(image)?;
            writer.write(&output_mat)
                .map_err(|e| ort::Error::new(format!("Failed to write frame: {}", e)))?;
        }
        Ok(())
    }

    pub fn release(self) {
        if let Some(mut writer) = self.writer {
            writer.release().unwrap_or(());
        }
    }
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
    let (_width, height) = image.dimensions();
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
