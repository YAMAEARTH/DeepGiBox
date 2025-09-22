use crate::packets::{RawFramePacket, PipelineError, PixelFormat};
use crate::pipeline::ProcessingStage;

use opencv::{
    core::{self, Mat, Rect, Scalar, Size, CV_8UC3},
    dnn, imgproc,
    prelude::*,
};

/// Configuration for preprocessing operations
#[derive(Debug, Clone)]
pub struct PreprocessingStageConfig {
    /// Pan X offset in pixels
    pub pan_x: i32,
    /// Pan Y offset in pixels  
    pub pan_y: i32,
    /// Zoom factor (1.0 = no zoom, >1.0 = zoom in, <1.0 = zoom out)
    pub zoom: f32,
    /// Target output size for inference (width, height)
    pub target_size: (u32, u32),
    /// Enable debug output
    pub debug: bool,
}

impl Default for PreprocessingStageConfig {
    fn default() -> Self {
        Self {
            pan_x: 0,
            pan_y: 0,
            zoom: 1.0,
            target_size: (512, 512),
            debug: false,
        }
    }
}

/// Preprocessing stage that converts RawFramePacket with OpenCV processing
pub struct PreprocessingStage {
    config: PreprocessingStageConfig,
    frame_count: u64,
}

impl PreprocessingStage {
    /// Create a new preprocessing stage with the given configuration
    pub fn new(config: PreprocessingStageConfig) -> Self {
        Self {
            config,
            frame_count: 0,
        }
    }
}

impl ProcessingStage for PreprocessingStage {
    fn process(&mut self, input: RawFramePacket) -> Result<RawFramePacket, PipelineError> {
        self.frame_count += 1;

        if self.config.debug {
            println!("Preprocessing frame {} (seq: {})", self.frame_count, input.meta.seq_no);
        }

        // For now, just return the original frame
        // In a real implementation, we would:
        // 1. Convert RawFramePacket data to OpenCV Mat
        // 2. Apply pan/zoom/resize operations
        // 3. Convert to tensor format for inference
        // 4. Return either the original frame or create a TensorInputPacket

        if self.config.debug {
            println!(
                "Preprocessing complete for frame {} ({}x{})",
                self.frame_count,
                input.meta.width,
                input.meta.height
            );
        }

        Ok(input)
    }

    fn name(&self) -> &str {
        "preprocessing"
    }
}

impl PreprocessingStage {
    /// Convert raw image data to OpenCV Mat (simplified version)
    fn create_mat_from_bgra(&self, data: &[u8], width: u32, height: u32) -> opencv::Result<Mat> {
        // Create a simple BGR Mat for processing
        let mut bgr_mat = Mat::zeros(height as i32, width as i32, CV_8UC3)?.to_mat()?;
        
        // In a real implementation, we'd convert BGRA data to BGR here
        // For now, just return an empty mat
        Ok(bgr_mat)
    }

    /// Perform preprocessing operations on the Mat
    fn preprocess_mat(&self, input_mat: &Mat) -> opencv::Result<Mat> {
        let (w, h) = (input_mat.cols(), input_mat.rows());
        
        // Calculate ROI based on pan/zoom
        let zoom = if self.config.zoom <= 0.0 { 1.0 } else { self.config.zoom };
        let tw = (w as f32 / zoom).round() as i32;
        let th = (h as f32 / zoom).round() as i32;
        let x = self.config.pan_x.clamp(0, (w - tw).max(0));
        let y = self.config.pan_y.clamp(0, (h - th).max(0));
        let roi = Rect::new(x, y, tw.max(1), th.max(1));
        
        // Extract ROI
        let roi_mat = Mat::roi(input_mat, roi)?;
        
        // Resize to target size
        let mut resized_mat = Mat::default();
        imgproc::resize(
            &roi_mat,
            &mut resized_mat,
            Size::new(self.config.target_size.0 as i32, self.config.target_size.1 as i32),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        
        // Create blob for inference
        let blob = dnn::blob_from_image(
            &resized_mat,
            1.0 / 255.0,
            Size::new(self.config.target_size.0 as i32, self.config.target_size.1 as i32),
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            true,  // swapRB => BGR->RGB
            false, // crop
            core::CV_32F,
        )?;
        
        Ok(blob)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packets::{FrameMeta, PixelFormat, ColorSpace, MemLoc};

    #[test]
    fn test_preprocessing_config_default() {
        let config = PreprocessingStageConfig::default();
        assert_eq!(config.target_size, (512, 512));
        assert_eq!(config.zoom, 1.0);
    }

    #[test]
    fn test_preprocessing_stage_creation() {
        let config = PreprocessingStageConfig::default();
        let stage = PreprocessingStage::new(config);
        assert_eq!(stage.name(), "preprocessing");
        assert_eq!(stage.frame_count, 0);
    }
}
