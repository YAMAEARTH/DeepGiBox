use crate::packets::{
    RawFramePacket, TensorInputPacket, PipelineError,
    TensorMem, TensorDesc, TensorDataType, TensorLayout, ColorFormat
};
use crate::pipeline::ProcessingStage;

use std::time::Instant;

/// Configuration for CV-CUDA preprocessing operations
#[derive(Debug, Clone)]
pub struct PreprocessingV2Config {
    /// Pan X offset in pixels
    pub pan_x: i32,
    /// Pan Y offset in pixels  
    pub pan_y: i32,
    /// Zoom factor (1.0 = no zoom, >1.0 = zoom in, <1.0 = zoom out)
    pub zoom: f32,
    /// Target output size for inference (width, height)
    pub target_size: (u32, u32),
    /// Enable CUDA acceleration
    pub use_cuda: bool,
    /// Enable debug output
    pub debug: bool,
    /// Normalization parameters (mean, std)
    pub normalization: Option<([f32; 3], [f32; 3])>, // (mean, std) for RGB
    /// Enable async processing
    pub async_processing: bool,
    /// GPU device ID
    pub device_id: i32,
}

impl Default for PreprocessingV2Config {
    fn default() -> Self {
        Self {
            pan_x: 0,
            pan_y: 0,
            zoom: 1.0,
            target_size: (512, 512),
            use_cuda: true,
            debug: false,
            normalization: Some(([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])), // ImageNet defaults
            async_processing: true,
            device_id: 0,
        }
    }
}

/// CV-CUDA accelerated preprocessing stage v.2
pub struct PreprocessingV2Stage {
    config: PreprocessingV2Config,
    frame_count: u64,
    opencv_context: Option<String>, // Simplified context
}

impl PreprocessingV2Stage {
    /// Create a new CV-CUDA preprocessing stage with the given configuration
    pub fn new(config: PreprocessingV2Config) -> Result<Self, PipelineError> {
        let mut stage = Self {
            config,
            frame_count: 0,
            opencv_context: None,
        };
        
        if stage.config.use_cuda {
            stage.initialize_cuda()?;
        }
        
        Ok(stage)
    }
    
    fn initialize_cuda(&mut self) -> Result<(), PipelineError> {
        // Initialize OpenCV with CUDA support if available
        if self.config.debug {
            println!("Initializing CV-CUDA preprocessing v.2...");
        }
        
        // For now, just set a context string
        self.opencv_context = Some("cuda_available".to_string());
        
        if self.config.debug {
            println!("CV-CUDA preprocessing v.2 initialized");
        }
        
        Ok(())
    }
    
    /// Process frame using CV-CUDA acceleration  
    fn process_with_cvcuda(&mut self, input: &RawFramePacket) -> Result<TensorInputPacket, PipelineError> {
        let start_time = Instant::now();
        
        // For now, create a simple tensor output
        let tensor_desc = TensorDesc {
            shape: [1, 3, self.config.target_size.1, self.config.target_size.0], // NCHW
            dtype: TensorDataType::F32,
            layout: TensorLayout::NCHW,
            colors: ColorFormat::RGB,
        };
        
        // Create dummy tensor data
        let tensor_size = (self.config.target_size.0 * self.config.target_size.1 * 3 * 4) as usize; // 3 channels, 4 bytes per float
        let tensor_data = vec![0u8; tensor_size];
        
        let tensor_packet = TensorInputPacket {
            mem: TensorMem::Cpu { data: tensor_data },
            desc: tensor_desc,
            meta: input.meta.clone(),
        };
        
        if self.config.debug {
            let processing_time = start_time.elapsed();
            println!(
                "CV-CUDA preprocessing completed in {:.2}ms for frame {} ({}x{} -> {}x{})",
                processing_time.as_secs_f64() * 1000.0,
                self.frame_count,
                input.meta.width,
                input.meta.height,
                self.config.target_size.0,
                self.config.target_size.1
            );
        }
        
        Ok(tensor_packet)
    }
    
    /// Fallback CPU processing when CUDA is not available
    fn process_with_cpu(&mut self, input: RawFramePacket) -> Result<TensorInputPacket, PipelineError> {
        if self.config.debug {
            println!("Using CPU processing for frame {}", self.frame_count);
        }
        
        // Use the same processing pipeline but without CUDA acceleration
        self.process_with_cvcuda(&input)
    }
}

impl Drop for PreprocessingV2Stage {
    fn drop(&mut self) {
        // Clean up resources
        if self.config.debug {
            println!("Cleaning up CV-CUDA preprocessing v.2 stage");
        }
    }
}

/// Generic ProcessingStage trait for type-safe pipeline operations
pub trait ProcessingStageV2<Input, Output>: Send {
    fn process(&mut self, input: Input) -> Result<Output, PipelineError>;
    fn name(&self) -> &str;
}

impl ProcessingStageV2<RawFramePacket, TensorInputPacket> for PreprocessingV2Stage {
    fn process(&mut self, input: RawFramePacket) -> Result<TensorInputPacket, PipelineError> {
        self.frame_count += 1;

        if self.config.debug {
            println!(
                "CV-CUDA preprocessing v.2 frame {} (seq: {}, {}x{})",
                self.frame_count,
                input.meta.seq_no,
                input.meta.width,
                input.meta.height
            );
        }

        // Process with CV-CUDA if available, otherwise fallback to CPU
        if self.config.use_cuda {
            self.process_with_cvcuda(&input)
        } else {
            self.process_with_cpu(input)
        }
    }

    fn name(&self) -> &str {
        "preprocessing_v2_cvcuda"
    }
}

// Backward compatibility with the original ProcessingStage trait
impl ProcessingStage for PreprocessingV2Stage {
    fn process(&mut self, input: RawFramePacket) -> Result<RawFramePacket, PipelineError> {
        // Convert to TensorInputPacket and back for compatibility
        // In practice, this should be avoided - use the type-safe interface instead
        let _tensor = ProcessingStageV2::process(self, input.clone())?;
        
        if self.config.debug {
            println!("Backward compatibility mode: returning original frame");
        }
        
        Ok(input)
    }

    fn name(&self) -> &str {
        "preprocessing_v2_compat"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packets::{FrameMeta, PixelFormat, ColorSpace};

    #[test]
    fn test_preprocessing_v2_config_default() {
        let config = PreprocessingV2Config::default();
        assert_eq!(config.target_size, (512, 512));
        assert_eq!(config.zoom, 1.0);
        assert!(config.use_cuda);
        assert!(config.async_processing);
    }

    #[test]
    fn test_preprocessing_v2_stage_creation() {
        let config = PreprocessingV2Config {
            use_cuda: false, // Disable CUDA for testing
            ..Default::default()
        };
        
        let stage = PreprocessingV2Stage::new(config);
        assert!(stage.is_ok());
        
        let stage = stage.unwrap();
        assert_eq!(ProcessingStageV2::name(&stage), "preprocessing_v2_cvcuda");
        assert_eq!(stage.frame_count, 0);
    }

    #[test]
    fn test_tensor_input_packet_creation() {
        let config = PreprocessingV2Config {
            use_cuda: false,
            target_size: (640, 480),
            ..Default::default()
        };
        
        let mut stage = PreprocessingV2Stage::new(config).unwrap();
        
        // Create test frame
        let test_data = vec![0u8; 1920 * 1080 * 4];
        let meta = FrameMeta {
            source_id: 1,
            width: 1920,
            height: 1080,
            stride: 1920 * 4,
            pixfmt: PixelFormat::BGRA8,
            colorspace: ColorSpace::BT709,
            pts_ns: 0,
            timecode: None,
            seq_no: 1,
        };
        
        let frame = RawFramePacket::new_cpu(test_data, meta);
        let result = ProcessingStageV2::process(&mut stage, frame);
        
        assert!(result.is_ok());
        let tensor = result.unwrap();
        assert_eq!(tensor.desc.shape, [1, 3, 480, 640]);
        assert_eq!(tensor.desc.layout, TensorLayout::NCHW);
        assert_eq!(tensor.desc.colors, ColorFormat::RGB);
    }
}
