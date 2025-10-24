/// Integration between decklink_output and DeepGiBox pipeline
/// 
/// This module connects the internal keying system with the main DeepGiBox pipeline,
/// allowing overlay graphics to come from either:
/// 1. The overlay_render stage (OverlayFramePacket)
/// 2. A static PNG file (foreground.png)

use crate::image_loader::BgraImage;
use crate::output::{GpuBuffer, OutputError, OutputSession};
use common_io::{MemLoc, MemRef, OverlayFramePacket, RawFramePacket};
use std::path::Path;

/// Source of overlay graphics
#[derive(Debug, Clone)]
pub enum OverlaySource {
    /// Use static PNG file
    StaticImage(String),
    /// Use dynamic overlay from pipeline
    Pipeline,
}

/// Compositor that integrates with DeepGiBox pipeline
pub struct PipelineCompositor {
    session: OutputSession,
    source: OverlaySource,
    overlay_buffer: Option<GpuBuffer>,
    width: u32,
    height: u32,
}

impl PipelineCompositor {
    /// Create compositor with static PNG overlay
    pub fn from_png<P: AsRef<Path>>(
        width: u32,
        height: u32,
        png_path: P,
    ) -> Result<Self, OutputError> {
        let png_image = BgraImage::load_from_file(png_path.as_ref())
            .map_err(|_| OutputError::InvalidDimensions)?;

        let session = OutputSession::new(width, height, &png_image)?;

        Ok(PipelineCompositor {
            session,
            source: OverlaySource::StaticImage(
                png_path.as_ref().to_string_lossy().to_string(),
            ),
            overlay_buffer: None,
            width,
            height,
        })
    }

    /// Create compositor that uses dynamic overlay from pipeline
    pub fn from_pipeline(width: u32, height: u32) -> Result<Self, OutputError> {
        // Create empty BGRA image as placeholder
        let empty_data = vec![0u8; (width * height * 4) as usize];
        let empty_image = BgraImage {
            data: empty_data,
            width,
            height,
            pitch: (width * 4) as usize,
        };

        let session = OutputSession::new(width, height, &empty_image)?;
        let overlay_buffer = GpuBuffer::new(width, height)?;

        Ok(PipelineCompositor {
            session,
            source: OverlaySource::Pipeline,
            overlay_buffer: Some(overlay_buffer),
            width,
            height,
        })
    }

    /// Update overlay from pipeline (only for Pipeline source)
    pub fn update_overlay(&mut self, overlay: &OverlayFramePacket) -> Result<(), OutputError> {
        match &self.source {
            OverlaySource::Pipeline => {
                // Validate dimensions
                if overlay.from.width != self.width || overlay.from.height != self.height {
                    return Err(OutputError::InvalidDimensions);
                }

                // Check if overlay is already on GPU
                match overlay.argb.loc {
                    MemLoc::Gpu { .. } => {
                        // Already on GPU, can use directly
                        // TODO: Update session's internal buffer pointer
                        Ok(())
                    }
                    MemLoc::Cpu => {
                        // Need to upload to GPU
                        if let Some(ref mut buffer) = self.overlay_buffer {
                            let bgra_data = convert_argb_to_bgra(
                                overlay.argb.ptr,
                                overlay.argb.len,
                                overlay.stride,
                            )?;

                            let image = BgraImage {
                                data: bgra_data,
                                width: self.width,
                                height: self.height,
                                pitch: overlay.stride,
                            };

                            buffer.upload(&image)?;
                        }
                        Ok(())
                    }
                }
            }
            OverlaySource::StaticImage(_) => {
                // Ignore updates when using static image
                Ok(())
            }
        }
    }

    /// Composite overlay onto video frame
    /// 
    /// # Arguments
    /// * `video_frame` - Input video frame (UYVY on GPU from DeckLink)
    /// 
    /// # Returns
    /// * Composited BGRA frame on GPU
    pub fn composite(&mut self, video_frame: &RawFramePacket) -> Result<MemRef, OutputError> {
        // Validate frame is on GPU
        match video_frame.data.loc {
            MemLoc::Gpu { .. } => {}
            MemLoc::Cpu => {
                return Err(OutputError::InvalidDimensions); // Should use appropriate error
            }
        }

        // Perform composite
        self.session.composite(
            video_frame.data.ptr,
            video_frame.data.stride,
        )?;

        // Return reference to output buffer
        Ok(MemRef {
            ptr: self.session.output_gpu_ptr() as *mut u8,
            len: self.session.output_buffer().size,
            stride: self.session.output_pitch(),
            loc: MemLoc::Gpu { device: 0 },
        })
    }

    /// Get output dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get overlay source type
    pub fn source(&self) -> &OverlaySource {
        &self.source
    }

    /// Download composited frame to CPU (for debugging/preview)
    pub fn download_output(&self) -> Result<Vec<u8>, OutputError> {
        self.session.download_output()
    }
}

/// Convert ARGB to BGRA format
/// 
/// Pipeline overlay uses ARGB, but CUDA kernel expects BGRA
fn convert_argb_to_bgra(
    argb_ptr: *mut u8,
    len: usize,
    stride: usize,
) -> Result<Vec<u8>, OutputError> {
    if argb_ptr.is_null() {
        return Err(OutputError::NullPointer);
    }

    let mut bgra_data = Vec::with_capacity(len);

    unsafe {
        let argb_slice = std::slice::from_raw_parts(argb_ptr, len);
        
        // Convert ARGB -> BGRA
        // ARGB: [A, R, G, B] -> BGRA: [B, G, R, A]
        for chunk in argb_slice.chunks_exact(4) {
            bgra_data.push(chunk[3]); // B
            bgra_data.push(chunk[2]); // G
            bgra_data.push(chunk[1]); // R
            bgra_data.push(chunk[0]); // A
        }
    }

    Ok(bgra_data)
}

/// Builder for easy compositor creation
pub struct CompositorBuilder {
    width: u32,
    height: u32,
    source: Option<OverlaySource>,
}

impl CompositorBuilder {
    pub fn new(width: u32, height: u32) -> Self {
        CompositorBuilder {
            width,
            height,
            source: None,
        }
    }

    /// Use static PNG file as overlay
    pub fn with_png<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.source = Some(OverlaySource::StaticImage(
            path.as_ref().to_string_lossy().to_string(),
        ));
        self
    }

    /// Use dynamic overlay from pipeline
    pub fn with_pipeline(mut self) -> Self {
        self.source = Some(OverlaySource::Pipeline);
        self
    }

    /// Build the compositor
    pub fn build(self) -> Result<PipelineCompositor, OutputError> {
        match self.source {
            Some(OverlaySource::StaticImage(ref path)) => {
                PipelineCompositor::from_png(self.width, self.height, path)
            }
            Some(OverlaySource::Pipeline) => {
                PipelineCompositor::from_pipeline(self.width, self.height)
            }
            None => {
                // Default to pipeline mode
                PipelineCompositor::from_pipeline(self.width, self.height)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compositor_builder() {
        let builder = CompositorBuilder::new(1920, 1080);
        assert_eq!(builder.width, 1920);
        assert_eq!(builder.height, 1080);
    }

    #[test]
    fn test_argb_to_bgra_conversion() {
        let argb = vec![
            0xFF, 0x00, 0x00, 0xFF, // A=255, R=0, G=0, B=255 (blue)
            0x80, 0xFF, 0x00, 0x00, // A=128, R=255, G=0, B=0 (red)
        ];

        let bgra = convert_argb_to_bgra(
            argb.as_ptr() as *mut u8,
            argb.len(),
            8,
        )
        .unwrap();

        // Expected BGRA: [B, G, R, A]
        assert_eq!(bgra[0], 0xFF); // B
        assert_eq!(bgra[1], 0x00); // G
        assert_eq!(bgra[2], 0x00); // R
        assert_eq!(bgra[3], 0xFF); // A

        assert_eq!(bgra[4], 0x00); // B
        assert_eq!(bgra[5], 0x00); // G
        assert_eq!(bgra[6], 0xFF); // R
        assert_eq!(bgra[7], 0x80); // A
    }
}
