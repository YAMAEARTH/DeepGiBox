/// Integration between decklink_output and DeepGiBox pipeline
/// 
/// This module connects the internal keying system with the main DeepGiBox pipeline,
/// allowing overlay graphics to come from either:
/// 1. The overlay_render stage (OverlayFramePacket)
/// 2. A static PNG file (foreground.png)

use crate::image_loader::BgraImage;
use crate::output::{GpuBuffer, OutputError, OutputSession};
use common_io::{MemLoc, MemRef, OverlayFramePacket, RawFramePacket};
use std::os::raw::{c_int, c_void};
use std::path::Path;
use std::ptr;

// ==================== GPU Compositor FFI ====================
extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaFree(dev_ptr: *mut c_void) -> c_int;
    fn cudaStreamCreate(stream: *mut *mut c_void) -> c_int;
    fn cudaStreamDestroy(stream: *mut c_void) -> c_int;
    fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;
    
    // จาก compositor_gpu.cu
    fn launch_composite_argb_overlay(
        decklink_uyvy_gpu: *const u8,
        overlay_argb_gpu: *const u8,
        output_bgra_gpu: *mut u8,
        width: c_int,
        height: c_int,
        decklink_pitch: c_int,
        overlay_pitch: c_int,
        output_pitch: c_int,
        stream: *mut c_void,
    );
}

const CUDA_SUCCESS: c_int = 0;

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
    // GPU compositor (ไม่ใช้ OutputSession แล้ว)
    output_buffer: Option<*mut u8>,
    stream: *mut c_void,
    width: u32,
    height: u32,
    use_gpu_compositor: bool,
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
            output_buffer: None,
            stream: ptr::null_mut(),
            width,
            height,
            use_gpu_compositor: false,
        })
    }

    /// Create compositor that uses dynamic overlay from pipeline
    pub fn from_pipeline(width: u32, height: u32) -> Result<Self, OutputError> {
        Self::from_pipeline_with_mode(width, height, true)
    }
    
    /// Create compositor with explicit GPU mode control
    pub fn from_pipeline_with_mode(width: u32, height: u32, use_gpu: bool) -> Result<Self, OutputError> {
        if use_gpu {
            // GPU compositor - ไม่ต้องใช้ OutputSession เลย
            // Allocate output buffer
            let stride = (width * 4) as usize;
            let size = stride * height as usize;
            let mut dev_ptr: *mut c_void = ptr::null_mut();
            
            let result = unsafe { cudaMalloc(&mut dev_ptr, size) };
            if result != CUDA_SUCCESS || dev_ptr.is_null() {
                return Err(OutputError::CudaAllocationFailed);
            }
            
            // Create CUDA stream
            let mut stream: *mut c_void = ptr::null_mut();
            let result = unsafe { cudaStreamCreate(&mut stream) };
            if result != CUDA_SUCCESS {
                unsafe { cudaFree(dev_ptr); }
                return Err(OutputError::CudaStreamFailed);
            }
            
            // Create dummy session (ไว้ compatibility)
            let empty_data = vec![0u8; (width * height * 4) as usize];
            let empty_image = BgraImage {
                data: empty_data,
                width,
                height,
                pitch: (width * 4) as usize,
            };
            let session = OutputSession::new(width, height, &empty_image)?;
            
            Ok(PipelineCompositor {
                session,
                source: OverlaySource::Pipeline,
                overlay_buffer: None,
                output_buffer: Some(dev_ptr as *mut u8),
                stream,
                width,
                height,
                use_gpu_compositor: true,
            })
        } else {
            // Legacy mode (CPU conversion)
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
                output_buffer: None,
                stream: ptr::null_mut(),
                width,
                height,
                use_gpu_compositor: false,
            })
        }
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

    /// Composite overlay onto video frame (GPU version - zero CPU copy)
    /// 
    /// # Arguments
    /// * `video_frame` - Input video frame (UYVY on GPU from DeckLink)
    /// * `overlay_frame` - Overlay (ARGB on GPU from overlay_render)
    /// 
    /// # Returns
    /// * Composited BGRA frame on GPU
    pub fn composite_gpu(
        &mut self,
        video_frame: &RawFramePacket,
        overlay_frame: &OverlayFramePacket,
    ) -> Result<MemRef, OutputError> {
        if !self.use_gpu_compositor {
            return Err(OutputError::InvalidDimensions); // fallback to old method
        }
        
        // Validate frames are on GPU
        match (&video_frame.data.loc, &overlay_frame.argb.loc) {
            (MemLoc::Gpu { .. }, MemLoc::Gpu { .. }) => {}
            _ => return Err(OutputError::NullPointer),
        }
        
        let output_ptr = self.output_buffer.ok_or(OutputError::NullPointer)?;
        let stride = (self.width * 4) as usize;
        
        // Launch GPU composite kernel
        unsafe {
            launch_composite_argb_overlay(
                video_frame.data.ptr,
                overlay_frame.argb.ptr,
                output_ptr,
                self.width as c_int,
                self.height as c_int,
                video_frame.data.stride as c_int,
                overlay_frame.stride as c_int,
                stride as c_int,
                self.stream,
            );
            
            // Synchronize
            let result = cudaStreamSynchronize(self.stream);
            if result != CUDA_SUCCESS {
                return Err(OutputError::CudaStreamFailed);
            }
        }
        
        Ok(MemRef {
            ptr: output_ptr,
            len: stride * self.height as usize,
            stride,
            loc: MemLoc::Gpu { device: 0 },
        })
    }
    
    /// Composite overlay onto video frame (legacy version)
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
        if self.use_gpu_compositor {
            // Download from GPU output buffer
            if let Some(ptr) = self.output_buffer {
                let size = (self.width * self.height * 4) as usize;
                let mut host_data = vec![0u8; size];
                
                unsafe {
                    let result = crate::output::cudaMemcpy(
                        host_data.as_mut_ptr() as *mut std::os::raw::c_void,
                        ptr as *const std::os::raw::c_void,
                        size,
                        2, // CUDA_MEMCPY_DEVICE_TO_HOST
                    );
                    if result != CUDA_SUCCESS {
                        return Err(OutputError::CudaMemcpyFailed);
                    }
                }
                
                Ok(host_data)
            } else {
                Err(OutputError::NullPointer)
            }
        } else {
            self.session.download_output()
        }
    }
}

impl Drop for PipelineCompositor {
    fn drop(&mut self) {
        if let Some(ptr) = self.output_buffer {
            unsafe {
                cudaFree(ptr as *mut c_void);
            }
        }
        if !self.stream.is_null() {
            unsafe {
                cudaStreamDestroy(self.stream);
            }
        }
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
