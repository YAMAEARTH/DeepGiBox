use crate::image_loader::BgraImage;
use std::os::raw::{c_int, c_void};
use std::{fmt, ptr};

extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaFree(dev_ptr: *mut c_void) -> c_int;
    fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
    ) -> c_int;
    fn cudaStreamCreate(stream: *mut *mut c_void) -> c_int;
    fn cudaStreamDestroy(stream: *mut c_void) -> c_int;
    fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;
    fn cudaGetErrorString(error: c_int) -> *const i8;
}

// CUDA memory copy kinds
const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
const CUDA_SUCCESS: c_int = 0;

extern "C" {
    fn launch_composite_png_over_decklink(
        decklink_uyvy: *const u8,
        png_bgra: *const u8,
        output_bgra: *mut u8,
        width: c_int,
        height: c_int,
        decklink_pitch: c_int,
        png_pitch: c_int,
        output_pitch: c_int,
        key_r: u8,
        key_g: u8,
        key_b: u8,
        threshold: f32,
        stream: *mut c_void,
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputError {
    CudaAllocationFailed,
    CudaMemcpyFailed,
    CudaStreamFailed,
    InvalidDimensions,
    NullPointer,
}

impl fmt::Display for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OutputError::CudaAllocationFailed => write!(f, "CUDA memory allocation failed"),
            OutputError::CudaMemcpyFailed => write!(f, "CUDA memcpy failed"),
            OutputError::CudaStreamFailed => write!(f, "CUDA stream operation failed"),
            OutputError::InvalidDimensions => write!(f, "Invalid image dimensions"),
            OutputError::NullPointer => write!(f, "Null pointer encountered"),
        }
    }
}

impl std::error::Error for OutputError {}

/// GPU buffer for BGRA image
pub struct GpuBuffer {
    pub ptr: *mut u8,
    pub size: usize,
    pub width: u32,
    pub height: u32,
    pub pitch: usize,
}

impl GpuBuffer {
    /// Allocate GPU memory for BGRA image
    pub fn new(width: u32, height: u32) -> Result<Self, OutputError> {
        if width == 0 || height == 0 {
            return Err(OutputError::InvalidDimensions);
        }

        let pitch = (width * 4) as usize;
        let size = pitch * height as usize;
        let mut dev_ptr: *mut c_void = ptr::null_mut();

        let result = unsafe { cudaMalloc(&mut dev_ptr, size) };
        if result != CUDA_SUCCESS || dev_ptr.is_null() {
            return Err(OutputError::CudaAllocationFailed);
        }

        Ok(GpuBuffer {
            ptr: dev_ptr as *mut u8,
            size,
            width,
            height,
            pitch,
        })
    }

    /// Upload BGRA image from host to GPU
    pub fn upload(&mut self, image: &BgraImage) -> Result<(), OutputError> {
        if image.width != self.width || image.height != self.height {
            return Err(OutputError::InvalidDimensions);
        }

        let result = unsafe {
            cudaMemcpy(
                self.ptr as *mut c_void,
                image.data.as_ptr() as *const c_void,
                image.data.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };

        if result != CUDA_SUCCESS {
            return Err(OutputError::CudaMemcpyFailed);
        }

        Ok(())
    }

    /// Download BGRA image from GPU to host
    pub fn download(&self) -> Result<Vec<u8>, OutputError> {
        let mut host_data = vec![0u8; self.size];

        let result = unsafe {
            cudaMemcpy(
                host_data.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                self.size,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };

        if result != CUDA_SUCCESS {
            return Err(OutputError::CudaMemcpyFailed);
        }

        Ok(host_data)
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudaFree(self.ptr as *mut c_void);
            }
        }
    }
}

unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

/// Chroma key configuration
#[derive(Debug, Clone, Copy)]
pub struct ChromaKey {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub threshold: f32,
}

impl ChromaKey {
    /// Green screen key (typical green screen color)
    pub fn green_screen() -> Self {
        ChromaKey {
            r: 0,
            g: 177,
            b: 64,
            threshold: 0.15,
        }
    }

    /// Blue screen key
    pub fn blue_screen() -> Self {
        ChromaKey {
            r: 0,
            g: 0,
            b: 255,
            threshold: 0.15,
        }
    }

    /// Custom key color
    pub fn custom(r: u8, g: u8, b: u8, threshold: f32) -> Self {
        ChromaKey {
            r,
            g,
            b,
            threshold,
        }
    }
}

/// Output session for compositing and keying
pub struct OutputSession {
    png_buffer: GpuBuffer,
    output_buffer: GpuBuffer,
    stream: *mut c_void,
    width: u32,
    height: u32,
}

impl OutputSession {
    /// Create new output session
    pub fn new(width: u32, height: u32, png_image: &BgraImage) -> Result<Self, OutputError> {
        if width == 0 || height == 0 {
            return Err(OutputError::InvalidDimensions);
        }

        // Allocate GPU buffers
        let mut png_buffer = GpuBuffer::new(width, height)?;
        let output_buffer = GpuBuffer::new(width, height)?;

        // Upload PNG to GPU
        png_buffer.upload(png_image)?;

        // Create CUDA stream
        let mut stream: *mut c_void = ptr::null_mut();
        let result = unsafe { cudaStreamCreate(&mut stream) };
        if result != CUDA_SUCCESS {
            return Err(OutputError::CudaStreamFailed);
        }

        Ok(OutputSession {
            png_buffer,
            output_buffer,
            stream,
            width,
            height,
        })
    }

    /// Composite PNG over DeckLink with chroma keying
    /// decklink_uyvy_gpu: pointer to DeckLink capture buffer on GPU (UYVY format)
    pub fn composite(
        &mut self,
        decklink_uyvy_gpu: *const u8,
        decklink_pitch: usize,
        chroma_key: ChromaKey,
    ) -> Result<(), OutputError> {
        if decklink_uyvy_gpu.is_null() {
            return Err(OutputError::NullPointer);
        }

        unsafe {
            launch_composite_png_over_decklink(
                decklink_uyvy_gpu,
                self.png_buffer.ptr,
                self.output_buffer.ptr,
                self.width as c_int,
                self.height as c_int,
                decklink_pitch as c_int,
                self.png_buffer.pitch as c_int,
                self.output_buffer.pitch as c_int,
                chroma_key.r,
                chroma_key.g,
                chroma_key.b,
                chroma_key.threshold,
                self.stream,
            );
        }

        // Synchronize stream
        let result = unsafe { cudaStreamSynchronize(self.stream) };
        if result != CUDA_SUCCESS {
            return Err(OutputError::CudaStreamFailed);
        }

        Ok(())
    }

    /// Get output buffer (BGRA on GPU)
    pub fn output_buffer(&self) -> &GpuBuffer {
        &self.output_buffer
    }

    /// Get GPU pointer for direct output (for zero-copy to SDI)
    pub fn output_gpu_ptr(&self) -> *const u8 {
        self.output_buffer.ptr
    }

    /// Get output pitch (bytes per row)
    pub fn output_pitch(&self) -> usize {
        self.output_buffer.pitch
    }

    /// Download output to host memory
    pub fn download_output(&self) -> Result<Vec<u8>, OutputError> {
        self.output_buffer.download()
    }
}

impl Drop for OutputSession {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                cudaStreamDestroy(self.stream);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chroma_key_presets() {
        let green = ChromaKey::green_screen();
        assert_eq!(green.g, 177);

        let blue = ChromaKey::blue_screen();
        assert_eq!(blue.b, 255);
    }
}
