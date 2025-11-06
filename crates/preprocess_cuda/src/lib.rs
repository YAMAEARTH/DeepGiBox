use anyhow::{anyhow, Result};
use common_io::{
    ColorSpace, DType, MemLoc, MemRef, PixelFormat, RawFramePacket, Stage, TensorDesc,
    TensorInputPacket,
};
use cudarc::driver::{CudaDevice, CudaFunction, CudaStream, DevicePtr, LaunchConfig};
use std::collections::HashMap;
use std::sync::Arc;

/// Chroma order for YUV422_8 pixel format
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaOrder {
    UYVY = 0,
    YUY2 = 1,
}

/// Camera-specific crop regions
/// Crop region format: [y_start:y_end, x_start:x_end] -> (x, y, width, height)
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
pub enum CropRegion {
    /// FUJI: [0:1080, 0:1376] -> crop left portion (1376×1080)
    #[serde(rename = "fuji")]
    Fuji,
    /// OLYMPUS: [0:1080, 548:1920] -> crop center portion (1372×1080)
    #[serde(rename = "olympus")]
    Olympus,
    /// PENTAX: [0:1080, 0:1376] -> crop left portion (1376×1080)
    #[serde(rename = "pentax")]
    Pentax,
    /// No cropping (use full frame)
    #[serde(rename = "none")]
    None,
    /// Custom crop region (x, y, width, height)
    Custom { x: u32, y: u32, width: u32, height: u32 },
}

impl CropRegion {
    /// Get crop coordinates (x, y, width, height) based on input frame size
    /// Supports both 1080p (1920×1080) and 4K (3840×2160) with same proportions
    pub fn get_coords(&self, input_width: u32, input_height: u32) -> (u32, u32, u32, u32) {
        // Detect if input is 4K (2x scale of 1080p)
        let scale = if input_width == 3840 && input_height == 2160 {
            2 // 4K
        } else {
            1 // 1080p or default
        };
        
        match self {
            CropRegion::Fuji => {
                // 1080p: [0:1080, 0:1376] -> (0, 0, 1376, 1080)
                // 4K:    [0:2160, 0:2752] -> (0, 0, 2752, 2160)
                (0, 0, 1376 * scale, 1080 * scale)
            }
            CropRegion::Olympus => {
                // 1080p: [0:1080, 548:1920] -> (548, 0, 1372, 1080)
                // 4K:    [0:2160, 1096:3840] -> (1096, 0, 2744, 2160)
                (548 * scale, 0, 1372 * scale, 1080 * scale)
            }
            CropRegion::Pentax => {
                // 1080p: [0:1080, 0:1376] -> (0, 0, 1376, 1080)
                // 4K:    [0:2160, 0:2752] -> (0, 0, 2752, 2160)
                (0, 0, 1376 * scale, 1080 * scale)
            }
            CropRegion::None => {
                // Full frame - use actual input size
                (0, 0, input_width, input_height)
            }
            CropRegion::Custom { x, y, width, height } => {
                // Custom crop region - use as-is
                (*x, *y, *width, *height)
            }
        }
    }
}

/// Configuration for preprocessing
#[derive(Clone, Debug, serde::Deserialize)]
pub struct PreprocessConfig {
    #[serde(default = "default_size")]
    pub size: [u32; 2],
    #[serde(default = "default_fp16")]
    pub fp16: bool,
    #[serde(default = "default_device")]
    pub device: u32,
    #[serde(default = "default_mean")]
    pub mean: [f32; 3],
    #[serde(default = "default_std")]
    pub std: [f32; 3],
    #[serde(default = "default_chroma")]
    pub chroma: String,
    #[serde(default = "default_crop")]
    pub crop_region: CropRegion,
}

fn default_size() -> [u32; 2] {
    [512, 512]
}
fn default_fp16() -> bool {
    true
}
fn default_device() -> u32 {
    0
}
fn default_mean() -> [f32; 3] {
    [0.0, 0.0, 0.0]
}
fn default_std() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}
fn default_chroma() -> String {
    "UYVY".to_string()
}
fn default_crop() -> CropRegion {
    CropRegion::None
}

/// Main preprocessing stage with GPU acceleration
pub struct Preprocessor {
    pub size: (u32, u32), // (width, height) - output size
    pub fp16: bool,
    pub device: u32,
    pub mean: [f32; 3], // RGB mean
    pub std: [f32; 3],  // RGB std
    pub chroma: ChromaOrder,
    pub crop_region: CropRegion, // Camera-specific crop region

    // Frame validation (for production video streams)
    pub expected_input_sizes: Vec<(u32, u32)>, // Empty = accept any size
    pub skip_invalid_frames: bool,              // true = skip, false = panic

    // CUDA resources
    cuda_device: Arc<CudaDevice>,
    cuda_stream: CudaStream,
    kernel_func: CudaFunction,

    // Buffer pool for output tensors (stores raw device pointers and sizes)
    buffer_pool: HashMap<(u32, u32, bool), (cudarc::driver::sys::CUdeviceptr, usize)>,
}

impl Preprocessor {
    /// Create a new preprocessor with specified configuration
    pub fn new(size: (u32, u32), fp16: bool, device: u32) -> Result<Self> {
        Self::with_params(
            size,
            fp16,
            device,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            ChromaOrder::UYVY,
        )
    }

    /// Create with full parameters
    pub fn with_params(
        size: (u32, u32),
        fp16: bool,
        device: u32,
        mean: [f32; 3],
        std: [f32; 3],
        chroma: ChromaOrder,
    ) -> Result<Self> {
        Self::with_crop_region(size, fp16, device, mean, std, chroma, CropRegion::None)
    }

    /// Create with crop region
    pub fn with_crop_region(
        size: (u32, u32),
        fp16: bool,
        device: u32,
        mean: [f32; 3],
        std: [f32; 3],
        chroma: ChromaOrder,
        crop_region: CropRegion,
    ) -> Result<Self> {
        // Initialize CUDA device
        let cuda_device = CudaDevice::new(device as usize)
            .map_err(|e| anyhow!("Failed to initialize CUDA device {}: {:?}", device, e))?;

        // Create CUDA stream
        let cuda_stream = cuda_device
            .fork_default_stream()
            .map_err(|e| anyhow!("Failed to create CUDA stream: {:?}", e))?;

        // Load PTX and get kernel function
        let ptx_path = std::path::Path::new(env!("OUT_DIR")).join("preprocess.ptx");
        let ptx = std::fs::read_to_string(&ptx_path).map_err(|e| {
            anyhow!(
                "Failed to read PTX file {:?}: {}. Make sure CUDA kernel compiled successfully.",
                ptx_path,
                e
            )
        })?;
        cuda_device
            .load_ptx(ptx.into(), "preprocess", &["fused_preprocess_kernel"])
            .map_err(|e| anyhow!("Failed to load PTX: {:?}", e))?;

        let kernel_func = cuda_device
            .get_func("preprocess", "fused_preprocess_kernel")
            .ok_or_else(|| anyhow!("Failed to get kernel function"))?;

        Ok(Self {
            size,
            fp16,
            device,
            mean,
            std,
            chroma,
            crop_region,
            expected_input_sizes: vec![(1920, 1080), (3840, 2160)], // Support both 1080p and 4K
            skip_invalid_frames: true,                               // Skip frames with unexpected size by default
            cuda_device,
            cuda_stream,
            kernel_func,
            buffer_pool: HashMap::new(),
        })
    }

    /// Set expected input size validation (empty vec = accept any size)
    pub fn with_input_validation(
        mut self,
        expected_sizes: Vec<(u32, u32)>,
        skip_invalid: bool,
    ) -> Self {
        self.expected_input_sizes = expected_sizes;
        self.skip_invalid_frames = skip_invalid;
        self
    }

    /// Update crop region dynamically (for runtime mode switching)
    pub fn update_crop_region(&mut self, new_crop: CropRegion) -> Result<()> {
        self.crop_region = new_crop;
        Ok(())
    }

    /// Get or create output buffer
    fn get_output_buffer(
        &mut self,
        w: u32,
        h: u32,
        fp16: bool,
    ) -> Result<(cudarc::driver::sys::CUdeviceptr, usize)> {
        let key = (w, h, fp16);

        if let Some(&buf) = self.buffer_pool.get(&key) {
            return Ok(buf);
        }

        // Calculate required size: 3 channels * w * h * dtype_size
        let dtype_size = if fp16 { 2 } else { 4 };
        let total_size = (3 * w * h * dtype_size) as usize;

        // Allocate new buffer
        let buf = self
            .cuda_device
            .alloc_zeros::<u8>(total_size)
            .map_err(|e| anyhow!("Failed to allocate GPU buffer: {:?}", e))?;

        let device_ptr = *buf.device_ptr() as cudarc::driver::sys::CUdeviceptr;

        // Keep buffer alive by leaking it (we'll manage lifecycle manually)
        std::mem::forget(buf);

        let buf_info = (device_ptr, total_size);
        self.buffer_pool.insert(key, buf_info);
        Ok(buf_info)
    }

    /// Launch preprocessing kernel
    fn launch_kernel(
        &self,
        input: &RawFramePacket,
        output_buf: cudarc::driver::sys::CUdeviceptr,
    ) -> Result<()> {
        let in_w = input.meta.width as i32;
        let in_h = input.meta.height as i32;
        let in_stride = input.meta.stride_bytes as i32;
        let out_w = self.size.0 as i32;
        let out_h = self.size.1 as i32;

        // Determine pixel format enum value
        let pixfmt = match input.meta.pixfmt {
            PixelFormat::BGRA8 => 0,
            PixelFormat::NV12 => 1,
            PixelFormat::YUV422_8 => 2,
            _ => return Err(anyhow!("Unsupported pixel format: {:?}", input.meta.pixfmt)),
        };

        let chroma_order = self.chroma as i32;

        // Get crop region coordinates based on input size
        let (crop_x, crop_y, crop_w, crop_h) = self.crop_region.get_coords(input.meta.width, input.meta.height);
        let crop_x_i32 = crop_x as i32;
        let crop_y_i32 = crop_y as i32;
        let crop_w_i32 = crop_w as i32;
        let crop_h_i32 = crop_h as i32;

        // For NV12, we need to calculate UV plane offset
        let (uv_plane_ptr, uv_stride) = if pixfmt == 1 {
            let y_plane_size = (in_h * in_stride) as usize;
            let uv_ptr = unsafe { input.data.ptr.add(y_plane_size) };
            (uv_ptr as *const u8, in_stride) // UV stride typically same as Y stride
        } else {
            (std::ptr::null(), 0)
        };

        // Launch configuration: 16x16 blocks
        let block_dim = (16, 16, 1);
        let grid_dim = (((out_w + 15) / 16) as u32, ((out_h + 15) / 16) as u32, 1);

        let config = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes: 0,
        };

        // Prepare kernel arguments
        // cudarc supports Vec<*mut c_void> for arbitrary number of parameters
        let in_ptr = input.data.ptr as cudarc::driver::sys::CUdeviceptr;
        let uv_ptr = uv_plane_ptr as cudarc::driver::sys::CUdeviceptr;
        let fp16_val = if self.fp16 { 1i32 } else { 0i32 };

        let mut params = vec![
            &in_ptr as *const _ as *mut std::ffi::c_void,
            &output_buf as *const _ as *mut std::ffi::c_void,
            &in_w as *const _ as *mut std::ffi::c_void,
            &in_h as *const _ as *mut std::ffi::c_void,
            &in_stride as *const _ as *mut std::ffi::c_void,
            &out_w as *const _ as *mut std::ffi::c_void,
            &out_h as *const _ as *mut std::ffi::c_void,
            &pixfmt as *const _ as *mut std::ffi::c_void,
            &chroma_order as *const _ as *mut std::ffi::c_void,
            &self.mean[0] as *const _ as *mut std::ffi::c_void,
            &self.mean[1] as *const _ as *mut std::ffi::c_void,
            &self.mean[2] as *const _ as *mut std::ffi::c_void,
            &self.std[0] as *const _ as *mut std::ffi::c_void,
            &self.std[1] as *const _ as *mut std::ffi::c_void,
            &self.std[2] as *const _ as *mut std::ffi::c_void,
            &fp16_val as *const _ as *mut std::ffi::c_void,
            &uv_ptr as *const _ as *mut std::ffi::c_void,
            &uv_stride as *const _ as *mut std::ffi::c_void,
            &crop_x_i32 as *const _ as *mut std::ffi::c_void,
            &crop_y_i32 as *const _ as *mut std::ffi::c_void,
            &crop_w_i32 as *const _ as *mut std::ffi::c_void,
            &crop_h_i32 as *const _ as *mut std::ffi::c_void,
        ];

        // Launch kernel using Vec of params
        use cudarc::driver::LaunchAsync;
        unsafe {
            self.kernel_func
                .clone()
                .launch(config, &mut params)
                .map_err(|e| anyhow!("Failed to launch kernel: {:?}", e))?;
        }

        // Synchronize device
        self.cuda_device
            .synchronize()
            .map_err(|e| anyhow!("Failed to synchronize device: {:?}", e))?;

        Ok(())
    }
}

impl Stage<RawFramePacket, TensorInputPacket> for Preprocessor {
    fn process(&mut self, input: RawFramePacket) -> TensorInputPacket {
        self.ensure_gpu_input(&input);
        if !self.dimensions_match(&input) {
            let expected_sizes_str = if self.expected_input_sizes.is_empty() {
                "any size".to_string()
            } else {
                self.expected_input_sizes
                    .iter()
                    .map(|(w, h)| format!("{}x{}", w, h))
                    .collect::<Vec<_>>()
                    .join(" or ")
            };
            panic!(
                "Frame #{} size mismatch: got {}x{}, expected {}",
                input.meta.frame_idx,
                input.meta.width,
                input.meta.height,
                expected_sizes_str
            );
        }
        self.process_inner(input)
    }
}

impl Preprocessor {
    fn ensure_gpu_input(&self, input: &RawFramePacket) {
        if !matches!(input.data.loc, MemLoc::Gpu { device } if device == self.device) {
            panic!(
                "Expected GPU input on device {}, got {:?}",
                self.device, input.data.loc
            );
        }
    }

    fn dimensions_match(&self, input: &RawFramePacket) -> bool {
        if self.expected_input_sizes.is_empty() {
            return true; // Accept any size
        }
        // Check if input matches any of the expected sizes
        self.expected_input_sizes
            .iter()
            .any(|&(w, h)| input.meta.width == w && input.meta.height == h)
    }

    fn process_inner(&mut self, input: RawFramePacket) -> TensorInputPacket {
        if !matches!(input.meta.colorspace, ColorSpace::BT709) {
            eprintln!(
                "Warning: Expected BT709 color space, got {:?}",
                input.meta.colorspace
            );
        }

        let expected_min_stride = match input.meta.pixfmt {
            PixelFormat::BGRA8 => input.meta.width * 4,
            PixelFormat::YUV422_8 => input.meta.width * 2,
            PixelFormat::NV12 => input.meta.width,
            _ => 0,
        };
        if input.meta.stride_bytes < expected_min_stride {
            panic!(
                "Invalid stride: {} < {} for format {:?}",
                input.meta.stride_bytes, expected_min_stride, input.meta.pixfmt
            );
        }

        let (output_buf, total_bytes) = self
            .get_output_buffer(self.size.0, self.size.1, self.fp16)
            .expect("Failed to get output buffer");

        self.launch_kernel(&input, output_buf)
            .expect("Failed to launch preprocessing kernel");

        let desc = TensorDesc {
            n: 1,
            c: 3,
            h: self.size.1,
            w: self.size.0,
            dtype: if self.fp16 { DType::Fp16 } else { DType::Fp32 },
            device: self.device,
        };

        let dtype_size = if self.fp16 { 2 } else { 4 };

        let data = MemRef {
            ptr: output_buf as *mut u8,
            len: total_bytes,
            stride: (self.size.0 * dtype_size) as usize,
            loc: MemLoc::Gpu {
                device: self.device,
            },
        };

        // Add crop region to metadata for downstream coordinate transformation
        let crop_info = if matches!(self.crop_region, CropRegion::None) {
            None
        } else {
            let (cx, cy, cw, ch) = self.crop_region.get_coords(input.meta.width, input.meta.height);
            Some((cx, cy, cw, ch))
        };
        
        let mut output_meta = input.meta;
        output_meta.crop_region = crop_info;

        TensorInputPacket {
            from: output_meta,
            desc,
            data,
        }
    }

    pub fn process_checked(&mut self, input: RawFramePacket) -> Option<TensorInputPacket> {
        self.ensure_gpu_input(&input);
        if !self.dimensions_match(&input) {
            let expected_sizes_str = if self.expected_input_sizes.is_empty() {
                "any size".to_string()
            } else {
                self.expected_input_sizes
                    .iter()
                    .map(|(w, h)| format!("{}x{}", w, h))
                    .collect::<Vec<_>>()
                    .join(" or ")
            };
            
            if self.skip_invalid_frames {
                eprintln!(
                    "⚠️  Skipping frame {} with size {}x{} (expected {})",
                    input.meta.frame_idx,
                    input.meta.width,
                    input.meta.height,
                    expected_sizes_str
                );
                return None;
            } else {
                panic!(
                    "Frame #{} size mismatch: got {}x{}, expected {}",
                    input.meta.frame_idx,
                    input.meta.width,
                    input.meta.height,
                    expected_sizes_str
                );
            }
        }
        Some(self.process_inner(input))
    }
}

/// Load preprocessor from TOML config file
pub fn from_path(cfg_path: &str) -> Result<Preprocessor> {
    let config_str = std::fs::read_to_string(cfg_path)
        .map_err(|e| anyhow!("Failed to read config file: {}", e))?;

    let config: toml::Value =
        toml::from_str(&config_str).map_err(|e| anyhow!("Failed to parse TOML: {}", e))?;

    let preprocess_config: PreprocessConfig = config
        .get("preprocess")
        .ok_or_else(|| anyhow!("Missing [preprocess] section"))?
        .clone()
        .try_into()
        .map_err(|e| anyhow!("Failed to parse preprocess config: {}", e))?;

    let chroma = match preprocess_config.chroma.to_uppercase().as_str() {
        "UYVY" => ChromaOrder::UYVY,
        "YUY2" => ChromaOrder::YUY2,
        _ => {
            return Err(anyhow!(
                "Invalid chroma order: {}",
                preprocess_config.chroma
            ))
        }
    };

    Preprocessor::with_params(
        (preprocess_config.size[0], preprocess_config.size[1]),
        preprocess_config.fp16,
        preprocess_config.device,
        preprocess_config.mean,
        preprocess_config.std,
        chroma,
    )
}

/// Alias for backward compatibility
pub type PreprocessStage = Preprocessor;
