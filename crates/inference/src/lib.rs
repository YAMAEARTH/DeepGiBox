use anyhow::Result;
use common_io::{DType, RawDetectionsPacket, Stage, TensorInputPacket};
use config::InferenceCfg;
use once_cell::sync::Lazy;
use ort::{
    execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider},
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
};
use telemetry::{now_ns, record_ms};

/// Global ORT environment - initialized once at first use
static ORT_ENVIRONMENT: Lazy<()> = Lazy::new(|| {
    ort::init()
        .with_name("deepgibox_inference")
        .commit()
        .expect("Failed to initialize ORT environment");
    
    println!("[ORT] Environment initialized");
});

pub struct OrtTrtEngine {
    session: Session,
    device: u32,
    fp16: bool,
    input_name: String,
    output_names: Vec<String>,
    gpu_mem_info: MemoryInfo,
    cpu_mem_info: MemoryInfo,
}

impl OrtTrtEngine {
    pub fn new(cfg: &InferenceCfg) -> Result<Self> {
        // Initialize ORT environment once (lazy initialization)
        Lazy::force(&ORT_ENVIRONMENT);

        println!("[Inference] Initializing with config:");
        println!("  Model:         {}", cfg.model);
        println!("  Device:        GPU {}", cfg.device);
        println!("  FP16:          {}", cfg.fp16);
        println!("  Engine Cache:  {}", cfg.engine_cache);
        println!("  Timing Cache:  {}", cfg.timing_cache);

        // Ensure cache directories exist
        std::fs::create_dir_all(&cfg.engine_cache)?;
        std::fs::create_dir_all(&cfg.timing_cache)?;

        // Build session with TensorRT EP
        let mut session_builder = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?;

        // Configure TensorRT EP
        let mut trt_provider = TensorRTExecutionProvider::default()
            .with_device_id(cfg.device as i32)
            .with_engine_cache(true)
            .with_engine_cache_path(&cfg.engine_cache)
            .with_timing_cache(true);

        if cfg.fp16 {
            trt_provider = trt_provider.with_fp16(true);
        }

        session_builder = session_builder.with_execution_providers([trt_provider.build()])?;

        // Optional: Add CUDA EP as fallback
        if cfg.enable_fallback_cuda.unwrap_or(true) {
            let cuda_provider = CUDAExecutionProvider::default()
                .with_device_id(cfg.device as i32);
            session_builder = session_builder.with_execution_providers([cuda_provider.build()])?;
        }

        let session = session_builder.commit_from_file(&cfg.model)?;

        // Get input/output names
        let input_name = cfg.input_name.clone().unwrap_or_else(|| "images".to_string());
        let output_names = cfg.output_names.clone().unwrap_or_else(|| vec!["output".to_string()]);

        println!("[Inference] Model loaded:");
        println!("  Input:  {}", input_name);
        println!("  Outputs: {:?}", output_names);

        // Create memory info for GPU and CPU
        let gpu_mem_info = MemoryInfo::new(
            AllocationDevice::CUDA,
            cfg.device as i32,
            AllocatorType::Device,
            MemoryType::Default
        )?;
        
        let cpu_mem_info = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::Default
        )?;

        let mut engine = Self {
            session,
            device: cfg.device,
            fp16: cfg.fp16,
            input_name,
            output_names,
            gpu_mem_info,
            cpu_mem_info,
        };

        // Warmup
        let warmup_runs = cfg.warmup_runs.unwrap_or(5);
        if warmup_runs > 0 {
            println!("[Inference] Warming up with {} runs...", warmup_runs);
            engine.warmup(warmup_runs)?;
            println!("[Inference] Warmup complete");
        }

        Ok(engine)
    }

    fn warmup(&mut self, runs: usize) -> Result<()> {
        use common_io::{FrameMeta, PixelFormat, ColorSpace, TensorDesc, MemRef, MemLoc};
        use cudarc::driver::DevicePtr;
        
        // Create a dummy input tensor for warmup
        let dummy_meta = FrameMeta {
            source_id: 0,
            frame_idx: 0,
            width: 512,
            height: 512,
            pixfmt: PixelFormat::RGB8,
            colorspace: ColorSpace::BT709,
            pts_ns: 0,
            t_capture_ns: 0,
            stride_bytes: 512 * 3,
            crop_region: None,
        };

        let shape = [1, 3, 512, 512];
        let total_elements: usize = shape.iter().product();
        let element_size = if self.fp16 { 2 } else { 4 };
        let total_bytes = total_elements * element_size;

        // Allocate GPU memory for dummy input
        let device = cudarc::driver::CudaDevice::new(self.device as usize)?;
        let dummy_buffer = device.alloc_zeros::<u8>(total_bytes)?;
        let gpu_ptr = *dummy_buffer.device_ptr() as *mut u8;

        let dummy_input = TensorInputPacket {
            from: dummy_meta,
            desc: TensorDesc {
                n: 1,
                c: 3,
                h: 512,
                w: 512,
                dtype: if self.fp16 { DType::Fp16 } else { DType::Fp32 },
                device: self.device,
            },
            data: MemRef {
                ptr: gpu_ptr,
                len: total_bytes,
                stride: 512 * 3,
                loc: MemLoc::Gpu { device: self.device },
            },
        };

        for i in 0..runs {
            let _ = self.run_inference(&dummy_input);
            if i == 0 {
                println!("  First run (building TRT engine - may take time)...");
            }
        }

        // Clean up
        drop(dummy_buffer);

        Ok(())
    }

    fn run_inference(&mut self, input_packet: &TensorInputPacket) -> Result<(Vec<f32>, Vec<usize>)> {
        let t_start = now_ns();
        
        let desc = &input_packet.desc;
        let data = &input_packet.data;

        // Validate input
        assert_eq!(desc.n, 1, "Batch size must be 1");
        assert!(matches!(desc.dtype, DType::Fp16 | DType::Fp32), "Only FP16/FP32 supported");

        let shape = [
            desc.n as usize,
            desc.c as usize,
            desc.h as usize,
            desc.w as usize,
        ];
        let total_elements = shape.iter().product::<usize>();

        // Check if we need to handle FP16 input
        let use_fp16 = matches!(desc.dtype, DType::Fp16);
        let is_gpu_memory = matches!(data.loc, common_io::MemLoc::Gpu { .. });

        // Bind input - Current implementation uses CPU staging
        // TODO: Implement true GPU-to-GPU I/O binding for zero-copy
        let t_bind_start = now_ns();
        let input_tensor = unsafe {
            use ort::value::Tensor;
            let mut staging = Tensor::<f32>::new(&Allocator::default(), shape)?;
            let (_, staging_data) = staging.try_extract_tensor_mut::<f32>()?;
            
            if use_fp16 && is_gpu_memory {
                // FP16 data on GPU - copy to CPU and convert
                let mut cpu_fp16_raw = vec![0u16; total_elements];
                cudarc::driver::result::memcpy_dtoh_sync(
                    &mut cpu_fp16_raw,
                    data.ptr as cudarc::driver::sys::CUdeviceptr,
                )?;
                
                for (i, &raw_val) in cpu_fp16_raw.iter().enumerate() {
                    let fp16_val = half::f16::from_bits(raw_val);
                    staging_data[i] = fp16_val.to_f32();
                }
            } else if use_fp16 && !is_gpu_memory {
                let fp16_slice = std::slice::from_raw_parts(data.ptr as *const half::f16, total_elements);
                for (i, &fp16_val) in fp16_slice.iter().enumerate() {
                    staging_data[i] = fp16_val.to_f32();
                }
            } else if !use_fp16 && is_gpu_memory {
                cudarc::driver::result::memcpy_dtoh_sync(
                    staging_data,
                    data.ptr as cudarc::driver::sys::CUdeviceptr,
                )?;
            } else {
                let slice = std::slice::from_raw_parts(data.ptr as *const f32, total_elements);
                staging_data.copy_from_slice(slice);
            }

            staging
        };
        record_ms("inference.bind", t_bind_start);

        // Create I/O binding
        let mut io_binding = self.session.create_binding()?;
        io_binding.bind_input(&self.input_name, &input_tensor)?;

        // Bind output to CPU for easy extraction
        for output_name in &self.output_names {
            io_binding.bind_output_to_device(output_name, &self.cpu_mem_info)?;
        }

        // Run inference
        let t_run_start = now_ns();
        let outputs = self.session.run_binding(&io_binding)?;
        record_ms("inference.run", t_run_start);

        // Extract outputs
        let t_extract_start = now_ns();
        let (predictions, output_shape) = if let Some(value) = outputs.get(&self.output_names[0]) {
            let (shape, data) = value.try_extract_tensor::<f32>()?;
            (data.to_vec(), shape.to_vec())
        } else {
            let value = outputs
                .values()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No output tensor"))?
                .try_upgrade()
                .map_err(|_| anyhow::anyhow!("Cannot access output tensor"))?;
            let (shape, data) = value.try_extract_tensor::<f32>()?;
            (data.to_vec(), shape.to_vec())
        };
        record_ms("inference.extract", t_extract_start);

        // Record total inference time
        record_ms("inference", t_start);

        // Convert shape from i64 to usize
        let output_shape_usize: Vec<usize> = output_shape.iter().map(|&x| x as usize).collect();
        Ok((predictions, output_shape_usize))
    }

    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }
}

impl Stage<TensorInputPacket, RawDetectionsPacket> for OrtTrtEngine {
    fn name(&self) -> &'static str {
        "OrtTrtEngine"
    }

    fn process(&mut self, input: TensorInputPacket) -> RawDetectionsPacket {
        match self.run_inference(&input) {
            Ok((predictions, output_shape)) => {
                RawDetectionsPacket {
                    from: input.from,
                    raw_output: predictions,
                    output_shape,
                }
            }
            Err(e) => {
                eprintln!("[Inference] Error: {}", e);
                RawDetectionsPacket {
                    from: input.from,
                    raw_output: Vec::new(),
                    output_shape: Vec::new(),
                }
            }
        }
    }
}

// Helper function to create from config path
pub fn from_path(cfg_path: &str) -> Result<OrtTrtEngine> {
    let app_cfg = config::AppConfig::from_file(cfg_path)?;
    let inf_cfg = app_cfg.inference
        .ok_or_else(|| anyhow::anyhow!("No [inference] section in config"))?;
    OrtTrtEngine::new(&inf_cfg)
}

// Backwards compatibility aliases
pub type InferenceEngine = OrtTrtEngine;
pub type InferStage = OrtTrtEngine;

#[cfg(test)]
mod tests;
