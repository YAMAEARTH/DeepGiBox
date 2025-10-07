use anyhow::Result;
use common_io::{Stage, TensorInputPacket, RawDetectionsPacket, DType, MemLoc};
use ort::{
    execution_providers::TensorRTExecutionProvider,
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
    value::Tensor,
};
use std::path::Path;

pub struct InferenceEngine {
    session: Session,
    gpu_allocator: Allocator,
    cpu_allocator: Allocator,
}

impl InferenceEngine {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        // Initialize ORT environment with TensorRT (matching your old fast config)
        let _ = ort::init()
            .with_name("tensorrt_iobinding")
            .with_execution_providers([
                TensorRTExecutionProvider::default()
                    .with_device_id(0)
                    .with_fp16(true)
                    .with_engine_cache(true)
                    .with_engine_cache_path("./trt_cache")
                    .build(),
            ])
            .commit();

        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path.as_ref())?;        let cpu_allocator = Allocator::new(
            &session,
            MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default)?,
        )?;

        let gpu_allocator = Allocator::new(
            &session,
            MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?,
        )?;

        Ok(Self {
            session,
            gpu_allocator,
            cpu_allocator,
        })
    }

    fn run_inference(&mut self, input_packet: &TensorInputPacket) -> Result<Vec<f32>> {
        let desc = &input_packet.desc;
        let data = &input_packet.data;

        let shape = [desc.n as usize, desc.c as usize, desc.h as usize, desc.w as usize];
        let total_elements = shape.iter().product::<usize>();

        // Create input tensor - optimized like your old fast code
        let input_tensor = match (&data.loc, desc.dtype) {
            // FAST PATH: Data already on GPU (zero-copy from preprocess_cuda)
            (MemLoc::Gpu { device: gpu_id }, DType::Fp32) if *gpu_id == 0 => {
                let tensor = Tensor::<f32>::new(&self.gpu_allocator, shape)?;
                unsafe {
                    // GPU→GPU memcpy is extremely fast (~0.1ms)
                    cudaMemcpy(
                        tensor.data_ptr() as *mut std::ffi::c_void,
                        data.ptr as *const std::ffi::c_void,
                        total_elements * std::mem::size_of::<f32>(),
                        cudaMemcpyDeviceToDevice,
                    );
                }
                tensor
            }
            
            // CPU input - use your old smart approach
            (MemLoc::Cpu, DType::Fp32) => {
                // Try to create GPU tensor first
                let mut input_tensor = Tensor::<f32>::new(&self.gpu_allocator, shape)?;
                
                if input_tensor.memory_info().is_cpu_accessible() {
                    // FAST: GPU tensor is CPU-accessible, direct copy
                    let (_, tensor_data) = input_tensor.try_extract_tensor_mut::<f32>()?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.ptr as *const f32,
                            tensor_data.as_mut_ptr(),
                            total_elements,
                        );
                    }
                } else {
                    // FALLBACK: Need staging for H2D transfer
                    let mut staging = Tensor::<f32>::new(&Allocator::default(), shape)?;
                    let (_, staging_data) = staging.try_extract_tensor_mut::<f32>()?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.ptr as *const f32,
                            staging_data.as_mut_ptr(),
                            total_elements,
                        );
                    }
                    staging.copy_into(&mut input_tensor)?;
                }
                
                input_tensor
            }
            
            (_, DType::Fp16) => {
                anyhow::bail!("FP16 input not yet supported - use FP32 from preprocess");
            }
            
            (MemLoc::Gpu { device }, _) => {
                anyhow::bail!("GPU device {} not supported (only device 0)", device);
            }
        };

        // Temporary CUDA FFI declarations (until proper CUDA binding)
        #[link(name = "cudart")]
        extern "C" {
            fn cudaMemcpy(
                dst: *mut std::ffi::c_void,
                src: *const std::ffi::c_void,
                count: usize,
                kind: i32,
            ) -> i32;
        }
        #[allow(non_upper_case_globals)]
        const cudaMemcpyDeviceToDevice: i32 = 3;

        // Run inference with IO binding (GPU input → TensorRT compute → CPU output)
        let mut io_binding = self.session.create_binding()?;
        io_binding.bind_input("images", &input_tensor)?;
        
        // Bind output to CPU for easy extraction (adds ~2-3ms but simplifies postprocess)
        io_binding.bind_output_to_device("output", &self.cpu_allocator.memory_info())?;

        let outputs = self.session.run_binding(&io_binding)?;

        // Extract predictions from CPU memory
        let predictions = if let Some(value) = outputs.get("output") {
            value.try_extract_tensor::<f32>()?.1.to_vec()
        } else {
            outputs
                .values()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No output tensor"))?
                .try_upgrade()
                .map_err(|_| anyhow::anyhow!("Cannot access output tensor"))?
                .try_extract_tensor::<f32>()?
                .1
                .to_vec()
        };

        Ok(predictions)
    }
}

impl Stage<TensorInputPacket, RawDetectionsPacket> for InferenceEngine {
    fn name(&self) -> &'static str {
        "InferenceEngine"
    }

    fn process(&mut self, input: TensorInputPacket) -> RawDetectionsPacket {
        let start = std::time::Instant::now();
        match self.run_inference(&input) {
            Ok(predictions) => {
                let duration = start.elapsed();
                
                // Print output info for debugging
                println!("  ✓ Inference time: {:.2}ms", duration.as_secs_f64() * 1000.0);
                println!("  ✓ Raw predictions: {} values", predictions.len());
                if predictions.len() >= 10 {
                    println!("  ✓ First 10: {:?}", &predictions[0..10]);
                }
                if predictions.len() > 10 {
                    println!("  ✓ Last 10: {:?}", &predictions[predictions.len()-10..]);
                }
                
                // Return predictions for postprocess stage
                RawDetectionsPacket { 
                    from: input.from,
                    raw_output: predictions,
                }
            }
            Err(e) => {
                eprintln!("Inference error: {}", e);
                RawDetectionsPacket { 
                    from: input.from,
                    raw_output: Vec::new(),
                }
            }
        }
    }
}

// Helper function to create from config path
pub fn from_path(model_path: &str) -> Result<InferenceEngine> {
    InferenceEngine::new(model_path)
}

// Type alias for backwards compatibility
pub type InferStage = InferenceEngine;
