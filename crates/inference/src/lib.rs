use anyhow::Result;
use common_io::{Stage, TensorInputPacket, RawDetectionsPacket};
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

        // Create GPU tensor - handle both CPU-accessible and pure GPU memory
        let tensor_start = std::time::Instant::now();
        let input_tensor = unsafe {
            let mut gpu_tensor = Tensor::<f32>::new(&self.gpu_allocator, shape)?;
            
            if gpu_tensor.memory_info().is_cpu_accessible() {
                // GPU memory is CPU-accessible - direct copy!
                let (_, tensor_data) = gpu_tensor.try_extract_tensor_mut::<f32>()?;
                std::ptr::copy_nonoverlapping(
                    data.ptr as *const f32,
                    tensor_data.as_mut_ptr(),
                    total_elements,
                );
            } else {
                // Pure GPU memory - use staging tensor for CPU→GPU or GPU→GPU
                let slice = std::slice::from_raw_parts(data.ptr as *const f32, total_elements);
                let mut staging = Tensor::<f32>::new(&Allocator::default(), shape)?;
                let (_, staging_data) = staging.try_extract_tensor_mut::<f32>()?;
                staging_data.copy_from_slice(slice);
                staging.copy_into(&mut gpu_tensor)?;
            }
            
            gpu_tensor
        };
        let tensor_time = tensor_start.elapsed();

        // Run inference with IO binding (GPU input → TensorRT compute → CPU output)
        let binding_start = std::time::Instant::now();
        let mut io_binding = self.session.create_binding()?;
        io_binding.bind_input("images", &input_tensor)?;
        
        // Bind output to CPU for easy extraction
        io_binding.bind_output_to_device("output", &self.cpu_allocator.memory_info())?;
        let binding_time = binding_start.elapsed();

        let inference_start = std::time::Instant::now();
        let outputs = self.session.run_binding(&io_binding)?;
        let inference_time = inference_start.elapsed();

        // Extract predictions from CPU memory
        let extract_start = std::time::Instant::now();
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
        let extract_time = extract_start.elapsed();

        // Print detailed timing breakdown
        println!("    [Timing] Tensor staging: {:.2}ms", tensor_time.as_secs_f64() * 1000.0);
        println!("    [Timing] IO binding setup: {:.2}ms", binding_time.as_secs_f64() * 1000.0);
        println!("    [Timing] TensorRT execution: {:.2}ms", inference_time.as_secs_f64() * 1000.0);
        println!("    [Timing] Output extraction: {:.2}ms", extract_time.as_secs_f64() * 1000.0);

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
