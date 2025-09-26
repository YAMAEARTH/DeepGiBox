use image::{imageops::FilterType, io::Reader as ImageReader, GenericImageView, RgbImage};
use ort::{
    execution_providers::TensorRTExecutionProvider,
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
    value::Tensor,
    Error,
    Result,
};
use std::{path::Path, time::Duration};

pub const INPUT_WIDTH: u32 = 512;
pub const INPUT_HEIGHT: u32 = 512;
pub const INPUT_SHAPE: [usize; 4] = [1, 3, INPUT_HEIGHT as usize, INPUT_WIDTH as usize];

pub struct PreprocessedImage {
    pub normalized: Vec<f32>,
    #[allow(dead_code)]
    pub original_size: (u32, u32),
    #[allow(dead_code)]
    pub letterbox_scale: f32,
    #[allow(dead_code)]
    pub letterbox_pad: (f32, f32),
}

pub struct InferenceContext {
    pub session: Session,
    gpu_allocator: Option<Allocator>,
    cpu_allocator: Allocator,
}

pub struct InferenceOutput {
    pub predictions: Vec<f32>,
    #[allow(dead_code)]
    pub shape: Vec<i64>,
    #[allow(dead_code)]
    pub duration: Duration,
}

pub fn init_environment() -> Result<()> {
    let committed = ort::init()
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

    if !committed {
        println!("ort environment already initialized; reusing existing configuration");
    }

    Ok(())
}

impl InferenceContext {
    pub fn new(model_path: &Path) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;

        let cpu_allocator = Allocator::new(
            &session,
            MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default)?,
        )?;

        let gpu_allocator = match Allocator::new(
            &session,
            MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?,
        ) {
            Ok(alloc) => Some(alloc),
            Err(err) => {
                println!("CUDA allocator unavailable; running inference on CPU ({err})");
                None
            }
        };

        Ok(Self {
            session,
            gpu_allocator,
            cpu_allocator,
        })
    }

    pub fn prepare_input(&self, input_data: &[f32]) -> Result<Tensor<f32>> {
        if input_data.len() != INPUT_SHAPE.iter().product() {
            return Err(Error::new(format!(
                "input data length {} does not match expected {}",
                input_data.len(),
                INPUT_SHAPE.iter().product::<usize>()
            )));
        }

        let allocator = self.gpu_allocator.as_ref().unwrap_or(&self.cpu_allocator);
        let mut input_tensor = Tensor::<f32>::new(allocator, INPUT_SHAPE)?;

        if input_tensor.memory_info().is_cpu_accessible() {
            let (_, tensor_data) = input_tensor.try_extract_tensor_mut::<f32>()?;
            tensor_data.copy_from_slice(input_data);
        } else {
            let mut staging_tensor = Tensor::<f32>::new(&Allocator::default(), INPUT_SHAPE)?;
            let (_, staging_data) = staging_tensor.try_extract_tensor_mut::<f32>()?;
            staging_data.copy_from_slice(input_data);
            staging_tensor.copy_into(&mut input_tensor)?;
        }

        Ok(input_tensor)
    }

    pub fn run(&mut self, input_tensor: &Tensor<f32>) -> Result<InferenceOutput> {
        let mut io_binding = self.session.create_binding()?;
        io_binding.bind_input("images", input_tensor)?;
        let cpu_mem_info = self.cpu_allocator.memory_info();
        io_binding.bind_output_to_device("output", &cpu_mem_info)?;

        let start = std::time::Instant::now();
        let outputs = self.session.run_binding(&io_binding)?;
        let duration = start.elapsed();

        let (tensor_shape, predictions) = if let Some(value) = outputs.get("output") {
            let (shape, data) = value.try_extract_tensor::<f32>()?;
            (shape.iter().copied().collect(), data.to_vec())
        } else if let Some(value_ref) = outputs.values().next() {
            let owned = value_ref
                .try_upgrade()
                .map_err(|_| Error::new("Unable to access output tensor"))?;
            let (shape, data) = owned.try_extract_tensor::<f32>()?;
            (shape.iter().copied().collect(), data.to_vec())
        } else {
            return Err(Error::new("No outputs returned from session"));
        };

        Ok(InferenceOutput {
            predictions,
            shape: tensor_shape,
            duration,
        })
    }
}


// this is just generated preprocess replace with kot's version
//***************************************
//*************************************** 
//*************************************** 
//*************************************** 
//***************************************  */
#[allow(dead_code)]
pub fn preprocess_image(image_path: &Path) -> Result<PreprocessedImage> {
    let original = ImageReader::open(image_path)
        .map_err(Error::wrap)?
        .decode()
        .map_err(Error::wrap)?;
    let original_size = original.dimensions();

    let (orig_w, orig_h) = (original_size.0 as f32, original_size.1 as f32);
    let scale = (INPUT_WIDTH as f32 / orig_w).min(INPUT_HEIGHT as f32 / orig_h);
    let new_w = (orig_w * scale).round() as u32;
    let new_h = (orig_h * scale).round() as u32;

    let resized = original
        .resize_exact(new_w.max(1), new_h.max(1), FilterType::Triangle)
        .to_rgb8();

    let pad_x = ((INPUT_WIDTH - new_w) / 2) as u32;
    let pad_y = ((INPUT_HEIGHT - new_h) / 2) as u32;

    let mut letterboxed = RgbImage::from_pixel(INPUT_WIDTH, INPUT_HEIGHT, image::Rgb([114, 114, 114]));
    for y in 0..new_h {
        for x in 0..new_w {
            let pixel = resized.get_pixel(x, y);
            letterboxed.put_pixel(x + pad_x, y + pad_y, *pixel);
        }
    }

    let mut normalized = vec![0.0f32; (INPUT_WIDTH * INPUT_HEIGHT * 3) as usize];
    let plane_size = (INPUT_WIDTH * INPUT_HEIGHT) as usize;
    for y in 0..INPUT_HEIGHT {
        for x in 0..INPUT_WIDTH {
            let pixel = letterboxed.get_pixel(x, y);
            let idx = (y * INPUT_WIDTH + x) as usize;
            normalized[idx] = pixel[0] as f32 / 255.0;
            normalized[idx + plane_size] = pixel[1] as f32 / 255.0;
            normalized[idx + 2 * plane_size] = pixel[2] as f32 / 255.0;
        }
    }

    Ok(PreprocessedImage {
        normalized,
        original_size,
        letterbox_scale: scale,
        letterbox_pad: (pad_x as f32, pad_y as f32),
    })
}
//*************************************** 
//*************************************** 
//*************************************** 
//*************************************** 
