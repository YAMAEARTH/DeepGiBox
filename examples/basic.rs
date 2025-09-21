use ort::{session::Session, value::Tensor, Result};
use ndarray::Array4;
use std::time::Instant;

fn main() -> Result<()> {
    let _committed = ort::init()
        .with_name("ort_binding_env")
        .with_telemetry(false)  // Disable telemetry
        .commit();

    let model_path = "./YOLOv5.onnx";
    
    let mut session = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .commit_from_file(model_path)?;

    
    println!("Model inputs:");
    for (i, input) in session.inputs.iter().enumerate() {
        println!("  Input {}: {} ({:?})", i, input.name, input.input_type);
    }
    
    println!("Model outputs:");
    for (i, output) in session.outputs.iter().enumerate() {
        println!("  Output {}: {} ({:?})", i, output.name, output.output_type);
    }

    let input_shape = (1, 3, 512, 512);
    let input_data: Array4<f32> = Array4::zeros(input_shape);
    
    let input_tensor = Tensor::from_array(input_data)?;
    
    session.run(ort::inputs![input_tensor])?;
    Ok(())
}
