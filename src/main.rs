mod inference;
mod post;
mod visualize;

use crate::inference::{init_environment, preprocess_image, InferenceContext, INPUT_HEIGHT, INPUT_WIDTH};
use crate::post::{postprocess_yolov5, SortTracker, YoloPostConfig};
use crate::visualize::visualize_detections;
use ort::Result;
use std::path::Path;

fn main() -> Result<()> {
    init_environment()?;

    let model_path = Path::new("./YOLOv5.onnx");
    let mut context = InferenceContext::new(model_path)?;

    let image_path = Path::new("/home/kot/Documents/pun/ort_binding/cat.jpg");

    //*************************************** 
    // generated preprocess replace with kot's version
    let preprocessed = preprocess_image(image_path)?;
    //***************************************  */

    let input_tensor = context.prepare_input(&preprocessed.normalized)?;
    let inference_output = context.run(&input_tensor)?;

    println!("Inference completed in {:?}", inference_output.duration);
    println!("Model output shape: {:?}", inference_output.shape);

    let num_classes = inference_output
        .shape
        .last()
        .map(|last| (*last as usize).saturating_sub(5))
        .filter(|&classes| classes > 0)
        .unwrap_or(1);

    let stride = 5 + num_classes;
    if let Some(first_detection) = inference_output.predictions.chunks(stride).next() {
        println!(
            "First raw detection chunk: {:?}",
            &first_detection[..first_detection.len().min(stride)]
        );
    }

    let mut tracker = SortTracker::new(3, 1, 0.3);

    let post_cfg = YoloPostConfig {
        num_classes,
        confidence_threshold: 0.35,
        nms_threshold: 0.45,
        max_detections: 20,
        input_size: (INPUT_WIDTH, INPUT_HEIGHT),
        original_size: preprocessed.original_size,
    };

    let post_result = postprocess_yolov5(&inference_output.predictions, &post_cfg, Some(&mut tracker));

    if post_result.detections.is_empty() {
        println!("No detections above threshold.");
    } else {
        println!("Detections:");
        for detection in post_result.detections.iter().take(10) {
            println!(
                "  class={} score={:.2} bbox=[{:.1}, {:.1}, {:.1}, {:.1}]",
                detection.class_id,
                detection.score,
                detection.bbox[0],
                detection.bbox[1],
                detection.bbox[2],
                detection.bbox[3]
            );
        }
        if post_result.detections.len() > 10 {
            println!("  … {} more detections", post_result.detections.len() - 10);
        }
    }

    if !post_result.tracks.is_empty() {
        println!("Tracked objects:");
        for tracked in post_result.tracks.iter().take(10) {
            println!(
                "  id={} class={} score={:.2} bbox=[{:.1}, {:.1}, {:.1}, {:.1}]",
                tracked.id,
                tracked.class_id,
                tracked.score,
                tracked.bbox[0],
                tracked.bbox[1],
                tracked.bbox[2],
                tracked.bbox[3]
            );
        }
        if post_result.tracks.len() > 10 {
            println!("  … {} more tracks", post_result.tracks.len() - 10);
        }
    }

    let output_path = Path::new("cat_detections.png");
    visualize_detections(&preprocessed.original, &post_result, output_path)?;
    println!("Visualization saved to {}", output_path.display());

    Ok(())
}
