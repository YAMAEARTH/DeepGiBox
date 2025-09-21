mod inference;
mod post;
mod visualize;

use crate::inference::{init_environment, preprocess_image, InferenceContext};
use crate::post::{postprocess_yolov5, SortTracker, YoloPostConfig};
use crate::visualize::visualize_detections;
use ort::Result;
use std::path::Path;

fn main() -> Result<()> {
    init_environment()?;

    let model_path = Path::new("./YOLOv5.onnx");
    let mut context = InferenceContext::new(model_path)?;

    let image_path = Path::new("/home/kot/Documents/pun/ort_binding/cat.jpg");
    let preprocessed = preprocess_image(image_path)?;
    println!(
        "Original image size: {:?}, letterbox scale: {:.3}, pad: {:?}",
        preprocessed.original_size, preprocessed.letterbox_scale, preprocessed.letterbox_pad
    );

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
        original_size: preprocessed.original_size,
        letterbox_scale: preprocessed.letterbox_scale,
        letterbox_pad: preprocessed.letterbox_pad,
    };

    let post_result = postprocess_yolov5(&inference_output.predictions, &post_cfg, Some(&mut tracker));

    if let Some(best_detection) = post_result.detections.first() {
        println!(
            "Top detection: class={} score={:.2} bbox=[{:.1}, {:.1}, {:.1}, {:.1}]",
            best_detection.class_id,
            best_detection.score,
            best_detection.bbox[0],
            best_detection.bbox[1],
            best_detection.bbox[2],
            best_detection.bbox[3]
        );
    } else {
        println!("No detections above threshold.");
    }

    if let Some(best_track) = post_result.tracks.first() {
        println!(
            "Top track: id={} class={} score={:.2} bbox=[{:.1}, {:.1}, {:.1}, {:.1}]",
            best_track.id,
            best_track.class_id,
            best_track.score,
            best_track.bbox[0],
            best_track.bbox[1],
            best_track.bbox[2],
            best_track.bbox[3]
        );
    }

    let output_path = Path::new("cat_detections.png");
    visualize_detections(image_path, &post_result, output_path)?;
    println!("Visualization saved to {}", output_path.display());

    Ok(())
}
