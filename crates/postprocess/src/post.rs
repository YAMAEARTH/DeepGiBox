use similari::utils::bbox::{BoundingBox, Universal2DBox};
use similari::utils::nms::nms;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct Detection {
    pub class_id: usize,
    pub score: f32,
    pub bbox: [f32; 4], // [left, top, right, bottom]
}

// Candidate struct used in NMS pipeline (guideline §5)
#[derive(Clone)]
pub struct Candidate {
    pub detection: Detection,
    pub bbox: Universal2DBox,
}

// Re-export config type alias for guideline compliance
pub type PostprocessConfig = YoloPostConfig;

// Temporal smoothing for video frame predictions
pub struct TemporalSmoother {
    window_size: usize,
    frame_scores: VecDeque<Vec<f32>>, // Sliding window of confidence scores per frame
}

impl TemporalSmoother {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            frame_scores: VecDeque::new(),
        }
    }

    pub fn add_frame_scores(&mut self, scores: Vec<f32>) {
        // Add new frame scores
        self.frame_scores.push_back(scores);

        // Remove oldest frame if window is full
        if self.frame_scores.len() > self.window_size {
            self.frame_scores.pop_front();
        }
    }

    pub fn get_smoothed_scores(&self, current_scores: &[f32]) -> Vec<f32> {
        if self.frame_scores.is_empty() {
            return current_scores.to_vec();
        }

        let mut smoothed = Vec::with_capacity(current_scores.len());

        for i in 0..current_scores.len() {
            let mut values = Vec::new();

            for frame in &self.frame_scores {
                if i < frame.len() {
                    values.push(frame[i]);
                }
            }

            values.push(current_scores[i]);

            // Compute median
            let median = if values.is_empty() {
                current_scores[i]
            } else {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mid = values.len() / 2;
                if values.len() % 2 == 0 && values.len() > 1 {
                    (values[mid - 1] + values[mid]) / 2.0
                } else {
                    values[mid]
                }
            };

            smoothed.push(median);
        }

        smoothed
    }
}

#[derive(Debug, Default, Clone)]
pub struct PostprocessingResult {
    pub detections: Vec<Detection>,
}

#[derive(Debug, Clone, Copy)]
pub struct YoloPostConfig {
    pub num_classes: usize,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    /// Maximum number of detections to retain after NMS (0 means no limit)
    pub max_detections: usize,
    pub letterbox_scale: f32,
    pub letterbox_pad: (f32, f32),
    pub original_size: (u32, u32),
    /// If true, use stretch resize coordinate transformation (no letterbox)
    /// If false, use letterbox coordinate transformation (default)
    pub use_stretch_resize: bool,
}

pub fn postprocess_yolov5_with_temporal_smoothing(
    predictions: &[f32],
    cfg: &YoloPostConfig,
    smoother: Option<&mut TemporalSmoother>,
) -> PostprocessingResult {
    let total_anchors = predictions.len() / (5 + cfg.num_classes);

    let mut decoded = decode_predictions(predictions, cfg);
    let decoded_count = decoded.len();

    println!(
        "    Postprocess stats: {} total anchors → {} passed confidence threshold ({:.1}%)",
        total_anchors,
        decoded_count,
        (decoded_count as f32 / total_anchors as f32) * 100.0
    );

    // Apply temporal smoothing if smoother is provided
    if let Some(smoother) = smoother {
        // Extract current confidence scores
        let current_scores: Vec<f32> = decoded.iter().map(|c| c.detection.score).collect();

        // Get smoothed scores
        let smoothed_scores = smoother.get_smoothed_scores(&current_scores);

        // Update detection scores with smoothed values
        for (candidate, &smoothed_score) in decoded.iter_mut().zip(smoothed_scores.iter()) {
            candidate.detection.score = smoothed_score;
        }

        // Debug: Print smoothing info for first few detections before moving current_scores
        if !current_scores.is_empty() && smoothed_scores.len() > 0 {
            let history_size = smoother.frame_scores.len();
            if history_size >= 2 {
                // Only print when we have enough history for actual smoothing
                println!(
                    "    Temporal smoothing active: {} frames in history",
                    history_size
                );
                // Show smoothing effect for first detection
                if current_scores.len() > 0 && smoothed_scores.len() > 0 {
                    println!(
                        "    First detection: original={:.4}, smoothed={:.4}",
                        current_scores[0], smoothed_scores[0]
                    );
                }
            }
        }

        // Add current scores to smoother's history
        smoother.add_frame_scores(current_scores);
    }

    // NMS re-enabled to filter overlapping bounding boxes
    let retained = apply_nms(decoded, cfg);
    let nms_count = retained.len();

    println!("    After NMS: {} detections retained", nms_count);

    let detections = retained
        .iter()
        .map(|candidate| candidate.detection.clone())
        .collect();

    PostprocessingResult { detections }
}

// decode_predictions now uses pub Candidate defined at top
// Optimized with early rejection and reduced allocations
pub fn decode_predictions(predictions: &[f32], cfg: &YoloPostConfig) -> Vec<Candidate> {
    let stride = 5 + cfg.num_classes;
    let inv_scale = if cfg.letterbox_scale > 0.0 {
        1.0 / cfg.letterbox_scale
    } else {
        1.0
    };
    let (pad_x, pad_y) = cfg.letterbox_pad;
    let (orig_w, orig_h) = (cfg.original_size.0 as f32, cfg.original_size.1 as f32);
    
    // Pre-allocate with reasonable capacity to avoid reallocation
    let estimated_capacity = (predictions.len() / stride / 10).max(100).min(1000);
    let mut candidates = Vec::with_capacity(estimated_capacity);

    for chunk in predictions.chunks(stride) {
        if chunk.len() < stride {
            continue;
        }

        // Early rejection: Check objectness first (cheapest operation)
        let objectness_raw = chunk[4];
        // Quick sigmoid approximation check for early rejection
        // If raw value < -5, sigmoid < 0.007, definitely below threshold
        if objectness_raw < -5.0 {
            continue;
        }

        let objectness = sigmoid(objectness_raw);
        // Early rejection: objectness alone must be above threshold/2 to have chance
        if objectness < cfg.confidence_threshold * 0.5 {
            continue;
        }

        let (best_class, class_conf) = best_class(chunk[5..].iter().copied());
        let score = objectness * class_conf;
        if score < cfg.confidence_threshold {
            continue;
        }

        let cx = chunk[0];
        let cy = chunk[1];
        let w = chunk[2].abs();
        let h = chunk[3].abs();

        // Early rejection: unreasonably small or large boxes
        if w < 1.0 || h < 1.0 || w > orig_w * 2.0 || h > orig_h * 2.0 {
            continue;
        }

        // Coordinate transformation: model space (512×512) → image space (original size)
        let (left, top, right, bottom) = if cfg.use_stretch_resize {
            // Stretch resize: direct scale (no padding)
            // model_coord → image_coord: img = model * (orig_size / model_size)
            const MODEL_SIZE: f32 = 512.0;
            let scale_x = orig_w / MODEL_SIZE;
            let scale_y = orig_h / MODEL_SIZE;
            
            let left = (cx - w * 0.5) * scale_x;
            let top = (cy - h * 0.5) * scale_y;
            let right = (cx + w * 0.5) * scale_x;
            let bottom = (cy + h * 0.5) * scale_y;
            
            (left, top, right, bottom)
        } else {
            // Letterbox: remove padding then scale
            // model_coord → remove_pad → scale_up: img = (model - pad) / letterbox_scale
            let mut left = cx - w * 0.5 - pad_x;
            let mut top = cy - h * 0.5 - pad_y;
            let mut right = cx + w * 0.5 - pad_x;
            let mut bottom = cy + h * 0.5 - pad_y;

            left *= inv_scale;
            top *= inv_scale;
            right *= inv_scale;
            bottom *= inv_scale;
            
            (left, top, right, bottom)
        };

        let left = left.clamp(0.0, orig_w);
        let top = top.clamp(0.0, orig_h);
        let right = right.clamp(0.0, orig_w);
        let bottom = bottom.clamp(0.0, orig_h);

        let width = (right - left).max(0.0);
        let height = (bottom - top).max(0.0);
        if width <= 0.0 || height <= 0.0 {
            continue;
        }

        let bbox = BoundingBox::new_with_confidence(left, top, width, height, score);
        let universal = bbox.as_xyaah();

        candidates.push(Candidate {
            detection: Detection {
                class_id: best_class,
                score,
                bbox: [left, top, right, bottom],
            },
            bbox: universal,
        });
    }
    
    candidates
}

pub fn apply_nms(candidates: Vec<Candidate>, cfg: &YoloPostConfig) -> Vec<Candidate> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let boxes: Vec<(Universal2DBox, Option<f32>)> = candidates
        .iter()
        .map(|candidate| (candidate.bbox.clone(), Some(candidate.detection.score)))
        .collect();

    let keep = nms(&boxes, cfg.nms_threshold, Some(cfg.confidence_threshold));

    let mut retained: Vec<Candidate> = candidates
        .into_iter()
        .filter(|candidate| keep.iter().any(|&kept| *kept == candidate.bbox))
        .collect();

    retained.sort_by(|a, b| {
        b.detection
            .score
            .partial_cmp(&a.detection.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if cfg.max_detections > 0 {
        retained.truncate(cfg.max_detections);
    }
    retained
}

fn best_class(scores: impl Iterator<Item = f32>) -> (usize, f32) {
    scores
        .enumerate()
        .map(|(idx, score)| (idx, sigmoid(score)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, 0.0))
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
