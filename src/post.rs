use similari::trackers::sort::{
    metric::DEFAULT_MINIMAL_SORT_CONFIDENCE,
    simple_api::Sort,
    PositionalMetricType,
};
use similari::utils::bbox::{BoundingBox, Universal2DBox};
use similari::utils::nms::nms;

#[derive(Debug, Clone)]
pub struct Detection {
    pub class_id: usize,
    pub score: f32,
    pub bbox: [f32; 4],
}

#[derive(Debug, Clone)]
pub struct TrackedObject {
    pub id: u64,
    pub class_id: usize,
    pub score: f32,
    pub bbox: [f32; 4],
}

#[derive(Debug, Default, Clone)]
pub struct PostprocessingResult {
    pub detections: Vec<Detection>,
    pub tracks: Vec<TrackedObject>,
}

#[derive(Debug, Clone, Copy)]
pub struct YoloPostConfig {
    pub num_classes: usize,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub max_detections: usize,
    pub letterbox_scale: f32,
    pub letterbox_pad: (f32, f32),
    pub original_size: (u32, u32),
}

pub struct SortTracker {
    sort: Sort,
}

impl SortTracker {
    pub fn new(max_idle_epochs: usize, _min_hits: usize, iou_threshold: f32) -> Self {
        let sort = Sort::new(
            1,
            1,
            max_idle_epochs.max(1),
            PositionalMetricType::IoU(iou_threshold.max(0.0)),
            DEFAULT_MINIMAL_SORT_CONFIDENCE,
            None,
        );
        Self { sort }
    }

    fn update(&mut self, retained: &[Candidate]) -> Vec<TrackedObject> {
        let inputs: Vec<(Universal2DBox, Option<i64>)> = retained
            .iter()
            .map(|candidate| {
                (
                    candidate.bbox.clone(),
                    Some(candidate.detection.class_id as i64),
                )
            })
            .collect();

        self.sort
            .predict(&inputs)
            .into_iter()
            .map(|track| {
                let observed = track.observed_bbox.clone();
                let width = observed.height * observed.aspect;
                let height = observed.height;
                let left = observed.xc - width * 0.5;
                let top = observed.yc - height * 0.5;
                let right = left + width;
                let bottom = top + height;
                let class_id = track
                    .custom_object_id
                    .and_then(|id| if id >= 0 { Some(id as usize) } else { None })
                    .unwrap_or(0);

                TrackedObject {
                    id: track.id,
                    class_id,
                    score: observed.confidence,
                    bbox: [left, top, right, bottom],
                }
            })
            .collect()
    }
}

pub fn postprocess_yolov5(
    predictions: &[f32],
    cfg: &YoloPostConfig,
    tracker: Option<&mut SortTracker>,
) -> PostprocessingResult {
    let decoded = decode_predictions(predictions, cfg);
    let retained = apply_nms(decoded, cfg);

    let detections = retained
        .iter()
        .map(|candidate| candidate.detection.clone())
        .collect();

    let tracks = tracker
        .map(|sort| sort.update(&retained))
        .unwrap_or_default();

    PostprocessingResult { detections, tracks }
}

#[derive(Clone)]
struct Candidate {
    detection: Detection,
    bbox: Universal2DBox,
}

fn decode_predictions(predictions: &[f32], cfg: &YoloPostConfig) -> Vec<Candidate> {
    let stride = 5 + cfg.num_classes;
    let inv_scale = if cfg.letterbox_scale > 0.0 {
        1.0 / cfg.letterbox_scale
    } else {
        1.0
    };
    let (pad_x, pad_y) = cfg.letterbox_pad;
    let (orig_w, orig_h) = (cfg.original_size.0 as f32, cfg.original_size.1 as f32);

    predictions
        .chunks(stride)
        .filter_map(|chunk| {
            if chunk.len() < stride {
                return None;
            }

            let objectness = sigmoid(chunk[4]);
            let (best_class, class_conf) = best_class(chunk[5..].iter().copied());
            let score = objectness * class_conf;
            if score < cfg.confidence_threshold {
                return None;
            }

            let cx = chunk[0];
            let cy = chunk[1];
            let w = chunk[2].abs();
            let h = chunk[3].abs();

            let mut left = cx - w * 0.5 - pad_x;
            let mut top = cy - h * 0.5 - pad_y;
            let mut right = cx + w * 0.5 - pad_x;
            let mut bottom = cy + h * 0.5 - pad_y;

            left *= inv_scale;
            top *= inv_scale;
            right *= inv_scale;
            bottom *= inv_scale;

            left = left.clamp(0.0, orig_w);
            top = top.clamp(0.0, orig_h);
            right = right.clamp(0.0, orig_w);
            bottom = bottom.clamp(0.0, orig_h);

            let width = (right - left).max(0.0);
            let height = (bottom - top).max(0.0);
            if width <= 0.0 || height <= 0.0 {
                return None;
            }

            let bbox = BoundingBox::new_with_confidence(left, top, width, height, score);
            let universal = bbox.as_xyaah();

            Some(Candidate {
                detection: Detection {
                    class_id: best_class,
                    score,
                    bbox: [left, top, right, bottom],
                },
                bbox: universal,
            })
        })
        .collect()
}

fn apply_nms(candidates: Vec<Candidate>, cfg: &YoloPostConfig) -> Vec<Candidate> {
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

    retained.sort_by(|a, b| b.detection.score.partial_cmp(&a.detection.score).unwrap_or(std::cmp::Ordering::Equal));
    retained.truncate(cfg.max_detections.max(1));
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
