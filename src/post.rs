use similari::trackers::sort::{metric::DEFAULT_MINIMAL_SORT_CONFIDENCE, simple_api::Sort, PositionalMetricType};
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
    pub input_size: (u32, u32),
    pub original_size: (u32, u32),
}

pub struct SortTracker {
    inner: Sort,
}

impl SortTracker {
    pub fn new(max_idle_epochs: usize, _min_hits: usize, iou_threshold: f32) -> Self {
        let inner = Sort::new(
            1,
            1,
            max_idle_epochs.max(1),
            PositionalMetricType::IoU(iou_threshold),
            DEFAULT_MINIMAL_SORT_CONFIDENCE,
            None,
        );

        Self { inner }
    }

    fn update(&mut self, candidates: &[Candidate]) -> Vec<TrackedObject> {
        let input: Vec<(Universal2DBox, Option<i64>)> = candidates
            .iter()
            .map(|c| (c.bbox.clone(), Some(c.detection.class_id as i64)))
            .collect();

        self.inner
            .predict(&input)
            .into_iter()
            .filter_map(|track| {
                let (left, top, width, height) = universal_to_ltwh(&track.observed_bbox);
                let right = left + width;
                let bottom = top + height;

                let class_id = track
                    .custom_object_id
                    .and_then(|id| if id >= 0 { Some(id as usize) } else { None })
                    .unwrap_or(0);

                Some(TrackedObject {
                    id: track.id,
                    class_id,
                    score: track.observed_bbox.confidence,
                    bbox: [left, top, right, bottom],
                })
            })
            .collect()
    }
}

pub fn postprocess_yolov5(
    predictions: &[f32],
    cfg: &YoloPostConfig,
    tracker: Option<&mut SortTracker>,
) -> PostprocessingResult {
    let decoded = decode_yolov5(predictions, cfg);
    let filtered = apply_nms(decoded, cfg);

    let tracks = tracker
        .map(|sort| sort.update(&filtered))
        .unwrap_or_default();

    let detections = filtered
        .into_iter()
        .map(|candidate| candidate.detection)
        .collect();

    PostprocessingResult { detections, tracks }
}

#[derive(Clone)]
struct Candidate {
    detection: Detection,
    bbox: Universal2DBox,
}

fn decode_yolov5(predictions: &[f32], cfg: &YoloPostConfig) -> Vec<Candidate> {
    let stride = 5 + cfg.num_classes;
    if stride <= 5 {
        return Vec::new();
    }

    let (input_w, input_h) = (cfg.input_size.0 as f32, cfg.input_size.1 as f32);
    let (orig_w, orig_h) = (cfg.original_size.0 as f32, cfg.original_size.1 as f32);
    let scale_x = orig_w / input_w;
    let scale_y = orig_h / input_h;

    predictions
        .chunks(stride)
        .filter_map(|chunk| {
            if chunk.len() < stride {
                return None;
            }

            let objectness = sigmoid(chunk[4]);
            let (best_class, class_score) = best_class(chunk[5..].iter().copied());
            let score = objectness * class_score;
            if score < cfg.confidence_threshold {
                return None;
            }

            let cx = chunk[0] * scale_x;
            let cy = chunk[1] * scale_y;
            let width = chunk[2].abs() * scale_x;
            let height = chunk[3].abs() * scale_y;

            let left = (cx - width * 0.5).max(0.0);
            let top = (cy - height * 0.5).max(0.0);
            let right = (cx + width * 0.5).min(orig_w);
            let bottom = (cy + height * 0.5).min(orig_h);
            let width = (right - left).max(0.0);
            let height = (bottom - top).max(0.0);

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

    let kept_shapes: Vec<(f32, f32, f32, f32)> = nms(&boxes, cfg.nms_threshold, Some(cfg.confidence_threshold))
        .iter()
        .map(|bbox| universal_to_ltwh(bbox))
        .collect();

    candidates
        .into_iter()
        .filter(|candidate| {
            let shape = universal_to_ltwh(&candidate.bbox);
            kept_shapes.iter().any(|kept| almost_equal(*kept, shape))
        })
        .take(cfg.max_detections.max(1))
        .collect()
}

fn best_class<'a>(scores: impl Iterator<Item = f32>) -> (usize, f32) {
    scores
        .enumerate()
        .map(|(idx, raw)| (idx, sigmoid(raw)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, 0.0))
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn universal_to_ltwh(b: &Universal2DBox) -> (f32, f32, f32, f32) {
    let width = b.height * b.aspect;
    let height = b.height;
    let left = b.xc - width * 0.5;
    let top = b.yc - height * 0.5;
    (left, top, width, height)
}

fn almost_equal(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> bool {
    const EPS: f32 = 1e-3;
    (a.0 - b.0).abs() <= EPS
        && (a.1 - b.1).abs() <= EPS
        && (a.2 - b.2).abs() <= EPS
        && (a.3 - b.3).abs() <= EPS
}
