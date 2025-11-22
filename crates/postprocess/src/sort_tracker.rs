use std::collections::HashMap;

use crate::{kalman::KalmanFilter2D, optical_flow::OpticalFlowEstimator};

/// Simple SORT-like tracker enhanced with Kalman filter smoothing, optical-flow
/// guided predictions, and optional velocity-based smoothing.
pub struct SortTrackerWrapper {
    tracks: HashMap<u64, Track>,
    next_id: u64,
    max_age: usize,
    min_hits: usize,
    iou_threshold: f32,
    frame_count: usize,
    motion: TrackerMotionOptions,
    flow_estimator: OpticalFlowEstimator,
}

#[derive(Clone)]
struct Track {
    id: u64,
    bbox: [f32; 4],
    predicted_bbox: [f32; 4],
    class_id: usize,
    score: f32,
    hits: usize,
    time_since_update: usize,
    kalman: Option<KalmanFilter2D>,
    velocity: (f32, f32),
}

#[derive(Clone, Copy, Debug)]
pub struct TrackerMotionOptions {
    pub use_kalman: bool,
    pub kalman_process_noise: f32,
    pub kalman_measurement_noise: f32,
    pub use_optical_flow: bool,
    pub optical_flow_alpha: f32,
    pub optical_flow_max_pixels: f32,
    pub use_velocity: bool,
    pub velocity_alpha: f32,
    pub velocity_max_delta: f32,
    pub velocity_decay: f32,
}

impl Default for TrackerMotionOptions {
    fn default() -> Self {
        Self {
            use_kalman: false,
            kalman_process_noise: 1.0,
            kalman_measurement_noise: 0.5,
            use_optical_flow: false,
            optical_flow_alpha: 0.35,
            optical_flow_max_pixels: 12.0,
            use_velocity: false,
            velocity_alpha: 0.35,
            velocity_max_delta: 12.0,
            velocity_decay: 0.15,
        }
    }
}

impl SortTrackerWrapper {
    pub fn new(max_idle_epochs: usize, _min_confidence: f32, iou_threshold: f32) -> Self {
        Self::with_motion(
            max_idle_epochs,
            _min_confidence,
            iou_threshold,
            TrackerMotionOptions::default(),
        )
    }

    pub fn with_motion(
        max_idle_epochs: usize,
        _min_confidence: f32,
        iou_threshold: f32,
        motion: TrackerMotionOptions,
    ) -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: 1,
            max_age: max_idle_epochs,
            min_hits: 1,
            iou_threshold,
            frame_count: 0,
            flow_estimator: OpticalFlowEstimator::new(
                motion.optical_flow_alpha,
                motion.optical_flow_max_pixels,
            ),
            motion,
        }
    }

    /// Update tracker with new detections
    /// Input: (x1, y1, x2, y2, score, class_id)
    /// Output: (track_id, x1, y1, x2, y2, class_id, score)
    pub fn update(
        &mut self,
        detections: &[(f32, f32, f32, f32, f32, usize)],
    ) -> Vec<(u64, f32, f32, f32, f32, usize, f32)> {
        self.frame_count += 1;

        let flow_delta = if self.motion.use_optical_flow {
            self.flow_estimator.current_flow()
        } else {
            (0.0, 0.0)
        };

        // Predict tracks forward
        for track in self.tracks.values_mut() {
            track.time_since_update += 1;
            track.predict(1.0, &self.motion);
            if self.motion.use_optical_flow {
                track.translate(flow_delta.0, flow_delta.1);
            }
        }

        // Match detections to existing tracks using IoU
        let mut matched_detections = vec![false; detections.len()];
        let mut matched_tracks = HashMap::new();

        for (&track_id, track) in &self.tracks {
            if let Some((det_idx, iou)) = detections
                .iter()
                .enumerate()
                .filter(|(idx, _)| !matched_detections[*idx])
                .map(|(idx, det)| {
                    let iou = calculate_iou(&track.predicted_bbox, &[det.0, det.1, det.2, det.3]);
                    (idx, iou)
                })
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            {
                if iou >= self.iou_threshold {
                    matched_detections[det_idx] = true;
                    matched_tracks.insert(track_id, det_idx);
                }
            }
        }

        let mut flow_samples = Vec::new();

        // Update matched tracks
        for (track_id, det_idx) in matched_tracks {
            if let Some(track) = self.tracks.get_mut(&track_id) {
                let det = &detections[det_idx];
                let measurement_bbox = [det.0, det.1, det.2, det.3];
                let pred_center = track.predicted_center();
                let meas_center = bbox_center(&measurement_bbox);
                flow_samples.push((meas_center.0 - pred_center.0, meas_center.1 - pred_center.1));
                track.apply_measurement(measurement_bbox, det.5, det.4, &self.motion);
            }
        }

        if self.motion.use_optical_flow {
            self.flow_estimator.update(&flow_samples);
        }

        // Create new tracks for unmatched detections
        for (idx, det) in detections.iter().enumerate() {
            if !matched_detections[idx] {
                let track = Track::new(
                    self.next_id,
                    [det.0, det.1, det.2, det.3],
                    det.5,
                    det.4,
                    &self.motion,
                );
                self.tracks.insert(self.next_id, track);
                self.next_id += 1;
            }
        }

        // Remove dead tracks
        self.tracks
            .retain(|_, track| track.time_since_update < self.max_age);

        // Return active tracks (with minimum hits requirement)
        self.tracks
            .values()
            .filter(|track| track.hits >= self.min_hits && track.time_since_update == 0)
            .map(|track| {
                (
                    track.id,
                    track.bbox[0],
                    track.bbox[1],
                    track.bbox[2],
                    track.bbox[3],
                    track.class_id,
                    track.score,
                )
            })
            .collect()
    }

    pub fn get_active_track_count(&self) -> usize {
        self.tracks
            .values()
            .filter(|track| track.time_since_update < self.max_age)
            .count()
    }
}

impl Track {
    fn new(
        id: u64,
        bbox: [f32; 4],
        class_id: usize,
        score: f32,
        motion: &TrackerMotionOptions,
    ) -> Self {
        let kalman = if motion.use_kalman {
            let center = bbox_center(&bbox);
            Some(KalmanFilter2D::new(
                center.0,
                center.1,
                motion.kalman_process_noise,
                motion.kalman_measurement_noise,
            ))
        } else {
            None
        };

        Self {
            id,
            bbox,
            predicted_bbox: bbox,
            class_id,
            score,
            hits: 1,
            time_since_update: 0,
            kalman,
            velocity: (0.0, 0.0),
        }
    }

    fn predict(&mut self, dt: f32, motion: &TrackerMotionOptions) {
        if motion.use_kalman {
            if let Some(filter) = &mut self.kalman {
                let (cx, cy) = filter.predict(dt);
                let (w, h) = bbox_dimensions(&self.bbox);
                self.predicted_bbox = bbox_from_center(cx, cy, w, h);
            } else {
                self.predicted_bbox = self.bbox;
            }
        } else {
            self.predicted_bbox = self.bbox;
        }

        self.apply_velocity_prediction(dt, motion);
    }

    fn apply_measurement(
        &mut self,
        measurement_bbox: [f32; 4],
        class_id: usize,
        score: f32,
        motion: &TrackerMotionOptions,
    ) {
        let previous_center = bbox_center(&self.bbox);
        self.bbox = measurement_bbox;
        self.class_id = class_id;
        self.score = score;
        self.hits += 1;
        self.time_since_update = 0;

        if motion.use_kalman {
            if let Some(filter) = &mut self.kalman {
                let center = bbox_center(&measurement_bbox);
                filter.correct(center.0, center.1);
            }
        }

        if motion.use_velocity {
            let new_center = bbox_center(&self.bbox);
            self.update_velocity(previous_center, new_center, motion);
        }
    }

    fn predicted_center(&self) -> (f32, f32) {
        bbox_center(&self.predicted_bbox)
    }

    fn translate(&mut self, dx: f32, dy: f32) {
        self.predicted_bbox[0] += dx;
        self.predicted_bbox[1] += dy;
        self.predicted_bbox[2] += dx;
        self.predicted_bbox[3] += dy;
        if self.time_since_update > 0 {
            self.bbox = self.predicted_bbox;
        }
    }

    fn update_velocity(
        &mut self,
        previous_center: (f32, f32),
        new_center: (f32, f32),
        motion: &TrackerMotionOptions,
    ) {
        let alpha = motion.velocity_alpha.clamp(0.0, 1.0);
        let delta = (
            new_center.0 - previous_center.0,
            new_center.1 - previous_center.1,
        );
        let blended = (
            self.velocity.0 * (1.0 - alpha) + delta.0 * alpha,
            self.velocity.1 * (1.0 - alpha) + delta.1 * alpha,
        );
        self.velocity = clamp_vector(blended, motion.velocity_max_delta);
    }

    fn apply_velocity_prediction(&mut self, dt: f32, motion: &TrackerMotionOptions) {
        if !motion.use_velocity {
            return;
        }

        if self.time_since_update > 0 {
            let decay = motion.velocity_decay.clamp(0.0, 1.0);
            let decay_factor = 1.0 - decay;
            self.velocity.0 *= decay_factor;
            self.velocity.1 *= decay_factor;
        }

        let dx = self.velocity.0 * dt;
        let dy = self.velocity.1 * dt;
        if dx.abs() > f32::EPSILON || dy.abs() > f32::EPSILON {
            self.translate(dx, dy);
        }
    }
}

/// Calculate Intersection over Union (IoU) between two bounding boxes
fn calculate_iou(bbox1: &[f32; 4], bbox2: &[f32; 4]) -> f32 {
    let x1_inter = bbox1[0].max(bbox2[0]);
    let y1_inter = bbox1[1].max(bbox2[1]);
    let x2_inter = bbox1[2].min(bbox2[2]);
    let y2_inter = bbox1[3].min(bbox2[3]);

    let inter_area = (x2_inter - x1_inter).max(0.0) * (y2_inter - y1_inter).max(0.0);

    if inter_area == 0.0 {
        return 0.0;
    }

    let area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
    let area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
    let union_area = area1 + area2 - inter_area;

    if union_area == 0.0 {
        return 0.0;
    }

    inter_area / union_area
}

fn bbox_center(bbox: &[f32; 4]) -> (f32, f32) {
    ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
}

fn bbox_dimensions(bbox: &[f32; 4]) -> (f32, f32) {
    (bbox[2] - bbox[0], bbox[3] - bbox[1])
}

fn bbox_from_center(cx: f32, cy: f32, w: f32, h: f32) -> [f32; 4] {
    [
        cx - 0.5 * w,
        cy - 0.5 * h,
        cx + 0.5 * w,
        cy + 0.5 * h,
    ]
}

fn clamp_vector(v: (f32, f32), max_len: f32) -> (f32, f32) {
    if max_len <= 0.0 {
        return v;
    }
    let len = (v.0 * v.0 + v.1 * v.1).sqrt();
    if len <= max_len || len <= f32::EPSILON {
        v
    } else {
        let scale = max_len / len;
        (v.0 * scale, v.1 * scale)
    }
}

