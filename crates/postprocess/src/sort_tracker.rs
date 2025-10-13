use std::collections::HashMap;

/// Simple SORT-like tracker using IoU matching
pub struct SortTrackerWrapper {
    tracks: HashMap<u64, Track>,
    next_id: u64,
    max_age: usize,
    min_hits: usize,
    iou_threshold: f32,
    frame_count: usize,
}

#[derive(Clone)]
struct Track {
    id: u64,
    bbox: [f32; 4], // [x1, y1, x2, y2]
    class_id: usize,
    score: f32,
    age: usize,  // Frames since last update
    hits: usize, // Total number of matches
    time_since_update: usize,
}

impl SortTrackerWrapper {
    pub fn new(max_idle_epochs: usize, _min_confidence: f32, iou_threshold: f32) -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: 1,
            max_age: max_idle_epochs,
            min_hits: 1, // Require at least 1 hit to create track
            iou_threshold,
            frame_count: 0,
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

        // Increment time since update for all tracks
        for track in self.tracks.values_mut() {
            track.time_since_update += 1;
        }

        // Match detections to existing tracks using IoU
        let mut matched_detections = vec![false; detections.len()];
        let mut matched_tracks = HashMap::new();

        // For each track, find best matching detection
        for (&track_id, track) in &self.tracks {
            if let Some((det_idx, iou)) = detections
                .iter()
                .enumerate()
                .filter(|(idx, _)| !matched_detections[*idx])
                .map(|(idx, det)| {
                    let iou = calculate_iou(&track.bbox, &[det.0, det.1, det.2, det.3]);
                    (idx, iou)
                })
                .max_by(|(_, iou1), (_, iou2)| iou1.partial_cmp(iou2).unwrap())
            {
                if iou >= self.iou_threshold {
                    matched_detections[det_idx] = true;
                    matched_tracks.insert(track_id, det_idx);
                }
            }
        }

        // Update matched tracks
        for (track_id, det_idx) in matched_tracks {
            if let Some(track) = self.tracks.get_mut(&track_id) {
                let det = &detections[det_idx];
                track.bbox = [det.0, det.1, det.2, det.3];
                track.class_id = det.5;
                track.score = det.4;
                track.hits += 1;
                track.time_since_update = 0;
            }
        }

        // Create new tracks for unmatched detections
        for (idx, det) in detections.iter().enumerate() {
            if !matched_detections[idx] {
                let track = Track {
                    id: self.next_id,
                    bbox: [det.0, det.1, det.2, det.3],
                    class_id: det.5,
                    score: det.4,
                    age: 0,
                    hits: 1,
                    time_since_update: 0,
                };
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

/// Calculate Intersection over Union (IoU) between two bounding boxes
fn calculate_iou(bbox1: &[f32; 4], bbox2: &[f32; 4]) -> f32 {
    // bbox format: [x1, y1, x2, y2]
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
