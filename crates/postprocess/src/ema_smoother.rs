/// EMA (Exponential Moving Average) Smoothing for Detection Bounding Boxes
/// 
/// Purpose: Reduce jitter/shake in detection boxes by applying exponential smoothing
/// to both position (x, y) and size (w, h) separately.
///
/// Algorithm:
///   smoothed_value = alpha * new_value + (1 - alpha) * old_value
///   where alpha ∈ (0, 1) controls responsiveness:
///     - alpha = 0.3 (position): Smooth position, reduce jitter
///     - alpha = 0.4 (size): Slightly more responsive for size changes
///
use std::collections::HashMap;

/// Smoothed bounding box with separate EMA for position and size
#[derive(Clone, Debug)]
struct SmoothedBBox {
    x: f32,      // Center X (smoothed)
    y: f32,      // Center Y (smoothed)
    w: f32,      // Width (smoothed)
    h: f32,      // Height (smoothed)
    class_id: i32,
    score: f32,  // Keep latest score (not smoothed)
    age: usize,  // Frames since last update
}

impl SmoothedBBox {
    fn new(x: f32, y: f32, w: f32, h: f32, class_id: i32, score: f32) -> Self {
        Self {
            x, y, w, h,
            class_id,
            score,
            age: 0,
        }
    }

    /// Update with new detection using EMA smoothing
    fn update(&mut self, new_x: f32, new_y: f32, new_w: f32, new_h: f32, new_score: f32, alpha_pos: f32, alpha_size: f32) {
        // Smooth position (center point)
        self.x = alpha_pos * new_x + (1.0 - alpha_pos) * self.x;
        self.y = alpha_pos * new_y + (1.0 - alpha_pos) * self.y;
        
        // Smooth size
        self.w = alpha_size * new_w + (1.0 - alpha_size) * self.w;
        self.h = alpha_size * new_h + (1.0 - alpha_size) * self.h;
        
        // Update score (no smoothing - use latest)
        self.score = new_score;
        
        // Reset age
        self.age = 0;
    }
}

/// EMA Smoother for all tracked detections
pub struct EmaSmoother {
    /// Map: track_id → SmoothedBBox
    boxes: HashMap<i32, SmoothedBBox>,
    
    /// Alpha for position smoothing (0.0 = no update, 1.0 = no smoothing)
    /// Default: 0.3 (smoother position, reduces jitter)
    alpha_position: f32,
    
    /// Alpha for size smoothing
    /// Default: 0.4 (slightly more responsive than position)
    alpha_size: f32,
    
    /// Maximum age (frames) before removing a track
    /// Default: 5 frames
    max_age: usize,
}

impl EmaSmoother {
    /// Create new EMA smoother with default parameters
    pub fn new() -> Self {
        Self {
            boxes: HashMap::new(),
            alpha_position: 0.3,
            alpha_size: 0.4,
            max_age: 5,
        }
    }

    /// Create with custom alpha values
    pub fn with_alpha(alpha_position: f32, alpha_size: f32) -> Self {
        Self {
            boxes: HashMap::new(),
            alpha_position: alpha_position.clamp(0.0, 1.0),
            alpha_size: alpha_size.clamp(0.0, 1.0),
            max_age: 5,
        }
    }

    /// Create with custom max age
    pub fn with_max_age(mut self, max_age: usize) -> Self {
        self.max_age = max_age;
        self
    }

    /// Process detections with EMA smoothing
    /// 
    /// Input: Vec of (track_id, x, y, w, h, class_id, score)
    /// Output: Vec of smoothed (track_id, x, y, w, h, class_id, score)
    pub fn smooth(&mut self, detections: Vec<(i32, f32, f32, f32, f32, i32, f32)>) -> Vec<(i32, f32, f32, f32, f32, i32, f32)> {
        // Increment age for all existing tracks
        for bbox in self.boxes.values_mut() {
            bbox.age += 1;
        }

        // Update or create boxes for current detections
        for (track_id, x, y, w, h, class_id, score) in detections.iter() {
            // Convert corner coordinates to center + size
            let center_x = x + w / 2.0;
            let center_y = y + h / 2.0;

            if let Some(bbox) = self.boxes.get_mut(track_id) {
                // Existing track - smooth update
                bbox.update(center_x, center_y, *w, *h, *score, self.alpha_position, self.alpha_size);
            } else {
                // New track - initialize without smoothing
                self.boxes.insert(*track_id, SmoothedBBox::new(center_x, center_y, *w, *h, *class_id, *score));
            }
        }

        // Remove stale tracks (not updated for max_age frames)
        self.boxes.retain(|_, bbox| bbox.age <= self.max_age);

        // Return smoothed detections (convert center → corner)
        self.boxes
            .iter()
            .map(|(track_id, bbox)| {
                let x = bbox.x - bbox.w / 2.0;
                let y = bbox.y - bbox.h / 2.0;
                (*track_id, x, y, bbox.w, bbox.h, bbox.class_id, bbox.score)
            })
            .collect()
    }

    /// Get number of active tracked boxes
    pub fn active_count(&self) -> usize {
        self.boxes.len()
    }

    /// Clear all tracked boxes
    pub fn clear(&mut self) {
        self.boxes.clear();
    }
}

impl Default for EmaSmoother {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_smoothing() {
        let mut smoother = EmaSmoother::new();

        // First detection: track_id=1, bbox=(100, 100, 50, 50)
        let detections = vec![(1, 100.0, 100.0, 50.0, 50.0, 0, 0.9)];
        let smoothed = smoother.smooth(detections);
        
        // First frame: no smoothing applied (initialization)
        assert_eq!(smoothed.len(), 1);
        assert_eq!(smoothed[0].0, 1); // track_id
        assert!((smoothed[0].1 - 100.0).abs() < 0.1); // x

        // Second detection: same track, moved to (110, 110, 55, 55)
        let detections = vec![(1, 110.0, 110.0, 55.0, 55.0, 0, 0.95)];
        let smoothed = smoother.smooth(detections);
        
        // Should be smoothed: not exactly at (110, 110, 55, 55)
        assert_eq!(smoothed.len(), 1);
        assert!(smoothed[0].1 > 100.0 && smoothed[0].1 < 110.0); // x smoothed between old and new
    }

    #[test]
    fn test_stale_track_removal() {
        let mut smoother = EmaSmoother::new().with_max_age(2);

        // Add track
        let detections = vec![(1, 100.0, 100.0, 50.0, 50.0, 0, 0.9)];
        smoother.smooth(detections);
        assert_eq!(smoother.active_count(), 1);

        // Don't update for 3 frames
        smoother.smooth(vec![]);
        smoother.smooth(vec![]);
        smoother.smooth(vec![]);
        
        // Should be removed (age > max_age)
        assert_eq!(smoother.active_count(), 0);
    }
}
