use anyhow::Result;
use common_io::{Stage, RawDetectionsPacket, DetectionsPacket, Detection, BBox};

mod post;
mod sort_tracker;

use post::{postprocess_yolov5_with_temporal_smoothing, YoloPostConfig, TemporalSmoother};
use sort_tracker::SortTrackerWrapper;

pub struct PostStage {
    config: YoloPostConfig,
    smoother: Option<TemporalSmoother>,
    tracker: Option<SortTrackerWrapper>,
    current_epoch: usize,
}

impl PostStage {
    pub fn new(config: YoloPostConfig) -> Self {
        Self {
            config,
            smoother: None,
            tracker: None,
            current_epoch: 0,
        }
    }

    pub fn with_temporal_smoothing(mut self, window_size: usize) -> Self {
        self.smoother = Some(TemporalSmoother::new(window_size));
        self
    }

    pub fn with_sort_tracking(
        mut self,
        max_idle_epochs: usize,
        min_confidence: f32,
        iou_threshold: f32,
    ) -> Self {
        self.tracker = Some(SortTrackerWrapper::new(
            max_idle_epochs,
            min_confidence,
            iou_threshold,
        ));
        self
    }
}

impl Stage<RawDetectionsPacket, DetectionsPacket> for PostStage {
    fn name(&self) -> &'static str {
        "PostStage"
    }

    fn process(&mut self, input: RawDetectionsPacket) -> DetectionsPacket {
        let start = std::time::Instant::now();
        
        // Process raw predictions (YOLO decode + NMS)
        let result = postprocess_yolov5_with_temporal_smoothing(
            &input.raw_output,
            &self.config,
            self.smoother.as_mut(),
        );
        
        let duration = start.elapsed();
        println!("  ✓ Postprocess time: {:.2}ms", duration.as_secs_f64() * 1000.0);
        println!("  ✓ Detections found: {}", result.detections.len());
        
        // Apply SORT tracking if enabled
        let items = if let Some(tracker) = &mut self.tracker {
            self.current_epoch += 1;
            
            // Convert detections to tracker format: (x1, y1, x2, y2, score, class_id)
            let detections_for_tracking: Vec<_> = result
                .detections
                .iter()
                .map(|d| (d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.score, d.class_id))
                .collect();
            
            // Update tracker and get tracked detections
            let tracked = tracker.update(&detections_for_tracking);
            
            println!("  ✓ SORT tracking: {} active tracks", tracker.get_active_track_count());
            
            // Convert tracked detections to common_io::Detection format
            tracked
                .iter()
                .map(|(track_id, x1, y1, x2, y2, class_id, score)| Detection {
                    bbox: BBox {
                        x: *x1,
                        y: *y1,
                        w: x2 - x1,
                        h: y2 - y1,
                    },
                    score: *score,
                    class_id: *class_id as i32,
                    track_id: Some(*track_id as i32), // ✅ Assign SORT track ID
                })
                .collect()
        } else {
            // No tracking - convert detections directly
            result
                .detections
                .iter()
                .map(|d| {
                    Detection {
                        bbox: BBox {
                            x: d.bbox[0],
                            y: d.bbox[1],
                            w: d.bbox[2] - d.bbox[0],
                            h: d.bbox[3] - d.bbox[1],
                        },
                        score: d.score,
                        class_id: d.class_id as i32,
                        track_id: None,
                    }
                })
                .collect()
        };

        DetectionsPacket {
            from: input.from,
            items,
        }
    }
}

// Helper function to create from config
pub fn from_path(_cfg: &str) -> Result<PostStage> {
    // Default YOLO config for 512x512 input
    let config = YoloPostConfig {
        num_classes: 80, // COCO dataset
        confidence_threshold: 0.25,
        nms_threshold: 0.45,
        max_detections: 100,
        letterbox_scale: 1.0,
        letterbox_pad: (0.0, 0.0),
        original_size: (512, 512),
    };
    
    Ok(PostStage::new(config).with_temporal_smoothing(4))
}

// Re-export for external use
pub use post::YoloPostConfig as PostprocessConfig;