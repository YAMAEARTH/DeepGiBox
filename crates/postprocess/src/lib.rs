use anyhow::Result;
use common_io::{BBox, Detection, DetectionsPacket, RawDetectionsPacket, Stage};
use telemetry::{now_ns, record_ms};

mod post;
mod sort_tracker;
mod fast_nms;

use post::{postprocess_yolov5_with_temporal_smoothing, TemporalSmoother, YoloPostConfig};
use sort_tracker::SortTrackerWrapper;

// Re-export config structs for external use
pub use post::PostprocessConfig;

/// Main postprocessing stage following postprocessing_guideline.md
/// Converts raw model outputs → DetectionsPacket with bbox/score/class/track
pub struct Postprocess {
    cfg: config::PostprocessCfg,
    // Legacy YOLO config (for backward compat)
    yolo_config: YoloPostConfig,
    smoother: Option<TemporalSmoother>,
    tracker: Option<SortTrackerWrapper>,
    current_epoch: usize,
    // Preallocated buffers to reduce allocation in hot path
    working_buffer: Vec<f32>,
}

/// Legacy struct for backward compatibility
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
        println!(
            "  ✓ Postprocess time: {:.2}ms",
            duration.as_secs_f64() * 1000.0
        );
        println!("  ✓ Detections found: {}", result.detections.len());

        // Apply SORT tracking if enabled
        let items = if let Some(tracker) = &mut self.tracker {
            self.current_epoch += 1;

            // Convert detections to tracker format: (x1, y1, x2, y2, score, class_id)
            let detections_for_tracking: Vec<_> = result
                .detections
                .iter()
                .map(|d| {
                    (
                        d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.score, d.class_id,
                    )
                })
                .collect();

            // Update tracker and get tracked detections
            let tracked = tracker.update(&detections_for_tracking);

            println!(
                "  ✓ SORT tracking: {} active tracks",
                tracker.get_active_track_count()
            );

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
                .map(|d| Detection {
                    bbox: BBox {
                        x: d.bbox[0],
                        y: d.bbox[1],
                        w: d.bbox[2] - d.bbox[0],
                        h: d.bbox[3] - d.bbox[1],
                    },
                    score: d.score,
                    class_id: d.class_id as i32,
                    track_id: None,
                })
                .collect()
        };

        DetectionsPacket {
            from: input.from,
            items,
        }
    }
}

// ============================================================================
// New guideline-compliant implementation
// ============================================================================

impl Postprocess {
    /// Create new postprocessing stage from config (guideline §5)
    pub fn new(cfg: config::PostprocessCfg) -> Result<Self> {
        // Convert config to internal YOLO config format
        let yolo_config = YoloPostConfig {
            num_classes: 80, // COCO default, should come from cfg if available
            confidence_threshold: cfg.score_thresh,
            nms_threshold: cfg.nms.iou_thresh,
            max_detections: cfg.nms.max_total,
            letterbox_scale: cfg.letterbox.scale,
            letterbox_pad: (cfg.letterbox.pad_x, cfg.letterbox.pad_y),
            original_size: (cfg.letterbox.original_w, cfg.letterbox.original_h),
        };

        Ok(Self {
            cfg,
            yolo_config,
            smoother: None,
            tracker: None,
            current_epoch: 0,
            working_buffer: Vec::with_capacity(100000), // Preallocate for ~10k anchors
        })
    }

    /// Enable temporal smoothing with window size (guideline §3.5)
    pub fn with_temporal_smoothing(mut self, window_size: usize) -> Self {
        self.smoother = Some(TemporalSmoother::new(window_size));
        self
    }

    /// Enable SORT tracking (guideline §3.5)
    pub fn with_tracking(mut self) -> Self {
        let tracking_cfg = &self.cfg.tracking;
        if tracking_cfg.enable {
            self.tracker = Some(SortTrackerWrapper::new(
                tracking_cfg.max_age,
                tracking_cfg.min_confidence,
                tracking_cfg.iou_match,
            ));
        }
        self
    }

    /// Decode model outputs to candidates (guideline §3.1)
    /// Returns (candidates, start_time_ns)
    fn decode(&mut self, raw: &RawDetectionsPacket) -> (Vec<post::Candidate>, u64) {
        let t0 = now_ns();
        
        // For YOLO: raw_output is flattened [N, (cx,cy,w,h,obj,cls...)]
        let candidates = post::decode_predictions(&raw.raw_output, &self.yolo_config);
        
        (candidates, t0)
    }

    /// Apply NMS (guideline §3.4)
    /// Returns (filtered_detections, start_time_ns)
    fn nms(&mut self, mut candidates: Vec<post::Candidate>) -> (Vec<Detection>, u64) {
        let t0 = now_ns();

        // Aggressive Top-K prefilter for speed (guideline §3.1, §6)
        // Reduced from 5000 to 300 for massive speed boost
        const TOP_K_THRESHOLD: usize = 300;
        if candidates.len() > TOP_K_THRESHOLD {
            // Partial sort by score, keep top K
            candidates.sort_unstable_by(|a, b| {
                b.detection.score
                    .partial_cmp(&a.detection.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(TOP_K_THRESHOLD);
        }

        // Use fast NMS implementation (3-5x faster than similari)
        let retained = fast_nms::fast_classwise_nms(
            &candidates,
            self.cfg.nms.iou_thresh,
            self.cfg.nms.max_per_class,
            self.cfg.nms.max_total,
        );

        // Convert to common_io::Detection
        let detections: Vec<Detection> = retained
            .iter()
            .map(|c| Detection {
                bbox: BBox {
                    x: c.detection.bbox[0],
                    y: c.detection.bbox[1],
                    w: c.detection.bbox[2] - c.detection.bbox[0],
                    h: c.detection.bbox[3] - c.detection.bbox[1],
                },
                score: c.detection.score,
                class_id: c.detection.class_id as i32,
                track_id: None,
            })
            .collect();

        (detections, t0)
    }

    /// Apply tracking (guideline §3.5)
    /// Returns (tracked_detections, start_time_ns)
    fn track(&mut self, detections: Vec<Detection>) -> (Vec<Detection>, u64) {
        let t0 = now_ns();

        let tracked = if let Some(tracker) = &mut self.tracker {
            self.current_epoch += 1;

            // Convert to tracker format
            let tracker_input: Vec<_> = detections
                .iter()
                .map(|d| {
                    let x1 = d.bbox.x;
                    let y1 = d.bbox.y;
                    let x2 = d.bbox.x + d.bbox.w;
                    let y2 = d.bbox.y + d.bbox.h;
                    (x1, y1, x2, y2, d.score, d.class_id as usize)
                })
                .collect();

            // Update tracker
            let tracked_results = tracker.update(&tracker_input);

            // Convert back with track IDs
            tracked_results
                .into_iter()
                .map(|(track_id, x1, y1, x2, y2, class_id, score)| Detection {
                    bbox: BBox {
                        x: x1,
                        y: y1,
                        w: x2 - x1,
                        h: y2 - y1,
                    },
                    score,
                    class_id: class_id as i32,
                    track_id: Some(track_id as i32),
                })
                .collect()
        } else {
            detections
        };

        (tracked, t0)
    }
}

impl Stage<RawDetectionsPacket, DetectionsPacket> for Postprocess {
    fn name(&self) -> &'static str {
        "Postprocess"
    }

    fn process(&mut self, input: RawDetectionsPacket) -> DetectionsPacket {
        let t_total = now_ns();

        // Stage 1: Decode (guideline §3.1)
        let (mut candidates, t_decode) = self.decode(&input);
        record_ms("post.decode", t_decode);

        // Stage 2: Apply temporal smoothing if enabled
        if let Some(smoother) = &mut self.smoother {
            let scores: Vec<f32> = candidates.iter().map(|c| c.detection.score).collect();
            let smoothed = smoother.get_smoothed_scores(&scores);
            for (c, &s) in candidates.iter_mut().zip(smoothed.iter()) {
                c.detection.score = s;
            }
            smoother.add_frame_scores(scores);
        }

        // Stage 3: NMS (guideline §3.4)
        let (detections, t_nms) = self.nms(candidates);
        record_ms("post.nms", t_nms);

        // Stage 4: Tracking (guideline §3.5)
        let (items, t_track) = self.track(detections);
        record_ms("post.track", t_track);

        // Total telemetry (guideline §8)
        record_ms("postprocess", t_total);

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
