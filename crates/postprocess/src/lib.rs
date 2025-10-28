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

        // Update config with actual frame dimensions and crop region from packet
        let mut config = self.config;
        config.original_size = (input.from.width, input.from.height);
        config.crop_region = input.from.crop_region;
        
        // ⚠️ IMPORTANT: CUDA preprocessor uses STRETCH RESIZE (not letterbox)
        // The preprocessing kernel in preprocess.cu does simple bilinear resize:
        //   float sx = (x + 0.5f) * ((float)in_w / (float)out_w) - 0.5f;
        // This means no aspect ratio preservation, no padding.
        // Therefore, we should NOT calculate letterbox parameters here.
        // The postprocessing will use stretch resize coordinate transformation.
        
        // Debug: Show stretch resize parameters (for first few frames)
        if input.from.frame_idx <= 5 {
            if let Some((cx, cy, cw, ch)) = config.crop_region {
                println!("    Crop+Stretch Resize [crop {}×{} at ({},{}) → 512x512]: scale to crop dims + offset", 
                         cw, ch, cx, cy);
            } else {
                println!("    Stretch Resize [{}x{} → 512x512]: direct scale (no letterbox)", 
                         input.from.width, input.from.height);
            }
        }

        // Process raw predictions (YOLO decode + NMS)
        let result = postprocess_yolov5_with_temporal_smoothing(
            &input.raw_output,
            &config,
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
            use_stretch_resize: true,  // CUDA preprocessor uses stretch resize, not letterbox
            skip_sigmoid: false,  // Default: apply sigmoid (for models that output logits)
            crop_region: None,  // No crop by default (set per-frame in process())
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
        
        // Update config with actual frame dimensions from packet
        let mut config = self.yolo_config;
        config.original_size = (raw.from.width, raw.from.height);

        // Derive number of classes dynamically from the inference output when possible.
        // TensorRT stage currently produces [1, N, values_per_det] where
        // values_per_det = 5 (box/objectness) + num_classes. When this value
        // doesn't match the static config (default 80), decoding mixes anchors
        // together and yields garbage bounding boxes. This happens with the
        // current 2-class model (values_per_det = 7).
        if let Some(values_per_det) = infer_values_per_candidate(raw) {
            let derived_classes = values_per_det.saturating_sub(5);
            if derived_classes > 0 && derived_classes != config.num_classes {
                println!(
                    "    Adjusting YOLO class count: inferred {} classes (was {})",
                    derived_classes, config.num_classes
                );
                self.yolo_config.num_classes = derived_classes;
                config.num_classes = derived_classes;
            }
        }
        
        // Calculate letterbox parameters based on actual preprocessing
        // Assuming model input is always 512x512 (from yolo_config)
        let model_w = 512.0;
        let model_h = 512.0;
        let orig_w = raw.from.width as f32;
        let orig_h = raw.from.height as f32;
        
        let scale = (model_w / orig_w).min(model_h / orig_h);
        let new_w = (orig_w * scale).round();
        let new_h = (orig_h * scale).round();
        let pad_x = (model_w - new_w) / 2.0;
        let pad_y = (model_h - new_h) / 2.0;
        
        config.letterbox_scale = scale;
        config.letterbox_pad = (pad_x, pad_y);
        
        // Debug: Show letterbox parameters
        println!("    Letterbox params: scale={:.4}, pad=({:.1}, {:.1}), orig_size={}x{}", 
                 scale, pad_x, pad_y, raw.from.width, raw.from.height);
        
        // For YOLO: raw_output is flattened [N, (cx,cy,w,h,obj,cls...)]
        let candidates = post::decode_predictions(&raw.raw_output, &config);
        
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
    // Optimized YOLO config for 512x512 input
    // ⚠️ CRITICAL: num_classes MUST match the model output!
    // For YOLOv5 model with output shape [1, 16128, 7]:
    // 7 values per detection = [cx, cy, w, h, objectness, class0, class1]
    // Therefore: num_classes = 7 - 5 = 2
    let config = YoloPostConfig {
        num_classes: 2, // ✅ FIXED: Changed from 80 to 2 to match actual model output
        confidence_threshold: 0.25, // ✅ Standard YOLO threshold (was 0.35, too high!)
        nms_threshold: 0.45,
        max_detections: 100,
        letterbox_scale: 1.0,
        letterbox_pad: (0.0, 0.0),
        original_size: (512, 512),
        use_stretch_resize: true,  // CUDA preprocessor uses stretch resize, not letterbox
        skip_sigmoid: true,  // v7_optimized model outputs probabilities, not logits
        crop_region: None,  // No crop by default (set per-frame in process())
    };

    Ok(PostStage::new(config).with_temporal_smoothing(4))
}

/// Infer the number of raw values per candidate detection from the inference output.
/// Returns Some(values_per_candidate) when the tensor shape provides enough information.
fn infer_values_per_candidate(raw: &RawDetectionsPacket) -> Option<usize> {
    // Prefer the last dimension of the output tensor (e.g., [1, N, 7]).
    if let Some(&values_per_det) = raw.output_shape.last() {
        if values_per_det >= 5 {
            return Some(values_per_det);
        }
    }

    // Fallback: derive from total float count and remaining dimensions (batch * anchors).
    if !raw.output_shape.is_empty() {
        let len = raw.output_shape.len();
        // Multiply every dimension except the last; default to 1 when shape has a single dim.
        let denom: usize = if len >= 2 {
            raw.output_shape[..len - 1].iter().product()
        } else {
            raw.output_shape[0]
        };

        if denom > 0 {
            let values_per_det = raw.raw_output.len() / denom;
            if values_per_det >= 5 {
                return Some(values_per_det);
            }
        }
    }

    None
}
