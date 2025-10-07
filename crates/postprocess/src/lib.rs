use anyhow::Result;
use common_io::{Stage, RawDetectionsPacket, DetectionsPacket, Detection, BBox};

mod post;
use post::{postprocess_yolov5_with_temporal_smoothing, YoloPostConfig, TemporalSmoother};

pub struct PostStage {
    config: YoloPostConfig,
    smoother: Option<TemporalSmoother>,
}

impl PostStage {
    pub fn new(config: YoloPostConfig) -> Self {
        Self {
            config,
            smoother: None,
        }
    }

    pub fn with_temporal_smoothing(mut self, window_size: usize) -> Self {
        self.smoother = Some(TemporalSmoother::new(window_size));
        self
    }
}

impl Stage<RawDetectionsPacket, DetectionsPacket> for PostStage {
    fn name(&self) -> &'static str {
        "PostStage"
    }

    fn process(&mut self, input: RawDetectionsPacket) -> DetectionsPacket {
        let start = std::time::Instant::now();
        
        // Process raw predictions
        let result = postprocess_yolov5_with_temporal_smoothing(
            &input.raw_output,
            &self.config,
            self.smoother.as_mut(),
        );
        
        let duration = start.elapsed();
        println!("  ✓ Postprocess time: {:.2}ms", duration.as_secs_f64() * 1000.0);
        println!("  ✓ Detections found: {}", result.detections.len());
        
        // Convert to common_io::Detection format
        let items = result
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
            .collect();

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