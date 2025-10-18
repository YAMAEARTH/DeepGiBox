use anyhow::Result;
use common_io::{DetectionsPacket, OverlayPlanPacket, DrawOp, Stage};

pub fn from_path(_cfg: &str) -> Result<PlanStage> {
    Ok(PlanStage {})
}

pub struct PlanStage {}

impl Stage<DetectionsPacket, OverlayPlanPacket> for PlanStage {
    fn process(&mut self, input: DetectionsPacket) -> OverlayPlanPacket {
        let mut ops = Vec::new();
        
        // Capture canvas dimensions before moving input.from
        let canvas = (input.from.width, input.from.height);
        
        // Convert each detection into drawing operations
        for det in &input.items {
            let class_label = match det.class_id {
                0 => "Hyper",
                1 => "Neo",
                _ => "Unknown",
            };
            
            let (x, y, w, h) = (det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h);
            
            // Add bounding box rectangle
            ops.push(DrawOp::Rect {
                xywh: (x, y, w, h),
                thickness: 2,
            });
            
            // Add label with confidence score
            let track_info = if let Some(track_id) = det.track_id {
                format!(" ID:{}", track_id)
            } else {
                String::new()
            };
            
            let label_text = format!("{} {:.2}{}", class_label, det.score, track_info);
            let label_y = if y > 20.0 { y - 10.0 } else { y + 20.0 };
            
            ops.push(DrawOp::Label {
                anchor: (x, label_y),
                text: label_text,
                font_px: 16,
            });
        }
        
        OverlayPlanPacket {
            from: input.from,
            ops,
            canvas,
        }
    }
}

