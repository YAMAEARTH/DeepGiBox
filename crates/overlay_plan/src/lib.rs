use anyhow::Result;
use common_io::{DetectionsPacket, OverlayPlanPacket, DrawOp, Stage};

pub fn from_path(cfg: &str) -> Result<PlanStage> {
    // Simple, permissive parser: enable full UI if cfg contains any of these keywords
    let cfg_lc = cfg.to_ascii_lowercase();
    let enable_full_ui = ["full", "full_ui", "ui=full", "hud"].iter().any(|k| cfg_lc.contains(k));
    Ok(PlanStage { enable_full_ui })
}

pub struct PlanStage {
    pub enable_full_ui: bool,
}

impl Stage<DetectionsPacket, OverlayPlanPacket> for PlanStage {
    fn process(&mut self, input: DetectionsPacket) -> OverlayPlanPacket {
        let mut ops = Vec::new();

        // Capture canvas dimensions before moving input.from
        let canvas = (input.from.width, input.from.height);
        let (cw, ch) = (canvas.0 as f32, canvas.1 as f32);

        // Determine primary detection (highest confidence) — only needed for full UI mode
        let primary = if self.enable_full_ui {
            input
                .items
                .iter()
                .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
        } else {
            None
        };

        // Convert each detection into drawing operations
        for det in &input.items {
            let (class_label, class_color) = match det.class_id {
                0 => ("Hyper", (255, 0, 255, 0)),      // Green
                1 => ("Neo", (255, 255, 0, 0)),        // Red
                _ => ("Unknown", (255, 128, 128, 128)),// Gray
            };
            let (x, y, w, h) = (det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h);
            // Add bounding box rectangle (baseline)
            ops.push(DrawOp::Rect {
                xywh: (x, y, w, h),
                thickness: 2,
                color: class_color,
            });
            // Corner brackets using Poly segments (approximate from temp.rs) — only in full UI
            if self.enable_full_ui {
                let corner_len = ((w.min(h) * 0.22).max(18.0)).min(40.0);
                let x2 = x + w;
                let y2 = y + h;
                // Top-left
                ops.push(DrawOp::Poly { pts: vec![(x, y), (x + corner_len, y)], thickness: 3, color: class_color });
                ops.push(DrawOp::Poly { pts: vec![(x, y), (x, y + corner_len)], thickness: 3, color: class_color });
                // Top-right
                ops.push(DrawOp::Poly { pts: vec![(x2, y), (x2 - corner_len, y)], thickness: 3, color: class_color });
                ops.push(DrawOp::Poly { pts: vec![(x2, y), (x2, y + corner_len)], thickness: 3, color: class_color });
                // Bottom-left
                ops.push(DrawOp::Poly { pts: vec![(x, y2), (x + corner_len, y2)], thickness: 3, color: class_color });
                ops.push(DrawOp::Poly { pts: vec![(x, y2), (x, y2 - corner_len)], thickness: 3, color: class_color });
                // Bottom-right
                ops.push(DrawOp::Poly { pts: vec![(x2, y2), (x2 - corner_len, y2)], thickness: 3, color: class_color });
                ops.push(DrawOp::Poly { pts: vec![(x2, y2), (x2, y2 - corner_len)], thickness: 3, color: class_color });
            }
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
                color: class_color,
            });
        }

        // HUD elements based on primary detection (top-left info, confidence bar, bottom class)
        if self.enable_full_ui {
            if let Some(p) = primary {
                // Mode text derived from class
                let mode_text = if p.class_id == 1 { "EGD" } else { "COLON" };
                let conf = p.score.clamp(0.0, 1.0);
                let class_text = match p.class_id {
                    0 => ("Hyperplastic", (255, 0, 255, 0)),      // Green
                    1 => ("Neoplastic", (255, 255, 0, 0)),        // Red
                    _ => ("Awaiting", (255, 128, 128, 128)),      // Gray
                };
                // Top-left info labels (date/time as in temp.rs) and mode + T value
                let left_x = 36.0;
                let mut y = 48.0;
                let label_color = (255, 255, 255, 255); // White
                ops.push(DrawOp::Label { anchor: (left_x, y), text: "27/11/2020".to_string(), font_px: 16, color: label_color });
                y += 32.0;
                ops.push(DrawOp::Label { anchor: (left_x, y), text: "10:21:16".to_string(), font_px: 16, color: label_color });
                y += 20.0;
                // Simple square icon (outline only)
                let icon_size = 30.0;
                let icon_top = y;
                let icon_color = (255, 255, 255, 255); // White
                ops.push(DrawOp::Rect { xywh: (left_x, icon_top, icon_size, icon_size), thickness: 2, color: icon_color });
                // Speaker-like shape using rectangles and lines (approximate)
                let speaker_x = left_x + icon_size + 16.0;
                ops.push(DrawOp::Rect { xywh: (speaker_x, icon_top + 4.0, 14.0, 20.0), thickness: 2, color: icon_color });
                ops.push(DrawOp::Poly { pts: vec![(speaker_x + 14.0, icon_top + 4.0), (speaker_x + 28.0, icon_top - 4.0)], thickness: 2, color: icon_color });
                ops.push(DrawOp::Poly { pts: vec![(speaker_x + 28.0, icon_top - 4.0), (speaker_x + 28.0, icon_top + icon_size - 4.0)], thickness: 2, color: icon_color });
                ops.push(DrawOp::Poly { pts: vec![(speaker_x + 14.0, icon_top + icon_size + 2.0), (speaker_x + 28.0, icon_top + icon_size - 4.0)], thickness: 2, color: icon_color });
                y = icon_top + icon_size + 20.0;
                ops.push(DrawOp::Label { anchor: (left_x, y), text: mode_text.to_string(), font_px: 18, color: label_color });
                y += 32.0;
                ops.push(DrawOp::Label { anchor: (left_x, y), text: format!("T = {:.2}", conf), font_px: 16, color: label_color });
                // Confidence bar on right side (3 segments stacked)
                if cw > 0.0 && ch > 0.0 {
                    let bar_x = (cw - 80.0).max(8.0);
                    let bar_height = 200.0;
                    let num_segments = 3;
                    let gap = 12.0;
                    let segment_height = ((bar_height - (num_segments as f32 - 1.0) * gap) / num_segments as f32).max(20.0);
                    let effective_bar_height = segment_height * num_segments as f32 + gap * (num_segments as f32 - 1.0);
                    let start_y = ((ch - effective_bar_height) / 2.0).max(32.0);
                    let active_segments = (conf * num_segments as f32).ceil().clamp(0.0, num_segments as f32) as i32;
                    for i in 0..num_segments {
                        let top = start_y + i as f32 * (segment_height + gap);
                        let seg_color = if (i as i32) < active_segments {
                            (255, 0, 255, 0) // Green for active
                        } else {
                            (128, 128, 128, 128) // Gray for inactive
                        };
                        // Filled bar
                        ops.push(DrawOp::FillRect { xywh: (bar_x, top, 28.0, segment_height), color: seg_color });
                        // Outline
                        ops.push(DrawOp::Rect { xywh: (bar_x, top, 28.0, segment_height), thickness: 2, color: (255, 255, 255, 255) });
                        // Optional: draw a short tick inside for active segments (visual hint)
                        if (i as i32) < active_segments {
                            ops.push(DrawOp::Poly { pts: vec![(bar_x + 4.0, top + 4.0), (bar_x + 24.0, top + segment_height - 4.0)], thickness: 2, color: (255, 255, 255, 255) });
                        }
                    }
                }
                // Bottom centered class box (outline + label)
                if cw > 0.0 && ch > 0.0 {
                    let (text, box_color) = class_text;
                    // Approximate a box sized to text: assume ~8 px per char width and 18 px height
                    let approx_w = (text.len() as f32 * 8.0) + 48.0; // padding
                    let approx_h = 18.0 + 36.0;
                    let origin_x = (cw - approx_w) / 2.0;
                    let origin_y = ch - approx_h - 24.0;
                    // Filled background
                    ops.push(DrawOp::FillRect { xywh: (origin_x, origin_y, approx_w, approx_h), color: box_color });
                    // Outline
                    ops.push(DrawOp::Rect { xywh: (origin_x, origin_y, approx_w, approx_h), thickness: 2, color: (255, 255, 255, 255) });
                    // Label
                    ops.push(DrawOp::Label { anchor: (origin_x + 24.0, origin_y + approx_h - 18.0), text: text.to_string(), font_px: 18, color: (255, 255, 255, 255) });
                }
            }
        }

        OverlayPlanPacket { from: input.from, ops, canvas }
    }
}

