use anyhow::Result;
use common_io::{DetectionsPacket, OverlayPlanPacket, DrawOp, Stage};
use std::cmp::Ordering;

pub fn from_path(cfg: &str) -> Result<PlanStage> {
    PlanStage::from_path(cfg)
}

pub struct PlanStage {
    pub enable_full_ui: bool,
}

fn class_palette(class_id: u32) -> (&'static str, &'static str, (u8, u8, u8, u8)) {
    match class_id {
        0 => ("Hyper", "Hyperplastic", (0, 255, 0, 255)),
        1 => ("Neo", "Neoplastic", (255, 64, 64, 255)),
        _ => ("Unknown", "Awaiting", (128, 128, 128, 255)),
    }
}

fn scaled_font_px(base_px: f32, scale: f32, min_px: i32) -> u16 {
    let px = (base_px * scale).round() as i32;
    let result = if px < min_px { min_px } else { px };
    result.max(1).min(u16::MAX as i32) as u16
}

fn scaled_thickness(base_px: f32, scale: f32) -> u8 {
    let px = (base_px * scale).round() as i32;
    let result = if px < 1 { 1 } else { px };
    result.min(u8::MAX as i32) as u8
}

impl Stage<DetectionsPacket, OverlayPlanPacket> for PlanStage {
    fn process(&mut self, input: DetectionsPacket) -> OverlayPlanPacket {
        let mut ops = Vec::new();

        let canvas = (input.from.width, input.from.height);
        let (cw, ch) = (canvas.0 as f32, canvas.1 as f32);

        if cw <= 0.0 || ch <= 0.0 {
            eprintln!("Invalid canvas dimensions: ({}, {})", cw, ch);
            return OverlayPlanPacket { from: input.from, ops, canvas };
        }

        let ui_scale = (ch / 1080.0).max(0.1);

        let best_detection = input
            .items
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal));
        let best_score = best_detection.map(|d| d.score.clamp(0.0, 1.0)).unwrap_or(0.0);
        let (_, best_long_label, best_color) = best_detection
            .map(|d| class_palette(d.class_id as u32))
            .unwrap_or(("None", "Awaiting", (80, 80, 80, 255)));

        if self.enable_full_ui {
            let label_color = (238, 240, 230, 255);
            let accent_color = (200, 200, 200, 255);
            let icon_outline = scaled_thickness(2.0, ui_scale);
            let icon_x = 24.0 * ui_scale;
            let icon_size = (28.0 * ui_scale).max(14.0);
            let text_x = icon_x + icon_size + 16.0 * ui_scale;
            let mut row_y = 44.0 * ui_scale;
            let row_gap = 42.0 * ui_scale;
            let font_regular = scaled_font_px(20.0, ui_scale, 14);

            ops.push(DrawOp::Rect {
                xywh: (icon_x, row_y, icon_size, icon_size),
                thickness: icon_outline,
                color: accent_color,
            });
            ops.push(DrawOp::Poly {
                pts: vec![
                    (icon_x + 4.0 * ui_scale, row_y + 6.0 * ui_scale),
                    (icon_x + icon_size - 4.0 * ui_scale, row_y + 6.0 * ui_scale),
                ],
                thickness: icon_outline,
                color: accent_color,
            });
            ops.push(DrawOp::Label {
                anchor: (text_x, row_y),
                text: "Patient ID".to_string(),
                font_px: font_regular,
                color: label_color,
            });
            row_y += row_gap;

            let mid_y = row_y + icon_size * 0.5;
            ops.push(DrawOp::Poly {
                pts: vec![
                    (icon_x + icon_size * 0.2, row_y + icon_size * 0.2),
                    (icon_x + icon_size * 0.5, row_y),
                    (icon_x + icon_size * 0.8, row_y + icon_size * 0.2),
                ],
                thickness: icon_outline,
                color: accent_color,
            });
            ops.push(DrawOp::Poly {
                pts: vec![
                    (icon_x + icon_size * 0.2, mid_y),
                    (icon_x + icon_size * 0.8, mid_y),
                ],
                thickness: icon_outline,
                color: accent_color,
            });
            ops.push(DrawOp::Label {
                anchor: (text_x, row_y),
                text: "Patient Name".to_string(),
                font_px: font_regular,
                color: label_color,
            });
            row_y += row_gap;

            ops.push(DrawOp::Label {
                anchor: (text_x, row_y),
                text: "Patient name (add. info)".to_string(),
                font_px: font_regular,
                color: label_color,
            });
            row_y += row_gap;

            ops.push(DrawOp::Rect {
                xywh: (icon_x, row_y, icon_size, icon_size),
                thickness: icon_outline,
                color: accent_color,
            });
            ops.push(DrawOp::Poly {
                pts: vec![
                    (icon_x, row_y + icon_size * 0.3),
                    (icon_x + icon_size, row_y + icon_size * 0.3),
                ],
                thickness: icon_outline,
                color: accent_color,
            });
            ops.push(DrawOp::Label {
                anchor: (text_x, row_y),
                text: "DOB".to_string(),
                font_px: font_regular,
                color: label_color,
            });
            ops.push(DrawOp::Label {
                anchor: (text_x + 148.0 * ui_scale, row_y),
                text: "Age".to_string(),
                font_px: font_regular,
                color: label_color,
            });
            row_y += row_gap;

            ops.push(DrawOp::Poly {
                pts: vec![
                    (icon_x + icon_size * 0.2, row_y),
                    (icon_x + icon_size * 0.8, row_y + icon_size),
                ],
                thickness: icon_outline,
                color: accent_color,
            });
            ops.push(DrawOp::Poly {
                pts: vec![
                    (icon_x + icon_size * 0.8, row_y),
                    (icon_x + icon_size * 0.2, row_y + icon_size),
                ],
                thickness: icon_outline,
                color: accent_color,
            });
            ops.push(DrawOp::Label {
                anchor: (text_x, row_y),
                text: "Sex".to_string(),
                font_px: font_regular,
                color: label_color,
            });
            row_y += row_gap;

            ops.push(DrawOp::Rect {
                xywh: (icon_x, row_y, icon_size, icon_size),
                thickness: icon_outline,
                color: accent_color,
            });
            ops.push(DrawOp::Label {
                anchor: (text_x, row_y),
                text: "0".to_string(),
                font_px: font_regular,
                color: label_color,
            });
            row_y += row_gap;

            let cam_w = icon_size * 1.6;
            let cam_h = icon_size * 0.9;
            let cam_y = row_y + (icon_size - cam_h) * 0.5;
            let cam_outline = std::cmp::max(icon_outline, 2);
            ops.push(DrawOp::FillRect {
                xywh: (icon_x, cam_y, cam_w, cam_h),
                color: (255, 200, 0, 255),
            });
            ops.push(DrawOp::Rect {
                xywh: (icon_x, cam_y, cam_w, cam_h),
                thickness: cam_outline,
                color: (0, 0, 0, 255),
            });
            ops.push(DrawOp::Poly {
                pts: vec![
                    (icon_x + cam_w * 0.2, cam_y + cam_h * 0.5),
                    (icon_x + cam_w * 0.8, cam_y + cam_h * 0.5),
                ],
                thickness: cam_outline,
                color: (0, 0, 0, 255),
            });
            ops.push(DrawOp::Label {
                anchor: (text_x, row_y),
                text: " ".to_string(),
                font_px: font_regular,
                color: label_color,
            });
            row_y += row_gap;

            ops.push(DrawOp::Label {
                anchor: (text_x, row_y),
                text: "OFF".to_string(),
                font_px: font_regular,
                color: label_color,
            });
            row_y += row_gap;

            let bar_width = 4.0 * ui_scale;
            for i in 0..3 {
                let height_factor = (i + 1) as f32 / 3.0;
                let bar_height = (icon_size * height_factor).max(4.0 * ui_scale);
                let bar_x = icon_x + i as f32 * (6.0 * ui_scale);
                let bar_y = row_y + icon_size - bar_height;
                ops.push(DrawOp::FillRect {
                    xywh: (bar_x, bar_y, bar_width, bar_height),
                    color: accent_color,
                });
            }
            ops.push(DrawOp::Label {
                anchor: (text_x, row_y),
                text: "".to_string(),
                font_px: font_regular,
                color: label_color,
            });
            row_y += row_gap;

            let bubble_h = icon_size * 0.7;
            ops.push(DrawOp::Rect {
                xywh: (icon_x, row_y, icon_size, bubble_h),
                thickness: icon_outline,
                color: accent_color,
            });
            ops.push(DrawOp::Poly {
                pts: vec![
                    (icon_x + icon_size * 0.3, row_y + bubble_h),
                    (icon_x + icon_size * 0.45, row_y + bubble_h + 6.0 * ui_scale),
                    (icon_x + icon_size * 0.55, row_y + bubble_h),
                ],
                thickness: icon_outline,
                color: accent_color,
            });
            ops.push(DrawOp::Label {
                anchor: (text_x, row_y),
                text: "Comment".to_string(),
                font_px: font_regular,
                color: label_color,
            });

            let bar_scale = ui_scale;
            let bar_x = (cw - 80.0 * bar_scale).max(8.0 * bar_scale);
            let bar_height = 200.0 * bar_scale;
            let num_segments = 3;
            let gap = 12.0 * bar_scale;
            let segment_height =
                ((bar_height - (num_segments as f32 - 1.0) * gap) / num_segments as f32)
                    .max(20.0 * bar_scale);
            let effective_bar_height =
                segment_height * num_segments as f32 + gap * (num_segments as f32 - 1.0);
            let start_y = ((ch - effective_bar_height) / 2.0).max(32.0 * bar_scale);
            let active_segments =
                (best_score * num_segments as f32).ceil().clamp(0.0, num_segments as f32) as i32;
            let bar_width = 28.0 * bar_scale;
            let diag_thickness = scaled_thickness(2.0, ui_scale);

            for i in 0..num_segments {
                let top = start_y + i as f32 * (segment_height + gap);
                let seg_color = if (i as i32) < active_segments {
                    (0, 255, 0, 255)
                } else {
                    (80, 80, 80, 255)
                };
                ops.push(DrawOp::FillRect {
                    xywh: (bar_x, top, bar_width, segment_height),
                    color: seg_color,
                });
                ops.push(DrawOp::Rect {
                    xywh: (bar_x, top, bar_width, segment_height),
                    thickness: icon_outline,
                    color: (220, 220, 220, 255),
                });
                if (i as i32) < active_segments {
                    ops.push(DrawOp::Poly {
                        pts: vec![
                            (bar_x + 6.0 * bar_scale, top + segment_height - 6.0 * bar_scale),
                            (bar_x + bar_width - 6.0 * bar_scale, top + 6.0 * bar_scale),
                        ],
                        thickness: diag_thickness,
                        color: (230, 255, 230, 255),
                    });
                }
            }

            let bottom_width = (240.0 * ui_scale).min(cw * 0.5).max(160.0 * ui_scale);
            let bottom_height = 36.0 * ui_scale;
            let bottom_x = (cw - bottom_width) / 2.0;
            let bottom_y = ch - bottom_height - 32.0 * ui_scale;
            ops.push(DrawOp::FillRect {
                xywh: (bottom_x, bottom_y, bottom_width, bottom_height),
                color: best_color,
            });
            ops.push(DrawOp::Rect {
                xywh: (bottom_x, bottom_y, bottom_width, bottom_height),
                thickness: icon_outline,
                color: (255, 255, 255, 255),
            });
            ops.push(DrawOp::Label {
                anchor: (bottom_x + 24.0 * ui_scale, bottom_y + bottom_height / 4.0),
                text: best_long_label.to_string(),
                font_px: scaled_font_px(18.0, ui_scale, 14),
                color: (255, 255, 255, 255),
            });
        }

        let bbox_scale = ui_scale;
        let bbox_outline = scaled_thickness(2.0, bbox_scale);
        let corner_outline = scaled_thickness(3.0, bbox_scale);
        let label_font = scaled_font_px(20.0, ui_scale, 14);

        for det in &input.items {
            let (short_label, _, class_color) = class_palette(det.class_id as u32);
            let (x, y, w, h) = (det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h);

            ops.push(DrawOp::Rect {
                xywh: (x, y, w, h),
                thickness: bbox_outline,
                color: class_color,
            });

            if self.enable_full_ui {
                let corner_len = (w.min(h) * 0.22)
                    .max(18.0 * ui_scale)
                    .min(40.0 * ui_scale);
                let x2 = x + w;
                let y2 = y + h;

                ops.push(DrawOp::Poly {
                    pts: vec![(x, y), (x + corner_len, y)],
                    thickness: corner_outline,
                    color: class_color,
                });
                ops.push(DrawOp::Poly {
                    pts: vec![(x, y), (x, y + corner_len)],
                    thickness: corner_outline,
                    color: class_color,
                });
                ops.push(DrawOp::Poly {
                    pts: vec![(x2, y), (x2 - corner_len, y)],
                    thickness: corner_outline,
                    color: class_color,
                });
                ops.push(DrawOp::Poly {
                    pts: vec![(x2, y), (x2, y + corner_len)],
                    thickness: corner_outline,
                    color: class_color,
                });
                ops.push(DrawOp::Poly {
                    pts: vec![(x, y2), (x + corner_len, y2)],
                    thickness: corner_outline,
                    color: class_color,
                });
                ops.push(DrawOp::Poly {
                    pts: vec![(x, y2), (x, y2 - corner_len)],
                    thickness: corner_outline,
                    color: class_color,
                });
                ops.push(DrawOp::Poly {
                    pts: vec![(x2, y2), (x2 - corner_len, y2)],
                    thickness: corner_outline,
                    color: class_color,
                });
                ops.push(DrawOp::Poly {
                    pts: vec![(x2, y2), (x2, y2 - corner_len)],
                    thickness: corner_outline,
                    color: class_color,
                });
            }

            let track_info = if let Some(track_id) = det.track_id {
                format!(" ID:{}", track_id)
            } else {
                String::new()
            };
            let label_text = format!("{} {:.2}{}", short_label, det.score, track_info);
            let preferred_top = y - 28.0 * ui_scale;
            let label_y = if preferred_top > 12.0 * ui_scale {
                preferred_top
            } else {
                y + h + 16.0 * ui_scale
            };
            ops.push(DrawOp::Label {
                anchor: (x, label_y),
                text: label_text,
                font_px: label_font,
                color: class_color,
            });
        }

        if self.enable_full_ui {
            let corner_padding = 32.0 * ui_scale;
            let icon_outline = scaled_thickness(2.0, ui_scale);
            let icon_size = (32.0 * ui_scale).max(14.0);
            let corner_x = cw - icon_size - corner_padding;
            let mut corner_y = corner_padding;

            ops.push(DrawOp::Rect {
                xywh: (corner_x, corner_y, icon_size, icon_size),
                thickness: icon_outline,
                color: (255, 255, 255, 255),
            });

            let speaker_x = corner_x + icon_size + 16.0 * ui_scale;
            let speaker_h = icon_size * 0.7;
            let speaker_body_w = icon_size * 0.3;
            let speaker_y = corner_y + (icon_size - speaker_h) / 2.0;

            ops.push(DrawOp::FillRect {
                xywh: (speaker_x, speaker_y, speaker_body_w, speaker_h),
                color: (255, 255, 255, 255),
            });

            let horn_pts = vec![
                (speaker_x + speaker_body_w, speaker_y),
                (
                    speaker_x + speaker_body_w + speaker_h * 0.55,
                    speaker_y + speaker_h * 0.5,
                ),
                (speaker_x + speaker_body_w, speaker_y + speaker_h),
            ];
            ops.push(DrawOp::Poly {
                pts: horn_pts,
                thickness: icon_outline,
                color: (255, 255, 255, 255),
            });

            let wave_inner = vec![
                (
                    speaker_x + speaker_body_w + speaker_h * 0.35,
                    speaker_y + speaker_h * 0.2,
                ),
                (
                    speaker_x + speaker_body_w + speaker_h * 0.58,
                    speaker_y + speaker_h * 0.5,
                ),
                (
                    speaker_x + speaker_body_w + speaker_h * 0.35,
                    speaker_y + speaker_h * 0.8,
                ),
            ];
            let wave_outer = vec![
                (
                    speaker_x + speaker_body_w + speaker_h * 0.55,
                    speaker_y + speaker_h * 0.05,
                ),
                (
                    speaker_x + speaker_body_w + speaker_h * 0.86,
                    speaker_y + speaker_h * 0.5,
                ),
                (
                    speaker_x + speaker_body_w + speaker_h * 0.55,
                    speaker_y + speaker_h * 0.95,
                ),
            ];
            ops.push(DrawOp::Poly {
                pts: wave_inner,
                thickness: icon_outline,
                color: (255, 255, 255, 255),
            });
            ops.push(DrawOp::Poly {
                pts: wave_outer,
                thickness: icon_outline,
                color: (255, 255, 255, 255),
            });

            corner_y += icon_size + 12.0 * ui_scale;

            let mode_text = best_detection
                .map(|d| if d.class_id == 1 { "EGD" } else { "COLON" })
                .unwrap_or("COLON");
            let font_regular = scaled_font_px(20.0, ui_scale, 14);
            ops.push(DrawOp::Label {
                anchor: (corner_x, corner_y),
                text: mode_text.to_string(),
                font_px: font_regular,
                color: (255, 255, 255, 255),
            });
            corner_y += 24.0 * ui_scale;
            ops.push(DrawOp::Label {
                anchor: (corner_x, corner_y),
                text: format!("T = {:.2}", best_score),
                font_px: scaled_font_px(18.0, ui_scale, 14),
                color: (255, 255, 255, 255),
            });
        }

        OverlayPlanPacket {
            from: input.from,
            ops,
            canvas,
        }
    }
}

impl PlanStage {
    fn parse_enable_full_ui(cfg: &str) -> bool {
        let cfg_lc = cfg.to_ascii_lowercase();
        ["full", "full_ui", "ui=full", "hud"]
            .iter()
            .any(|key| cfg_lc.contains(key))
    }

    pub fn from_path(cfg: &str) -> Result<Self> {
        Ok(Self {
            enable_full_ui: Self::parse_enable_full_ui(cfg),
        })
    }
}
