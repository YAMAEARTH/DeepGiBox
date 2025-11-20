use anyhow::Result;
use common_io::{DetectionsPacket, OverlayPlanPacket, DrawOp, Stage};
use std::cmp::Ordering;

// GPU overlay planning module (optional)
#[cfg(feature = "gpu")]
pub mod gpu_overlay;

// Re-export GPU utilities for easy access
#[cfg(feature = "gpu")]
pub use gpu_overlay::{
    GpuOverlayPlanner, 
    GpuOverlayPlan,
    convert_gpu_to_overlay_plan,
    prepare_detections_for_gpu,
};

/// Helper function: Process detections with GPU overlay planning
/// Returns OverlayPlanPacket compatible with existing render pipeline
#[cfg(feature = "gpu")]
pub fn process_with_gpu_planning(
    gpu_planner: &mut GpuOverlayPlanner,
    detections: DetectionsPacket,
) -> Result<OverlayPlanPacket> {
    let canvas = (detections.from.width, detections.from.height);
    
    // Generate GPU overlay plan
    let gpu_plan = gpu_planner.plan_from_detections(&detections)?;
    
    // Convert to CPU format for render stage
    convert_gpu_to_overlay_plan(gpu_plan, canvas)
}

pub fn from_path(cfg: &str) -> Result<PlanStage> {
    PlanStage::from_path(cfg)
}

pub struct PlanStage {
    pub enable_full_ui: bool,
    pub spk: bool,  // ตัวแปรเปิด/ปิดไอคอนลำโพง (speaker on/off)
    pub display_mode: String,  // "COLON" หรือ "EGD" - แสดงบน UI
    pub confidence_threshold: f32,  // ค่า threshold จาก config (แสดงบน UI)
    pub endoscope_mode: String,  // "OLYMPUS" หรือ "FUJI" - กำหนดตำแหน่ง overlay
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
        // ops = รายการคำสั่งวาด (Drawing Operations)
        let mut ops = Vec::new();

        // canvas = ขนาดภาพ (width x height)
        let canvas = (input.from.width, input.from.height);
        // cw, ch = ความกว้าง, ความสูงของภาพ (canvas width, canvas height)
        let (cw, ch) = (canvas.0 as f32, canvas.1 as f32);

        if cw <= 0.0 || ch <= 0.0 {
            eprintln!("Invalid canvas dimensions: ({}, {})", cw, ch);
            return OverlayPlanPacket { from: input.from, ops, canvas };
        }

        // ui_scale = อัตราส่วนการขยาย UI ตามความสูงของภาพ (1080p = scale 1.0)
        let ui_scale = (ch / 1080.0).max(0.1);

        // best_detection = การตรวจจับที่มี confidence score สูงสุด
        let best_detection = input
            .items
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal));
        // best_score = คะแนนความมั่นใจสูงสุด (0.0 - 1.0)
        let best_score = best_detection.map(|d| d.score.clamp(0.0, 1.0)).unwrap_or(0.0);
        // best_long_label = ชื่อประเภทของการตรวจจับ, _best_color = สีของการตรวจจับ (ไม่ใช้)
        let (_, best_long_label, _best_color) = best_detection
            .map(|d| class_palette(d.class_id as u32))
            .unwrap_or(("None", "Awaiting", (80, 80, 80, 255)));

        if self.enable_full_ui {
            // outline = ความหนาของเส้นขอบ
            let outline = scaled_thickness(2.0, ui_scale);
            
            // ============================================================
            // CONFIDENCE BAR (แถบแสดงความมั่นใจ) - 4 ช่อง แนวตั้ง
            // ตำแหน่งปรับตาม endoscope_mode: OLYMPUS=ซ้าย, FUJI=ขวา
            // ============================================================
            let bar_scale = ui_scale;  // อัตราส่วนการขยายแถบ
            // bar_x = ตำแหน่ง X (ซ้าย=OLYMPUS, ขวา=FUJI)
            let bar_x = if self.endoscope_mode == "FUJI" {
                cw - 120.0 * ui_scale  // FUJI: ชิดขวา
            } else {
                400.0 * ui_scale  // OLYMPUS: ซ้ายมือ
            };
            let bar_height = 260.0 * bar_scale;  // ความสูงรวมของแถบ
            let num_segments = 4;  // จำนวนช่องในแถบ
            let gap = 10.0 * bar_scale;  // ช่องว่างระหว่างแต่ละช่อง
            // segment_height = ความสูงของแต่ละช่อง
            let segment_height =
                ((bar_height - (num_segments as f32 - 1.0) * gap) / num_segments as f32)
                    .max(20.0 * bar_scale);
            // effective_bar_height = ความสูงรวมที่แท้จริง (รวมช่องว่าง)
            let effective_bar_height =
                segment_height * num_segments as f32 + gap * (num_segments as f32 - 1.0);
            // start_y = ตำแหน่ง Y เริ่มต้น (กึ่งกลางแนวตั้ง)
            let start_y = ((ch - effective_bar_height) / 2.0).max(32.0 * bar_scale);
            // active_segments = จำนวนช่องที่ active (ตามคะแนน confidence)
            let active_segments =
                (best_score * num_segments as f32).ceil().clamp(0.0, num_segments as f32) as i32;
            // bar_width = ความกว้างของแถบ
            let bar_width = 28.0 * bar_scale;

            // วนวาดแต่ละช่องของ Confidence Bar (จากล่างขึ้นบน)
            for i in 0..num_segments {
                // reversed_i = ดัชนีกลับหัว (ช่องล่างสุด = 0, ช่องบนสุด = 3)
                let reversed_i = num_segments - 1 - i;
                // top = ตำแหน่ง Y ของช่องปัจจุบัน (นับจากล่างขึ้นบน)
                let top = start_y + reversed_i as f32 * (segment_height + gap);
                // seg_color = สีของช่อง (ขาวเข้มถ้า active, เทาอ่อนถ้า inactive)
                let seg_color = if (i as i32) < active_segments {
                    (255, 255, 255, 255)  // สีขาวเข้ม สำหรับช่อง active
                } else {
                    (80, 80, 80, 255)  // สีเทาอ่อน สำหรับช่อง inactive
                };
                // วาดสี่เหลี่ยมเติมสี (ตัวช่อง)
                ops.push(DrawOp::FillRect {
                    xywh: (bar_x, top, bar_width, segment_height),
                    color: seg_color,
                });
                // วาดเส้นขอบช่อง
                ops.push(DrawOp::Rect {
                    xywh: (bar_x, top, bar_width, segment_height),
                    thickness: outline,
                    color: (220, 220, 220, 255),
                });
                // ไม่มีเส้นทแยงมุม - ลบออกแล้ว
            }

            // ============================================================
            // TEXT DISPLAY (แสดงข้อความด้วย DrawOp::Label - ตอนนี้ implement แล้ว!)
            // ตำแหน่งปรับตาม endoscope_mode: OLYMPUS=ซ้าย, FUJI=กลาง
            // ============================================================
            // label_x = ตำแหน่ง X ของ label (FUJI=กลางจอ, OLYMPUS=ซ้ายมือ)
            let label_x = if self.endoscope_mode == "FUJI" {
                cw / 2.0 - 100.0 * ui_scale  // FUJI: กลางหน้าจอ
            } else {
                bar_x - 200.0  // OLYMPUS: ซ้ายมือ
            };
            // label_y = ตำแหน่ง Y ของ label (FUJI=ล่างสุด, OLYMPUS=ใต้แถบ)
            let label_y = if self.endoscope_mode == "FUJI" {
                ch - 80.0 * ui_scale  // FUJI: ใกล้ขอบล่าง
            } else {
                start_y + effective_bar_height + 28.0 * ui_scale  // OLYMPUS: ใต้แถบ confidence
            };
            
            // === CLASS LABEL (แสดงประเภท Hyperplastic/Neoplastic) ===
            ops.push(DrawOp::Label {
                anchor: (label_x, label_y),
                text: best_long_label.to_string(),
                font_px: scaled_font_px(54.0, ui_scale, 32),
                color: (255, 255, 0, 255),  // Yellow - สีเหลืองสด
            });
        }

        // ============================================================
        // MODE & THRESHOLD TEXT
        // ตำแหน่งปรับตาม endoscope_mode: OLYMPUS=ขวาบน, FUJI=ซ้ายบน
        // ============================================================
        if self.enable_full_ui {
            let controls_margin = 80.0 * ui_scale;
            let icon_size = (64.0 * ui_scale).max(28.0);
            let square_size = (48.0 * ui_scale).max(22.0);
            
            // ตำแหน่ง X และ Y ปรับตาม endoscope_mode
            let (text_x, text_start_y) = if self.endoscope_mode == "FUJI" {
                // FUJI: มุมซ้ายบน
                (controls_margin, 80.0 * ui_scale)
            } else {
                // OLYMPUS: มุมขวาบน (ใต้ลำโพง)
                let right_x = cw - controls_margin - square_size - 16.0 * ui_scale - icon_size + icon_size + 12.0 * ui_scale;
                (right_x, controls_margin)
            };

            // === MODE TEXT (แสดง COLON/EGD จาก config) ===
            ops.push(DrawOp::Label {
                anchor: (text_x, text_start_y),
                text: self.display_mode.clone(),
                font_px: scaled_font_px(35.0, ui_scale, 36),
                color: (255, 255, 255, 255),  // White - สีขาว
            });

            // === THRESHOLD TEXT (แสดงค่า T = 0.xx จาก config) ===
            let score_text = format!("T = {:.2}", self.confidence_threshold);
            let score_y = text_start_y + 50.0 * ui_scale;
            ops.push(DrawOp::Label {
                anchor: (text_x, score_y),
                text: score_text,
                font_px: scaled_font_px(35.0, ui_scale, 30),
                color: (255, 255, 255, 255),  // White - สีขาว
            });
        }

        // ============================================================
        // BOUNDING BOXES (กรอบวัตถุที่ตรวจพบ)
        // ============================================================
        let _bbox_scale = ui_scale;  // อัตราส่วนการขยายกรอบ (ไม่ใช้แล้ว)
        let corner_outline = scaled_thickness(3.0, ui_scale);  // ความหนาเส้นมุม

        // วนวาดกรอบสำหรับแต่ละวัตถุที่ตรวจพบ
        for det in &input.items {
            // box_color = สีฟ้าสำหรับกรอบ (ไม่แยกตามคลาส)
            let box_color = (0, 180, 255, 255);  // สีฟ้า (Cyan Blue)
            // x, y, w, h = ตำแหน่งและขนาดของกรอบ
            let (x, y, w, h) = (det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h);

            // ไม่วาดกรอบสี่เหลี่ยมเต็มรูป - ใช้เฉพาะเส้นมุมโค้งมนแทน

            // วาดเส้นมุมโค้งมน (Rounded Corner Brackets) - 4 มุมของกรอบ สีฟ้า
            if self.enable_full_ui {
                // corner_len = ความยาวของเส้นมุม (22% ของด้านที่สั้นกว่า)
                let corner_len = (w.min(h) * 0.22)
                    .max(18.0 * ui_scale)
                    .min(40.0 * ui_scale);
                // radius = รัศมีความโค้งของมุม (ประมาณ 30% ของความยาวเส้น)
                let radius = (corner_len * 0.3).max(4.0 * ui_scale);
                // x2, y2 = ตำแหน่งมุมขวาล่าง
                let x2 = x + w;
                let y2 = y + h;

                // === มุมซ้ายบน (Top-Left) - มุมโค้งมน ===
                // เส้นนอน (แนวนอนจากซ้าย) โค้งเข้ามุม
                ops.push(DrawOp::Poly {
                    pts: vec![
                        (x, y + radius),
                        (x, y + radius * 0.5),
                        (x + radius * 0.5, y),
                        (x + radius, y),
                    ],
                    thickness: corner_outline,
                    color: box_color,
                });
                // เส้นนอนยาวออกไป
                ops.push(DrawOp::Poly {
                    pts: vec![(x + radius, y), (x + corner_len, y)],
                    thickness: corner_outline,
                    color: box_color,
                });
                // เส้นตั้งยาวลงมา
                ops.push(DrawOp::Poly {
                    pts: vec![(x, y + radius), (x, y + corner_len)],
                    thickness: corner_outline,
                    color: box_color,
                });

                // === มุมขวาบน (Top-Right) - มุมโค้งมน ===
                ops.push(DrawOp::Poly {
                    pts: vec![
                        (x2 - radius, y),
                        (x2 - radius * 0.5, y),
                        (x2, y + radius * 0.5),
                        (x2, y + radius),
                    ],
                    thickness: corner_outline,
                    color: box_color,
                });
                ops.push(DrawOp::Poly {
                    pts: vec![(x2 - corner_len, y), (x2 - radius, y)],
                    thickness: corner_outline,
                    color: box_color,
                });
                ops.push(DrawOp::Poly {
                    pts: vec![(x2, y + radius), (x2, y + corner_len)],
                    thickness: corner_outline,
                    color: box_color,
                });

                // === มุมซ้ายล่าง (Bottom-Left) - มุมโค้งมน ===
                ops.push(DrawOp::Poly {
                    pts: vec![
                        (x, y2 - radius),
                        (x, y2 - radius * 0.5),
                        (x + radius * 0.5, y2),
                        (x + radius, y2),
                    ],
                    thickness: corner_outline,
                    color: box_color,
                });
                ops.push(DrawOp::Poly {
                    pts: vec![(x, y2 - corner_len), (x, y2 - radius)],
                    thickness: corner_outline,
                    color: box_color,
                });
                ops.push(DrawOp::Poly {
                    pts: vec![(x + radius, y2), (x + corner_len, y2)],
                    thickness: corner_outline,
                    color: box_color,
                });

                // === มุมขวาล่าง (Bottom-Right) - มุมโค้งมน ===
                ops.push(DrawOp::Poly {
                    pts: vec![
                        (x2 - radius, y2),
                        (x2 - radius * 0.5, y2),
                        (x2, y2 - radius * 0.5),
                        (x2, y2 - radius),
                    ],
                    thickness: corner_outline,
                    color: box_color,
                });
                ops.push(DrawOp::Poly {
                    pts: vec![(x2 - corner_len, y2), (x2 - radius, y2)],
                    thickness: corner_outline,
                    color: box_color,
                });
                ops.push(DrawOp::Poly {
                    pts: vec![(x2, y2 - corner_len), (x2, y2 - radius)],
                    thickness: corner_outline,
                    color: box_color,
                });
            }

            // ============================================================
            // DETECTION LABEL (ข้อความข้างกรอบวัตถุ) - ลบออกแล้ว
            // ============================================================
            // ไม่แสดงข้อความบนกรอบสีฟ้าอีกต่อไป
        }

        if self.enable_full_ui {
            // ============================================================
            // CONTROL BOX + SPEAKER ICON
            // ตำแหน่งปรับตาม endoscope_mode: OLYMPUS=ขวาบน, FUJI=ซ้ายบน
            // ============================================================
            let controls_margin = 10.0 * ui_scale;  // ระยะห่างจากขอบ
            let icon_size = (64.0 * ui_scale).max(28.0);
            let square_size = (48.0 * ui_scale).max(22.0);
            
            // ตำแหน่ง block_x, speaker_x ปรับตาม endoscope_mode
            let (block_x, speaker_x) = if self.endoscope_mode == "FUJI" {
                // FUJI: มุมซ้ายบน (กล่องซ้ายสุด, ลำโพงขวากล่อง)
                let blk_x = controls_margin + 70.0 * ui_scale;  // ชิดซ้าย (เว้นระยะให้ MODE TEXT)
                let spk_x = blk_x + square_size + 16.0 * ui_scale;
                (blk_x, spk_x)
            } else {
                // OLYMPUS: มุมขวาบน (ลำโพงขวาสุด, กล่องซ้ายลำโพง)
                let spk_x = cw - controls_margin - icon_size;
                let blk_x = spk_x - 16.0 * ui_scale - square_size;
                (blk_x, spk_x)
            };
            let block_y = controls_margin;

            // กล่องสีเทา
            ops.push(DrawOp::FillRect {
                xywh: (block_x, block_y, square_size, square_size),
                color: (58, 58, 58, 255),
            });
            ops.push(DrawOp::Rect {
                xywh: (block_x, block_y, square_size, square_size),
                thickness: scaled_thickness(2.0, ui_scale),
                color: (150, 150, 150, 255),
            });

            // ไอคอนลำโพง (วาดด้วย shapes แบบง่ายๆ ให้เห็นชัด)
            if self.spk {
                let center_y = block_y + square_size / 2.0;
                let spk_size = square_size * 0.6;
                let spk_x = speaker_x + (icon_size - spk_size) / 2.0;
                
                // === ตัวลำโพง (สี่เหลี่ยม) ===
                let body_w = spk_size * 0.25;
                let body_h = spk_size * 0.50;
                let body_x = spk_x;
                let body_y = center_y - body_h / 2.0;
                
                ops.push(DrawOp::FillRect {
                    xywh: (body_x, body_y, body_w, body_h),
                    color: (255, 255, 255, 255),
                });
                
                // === แตร (trapezoid) ===
                let horn_x = body_x + body_w;
                let horn_w = spk_size * 0.40;
                
                ops.push(DrawOp::Poly {
                    pts: vec![
                        (horn_x, body_y),
                        (horn_x + horn_w, body_y - body_h * 0.30),
                        (horn_x + horn_w, body_y + body_h + body_h * 0.30),
                        (horn_x, body_y + body_h),
                        (horn_x, body_y),
                    ],
                    thickness: 0,
                    color: (255, 255, 255, 255),
                });
                
                // เติมสีแตร
                for i in 0..10 {
                    let t = i as f32 / 10.0;
                    let x = horn_x + horn_w * t;
                    let h_expand = body_h * 0.30 * t;
                    let y1 = body_y - h_expand;
                    let y2 = body_y + body_h + h_expand;
                    
                    ops.push(DrawOp::Poly {
                        pts: vec![(x, y1), (x, y2)],
                        thickness: scaled_thickness(2.0, ui_scale),
                        color: (255, 255, 255, 255),
                    });
                }
                
                // === คลื่นเสียง 3 เส้น ===
                let wave_x = horn_x + horn_w + spk_size * 0.15;
                let wave_gap = spk_size * 0.10;
                
                for i in 0..3 {
                    let x = wave_x + i as f32 * wave_gap;
                    let h = body_h * 0.30 + i as f32 * body_h * 0.15;
                    
                    ops.push(DrawOp::Poly {
                        pts: vec![
                            (x, center_y - h),
                            (x + wave_gap * 0.4, center_y),
                            (x, center_y + h),
                        ],
                        thickness: scaled_thickness(2.5, ui_scale),
                        color: (255, 255, 255, 255),
                    });
                }
            }
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
            spk: true,  // ค่าเริ่มต้น: เปิดลำโพง (แสดงไอคอน)
            display_mode: "COLON".to_string(),  // ค่าเริ่มต้น (จะถูก override จาก config)
            confidence_threshold: 0.25,  // ค่าเริ่มต้น (จะถูก override จาก config)
            endoscope_mode: "OLYMPUS".to_string(),  // ค่าเริ่มต้น (จะถูก override จาก main.rs)
        })
    }
}
