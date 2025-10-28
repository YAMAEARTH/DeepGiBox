use anyhow::Result;
use common_io::{MemLoc, MemRef, OverlayFramePacket, OverlayPlanPacket, Stage, DrawOp};

// ----- Minimal CPU drawing helpers (ARGB) -----
fn put_pixel(buf: &mut [u8], stride: usize, w: usize, h: usize, x: i32, y: i32, a: u8, r: u8, g: u8, b: u8) {
    if x < 0 || y < 0 { return; }
    let (x, y) = (x as usize, y as usize);
    if x >= w || y >= h { return; }
    let idx = y * stride + x * 4;
    buf[idx + 0] = a;
    buf[idx + 1] = r;
    buf[idx + 2] = g;
    buf[idx + 3] = b;
}

fn draw_line(buf: &mut [u8], stride: usize, w: usize, h: usize, x0: i32, y0: i32, x1: i32, y1: i32, thickness: i32, color: (u8, u8, u8, u8)) {
    // Bresenham with naive thickness (square brush)
    let (mut x0, mut y0, x1, y1) = (x0, y0, x1, y1);
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let rad = thickness.max(1) / 2;
    loop {
        for oy in -rad..=rad {
            for ox in -rad..=rad {
                put_pixel(buf, stride, w, h, x0 + ox, y0 + oy, color.0, color.1, color.2, color.3);
            }
        }
        if x0 == x1 && y0 == y1 { break; }
        let e2 = 2 * err;
        if e2 >= dy { err += dy; x0 += sx; }
        if e2 <= dx { err += dx; y0 += sy; }
    }
}

fn draw_rect_outline(buf: &mut [u8], stride: usize, w: usize, h: usize, x: i32, y: i32, ww: i32, hh: i32, thickness: i32, color: (u8, u8, u8, u8)) {
    if ww <= 0 || hh <= 0 { return; }
    let x2 = x + ww;
    let y2 = y + hh;
    draw_line(buf, stride, w, h, x, y, x2, y, thickness, color);
    draw_line(buf, stride, w, h, x2, y, x2, y2, thickness, color);
    draw_line(buf, stride, w, h, x2, y2, x, y2, thickness, color);
    draw_line(buf, stride, w, h, x, y2, x, y, thickness, color);
}

pub fn from_path(_cfg: &str) -> Result<RenderStage> {
    Ok(RenderStage {})
}
pub struct RenderStage {}
impl Stage<OverlayPlanPacket, OverlayFramePacket> for RenderStage {
    fn process(&mut self, input: OverlayPlanPacket) -> OverlayFramePacket {
        // Create ARGB buffer
        let w = input.canvas.0 as usize;
        let h = input.canvas.1 as usize;
        let stride = w * 4;

        let mut buf = vec![0u8; h * stride];

        // Execute ops
        for op in &input.ops {
            match op {
                DrawOp::Rect { xywh, thickness, color } => {
                    let (x, y, ww, hh) = (xywh.0 as i32, xywh.1 as i32, xywh.2 as i32, xywh.3 as i32);
                    draw_rect_outline(&mut buf, stride, w, h, x, y, ww, hh, *thickness as i32, *color);
                }
                DrawOp::FillRect { xywh, color } => {
                    let (x, y, ww, hh) = (xywh.0 as i32, xywh.1 as i32, xywh.2 as i32, xywh.3 as i32);
                    for yy in y.max(0)..(y+hh).min(h as i32) {
                        for xx in x.max(0)..(x+ww).min(w as i32) {
                            put_pixel(&mut buf, stride, w, h, xx, yy, color.0, color.1, color.2, color.3);
                        }
                    }
                }
                DrawOp::Poly { pts, thickness, color } => {
                    if pts.len() >= 2 {
                        let (x0, y0) = (pts[0].0 as i32, pts[0].1 as i32);
                        let (x1, y1) = (pts[1].0 as i32, pts[1].1 as i32);
                        draw_line(&mut buf, stride, w, h, x0, y0, x1, y1, *thickness as i32, *color);
                    }
                }
                DrawOp::Label { anchor: _, text: _, font_px: _, color: _ } => {
                    // No-op for text in minimal renderer (color parsed)
                }
            }
        }

        // Leak buffer to keep memory alive (since MemRef carries a raw pointer w/o ownership)
        let len = buf.len();
        let boxed = buf.into_boxed_slice();
        let ptr = Box::into_raw(boxed) as *mut u8;

        OverlayFramePacket {
            from: input.from,
            argb: MemRef { ptr, len, stride, loc: MemLoc::Cpu },
            stride,
        }
    }
}
