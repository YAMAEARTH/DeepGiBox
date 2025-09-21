use crate::post::PostprocessingResult;
use image::{DynamicImage, Rgb, RgbImage};
use ort::{Error, Result};
use std::path::Path;

const BOX_THICKNESS: u32 = 2;

pub fn visualize_detections(image: &DynamicImage, result: &PostprocessingResult, output_path: &Path) -> Result<()> {
    let mut canvas: RgbImage = image.to_rgb8();

    if !result.tracks.is_empty() {
        for tracked in &result.tracks {
            let color = color_for_id(tracked.id);
            draw_bbox(&mut canvas, tracked.bbox, color);
        }
    } else {
        for (idx, detection) in result.detections.iter().enumerate() {
            let color = color_for_id(idx as u64 + 1);
            draw_bbox(&mut canvas, detection.bbox, color);
        }
    }

    canvas
        .save(output_path)
        .map_err(Error::wrap)?;
    Ok(())
}

fn draw_bbox(image: &mut RgbImage, bbox: [f32; 4], color: [u8; 3]) {
    let (width, height) = (image.width(), image.height());

    let x_min = bbox[0].floor().max(0.0) as i32;
    let y_min = bbox[1].floor().max(0.0) as i32;
    let x_max = bbox[2].ceil().min((width - 1) as f32) as i32;
    let y_max = bbox[3].ceil().min((height - 1) as f32) as i32;

    if x_min >= x_max || y_min >= y_max {
        return;
    }

    for offset in 0..BOX_THICKNESS as i32 {
        draw_horizontal_line(image, y_min + offset, x_min, x_max, color);
        draw_horizontal_line(image, y_max - offset, x_min, x_max, color);
        draw_vertical_line(image, x_min + offset, y_min, y_max, color);
        draw_vertical_line(image, x_max - offset, y_min, y_max, color);
    }
}

fn draw_horizontal_line(image: &mut RgbImage, y: i32, x_start: i32, x_end: i32, color: [u8; 3]) {
    if y < 0 || y >= image.height() as i32 {
        return;
    }
    let y = y as u32;
    let start = x_start.max(0) as u32;
    let end = x_end.min(image.width() as i32 - 1) as u32;
    for x in start..=end {
        image.put_pixel(x, y, Rgb(color));
    }
}

fn draw_vertical_line(image: &mut RgbImage, x: i32, y_start: i32, y_end: i32, color: [u8; 3]) {
    if x < 0 || x >= image.width() as i32 {
        return;
    }
    let x = x as u32;
    let start = y_start.max(0) as u32;
    let end = y_end.min(image.height() as i32 - 1) as u32;
    for y in start..=end {
        image.put_pixel(x, y, Rgb(color));
    }
}

fn color_for_id(id: u64) -> [u8; 3] {
    let mut hash = id.wrapping_mul(0x9E3779B185EBCA87);
    let r = ((hash & 0xFF) as u8) | 0x50;
    hash >>= 8;
    let g = ((hash & 0xFF) as u8) | 0x50;
    hash >>= 8;
    let b = ((hash & 0xFF) as u8) | 0x50;
    [r, g, b]
}
