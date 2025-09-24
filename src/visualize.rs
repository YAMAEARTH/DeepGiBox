use crate::post::PostprocessingResult;
use image::{io::Reader as ImageReader, Rgb, RgbImage};
use ort::{Error, Result};
use std::path::Path;

const BOX_COLOR: [u8; 3] = [255, 99, 132];

pub fn visualize_detections(
    image_path: &Path,
    result: &PostprocessingResult,
    output_path: &Path,
) -> Result<()> {
    let mut image = load_image(image_path)?;

    if let Some(best_detection) = result.detections.first() {
        draw_bbox(&mut image, best_detection.bbox, BOX_COLOR);
    }

    image
        .save(output_path)
        .map_err(Error::wrap)?;

    Ok(())
}

pub fn visualize_detections_on_image(
    image: &RgbImage,
    result: &PostprocessingResult,
) -> Result<RgbImage> {
    let mut output_image = image.clone();
    
    // Draw all detections
    for detection in &result.detections {
        draw_bbox(&mut output_image, detection.bbox, BOX_COLOR);
    }
    
    // Draw all tracks (if any)
    for track in &result.tracks {
        // Use different color for tracks
        const TRACK_COLOR: [u8; 3] = [0, 255, 0]; // Green for tracks
        draw_bbox(&mut output_image, track.bbox, TRACK_COLOR);
        
        // You could also draw track ID here if needed
        // draw_text(&mut output_image, &track.id.to_string(), track.bbox);
    }
    
    Ok(output_image)
}

fn load_image(path: &Path) -> Result<RgbImage> {
    ImageReader::open(path)
        .map_err(Error::wrap)?
        .decode()
        .map(|img| img.to_rgb8())
        .map_err(Error::wrap)
}

fn draw_bbox(image: &mut RgbImage, bbox: [f32; 4], color: [u8; 3]) {
    let (width, height) = (image.width(), image.height());

    let mut left = bbox[0].floor() as i32;
    let mut top = bbox[1].floor() as i32;
    let mut right = bbox[2].ceil() as i32;
    let mut bottom = bbox[3].ceil() as i32;

    left = left.clamp(0, width.saturating_sub(1) as i32);
    top = top.clamp(0, height.saturating_sub(1) as i32);
    right = right.clamp(0, width.saturating_sub(1) as i32);
    bottom = bottom.clamp(0, height.saturating_sub(1) as i32);

    if left >= right || top >= bottom {
        return;
    }

    let color = Rgb(color);

    for x in left..=right {
        put_pixel(image, x, top, color);
        put_pixel(image, x, bottom, color);
    }

    for y in top..=bottom {
        put_pixel(image, left, y, color);
        put_pixel(image, right, y, color);
    }
}

fn put_pixel(image: &mut RgbImage, x: i32, y: i32, color: Rgb<u8>) {
    if x >= 0 && y >= 0 {
        let (x_u32, y_u32) = (x as u32, y as u32);
        if x_u32 < image.width() && y_u32 < image.height() {
            image.put_pixel(x_u32, y_u32, color);
        }
    }
}
