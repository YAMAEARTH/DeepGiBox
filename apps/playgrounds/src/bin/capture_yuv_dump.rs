use std::error::Error;
use std::thread;
use std::time::Duration;

use decklink_input::capture::CaptureSession;
use decklink_input::{copy_frame_region_to_host, RawFramePacket};

fn clamp_u8(v: i32) -> u8 {
    if v < 0 {
        0
    } else if v > 255 {
        255
    } else {
        v as u8
    }
}

fn yuv_to_rgb(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let c = y as i32 - 16;
    let d = u as i32 - 128;
    let e = v as i32 - 128;
    let r = (298 * c + 409 * e + 128) >> 8;
    let g = (298 * c - 100 * d - 208 * e + 128) >> 8;
    let b = (298 * c + 516 * d + 128) >> 8;
    (clamp_u8(r), clamp_u8(g), clamp_u8(b))
}

fn fetch_sample_bytes(
    packet: &RawFramePacket,
    offset: usize,
    len: usize,
) -> Result<Vec<u8>, String> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let mut buf = vec![0u8; len];
    copy_frame_region_to_host(packet, offset, &mut buf).map_err(|e| e.to_string())?;
    Ok(buf)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut session = CaptureSession::open(0)?;
    println!("Opened capture session on device 0");

    for attempt in 0..60 {
        if let Some(packet) = session.get_frame()? {
            println!(
                "Frame idx={} size={}x{} stride={} buffer_len={} bytes pixfmt={:?} colorspace={:?} loc={:?}",
                packet.meta.frame_idx,
                packet.meta.width,
                packet.meta.height,
                packet.meta.stride_bytes,
                packet.data.len,
                packet.meta.pixfmt,
                packet.meta.colorspace,
                packet.data.loc,
            );

            let stride = packet.data.stride;
            let width = packet.meta.width as usize;
            let height = packet.meta.height as usize;
            if stride == 0 || width == 0 || height == 0 {
                println!("Frame dimensions invalid (stride/width/height)");
            } else {
                let bytes_per_pixel: usize = 2; // DeckLink 8-bit YUV (2vuy / UYVY)
                let expected_row = width.saturating_mul(bytes_per_pixel);
                if stride < expected_row {
                    println!(
                        "Warning: stride {} smaller than expected row bytes {}",
                        stride, expected_row
                    );
                }

                let mid_y = height / 2;
                let mut sample_pixels: usize = usize::min(6, width);
                if sample_pixels % 2 != 0 {
                    sample_pixels -= 1;
                }
                if sample_pixels == 0 {
                    println!("Frame too narrow to sample");
                } else {
                    let mid_x = width / 2;
                    let half = sample_pixels / 2;
                    let mut start_x = mid_x.saturating_sub(half);
                    if start_x + sample_pixels > width {
                        start_x = width - sample_pixels;
                    }
                    start_x &= !1; // align to even pixel index

                    let offset = mid_y
                        .saturating_mul(stride)
                        .saturating_add(start_x.saturating_mul(bytes_per_pixel));
                    let needed_bytes = sample_pixels.saturating_mul(bytes_per_pixel);
                    if offset.saturating_add(needed_bytes) > packet.data.len {
                        println!(
                            "Not enough data to sample center pixels (need {needed_bytes} bytes)"
                        );
                    } else {
                        match fetch_sample_bytes(&packet, offset, needed_bytes) {
                            Ok(raw) => {
                                for (pair_idx, chunk) in raw.chunks_exact(4).enumerate() {
                                    let u = chunk[0];
                                    let y0 = chunk[1];
                                    let v = chunk[2];
                                    let y1 = chunk[3];
                                    let x0 = start_x + pair_idx * 2;
                                    let x1 = x0 + 1;
                                    let (r0, g0, b0) = yuv_to_rgb(y0, u, v);
                                    let (r1, g1, b1) = yuv_to_rgb(y1, u, v);
                                    println!(
                                        "pixel[y={mid_y} x={x0}] = Y:{y0:3} U:{u:3} V:{v:3} -> R:{r0:3} G:{g0:3} B:{b0:3}"
                                    );
                                    if x1 < width {
                                        println!(
                                            "pixel[y={mid_y} x={x1}] = Y:{y1:3} U:{u:3} V:{v:3} -> R:{r1:3} G:{g1:3} B:{b1:3}"
                                        );
                                    }
                                }
                            }
                            Err(err) => {
                                println!("Failed to fetch sample bytes: {err}");
                            }
                        }
                    }
                }
            }
        } else {
            println!("No frame yet (attempt {attempt})");
        }
        thread::sleep(Duration::from_millis(100));
    }

    Err("Timed out waiting for a frame".into())
}
