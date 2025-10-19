/// Test program for DeckLink internal keying
/// 
/// This program demonstrates how to use the internal keying functionality
/// by creating synthetic fill (YUV8) and key (BGRA) frames and outputting them
/// through a DeckLink device.
/// 
/// Usage:
///   cargo run --bin test_internal_keying [device_index]
/// 
/// The program will:
/// 1. Open the DeckLink output device
/// 2. Generate test patterns (color bars for fill, moving box with alpha for key)
/// 3. Output frames with internal keying enabled
/// 4. Run for a few seconds then exit

use anyhow::Result;
use decklink_output::keying::InternalKeyingOutput;
use common_io::{FrameMeta, MemLoc, OverlayFramePacket, PixelFormat, RawFramePacket, MemRef, ColorSpace};
use std::env;
use std::thread;
use std::time::Duration;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let device_index: i32 = if args.len() > 1 {
        args[1].parse().unwrap_or(0)
    } else {
        0
    };

    println!("DeckLink Internal Keying Test");
    println!("==============================");
    println!("Device index: {}", device_index);
    println!();

    // Configuration: 1080p60
    let width = 1920;
    let height = 1080;
    let fps_num = 60;
    let fps_den = 1;

    println!("Creating output stage for {}x{} @ {}/{} fps", width, height, fps_num, fps_den);

    // Create and open output
    let mut output = InternalKeyingOutput::new(device_index, width, height, fps_num, fps_den);
    output.open()?;

    println!("Output device opened successfully");
    println!("Generating test frames...");

    // Allocate buffers for fill and key frames
    let fill_stride = width * 2; // YUV422: 2 bytes per pixel
    let mut fill_buffer = vec![0u8; (fill_stride * height) as usize];

    let key_stride = width * 4; // BGRA: 4 bytes per pixel
    let mut key_buffer = vec![0u8; (key_stride * height) as usize];

    // Generate fill frame (YUV color bars)
    generate_yuv_color_bars(&mut fill_buffer, width, height, fill_stride);

    // Start playback
    println!("Starting playback...");
    output.start_playback()?;

    // Animate for a few seconds
    let total_frames = fps_num * 3; // 3 seconds
    let frame_duration = Duration::from_secs_f64(1.0 / fps_num as f64);

    for frame_num in 0..total_frames {
        // Generate key frame with moving box
        generate_bgra_moving_box(&mut key_buffer, width, height, key_stride, frame_num, fps_num);

        // Create packets
        let fill_packet = RawFramePacket {
            meta: FrameMeta {
                source_id: 0,
                width: width as u32,
                height: height as u32,
                pixfmt: PixelFormat::YUV422_8,
                colorspace: ColorSpace::BT709,
                stride_bytes: fill_stride as u32,
                frame_idx: frame_num as u64,
                pts_ns: (frame_num as u64 * 1_000_000_000) / fps_num as u64,
                t_capture_ns: 0,
            },
            data: MemRef {
                ptr: fill_buffer.as_mut_ptr(),
                len: fill_buffer.len(),
                stride: fill_stride as usize,
                loc: MemLoc::Cpu,
            },
        };

        let key_packet = OverlayFramePacket {
            from: FrameMeta {
                source_id: 0,
                width: width as u32,
                height: height as u32,
                pixfmt: PixelFormat::BGRA8,
                colorspace: ColorSpace::BT709,
                stride_bytes: key_stride as u32,
                frame_idx: frame_num as u64,
                pts_ns: (frame_num as u64 * 1_000_000_000) / fps_num as u64,
                t_capture_ns: 0,
            },
            argb: MemRef {
                ptr: key_buffer.as_mut_ptr(),
                len: key_buffer.len(),
                stride: key_stride as usize,
                loc: MemLoc::Cpu,
            },
            stride: key_stride as usize,
        };

        // Submit frames
        output.submit_keying_frames(&fill_packet, &key_packet)?;

        // Print progress
        if frame_num % fps_num == 0 {
            let buffered = output.buffered_frame_count();
            println!("Frame {}/{} (buffered: {})", frame_num, total_frames, buffered);
        }

        // Wait for next frame
        thread::sleep(frame_duration);
    }

    println!("Stopping playback...");
    output.stop_playback()?;

    println!("Test completed successfully!");
    println!();
    println!("If you saw a moving colored box over color bars on your DeckLink output,");
    println!("the internal keying is working correctly!");

    Ok(())
}

/// Generate YUV422 color bars
fn generate_yuv_color_bars(buffer: &mut [u8], width: i32, height: i32, stride: i32) {
    // Standard SMPTE color bars (8 colors)
    // Colors in YUV (Y, U, V): White, Yellow, Cyan, Green, Magenta, Red, Blue, Black
    let colors_yuv: [(u8, u8, u8); 8] = [
        (235, 128, 128), // White
        (210, 16, 146),  // Yellow
        (170, 166, 16),  // Cyan
        (145, 54, 34),   // Green
        (106, 202, 222), // Magenta
        (81, 90, 240),   // Red
        (41, 240, 110),  // Blue
        (16, 128, 128),  // Black
    ];

    let bar_width = width / 8;

    for y in 0..height {
        let row_offset = (y * stride) as usize;
        
        for x in 0..width {
            let bar_index = (x / bar_width).min(7) as usize;
            let (y_val, u_val, v_val) = colors_yuv[bar_index];
            
            // YUV422 format: UYVY (U Y V Y)
            let pixel_pair = (x / 2) as usize;
            let base_offset = row_offset + pixel_pair * 4;
            
            if x % 2 == 0 {
                // First pixel of pair
                buffer[base_offset + 0] = u_val; // U
                buffer[base_offset + 1] = y_val; // Y0
            } else {
                // Second pixel of pair
                buffer[base_offset + 2] = v_val; // V
                buffer[base_offset + 3] = y_val; // Y1
            }
        }
    }
}

/// Generate BGRA frame with moving box and alpha channel
fn generate_bgra_moving_box(buffer: &mut [u8], width: i32, height: i32, stride: i32, 
                             frame_num: i32, fps: i32) {
    // Clear to transparent
    for pixel in buffer.chunks_exact_mut(4) {
        pixel[0] = 0;   // B
        pixel[1] = 0;   // G
        pixel[2] = 0;   // R
        pixel[3] = 0;   // A (transparent)
    }

    // Calculate box position (moving horizontally)
    let box_size = 400;
    let margin = 100;
    let travel_distance = width - box_size - 2 * margin;
    let cycle_frames = fps * 2; // 2 second cycle
    
    let progress = (frame_num % cycle_frames) as f32 / cycle_frames as f32;
    let box_x = margin + (progress * travel_distance as f32) as i32;
    let box_y = (height - box_size) / 2;

    // Draw semi-transparent box with colored border
    for y in 0..height {
        let row_offset = (y * stride) as usize;
        
        for x in 0..width {
            if x >= box_x && x < box_x + box_size && y >= box_y && y < box_y + box_size {
                let pixel_offset = row_offset + (x * 4) as usize;
                
                // Distance from edge
                let dist_from_edge = ((x - box_x).min(box_x + box_size - x - 1)
                                     .min(y - box_y)
                                     .min(box_y + box_size - y - 1)) as f32;
                
                if dist_from_edge < 10.0 {
                    // Border: bright color with full opacity
                    buffer[pixel_offset + 0] = 255; // B
                    buffer[pixel_offset + 1] = 200; // G
                    buffer[pixel_offset + 2] = 0;   // R
                    buffer[pixel_offset + 3] = 255; // A (opaque)
                } else {
                    // Interior: semi-transparent color
                    let fade = ((dist_from_edge - 10.0) / 50.0).min(1.0);
                    buffer[pixel_offset + 0] = 100; // B
                    buffer[pixel_offset + 1] = 150; // G
                    buffer[pixel_offset + 2] = 255; // R
                    buffer[pixel_offset + 3] = (200.0 * fade) as u8; // A (semi-transparent)
                }
            }
        }
    }
}
