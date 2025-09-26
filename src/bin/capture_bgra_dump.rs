use std::error::Error;
use std::thread;
use std::time::Duration;

use decklink_rust::capture::CaptureSession;

fn main() -> Result<(), Box<dyn Error>> {
    let mut session = CaptureSession::open(0)?;
    println!("Opened capture session on device 0");

    for attempt in 0..60 {
        if let Some(frame) = session.get_frame()? {
            println!(
                "Frame seq={} size={}x{} row_bytes={} buffer_len={} bytes",
                frame.seq, frame.width, frame.height, frame.row_bytes, frame.data_len
            );

            if frame.data_len >= 4 {
                let sample_count = usize::min(5, frame.data_len / 4);
                // let raw = unsafe { std::slice::from_raw_parts(frame.data_ptr, sample_count * 4) };
                // for (idx, chunk) in raw.chunks(4).enumerate() {
                //     let b = chunk[0];
                //     let g = chunk[1];
                //     let r = chunk[2];
                //     let a = chunk[3];
                //     println!("pixel[{idx}] = B:{b:3} G:{g:3} R:{r:3} A:{a:3}");
                let stride = frame.row_bytes as usize;
                let width = frame.width as usize;
                let height = frame.height as usize;

                let mid_y = if height > 0 { height / 2 } else { 0 };
                let mid_x = if width > 0 { width / 2 } else { 0 };
                let start_x = if width > sample_count {
                    let half = sample_count / 2;
                    let min_x = mid_x.saturating_sub(half);
                    let max_start = width - sample_count;
                    std::cmp::min(min_x, max_start)
                } else {
                    0
                };
                let offset = mid_y.saturating_mul(stride).saturating_add(start_x.saturating_mul(4));
                let max_bytes = std::cmp::min(frame.data_len, offset.saturating_add(sample_count * 4));
                let available = max_bytes.saturating_sub(offset);
                let actual_pixels = available / 4;
                if actual_pixels > 0 {
                    let raw = unsafe {
                        std::slice::from_raw_parts(frame.data_ptr.add(offset), actual_pixels * 4)
                    };
                    for (i, chunk) in raw.chunks(4).enumerate() {
                        let x = start_x + i;
                        let y = mid_y;
                        let b = chunk[0];
                        let g = chunk[1];
                        let r = chunk[2];
                        let a = chunk[3];
                        println!("pixel[y={y} x={x}] = B:{b:3} G:{g:3} R:{r:3} A:{a:3}");
                    }
                } else {
                    println!("No center pixels available within buffer bounds");
                }
            } else {
                println!("Frame buffer shorter than one BGRA pixel");
            }

            // session.close();
            // return Ok(());
        }
        println!("No frame yet (attempt {attempt})");
        thread::sleep(Duration::from_millis(100));
    }

    Err("Timed out waiting for a frame".into())
}
