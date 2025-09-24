use std::error::Error;
use std::thread;
use std::time::Duration;

use decklink_rust::capture::CaptureSession;

fn main() -> Result<(), Box<dyn Error>> {
    let mut session = CaptureSession::open(0)?;
    println!("Opened capture session on device 0");

    for attempt in 0..5 {
        match session.get_frame()? {
            Some(frame) => {
                println!(
                    "Frame seq={} size={}x{} stride={} bytes buffer_ptr={:?} buffer_len={} bytes",
                    frame.seq,
                    frame.width,
                    frame.height,
                    frame.row_bytes,
                    frame.data_ptr,
                    frame.data_len,
                );
            }
            None => println!("No frame available yet (attempt {attempt})"),
        }
        thread::sleep(Duration::from_millis(200));
    }

    session.close();
    Ok(())
}
