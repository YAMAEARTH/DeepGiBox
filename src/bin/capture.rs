use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use decklink_rust::capture::CaptureSession;

fn main() -> Result<()> {
    let session = CaptureSession::open(0)?;
    println!(
        "DeckLink capture started on device {}",
        session.device_index()
    );
    println!("Press Ctrl+C to stop.");

    let mut last_seq = 0u64;
    let mut last_log = Instant::now();

    loop {
        if let Some(frame) = session.get_frame() {
            if frame.seq != 0 && frame.seq != last_seq {
                last_seq = frame.seq;
                if last_log.elapsed() >= Duration::from_secs(1) {
                    let ptr = frame.data as usize;
                    println!(
                        "seq={}, size={}x{}, row_bytes={}, ptr=0x{ptr:x}",
                        frame.seq, frame.width, frame.height, frame.row_bytes
                    );
                    last_log = Instant::now();
                }
            }
        }

        thread::sleep(Duration::from_millis(16));
    }
}
