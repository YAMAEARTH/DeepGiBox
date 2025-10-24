use std::thread;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use common_io::{MemLoc, MemRef, PixelFormat, RawFramePacket};
use decklink_input::capture::CaptureSession;
use decklink_input::copy_frame_region_to_host;
use decklink_output::{self, OutputFrame, OutputRequest, VideoFrame};

fn main() -> Result<()> {
    println!("=== DeckLink Output playground (capture passthrough) ===");
    println!("กด Ctrl+C เพื่อหยุดโปรแกรม\n");

    let mut output = decklink_output::from_path("configs/dev_1080p60_yuv422_fp16_trt.toml")?;
    let mut session = CaptureSession::open(0)?;
    println!("✓ เปิด DeckLink capture device #0 สำเร็จ\n");

    let mut submitted = 0usize;

    loop {
        match session.get_frame()? {
            Some(packet) => {
                let frame_idx = packet.meta.frame_idx;
                let pts = packet.meta.pts_ns;

                let (cpu_packet, backing) = ensure_cpu_packet(packet)
                    .with_context(|| format!("prepare frame #{frame_idx} for output"))?;

                // Submit frame, skip if size doesn't match (e.g. during resolution change)
                match output.submit(OutputRequest {
                    video: Some(&cpu_packet),
                    overlay: None,
                }) {
                    Ok(_) => {
                        submitted += 1;
                        if submitted == 1 || submitted % 30 == 0 {
                            if let Some(out) = output.last_frame() {
                                print_output_summary(out, submitted, frame_idx, pts);
                            }
                        }
                    }
                    Err(e) => {
                        // Skip frames with wrong size (happens during format change)
                        if submitted == 0 {
                            println!("⏳ Waiting for correct frame size... ({})", e);
                        }
                    }
                }

                drop(backing);
            }
            None => {
                thread::sleep(Duration::from_millis(5));
            }
        }
    }
}

fn ensure_cpu_packet(mut packet: RawFramePacket) -> Result<(RawFramePacket, Option<Vec<u8>>)> {
    match packet.data.loc {
        MemLoc::Cpu => {
            packet.meta.stride_bytes = packet.data.stride as u32;
            Ok((packet, None))
        }
        MemLoc::Gpu { .. } => {
            let height = packet.meta.height as usize;
            let total_bytes = packet
                .data
                .stride
                .checked_mul(height)
                .ok_or_else(|| anyhow!("frame size overflow"))?;

            let mut buf = vec![0u8; total_bytes];
            copy_frame_region_to_host(&packet, 0, &mut buf).context("copy GPU frame to host")?;

            packet.data = MemRef {
                ptr: buf.as_mut_ptr(),
                len: buf.len(),
                stride: packet.data.stride,
                loc: MemLoc::Cpu,
            };
            packet.meta.stride_bytes = packet.data.stride as u32;
            Ok((packet, Some(buf)))
        }
    }
}

fn print_output_summary(
    frame: &OutputFrame,
    submitted: usize,
    frame_idx: u64,
    pts_ns: u64,
) {
    println!("---");
    println!(
        "#{} ส่งต่อเฟรม decklink_input idx={} pts={} ns",
        submitted, frame_idx, pts_ns
    );
    if let Some(video) = &frame.video {
        print_video_frame(video);
    } else {
        println!("(ไม่มีเฟรมวิดีโอใน OutputFrame)");
    }
    if frame.overlay.is_some() {
        println!("(พบ overlay แต่ playground นี้ยังไม่พิมพ์รายละเอียด)");
    }
}

fn print_video_frame(frame: &VideoFrame) {
    println!(
        "วิดีโอ: {}x{} {:?} stride={} bytes",
        frame.meta.width, frame.meta.height, frame.meta.pixfmt, frame.stride
    );
    println!(
        "source_id={} frame_idx={} t_capture_ns={}",
        frame.meta.source_id, frame.meta.frame_idx, frame.meta.t_capture_ns
    );

    if matches!(frame.meta.pixfmt, PixelFormat::BGRA8)
        && frame.stride >= frame.meta.width as usize * 4
    {
        let width = frame.meta.width as usize;
        let row_bytes = frame.stride;
        let sample_row = (frame.meta.height as usize) / 2;
        let base = sample_row * row_bytes;
        let row = &frame.bytes[base..base + width * 4];
        println!("ตัวอย่างแถวกลาง (BGRA):");
        for (i, px) in row.chunks_exact(4).enumerate().take(4) {
            println!(
                "  x={} -> B{:3} G{:3} R{:3} A{:3}",
                i, px[0], px[1], px[2], px[3]
            );
        }
    }
}
