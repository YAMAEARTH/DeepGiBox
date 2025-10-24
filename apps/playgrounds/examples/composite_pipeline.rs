/// Example: Using DeckLink Output with Pipeline Overlay
/// 
/// This example shows how to composite dynamic overlay graphics
/// from the DeepGiBox pipeline (e.g., bounding boxes, labels)
/// over captured DeckLink video frames.

use anyhow::Result;
use decklink_output::CompositorBuilder;
use decklink_input::CaptureSession;
use common_io::{MemLoc, OverlayFramePacket, OverlayPlanPacket};
use overlay_render::RenderStage;
use std::time::Instant;

fn main() -> Result<()> {
    println!("DeckLink Output - Pipeline Overlay Example");
    println!("==========================================\n");

    // Initialize DeckLink capture
    println!("Initializing DeckLink capture...");
    let mut capture = CaptureSession::new(0)?;
    println!("✓ DeckLink capture ready\n");

    // Initialize overlay renderer
    println!("Initializing overlay renderer...");
    let mut overlay_render = RenderStage::new()?;
    println!("✓ Overlay renderer ready\n");

    // Create compositor for pipeline overlay
    println!("Creating compositor for pipeline overlay...");
    let mut compositor = CompositorBuilder::new(1920, 1080)
        .with_pipeline()
        .build()?;
    println!("✓ Compositor ready (pipeline mode)\n");

    println!("Starting compositing loop (Press Ctrl+C to exit)...\n");

    let mut frame_count = 0;
    let start_time = Instant::now();
    let mut composite_times = Vec::new();
    let mut overlay_times = Vec::new();

    loop {
        // 1. Capture video frame from DeckLink
        let video_frame = match capture.next_frame() {
            Ok(frame) => frame,
            Err(e) => {
                eprintln!("Capture error: {}", e);
                continue;
            }
        };

        // 2. Generate overlay plan (from detections, etc.)
        // In real pipeline, this comes from inference + postprocess
        let overlay_plan = create_sample_overlay_plan(&video_frame.meta);

        // 3. Render overlay to ARGB
        let overlay_start = Instant::now();
        let overlay_frame: OverlayFramePacket = overlay_render.process(overlay_plan);
        let overlay_time = overlay_start.elapsed();
        overlay_times.push(overlay_time.as_micros() as f64 / 1000.0);

        // 4. Update compositor with new overlay
        compositor.update_overlay(&overlay_frame)?;

        // 5. Composite overlay onto video
        let comp_start = Instant::now();
        let composited = compositor.composite(&video_frame)?;
        let comp_time = comp_start.elapsed();
        composite_times.push(comp_time.as_micros() as f64 / 1000.0);

        frame_count += 1;

        // Print stats every second
        if frame_count % 60 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let fps = frame_count as f64 / elapsed;
            let avg_overlay = overlay_times.iter().sum::<f64>() / overlay_times.len() as f64;
            let avg_comp = composite_times.iter().sum::<f64>() / composite_times.len() as f64;

            println!(
                "Frame {}: {:.1} fps | Overlay: {:.2}ms | Composite: {:.2}ms | Total: {:.2}ms",
                frame_count,
                fps,
                avg_overlay,
                avg_comp,
                avg_overlay + avg_comp,
            );

            overlay_times.clear();
            composite_times.clear();
        }

        // Optional: Save first frame
        if frame_count == 1 {
            println!("\nSaving first frame as debug_pipeline.png...");
            let bgra_data = compositor.download_output()?;
            save_bgra_as_png(&bgra_data, compositor.dimensions(), "debug_pipeline.png")?;
            println!("✓ Saved debug_pipeline.png\n");
        }

        // Send to output...
    }
}

/// Create sample overlay plan (for demo purposes)
/// In real pipeline, this comes from detection postprocessing
fn create_sample_overlay_plan(meta: &common_io::FrameMeta) -> OverlayPlanPacket {
    use common_io::DrawOp;

    OverlayPlanPacket {
        from: meta.clone(),
        ops: vec![
            // Bounding box
            DrawOp::Rect {
                xywh: (100.0, 100.0, 300.0, 200.0),
                thickness: 3,
            },
            // Label
            DrawOp::Label {
                anchor: (100.0, 80.0),
                text: "Person (0.95)".to_string(),
                font_px: 24,
            },
            // Another box
            DrawOp::Rect {
                xywh: (500.0, 300.0, 250.0, 180.0),
                thickness: 3,
            },
            DrawOp::Label {
                anchor: (500.0, 280.0),
                text: "Car (0.88)".to_string(),
                font_px: 24,
            },
        ],
        canvas: (meta.width, meta.height),
    }
}

fn save_bgra_as_png(
    bgra_data: &[u8],
    (width, height): (u32, u32),
    path: &str,
) -> Result<()> {
    use image::{ImageBuffer, RgbaImage};

    // Convert BGRA to RGBA
    let mut rgba_data = Vec::with_capacity(bgra_data.len());
    for chunk in bgra_data.chunks_exact(4) {
        rgba_data.push(chunk[2]); // R
        rgba_data.push(chunk[1]); // G
        rgba_data.push(chunk[0]); // B
        rgba_data.push(chunk[3]); // A
    }

    let img: RgbaImage = ImageBuffer::from_raw(width, height, rgba_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;

    img.save(path)?;
    Ok(())
}
