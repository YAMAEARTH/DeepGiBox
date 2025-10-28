// test_overlay_render.rs - Generate overlay plan and render to PNG for inspection
use anyhow::Result;
use common_io::{BBox, ColorSpace, Detection, DetectionsPacket, FrameMeta, PixelFormat, Stage};
use image::RgbaImage;
use overlay_plan;
use overlay_render;
use std::path::Path;

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║           OVERLAY RENDER VISUAL CHECK (MOCK DATA)        ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    std::fs::create_dir_all("output")?;

    let mut plan_stage = overlay_plan::from_path("full_ui")?;
    let mut render_stage = overlay_render::from_path("")?;

    let detections_packet = sample_packet();
    print_detections(&detections_packet);

    let overlay_plan = plan_stage.process(detections_packet);
    println!(
        "\n🎨 Overlay plan ready: canvas={}x{}, ops={}",
        overlay_plan.canvas.0,
        overlay_plan.canvas.1,
        overlay_plan.ops.len()
    );

    let overlay_frame = render_stage.process(overlay_plan);
    let output_path = Path::new("output/overlay_render_test.png");
    save_argb_buffer_as_png(&overlay_frame, output_path)?;

    println!(
        "🖼️  Overlay rendered and saved to {}\n    (Open the image to verify HUD elements and boxes)",
        output_path.display()
    );

    Ok(())
}

fn sample_packet() -> DetectionsPacket {
    let frame_meta = FrameMeta {
        source_id: 0,
        width: 1920,
        height: 1080,
        pixfmt: PixelFormat::BGRA8,
        colorspace: ColorSpace::BT709,
        frame_idx: 123,
        pts_ns: 0,
        t_capture_ns: 0,
        stride_bytes: 1920 * 4,
    };

    let detections = vec![
        Detection {
            bbox: BBox {
                x: 380.0,
                y: 260.0,
                w: 220.0,
                h: 160.0,
            },
            score: 0.92,
            class_id: 0,
            track_id: Some(11),
        },
        Detection {
            bbox: BBox {
                x: 1050.0,
                y: 220.0,
                w: 260.0,
                h: 210.0,
            },
            score: 0.81,
            class_id: 1,
            track_id: Some(12),
        },
        Detection {
            bbox: BBox {
                x: 1280.0,
                y: 640.0,
                w: 180.0,
                h: 230.0,
            },
            score: 0.69,
            class_id: 1,
            track_id: None,
        },
    ];

    DetectionsPacket {
        from: frame_meta,
        items: detections,
    }
}

fn save_argb_buffer_as_png(
    frame: &common_io::OverlayFramePacket,
    output_path: &Path,
) -> Result<()> {
    let width = frame.from.width as usize;
    let height = frame.from.height as usize;
    let stride = frame.stride;

    // Take ownership of the ARGB buffer that overlay_render leaked for pipeline compatibility.
    let argb = unsafe { Vec::from_raw_parts(frame.argb.ptr, frame.argb.len, frame.argb.len) };

    let mut rgba = vec![0u8; width * height * 4];
    for y in 0..height {
        let src_row = &argb[y * stride..y * stride + width * 4];
        let dst_row = &mut rgba[y * width * 4..(y + 1) * width * 4];

        for (dst, src) in dst_row.chunks_exact_mut(4).zip(src_row.chunks_exact(4)) {
            let (a, r, g, b) = (src[0], src[1], src[2], src[3]);
            dst.copy_from_slice(&[r, g, b, a]);
        }
    }

    let image = RgbaImage::from_raw(frame.from.width, frame.from.height, rgba)
        .expect("buffer has correct dimensions");
    image.save(output_path)?;
    Ok(())
}

fn print_detections(packet: &DetectionsPacket) {
    println!(
        "📦 Frame {} | {} detections",
        packet.from.frame_idx,
        packet.items.len()
    );
    for (idx, det) in packet.items.iter().enumerate() {
        let class_name = match det.class_id {
            0 => "Hyper",
            1 => "Neo",
            _ => "Unknown",
        };
        let track_info = det
            .track_id
            .map(|t| format!("ID:{}", t))
            .unwrap_or_else(|| "No track".into());
        println!(
            "  [{}] {:<6} score={:.2} bbox=({:.0},{:.0},{:.0},{:.0}) {}",
            idx + 1,
            class_name,
            det.score,
            det.bbox.x,
            det.bbox.y,
            det.bbox.w,
            det.bbox.h,
            track_info
        );
    }
}
