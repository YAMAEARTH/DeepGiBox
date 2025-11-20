/// Test class label text rendering for Hyperplastic and Neoplastic
use overlay_plan::PlanStage;
use common_io::{DetectionsPacket, Detection, BBox, FrameMeta, Stage, DrawOp, PixelFormat, ColorSpace};

fn create_meta(w: u32, h: u32) -> FrameMeta {
    FrameMeta {
        source_id: 0, width: w, height: h,
        pixfmt: PixelFormat::YUV422_8, colorspace: ColorSpace::BT709,
        frame_idx: 0, pts_ns: 0, t_capture_ns: 0,
        stride_bytes: w * 2, crop_region: None,
    }
}

#[test]
fn test_hyperplastic_label() {
    println!("\n=== TEST: Hyperplastic Label (Class 0) ===");
    let mut stage = PlanStage {
        enable_full_ui: true, spk: true,
        display_mode: "COLON".to_string(),
        confidence_threshold: 0.25,
        endoscope_mode: "OLYMPUS".to_string(),
    };
    let detection = Detection {
        bbox: BBox { x: 100.0, y: 100.0, w: 200.0, h: 150.0 },
        score: 0.85, class_id: 0, track_id: None,
    };
    let packet = DetectionsPacket {
        from: create_meta(1920, 1080),
        items: vec![detection],
    };
    let result = stage.process(packet);
    let labels: Vec<_> = result.ops.iter().filter_map(|op| {
        if let DrawOp::Label { text, color, .. } = op {
            Some((text.clone(), *color))
        } else { None }
    }).collect();
    println!("Labels: {:?}", labels.iter().map(|(t, _)| t).collect::<Vec<_>>());
    assert!(labels.iter().any(|(t, _)| t == "Hyperplastic"), "Missing Hyperplastic!");
    if let Some((_, c)) = labels.iter().find(|(t, _)| t == "Hyperplastic") {
        assert_eq!((c.0, c.1, c.2), (255, 255, 0), "Wrong color!");
        println!("✅ Hyperplastic: Yellow (255,255,0)");
    }
    println!("✅ TEST PASSED\n");
}

#[test]
fn test_neoplastic_label() {
    println!("\n=== TEST: Neoplastic Label (Class 1) ===");
    let mut stage = PlanStage {
        enable_full_ui: true, spk: true,
        display_mode: "EGD".to_string(),
        confidence_threshold: 0.30,
        endoscope_mode: "FUJI".to_string(),
    };
    let detection = Detection {
        bbox: BBox { x: 500.0, y: 400.0, w: 300.0, h: 250.0 },
        score: 0.92, class_id: 1, track_id: None,
    };
    let packet = DetectionsPacket {
        from: create_meta(3840, 2160),
        items: vec![detection],
    };
    let result = stage.process(packet);
    let labels: Vec<_> = result.ops.iter().filter_map(|op| {
        if let DrawOp::Label { text, color, .. } = op {
            Some((text.clone(), *color))
        } else { None }
    }).collect();
    println!("Labels: {:?}", labels.iter().map(|(t, _)| t).collect::<Vec<_>>());
    assert!(labels.iter().any(|(t, _)| t == "Neoplastic"), "Missing Neoplastic!");
    if let Some((_, c)) = labels.iter().find(|(t, _)| t == "Neoplastic") {
        assert_eq!((c.0, c.1, c.2), (255, 255, 0), "Wrong color!");
        println!("✅ Neoplastic: Yellow (255,255,0)");
    }
    println!("✅ TEST PASSED\n");
}

#[test]
fn test_best_score_selection() {
    println!("\n=== TEST: Multiple Detections - Best Score ===");
    let mut stage = PlanStage {
        enable_full_ui: true, spk: false,
        display_mode: "COLON".to_string(),
        confidence_threshold: 0.25,
        endoscope_mode: "OLYMPUS".to_string(),
    };
    let detections = vec![
        Detection { bbox: BBox { x: 100.0, y: 100.0, w: 150.0, h: 150.0 },
            score: 0.65, class_id: 0, track_id: None },
        Detection { bbox: BBox { x: 500.0, y: 400.0, w: 200.0, h: 180.0 },
            score: 0.88, class_id: 1, track_id: None },  // Best
        Detection { bbox: BBox { x: 800.0, y: 300.0, w: 180.0, h: 160.0 },
            score: 0.72, class_id: 0, track_id: None },
    ];
    let packet = DetectionsPacket {
        from: create_meta(1920, 1080),
        items: detections,
    };
    let result = stage.process(packet);
    let class_labels: Vec<_> = result.ops.iter().filter_map(|op| {
        if let DrawOp::Label { text, .. } = op {
            if text == "Hyperplastic" || text == "Neoplastic" || text == "Awaiting" {
                Some(text.clone())
            } else { None }
        } else { None }
    }).collect();
    println!("Class labels: {:?}", class_labels);
    assert!(class_labels.contains(&"Neoplastic".to_string()), "Should show Neoplastic!");
    assert_eq!(class_labels.iter().filter(|&s| s == "Neoplastic").count(), 1, "Should have 1 label");
    println!("✅ TEST PASSED: Neoplastic (score 0.88)\n");
}

#[test]
fn test_unknown_class() {
    println!("\n=== TEST: Unknown Class ===");
    let mut stage = PlanStage {
        enable_full_ui: true, spk: true,
        display_mode: "EGD".to_string(),
        confidence_threshold: 0.25,
        endoscope_mode: "OLYMPUS".to_string(),
    };
    let detection = Detection {
        bbox: BBox { x: 300.0, y: 300.0, w: 200.0, h: 200.0 },
        score: 0.75, class_id: 99, track_id: None,
    };
    let packet = DetectionsPacket {
        from: create_meta(1920, 1080),
        items: vec![detection],
    };
    let result = stage.process(packet);
    let class_labels: Vec<_> = result.ops.iter().filter_map(|op| {
        if let DrawOp::Label { text, color, .. } = op {
            if text == "Hyperplastic" || text == "Neoplastic" || text == "Awaiting" {
                Some((text.clone(), *color))
            } else { None }
        } else { None }
    }).collect();
    println!("Labels: {:?}", class_labels.iter().map(|(t, _)| t).collect::<Vec<_>>());
    assert!(class_labels.iter().any(|(t, _)| t == "Awaiting"), "Should show Awaiting!");
    if let Some((_, c)) = class_labels.iter().find(|(t, _)| t == "Awaiting") {
        assert_eq!((c.0, c.1, c.2), (255, 255, 0), "Wrong color!");
        println!("✅ Awaiting: Yellow (255,255,0)");
    }
    println!("✅ TEST PASSED\n");
}

#[test]
fn test_no_detection() {
    println!("\n=== TEST: No Detection ===");
    let mut stage = PlanStage {
        enable_full_ui: true, spk: true,
        display_mode: "COLON".to_string(),
        confidence_threshold: 0.25,
        endoscope_mode: "FUJI".to_string(),
    };
    let packet = DetectionsPacket {
        from: create_meta(1920, 1080),
        items: vec![],
    };
    let result = stage.process(packet);
    let class_labels: Vec<_> = result.ops.iter().filter_map(|op| {
        if let DrawOp::Label { text, .. } = op {
            if text == "Hyperplastic" || text == "Neoplastic" || text == "Awaiting" {
                Some(text.clone())
            } else { None }
        } else { None }
    }).collect();
    println!("Labels: {:?}", class_labels);
    assert!(class_labels.iter().any(|t| t == "Awaiting"), "Should show Awaiting!");
    println!("✅ TEST PASSED: Awaiting when no detection\n");
}
