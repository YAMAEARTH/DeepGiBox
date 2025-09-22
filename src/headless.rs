use std::time::{Duration, Instant};
use std::collections::HashMap;
use anyhow::Result;

use crate::packets::{
    RawFramePacket, TensorInputPacket, RawDetectionsPacket, DetectionsPacket, 
    OverlayPlanPacket, KeyingPacket, PipelineError, PipelineStage,
    TensorDesc, TensorMem, TensorDataType, TensorLayout, ColorFormat,
    RawDetection, Detection, OverlayOp, OverlayFrame
};
use crate::capture::{CaptureStage, CaptureConfig};
use crate::ColorSpace;

/// Stage performance metrics following DeepGI standards
#[derive(Debug, Clone, Default)]
pub struct StageMetrics {
    pub frames_processed: u64,
    pub frames_failed: u64,
    pub total_latency_ns: u64,
    pub min_latency_ns: u64,
    pub max_latency_ns: u64,
    pub last_latency_ns: u64,
    pub throughput_fps: f64,
}

impl StageMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn average_latency_ms(&self) -> f64 {
        if self.frames_processed > 0 {
            (self.total_latency_ns as f64 / self.frames_processed as f64) / 1_000_000.0
        } else {
            0.0
        }
    }

    pub fn last_latency_ms(&self) -> f64 {
        self.last_latency_ns as f64 / 1_000_000.0
    }

    pub fn min_latency_ms(&self) -> f64 {
        if self.min_latency_ns > 0 {
            self.min_latency_ns as f64 / 1_000_000.0
        } else {
            0.0
        }
    }

    pub fn max_latency_ms(&self) -> f64 {
        self.max_latency_ns as f64 / 1_000_000.0
    }

    pub fn success_rate(&self) -> f64 {
        let total = self.frames_processed + self.frames_failed;
        if total > 0 {
            (self.frames_processed as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    }

    pub fn update(&mut self, latency_ns: u64, success: bool) {
        if success {
            self.frames_processed += 1;
            self.total_latency_ns += latency_ns;
            self.last_latency_ns = latency_ns;
            
            if self.frames_processed == 1 {
                self.min_latency_ns = latency_ns;
                self.max_latency_ns = latency_ns;
            } else {
                self.min_latency_ns = self.min_latency_ns.min(latency_ns);
                self.max_latency_ns = self.max_latency_ns.max(latency_ns);
            }
        } else {
            self.frames_failed += 1;
        }
    }

    pub fn update_throughput(&mut self, elapsed_seconds: f64) {
        if elapsed_seconds > 0.0 {
            self.throughput_fps = self.frames_processed as f64 / elapsed_seconds;
        }
    }
}

/// Preprocessing stage - converts RawFramePacket to TensorInputPacket
pub struct PreprocessingStage {
    target_width: u32,
    target_height: u32,
    normalize: bool,
}

impl PreprocessingStage {
    pub fn new(width: u32, height: u32, normalize: bool) -> Self {
        Self {
            target_width: width,
            target_height: height,
            normalize,
        }
    }
}

impl PipelineStage<RawFramePacket, TensorInputPacket> for PreprocessingStage {
    type Error = PipelineError;

    fn process(&mut self, input: RawFramePacket) -> Result<TensorInputPacket, Self::Error> {
        // Simulate preprocessing operations
        std::thread::sleep(Duration::from_millis(2)); // Color space conversion
        std::thread::sleep(Duration::from_millis(1)); // Resize/crop
        if self.normalize {
            std::thread::sleep(Duration::from_micros(500)); // Normalization
        }

        // Create tensor description
        let tensor_desc = TensorDesc {
            shape: [1, 3, self.target_height, self.target_width],
            dtype: TensorDataType::F32,
            layout: TensorLayout::NCHW,
            colors: ColorFormat::RGB,
        };

        // Simulate GPU tensor creation
        let tensor_packet = TensorInputPacket {
            mem: TensorMem::Cuda { device_ptr: 0x12345678 }, // Mock GPU pointer
            desc: tensor_desc,
            meta: input.meta,
        };

        Ok(tensor_packet)
    }
}

/// AI Inference stage - processes TensorInputPacket to RawDetectionsPacket
pub struct InferenceStage {
    model_name: String,
    confidence_threshold: f32,
}

impl InferenceStage {
    pub fn new(model_name: String, confidence_threshold: f32) -> Self {
        Self {
            model_name,
            confidence_threshold,
        }
    }
}

impl PipelineStage<TensorInputPacket, RawDetectionsPacket> for InferenceStage {
    type Error = PipelineError;

    fn process(&mut self, input: TensorInputPacket) -> Result<RawDetectionsPacket, Self::Error> {
        // Simulate AI model inference with varying duration
        let inference_duration = Duration::from_millis(15 + (input.meta.seq_no % 10));
        std::thread::sleep(inference_duration);

        // Simulate occasional inference failures
        if input.meta.seq_no % 200 == 0 {
            return Err(PipelineError::Processing(format!(
                "AI model {} inference timeout", self.model_name
            )));
        }

        // Generate mock detection results
        let mut detections = Vec::new();
        
        // Simulate 0-5 detections per frame
        let num_detections = (input.meta.seq_no % 6) as usize;
        for i in 0..num_detections {
            let det = RawDetection {
                cx: 320.0 + (i as f32 * 100.0),
                cy: 240.0 + (i as f32 * 50.0),
                w: 80.0 + (i as f32 * 20.0),
                h: 120.0 + (i as f32 * 30.0),
                obj_conf: 0.7 + (i as f32 * 0.05),
                class_conf: 0.8 + (i as f32 * 0.02),
                class_id: i as i32,
            };
            
            if det.obj_conf >= self.confidence_threshold {
                detections.push(det);
            }
        }

        Ok(RawDetectionsPacket {
            dets: detections,
            meta: input.meta,
        })
    }
}

/// Post-processing stage - applies NMS and filtering to RawDetectionsPacket
pub struct PostProcessingStage {
    nms_threshold: f32,
    max_detections: usize,
}

impl PostProcessingStage {
    pub fn new(nms_threshold: f32, max_detections: usize) -> Self {
        Self {
            nms_threshold,
            max_detections,
        }
    }

    fn apply_nms(&self, detections: &[RawDetection]) -> Vec<Detection> {
        // Simulate NMS processing
        std::thread::sleep(Duration::from_millis(3));
        
        // Convert to processed detections (simplified NMS simulation)
        let mut processed: Vec<Detection> = detections
            .iter()
            .enumerate()
            .map(|(i, det)| Detection {
                bbox: det.to_bbox(),
                score: det.obj_conf * det.class_conf,
                class_id: det.class_id,
                track_id: Some(i as i64 + 1000), // Mock track ID
                label: Some(format!("Object_{}", det.class_id)),
            })
            .collect();

        // Sort by score and take top N
        processed.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        processed.truncate(self.max_detections);

        processed
    }
}

impl PipelineStage<RawDetectionsPacket, DetectionsPacket> for PostProcessingStage {
    type Error = PipelineError;

    fn process(&mut self, input: RawDetectionsPacket) -> Result<DetectionsPacket, Self::Error> {
        // Apply NMS and filtering
        let processed_detections = self.apply_nms(&input.dets);
        
        // Simulate confidence filtering
        std::thread::sleep(Duration::from_millis(1));
        
        // Simulate result validation
        std::thread::sleep(Duration::from_micros(500));

        Ok(DetectionsPacket {
            dets: processed_detections,
            meta: input.meta,
        })
    }
}

/// Object tracking stage - adds temporal consistency to detections
pub struct ObjectTrackingStage {
    next_track_id: i64,
    track_memory: HashMap<i64, Detection>,
}

impl ObjectTrackingStage {
    pub fn new() -> Self {
        Self {
            next_track_id: 1,
            track_memory: HashMap::new(),
        }
    }
}

impl PipelineStage<DetectionsPacket, DetectionsPacket> for ObjectTrackingStage {
    type Error = PipelineError;

    fn process(&mut self, mut input: DetectionsPacket) -> Result<DetectionsPacket, Self::Error> {
        // Simulate tracking operations
        std::thread::sleep(Duration::from_millis(2)); // Kalman filter prediction
        std::thread::sleep(Duration::from_millis(3)); // Data association
        std::thread::sleep(Duration::from_millis(1)); // Track state update

        // Simulate tracking failures for complex scenes
        if input.meta.seq_no % 300 == 0 {
            return Err(PipelineError::Processing(
                "Tracking lost for complex scene".to_string()
            ));
        }

        // Simple tracking simulation - assign track IDs
        for detection in &mut input.dets {
            if detection.track_id.is_none() {
                detection.track_id = Some(self.next_track_id);
                self.next_track_id += 1;
            }
        }

        Ok(input)
    }
}

/// Overlay planning stage - creates rendering instructions
pub struct OverlayPlanningStage {
    show_labels: bool,
    show_confidence: bool,
}

impl OverlayPlanningStage {
    pub fn new(show_labels: bool, show_confidence: bool) -> Self {
        Self {
            show_labels,
            show_confidence,
        }
    }
}

impl PipelineStage<DetectionsPacket, OverlayPlanPacket> for OverlayPlanningStage {
    type Error = PipelineError;

    fn process(&mut self, input: DetectionsPacket) -> Result<OverlayPlanPacket, Self::Error> {
        // Simulate overlay generation time
        std::thread::sleep(Duration::from_millis(2)); // Bounding box planning
        std::thread::sleep(Duration::from_millis(1)); // Text layout
        
        let mut ops = Vec::new();
        
        for detection in &input.dets {
            // Add bounding box
            ops.push(OverlayOp::Rect {
                bbox: detection.bbox,
                thickness: 2,
                alpha: 0.8,
                color: [0, 255, 0], // Green
            });
            
            if self.show_labels {
                if let Some(ref label) = detection.label {
                    let text = if self.show_confidence {
                        format!("{} {:.2}", label, detection.score)
                    } else {
                        label.clone()
                    };
                    
                    ops.push(OverlayOp::Text {
                        x: detection.bbox.x1 as i32,
                        y: (detection.bbox.y1 - 20.0) as i32,
                        text,
                        font_px: 16,
                        alpha: 0.9,
                        color: [255, 255, 255], // White
                    });
                }
            }
        }
        
        Ok(OverlayPlanPacket {
            size: (input.meta.width, input.meta.height),
            ops,
            meta: input.meta,
        })
    }
}

/// Keying stage - combines original frame with overlay
pub struct KeyingStage {
    enable_chroma_key: bool,
}

impl KeyingStage {
    pub fn new(enable_chroma_key: bool) -> Self {
        Self { enable_chroma_key }
    }
}

impl PipelineStage<(RawFramePacket, OverlayPlanPacket), KeyingPacket> for KeyingStage {
    type Error = PipelineError;

    fn process(&mut self, input: (RawFramePacket, OverlayPlanPacket)) -> Result<KeyingPacket, Self::Error> {
        let (frame, overlay_plan) = input;
        
        // Simulate keying operations based on frame size
        if let Some(data) = frame.as_slice() {
            let pixel_count = data.len() / 4; // BGRA format
            let processing_time_us = pixel_count / 500; // 2μs per 1000 pixels
            std::thread::sleep(Duration::from_micros(processing_time_us as u64));
        }
        
        if self.enable_chroma_key {
            std::thread::sleep(Duration::from_millis(2)); // Chroma key processing
        }
        
        std::thread::sleep(Duration::from_millis(1)); // Alpha blending
        
        // Create mock overlay frame
        let overlay_size = (overlay_plan.size.0 * overlay_plan.size.1 * 4) as usize;
        let overlay_buffer = vec![0u8; overlay_size]; // Transparent overlay
        
        let overlay_frame = OverlayFrame::new_cpu(
            overlay_buffer,
            overlay_plan.size.0,
            overlay_plan.size.1,
            overlay_plan.size.0 * 4,
            false,
        );
        
        Ok(KeyingPacket {
            passthrough: frame,
            overlay: overlay_frame,
            meta: overlay_plan.meta,
        })
    }
}

/// Output stage - handles final encoding and transmission
pub struct OutputStage {
    output_format: String,
    enable_streaming: bool,
}

impl OutputStage {
    pub fn new(output_format: String, enable_streaming: bool) -> Self {
        Self {
            output_format,
            enable_streaming,
        }
    }
}

impl PipelineStage<KeyingPacket, ()> for OutputStage {
    type Error = PipelineError;

    fn process(&mut self, input: KeyingPacket) -> Result<(), Self::Error> {
        // Simulate output operations
        std::thread::sleep(Duration::from_millis(1)); // Format conversion
        std::thread::sleep(Duration::from_millis(3)); // Encoding/compression
        
        if self.enable_streaming {
            std::thread::sleep(Duration::from_micros(500)); // Network transmission prep
        }
        
        // Simulate occasional network issues
        if input.meta.seq_no % 500 == 0 {
            return Err(PipelineError::Processing(
                "Output buffer full - network congestion".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Complete pipeline configuration following DeepGI standards
#[derive(Debug, Clone)]
pub struct HeadlessConfig {
    pub capture_config: CaptureConfig,
    pub preprocessing: Option<PreprocessingConfig>,
    pub inference: Option<InferenceConfig>,
    pub postprocessing: Option<PostProcessingConfig>,
    pub tracking: Option<TrackingConfig>,
    pub overlay: Option<OverlayConfig>,
    pub keying: Option<KeyingConfig>,
    pub output: Option<OutputConfig>,
    pub stats_interval: Duration,
    pub max_runtime: Option<Duration>,
    pub max_frames: Option<u64>,
    pub enable_detailed_logging: bool,
}

#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    pub target_width: u32,
    pub target_height: u32,
    pub normalize: bool,
}

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub model_name: String,
    pub confidence_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct PostProcessingConfig {
    pub nms_threshold: f32,
    pub max_detections: usize,
}

#[derive(Debug, Clone)]
pub struct TrackingConfig {
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct OverlayConfig {
    pub show_labels: bool,
    pub show_confidence: bool,
}

#[derive(Debug, Clone)]
pub struct KeyingConfig {
    pub enable_chroma_key: bool,
}

#[derive(Debug, Clone)]
pub struct OutputConfig {
    pub format: String,
    pub enable_streaming: bool,
}

impl Default for HeadlessConfig {
    fn default() -> Self {
        Self {
            capture_config: CaptureConfig {
                device_index: 0,
                source_id: 100,
                expected_colorspace: ColorSpace::BT709,
            },
            preprocessing: Some(PreprocessingConfig {
                target_width: 640,
                target_height: 480,
                normalize: true,
            }),
            inference: Some(InferenceConfig {
                model_name: "yolov8n".to_string(),
                confidence_threshold: 0.5,
            }),
            postprocessing: Some(PostProcessingConfig {
                nms_threshold: 0.45,
                max_detections: 100,
            }),
            tracking: Some(TrackingConfig { enabled: true }),
            overlay: Some(OverlayConfig {
                show_labels: true,
                show_confidence: true,
            }),
            keying: Some(KeyingConfig {
                enable_chroma_key: false,
            }),
            output: Some(OutputConfig {
                format: "h264".to_string(),
                enable_streaming: false,
            }),
            stats_interval: Duration::from_secs(1),
            max_runtime: Some(Duration::from_secs(30)),
            max_frames: None,
            enable_detailed_logging: false,
        }
    }
}

/// Enhanced headless processor following DeepGI pipeline standards
pub struct HeadlessProcessor {
    config: HeadlessConfig,
    capture_stage: CaptureStage,
    metrics: HashMap<String, StageMetrics>,
    start_time: Instant,
    total_frames: u64,
}

impl HeadlessProcessor {
    pub fn new(config: HeadlessConfig) -> Self {
        let capture_stage = CaptureStage::new(config.capture_config.clone());
        let metrics = HashMap::new();
        
        Self {
            config,
            capture_stage,
            metrics,
            start_time: Instant::now(),
            total_frames: 0,
        }
    }

    /// Run the complete DeepGI pipeline with standard I/O packets
    pub fn run(&mut self) -> Result<()> {
        println!("🚀 Starting DeepGI Standard I/O Pipeline");
        println!("=========================================");
        self.print_configuration();
        
        // Initialize stages based on configuration
        let mut preprocessing_stage = self.config.preprocessing.as_ref()
            .map(|cfg| PreprocessingStage::new(cfg.target_width, cfg.target_height, cfg.normalize));
        
        let mut inference_stage = self.config.inference.as_ref()
            .map(|cfg| InferenceStage::new(cfg.model_name.clone(), cfg.confidence_threshold));
        
        let mut postprocessing_stage = self.config.postprocessing.as_ref()
            .map(|cfg| PostProcessingStage::new(cfg.nms_threshold, cfg.max_detections));
        
        let mut tracking_stage = self.config.tracking.as_ref()
            .filter(|cfg| cfg.enabled)
            .map(|_| ObjectTrackingStage::new());
        
        let mut overlay_stage = self.config.overlay.as_ref()
            .map(|cfg| OverlayPlanningStage::new(cfg.show_labels, cfg.show_confidence));
        
        let mut keying_stage = self.config.keying.as_ref()
            .map(|cfg| KeyingStage::new(cfg.enable_chroma_key));
        
        let mut output_stage = self.config.output.as_ref()
            .map(|cfg| OutputStage::new(cfg.format.clone(), cfg.enable_streaming));

        // Initialize metrics
        if preprocessing_stage.is_some() { 
            self.metrics.insert("preprocessing".to_string(), StageMetrics::new()); 
        }
        if inference_stage.is_some() { 
            self.metrics.insert("inference".to_string(), StageMetrics::new()); 
        }
        if postprocessing_stage.is_some() { 
            self.metrics.insert("postprocessing".to_string(), StageMetrics::new()); 
        }
        if tracking_stage.is_some() { 
            self.metrics.insert("tracking".to_string(), StageMetrics::new()); 
        }
        if overlay_stage.is_some() { 
            self.metrics.insert("overlay".to_string(), StageMetrics::new()); 
        }
        if keying_stage.is_some() { 
            self.metrics.insert("keying".to_string(), StageMetrics::new()); 
        }
        if output_stage.is_some() { 
            self.metrics.insert("output".to_string(), StageMetrics::new()); 
        }

        // Start capture
        self.capture_stage.start().map_err(|e| anyhow::anyhow!("Failed to start capture: {}", e))?;
        println!("✅ Capture started successfully");
        
        let mut last_stats_time = Instant::now();
        
        // Main processing loop
        loop {
            // Check termination conditions
            if let Some(max_runtime) = self.config.max_runtime {
                if self.start_time.elapsed() >= max_runtime {
                    println!("⏰ Maximum runtime reached");
                    break;
                }
            }
            
            if let Some(max_frames) = self.config.max_frames {
                if self.total_frames >= max_frames {
                    println!("🎬 Maximum frames processed");
                    break;
                }
            }
            
            // Get next frame
            match self.capture_stage.get_next_frame() {
                Ok(Some(raw_frame)) => {
                    if let Err(e) = self.process_frame_pipeline(
                        raw_frame,
                        &mut preprocessing_stage,
                        &mut inference_stage,
                        &mut postprocessing_stage,
                        &mut tracking_stage,
                        &mut overlay_stage,
                        &mut keying_stage,
                        &mut output_stage,
                    ) {
                        if self.config.enable_detailed_logging {
                            eprintln!("❌ Frame processing failed: {}", e);
                        }
                    }
                    self.total_frames += 1;
                }
                Ok(None) => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(e) => {
                    eprintln!("❌ Capture error: {}", e);
                    break;
                }
            }
            
            // Print periodic stats
            if last_stats_time.elapsed() >= self.config.stats_interval {
                self.print_status();
                last_stats_time = Instant::now();
            }
        }
        
        // Cleanup
        self.capture_stage.stop().map_err(|e| anyhow::anyhow!("Failed to stop capture: {}", e))?;
        println!("✅ Capture stopped successfully");
        
        self.print_final_summary();
        
        Ok(())
    }

    fn process_frame_pipeline(
        &mut self,
        raw_frame: RawFramePacket,
        preprocessing_stage: &mut Option<PreprocessingStage>,
        inference_stage: &mut Option<InferenceStage>,
        postprocessing_stage: &mut Option<PostProcessingStage>,
        tracking_stage: &mut Option<ObjectTrackingStage>,
        overlay_stage: &mut Option<OverlayPlanningStage>,
        keying_stage: &mut Option<KeyingStage>,
        output_stage: &mut Option<OutputStage>,
    ) -> Result<(), PipelineError> {
        let frame_start = Instant::now();
        let current_frame = raw_frame.clone();
        
        if self.config.enable_detailed_logging {
            println!("📦 Processing frame {}", self.total_frames + 1);
        }

        // Stage 1: Preprocessing (RawFramePacket -> TensorInputPacket)
        let tensor_input = if let Some(ref mut stage) = preprocessing_stage {
            let stage_start = Instant::now();
            match stage.process(current_frame.clone()) {
                Ok(tensor) => {
                    let latency_ns = stage_start.elapsed().as_nanos() as u64;
                    self.metrics.get_mut("preprocessing").unwrap().update(latency_ns, true);
                    if self.config.enable_detailed_logging {
                        println!("  ✓ Preprocessing completed in {:.2}ms", latency_ns as f64 / 1_000_000.0);
                    }
                    Some(tensor)
                }
                Err(e) => {
                    let latency_ns = stage_start.elapsed().as_nanos() as u64;
                    self.metrics.get_mut("preprocessing").unwrap().update(latency_ns, false);
                    if self.config.enable_detailed_logging {
                        eprintln!("  ❌ Preprocessing failed: {}", e);
                    }
                    return Err(e);
                }
            }
        } else {
            None
        };

        // Stage 2: Inference (TensorInputPacket -> RawDetectionsPacket)
        let raw_detections = if let (Some(ref mut stage), Some(tensor)) = (inference_stage, tensor_input) {
            let stage_start = Instant::now();
            match stage.process(tensor) {
                Ok(detections) => {
                    let latency_ns = stage_start.elapsed().as_nanos() as u64;
                    self.metrics.get_mut("inference").unwrap().update(latency_ns, true);
                    if self.config.enable_detailed_logging {
                        println!("  ✓ Inference completed in {:.2}ms ({} detections)", 
                            latency_ns as f64 / 1_000_000.0, detections.dets.len());
                    }
                    Some(detections)
                }
                Err(e) => {
                    let latency_ns = stage_start.elapsed().as_nanos() as u64;
                    self.metrics.get_mut("inference").unwrap().update(latency_ns, false);
                    if self.config.enable_detailed_logging {
                        eprintln!("  ❌ Inference failed: {}", e);
                    }
                    return Err(e);
                }
            }
        } else {
            None
        };

        // Stage 3: Post-processing (RawDetectionsPacket -> DetectionsPacket)
        let mut processed_detections = if let (Some(ref mut stage), Some(raw_dets)) = (postprocessing_stage, raw_detections) {
            let stage_start = Instant::now();
            match stage.process(raw_dets) {
                Ok(detections) => {
                    let latency_ns = stage_start.elapsed().as_nanos() as u64;
                    self.metrics.get_mut("postprocessing").unwrap().update(latency_ns, true);
                    if self.config.enable_detailed_logging {
                        println!("  ✓ Post-processing completed in {:.2}ms ({} final detections)", 
                            latency_ns as f64 / 1_000_000.0, detections.dets.len());
                    }
                    Some(detections)
                }
                Err(e) => {
                    let latency_ns = stage_start.elapsed().as_nanos() as u64;
                    self.metrics.get_mut("postprocessing").unwrap().update(latency_ns, false);
                    if self.config.enable_detailed_logging {
                        eprintln!("  ❌ Post-processing failed: {}", e);
                    }
                    return Err(e);
                }
            }
        } else {
            None
        };

        // Stage 4: Object Tracking (DetectionsPacket -> DetectionsPacket)
        let tracked_detections = if let (Some(ref mut stage), Some(detections)) = (tracking_stage, processed_detections.take()) {
            let stage_start = Instant::now();
            match stage.process(detections) {
                Ok(tracked) => {
                    let latency_ns = stage_start.elapsed().as_nanos() as u64;
                    self.metrics.get_mut("tracking").unwrap().update(latency_ns, true);
                    if self.config.enable_detailed_logging {
                        println!("  ✓ Tracking completed in {:.2}ms", latency_ns as f64 / 1_000_000.0);
                    }
                    Some(tracked)
                }
                Err(e) => {
                    let latency_ns = stage_start.elapsed().as_nanos() as u64;
                    self.metrics.get_mut("tracking").unwrap().update(latency_ns, false);
                    if self.config.enable_detailed_logging {
                        eprintln!("  ❌ Tracking failed: {}", e);
                    }
                    return Err(e);
                }
            }
        } else {
            processed_detections
        };

        // Stage 5: Overlay Planning (DetectionsPacket -> OverlayPlanPacket)
        let overlay_plan = if let (Some(ref mut stage), Some(detections)) = (overlay_stage, tracked_detections) {
            let stage_start = Instant::now();
            match stage.process(detections) {
                Ok(plan) => {
                    let latency_ns = stage_start.elapsed().as_nanos() as u64;
                    self.metrics.get_mut("overlay").unwrap().update(latency_ns, true);
                    if self.config.enable_detailed_logging {
                        println!("  ✓ Overlay planning completed in {:.2}ms ({} operations)", 
                            latency_ns as f64 / 1_000_000.0, plan.ops.len());
                    }
                    Some(plan)
                }
                Err(e) => {
                    let latency_ns = stage_start.elapsed().as_nanos() as u64;
                    self.metrics.get_mut("overlay").unwrap().update(latency_ns, false);
                    if self.config.enable_detailed_logging {
                        eprintln!("  ❌ Overlay planning failed: {}", e);
                    }
                    return Err(e);
                }
            }
        } else {
            None
        };

        // Stage 6: Keying ((RawFramePacket, OverlayPlanPacket) -> KeyingPacket)
        let keying_result = if let (Some(ref mut stage), Some(plan)) = (keying_stage, overlay_plan) {
            let stage_start = Instant::now();
            match stage.process((current_frame.clone(), plan)) {
                Ok(keyed) => {
                    let latency_ns = stage_start.elapsed().as_nanos() as u64;
                    self.metrics.get_mut("keying").unwrap().update(latency_ns, true);
                    if self.config.enable_detailed_logging {
                        println!("  ✓ Keying completed in {:.2}ms", latency_ns as f64 / 1_000_000.0);
                    }
                    Some(keyed)
                }
                Err(e) => {
                    let latency_ns = stage_start.elapsed().as_nanos() as u64;
                    self.metrics.get_mut("keying").unwrap().update(latency_ns, false);
                    if self.config.enable_detailed_logging {
                        eprintln!("  ❌ Keying failed: {}", e);
                    }
                    return Err(e);
                }
            }
        } else {
            None
        };

        // Stage 7: Output (KeyingPacket -> ())
        if let (Some(ref mut stage), Some(keyed)) = (output_stage, keying_result) {
            let stage_start = Instant::now();
            match stage.process(keyed) {
                Ok(_) => {
                    let latency_ns = stage_start.elapsed().as_nanos() as u64;
                    self.metrics.get_mut("output").unwrap().update(latency_ns, true);
                    if self.config.enable_detailed_logging {
                        println!("  ✓ Output completed in {:.2}ms", latency_ns as f64 / 1_000_000.0);
                    }
                }
                Err(e) => {
                    let latency_ns = stage_start.elapsed().as_nanos() as u64;
                    self.metrics.get_mut("output").unwrap().update(latency_ns, false);
                    if self.config.enable_detailed_logging {
                        eprintln!("  ❌ Output failed: {}", e);
                    }
                    return Err(e);
                }
            }
        }

        let total_latency = frame_start.elapsed();
        if self.config.enable_detailed_logging {
            println!("  📋 Total pipeline latency: {:.2}ms", total_latency.as_secs_f64() * 1000.0);
        }

        Ok(())
    }

    fn print_configuration(&self) {
        println!("Configuration:");
        println!("  Device: {}", self.config.capture_config.device_index);
        println!("  Colorspace: {:?}", self.config.capture_config.expected_colorspace);
        
        let mut enabled_stages = Vec::new();
        if self.config.preprocessing.is_some() { enabled_stages.push("Preprocessing"); }
        if self.config.inference.is_some() { enabled_stages.push("Inference"); }
        if self.config.postprocessing.is_some() { enabled_stages.push("Post-processing"); }
        if self.config.tracking.is_some() { enabled_stages.push("Tracking"); }
        if self.config.overlay.is_some() { enabled_stages.push("Overlay"); }
        if self.config.keying.is_some() { enabled_stages.push("Keying"); }
        if self.config.output.is_some() { enabled_stages.push("Output"); }
        
        println!("  Enabled Stages: [{}]", enabled_stages.join(" -> "));
        
        if let Some(max_runtime) = self.config.max_runtime {
            println!("  Max Runtime: {:.1}s", max_runtime.as_secs_f64());
        }
        if let Some(max_frames) = self.config.max_frames {
            println!("  Max Frames: {}", max_frames);
        }
        println!();
    }

    fn print_status(&mut self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let fps = self.total_frames as f64 / elapsed;
        
        println!("📊 Pipeline Status - Runtime: {:.1}s, Frames: {}, FPS: {:.1}", 
            elapsed, self.total_frames, fps);
        
        // Update throughput for all stages
        for (_, metrics) in &mut self.metrics {
            metrics.update_throughput(elapsed);
        }
        
        for (stage_name, metrics) in &self.metrics {
            let status_icon = if metrics.success_rate() > 95.0 { "✅" }
                            else if metrics.success_rate() > 80.0 { "⚠️" }
                            else { "❌" };
            
            println!("  {} {} - Processed: {}, Failed: {}, Success: {:.1}%, Avg: {:.2}ms, Throughput: {:.1} FPS",
                status_icon,
                stage_name,
                metrics.frames_processed,
                metrics.frames_failed,
                metrics.success_rate(),
                metrics.average_latency_ms(),
                metrics.throughput_fps
            );
        }
        println!();
    }

    fn print_final_summary(&mut self) {
        println!();
        println!("🏁 Final DeepGI Pipeline Summary");
        println!("================================");
        
        let total_runtime = self.start_time.elapsed().as_secs_f64();
        let overall_fps = self.total_frames as f64 / total_runtime;
        
        // Calculate total pipeline latency
        let total_avg_latency: f64 = self.metrics.values()
            .map(|m| m.average_latency_ms())
            .sum();
        
        println!("Overall Performance:");
        println!("  Total Runtime: {:.2}s", total_runtime);
        println!("  Total Frames: {}", self.total_frames);
        println!("  Average FPS: {:.2}", overall_fps);
        println!("  Total Pipeline Latency: {:.2}ms", total_avg_latency);
        println!();
        
        println!("Stage Performance (following DeepGI I/O Standards):");
        for (stage_name, metrics) in &self.metrics {
            println!("  🔧 {}:", stage_name);
            println!("    Processed: {} frames", metrics.frames_processed);
            println!("    Failed: {} frames", metrics.frames_failed);
            println!("    Success Rate: {:.1}%", metrics.success_rate());
            println!("    Latency - Avg: {:.2}ms, Min: {:.2}ms, Max: {:.2}ms", 
                metrics.average_latency_ms(),
                metrics.min_latency_ms(),
                metrics.max_latency_ms()
            );
            println!("    Throughput: {:.1} FPS", metrics.throughput_fps);
            println!();
        }
        
        println!("🎯 Pipeline followed DeepGI Standard I/O Packet Flow:");
        println!("  RawFramePacket -> TensorInputPacket -> RawDetectionsPacket");
        println!("  -> DetectionsPacket -> OverlayPlanPacket -> KeyingPacket -> Output");
    }
}
