use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    pub inference: Option<InferenceCfg>,
    pub postprocess: Option<PostprocessCfg>,
}

impl AppConfig {
    pub fn from_file(p: &str) -> Result<Self> {
        let content = std::fs::read_to_string(p)?;
        let config: AppConfig = toml::from_str(&content)?;
        Ok(config)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct InferenceCfg {
    pub backend: String,
    pub model: String,
    pub device: u32,
    pub fp16: bool,
    pub engine_cache: String,
    pub timing_cache: String,
    pub max_workspace_mb: usize,
    pub enable_fallback_cuda: Option<bool>,
    pub warmup_runs: Option<usize>,
    pub input_name: Option<String>,
    pub output_names: Option<Vec<String>>,
}

impl Default for InferenceCfg {
    fn default() -> Self {
        Self {
            backend: "onnxruntime_trt".to_string(),
            model: "models/model.onnx".to_string(),
            device: 0,
            fp16: true,
            engine_cache: "./trt_cache".to_string(),
            timing_cache: "./trt_cache".to_string(),
            max_workspace_mb: 2048,
            enable_fallback_cuda: Some(true),
            warmup_runs: Some(5),
            input_name: Some("images".to_string()),
            output_names: Some(vec!["output".to_string()]),
        }
    }
}

// Postprocessing configuration following postprocessing_guideline.md
#[derive(Debug, Clone, Deserialize)]
pub struct PostprocessCfg {
    #[serde(rename = "type")]
    pub model_type: String, // "yolo" | "detr" | "seg" | "keypoints"
    pub score_thresh: f32,
    pub max_dets: usize,
    pub nms: NmsCfg,
    #[serde(default)]
    pub tracking: TrackingCfg,
    #[serde(default)]
    pub letterbox: LetterboxCfg,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NmsCfg {
    #[serde(rename = "type")]
    pub nms_type: String, // "classwise" | "soft" | "diou" | "ciou"
    pub iou_thresh: f32,
    pub max_per_class: usize,
    pub max_total: usize,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct TrackingCfg {
    #[serde(default)]
    pub enable: bool,
    #[serde(default = "default_iou_match")]
    pub iou_match: f32,
    #[serde(default = "default_max_age")]
    pub max_age: usize,
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f32,
    #[serde(default)]
    pub motion: TrackingMotionCfg,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrackingMotionCfg {
    #[serde(default = "default_use_kalman")]
    pub use_kalman: bool,
    #[serde(default = "default_kalman_process_noise")]
    pub kalman_process_noise: f32,
    #[serde(default = "default_kalman_measurement_noise")]
    pub kalman_measurement_noise: f32,
    #[serde(default = "default_use_optical_flow")]
    pub use_optical_flow: bool,
    #[serde(default = "default_optical_flow_alpha")]
    pub optical_flow_alpha: f32,
    #[serde(default = "default_optical_flow_max_pixels")]
    pub optical_flow_max_pixels: f32,
    #[serde(default = "default_use_velocity")]
    pub use_velocity: bool,
    #[serde(default = "default_velocity_alpha")]
    pub velocity_alpha: f32,
    #[serde(default = "default_velocity_max_delta")]
    pub velocity_max_delta: f32,
    #[serde(default = "default_velocity_decay")]
    pub velocity_decay: f32,
}

impl Default for TrackingMotionCfg {
    fn default() -> Self {
        Self {
            use_kalman: default_use_kalman(),
            kalman_process_noise: default_kalman_process_noise(),
            kalman_measurement_noise: default_kalman_measurement_noise(),
            use_optical_flow: default_use_optical_flow(),
            optical_flow_alpha: default_optical_flow_alpha(),
            optical_flow_max_pixels: default_optical_flow_max_pixels(),
            use_velocity: default_use_velocity(),
            velocity_alpha: default_velocity_alpha(),
            velocity_max_delta: default_velocity_max_delta(),
            velocity_decay: default_velocity_decay(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct LetterboxCfg {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub original_w: u32,
    pub original_h: u32,
}

impl Default for LetterboxCfg {
    fn default() -> Self {
        Self {
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
            original_w: 512,
            original_h: 512,
        }
    }
}

impl Default for PostprocessCfg {
    fn default() -> Self {
        Self {
            model_type: "yolo".to_string(),
            score_thresh: 0.25,
            max_dets: 300,
            nms: NmsCfg::default(),
            tracking: TrackingCfg::default(),
            letterbox: LetterboxCfg::default(),
        }
    }
}

impl Default for NmsCfg {
    fn default() -> Self {
        Self {
            nms_type: "classwise".to_string(),
            iou_thresh: 0.45,
            max_per_class: 100,
            max_total: 300,
        }
    }
}

fn default_iou_match() -> f32 {
    0.5
}

fn default_max_age() -> usize {
    30
}

fn default_min_confidence() -> f32 {
    0.25
}

fn default_use_kalman() -> bool {
    true
}

fn default_kalman_process_noise() -> f32 {
    1.0
}

fn default_kalman_measurement_noise() -> f32 {
    0.4
}

fn default_use_optical_flow() -> bool {
    true
}

fn default_optical_flow_alpha() -> f32 {
    0.35
}

fn default_optical_flow_max_pixels() -> f32 {
    12.0
}

fn default_use_velocity() -> bool {
    true
}

fn default_velocity_alpha() -> f32 {
    0.35
}

fn default_velocity_max_delta() -> f32 {
    18.0
}

fn default_velocity_decay() -> f32 {
    0.15
}
