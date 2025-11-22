//! Configuration loader for DeepGiBox Runner
//!
//! Loads and parses TOML configuration files

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineMode {
    #[serde(rename = "hardware_keying")]
    HardwareKeying,
    #[serde(rename = "inference_only")]
    InferenceOnly,
    #[serde(rename = "visualization")]
    Visualization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndoscopeMode {
    #[serde(rename = "fuji")]
    Fuji,
    #[serde(rename = "olympus")]
    Olympus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisplayMode {
    #[serde(rename = "colon")]
    Colon,
    #[serde(rename = "egd")]
    Egd,
}

impl DisplayMode {
    /// Get display mode name for UI
    pub fn name(&self) -> &'static str {
        match self {
            DisplayMode::Colon => "COLON",
            DisplayMode::Egd => "EGD",
        }
    }
}

impl Default for DisplayMode {
    fn default() -> Self {
        DisplayMode::Colon
    }
}

impl EndoscopeMode {
    /// Get crop region coordinates (x, y, width, height)
    /// Note: These are 1080p base coordinates, use get_crop_region_preset() for auto-scaling
    pub fn get_crop_region(&self) -> (u32, u32, u32, u32) {
        match self {
            EndoscopeMode::Fuji => (1032, 326, 848, 848),
            EndoscopeMode::Olympus => (830, 330, 655, 490),
        }
    }

    /// Get CropRegion preset enum (with auto-scaling for 1080p/4K)
    pub fn get_crop_region_preset(&self) -> preprocess_cuda::CropRegion {
        match self {
            EndoscopeMode::Fuji => preprocess_cuda::CropRegion::Fuji,
            EndoscopeMode::Olympus => preprocess_cuda::CropRegion::Olympus,
        }
    }

    /// Get overlay plan name (prepared for future differentiation)
    pub fn get_overlay_plan(&self) -> &'static str {
        "default" // Currently all modes use the same plan
    }

    /// Get endoscope mode name for display
    pub fn name(&self) -> &'static str {
        match self {
            EndoscopeMode::Fuji => "FUJI",
            EndoscopeMode::Olympus => "OLYMPUS",
        }
    }
}

impl Default for EndoscopeMode {
    fn default() -> Self {
        EndoscopeMode::Olympus
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct GeneralConfig {
    #[serde(default)]
    pub test_duration_seconds: u64,
    #[serde(default)]
    pub enable_debug_dumps: bool,
    #[serde(default = "default_debug_dump_frame_count")]
    pub debug_dump_frame_count: u64,
    #[serde(default = "default_stats_print_interval")]
    pub stats_print_interval: u64,
    #[serde(default = "default_enable_runtime_statistics")]
    pub enable_runtime_statistics: bool,
    #[serde(default = "default_enable_final_summary")]
    pub enable_final_summary: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CaptureConfig {
    #[serde(default)]
    pub device_index: usize,
    #[serde(default = "default_expected_resolution")]
    pub expected_resolution: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PreprocessingConfig {
    #[serde(default = "default_output_width")]
    pub output_width: u32,
    #[serde(default = "default_output_height")]
    pub output_height: u32,
    #[serde(default)]
    pub use_fp16: bool,
    #[serde(default)]
    pub cuda_device: usize,
    #[serde(default = "default_mean")]
    pub mean: [f32; 3],
    #[serde(default = "default_std")]
    pub std: [f32; 3],
    #[serde(default = "default_chroma_order")]
    pub chroma_order: String,
    #[serde(default = "default_crop_region")]
    pub crop_region: String,
    /// Initial endoscope mode (can be changed at runtime with keys 1, 2)
    #[serde(default)]
    pub initial_endoscope_mode: EndoscopeMode,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InferenceConfig {
    pub engine_path: String,
    pub lib_path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TemporalSmoothingConfig {
    #[serde(default)]
    pub enable: bool,
    #[serde(default = "default_window_size")]
    pub window_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrackingConfig {
    #[serde(default)]
    pub enable: bool,
    #[serde(default = "default_max_age")]
    pub max_age: usize,
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f32,
    #[serde(default = "default_iou_threshold")]
    pub iou_threshold: f32,
    #[serde(default)]
    pub motion: TrackingMotionConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrackingMotionConfig {
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

impl Default for TrackingMotionConfig {
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
pub struct EmaSmoothingConfig {
    #[serde(default)]
    pub enable: bool,
    #[serde(default = "default_alpha_position")]
    pub alpha_position: f32,
    #[serde(default = "default_alpha_size")]
    pub alpha_size: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PostprocessingConfig {
    #[serde(default = "default_num_classes")]
    pub num_classes: usize,
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,
    #[serde(default = "default_nms_threshold")]
    pub nms_threshold: f32,
    #[serde(default = "default_max_detections")]
    pub max_detections: usize,
    #[serde(default)]
    pub temporal_smoothing: TemporalSmoothingConfig,
    #[serde(default)]
    pub tracking: TrackingConfig,
    #[serde(default)]
    pub ema_smoothing: EmaSmoothingConfig,
    /// Print verbose statistics (anchor stats, temporal smoothing, NMS details, etc.)
    #[serde(default)]
    pub verbose_stats: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BBoxConfig {
    #[serde(default = "default_base_thickness")]
    pub base_thickness: u32,
    #[serde(default = "default_corner_length_factor")]
    pub corner_length_factor: f32,
    #[serde(default = "default_corner_min_length")]
    pub corner_min_length: f32,
    #[serde(default = "default_corner_max_length")]
    pub corner_max_length: f32,
    #[serde(default = "default_corner_thickness")]
    pub corner_thickness: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LabelConfig {
    #[serde(default = "default_font_size")]
    pub font_size: u32,
    #[serde(default = "default_show_confidence")]
    pub show_confidence: bool,
    #[serde(default = "default_show_track_id")]
    pub show_track_id: bool,
    #[serde(default = "default_label_offset_y")]
    pub label_offset_y: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OverlayConfig {
    #[serde(default = "default_enable_full_ui")]
    pub enable_full_ui: bool,
    #[serde(default)]
    pub display_mode: DisplayMode,
    #[serde(default = "default_show_speaker")]
    pub show_speaker: bool,
    #[serde(default)]
    pub bbox: BBoxConfig,
    #[serde(default)]
    pub label: LabelConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RenderingConfig {
    pub font_path: Option<String>,
    #[serde(default = "default_text_antialiasing")]
    pub text_antialiasing: bool,
    #[serde(default = "default_debug_rendering")]
    pub debug_rendering: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KeyingConfig {
    #[serde(default = "default_keyer_level")]
    pub keyer_level: u8,
    #[serde(default = "default_enable_internal_keying")]
    pub enable_internal_keying: bool,
    pub config_path_1080p: Option<String>,
    pub config_path_4k: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PerformanceConfig {
    #[serde(default = "default_gpu_buffer_pool_size")]
    pub gpu_buffer_pool_size: usize,
    #[serde(default = "default_gpu_buffer_reuse")]
    pub gpu_buffer_reuse: bool,
    #[serde(default = "default_print_detailed_timings")]
    pub print_detailed_timings: bool,
    #[serde(default = "default_print_per_frame_latency")]
    pub print_per_frame_latency: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClassesConfig {
    pub labels: Vec<String>,
    pub modes: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PipelineConfig {
    pub mode: PipelineMode,
    #[serde(default)]
    pub general: GeneralConfig,
    #[serde(default)]
    pub capture: CaptureConfig,
    #[serde(default)]
    pub preprocessing: PreprocessingConfig,
    pub inference: InferenceConfig,
    #[serde(default)]
    pub postprocessing: PostprocessingConfig,
    #[serde(default)]
    pub overlay: OverlayConfig,
    #[serde(default)]
    pub rendering: RenderingConfig,
    #[serde(default)]
    pub keying: KeyingConfig,
    #[serde(default)]
    pub performance: PerformanceConfig,
    pub classes: Option<ClassesConfig>,
}

// Default value functions
fn default_debug_dump_frame_count() -> u64 {
    5
}
fn default_stats_print_interval() -> u64 {
    60
}
fn default_enable_runtime_statistics() -> bool {
    true
}
fn default_enable_final_summary() -> bool {
    true
}
fn default_expected_resolution() -> String {
    "1080p60".to_string()
}
fn default_output_width() -> u32 {
    512
}
fn default_output_height() -> u32 {
    512
}
fn default_mean() -> [f32; 3] {
    [0.0, 0.0, 0.0]
}
fn default_std() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}
fn default_chroma_order() -> String {
    "UYVY".to_string()
}
fn default_crop_region() -> String {
    "Olympus".to_string()
}
fn default_window_size() -> usize {
    4
}
fn default_max_age() -> usize {
    30
}
fn default_min_confidence() -> f32 {
    0.3
}
fn default_iou_threshold() -> f32 {
    0.3
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
fn default_alpha_position() -> f32 {
    0.3
}
fn default_alpha_size() -> f32 {
    0.4
}
fn default_num_classes() -> usize {
    2
}
fn default_confidence_threshold() -> f32 {
    0.25
}
fn default_nms_threshold() -> f32 {
    0.45
}
fn default_max_detections() -> usize {
    100
}
fn default_base_thickness() -> u32 {
    2
}
fn default_corner_length_factor() -> f32 {
    0.22
}
fn default_corner_min_length() -> f32 {
    18.0
}
fn default_corner_max_length() -> f32 {
    40.0
}
fn default_corner_thickness() -> u32 {
    3
}
fn default_font_size() -> u32 {
    16
}
fn default_show_confidence() -> bool {
    true
}
fn default_show_track_id() -> bool {
    true
}
fn default_label_offset_y() -> f32 {
    10.0
}
fn default_enable_full_ui() -> bool {
    true
}
fn default_show_speaker() -> bool {
    true
}
fn default_text_antialiasing() -> bool {
    true
}
fn default_debug_rendering() -> bool {
    false
}
fn default_keyer_level() -> u8 {
    255
}
fn default_enable_internal_keying() -> bool {
    true
}
fn default_gpu_buffer_pool_size() -> usize {
    10
}
fn default_gpu_buffer_reuse() -> bool {
    true
}
fn default_print_detailed_timings() -> bool {
    true
}
fn default_print_per_frame_latency() -> bool {
    true
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            test_duration_seconds: 0,
            enable_debug_dumps: false,
            debug_dump_frame_count: default_debug_dump_frame_count(),
            stats_print_interval: default_stats_print_interval(),
            enable_runtime_statistics: default_enable_runtime_statistics(),
            enable_final_summary: default_enable_final_summary(),
        }
    }
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            device_index: 0,
            expected_resolution: default_expected_resolution(),
        }
    }
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            output_width: default_output_width(),
            output_height: default_output_height(),
            use_fp16: false,
            cuda_device: 0,
            mean: default_mean(),
            std: default_std(),
            chroma_order: default_chroma_order(),
            crop_region: default_crop_region(),
            initial_endoscope_mode: EndoscopeMode::default(),
        }
    }
}

impl Default for TemporalSmoothingConfig {
    fn default() -> Self {
        Self {
            enable: true,
            window_size: default_window_size(),
        }
    }
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            enable: true,
            max_age: default_max_age(),
            min_confidence: default_min_confidence(),
            iou_threshold: default_iou_threshold(),
            motion: TrackingMotionConfig::default(),
        }
    }
}

impl Default for EmaSmoothingConfig {
    fn default() -> Self {
        Self {
            enable: true,
            alpha_position: default_alpha_position(),
            alpha_size: default_alpha_size(),
        }
    }
}

impl Default for PostprocessingConfig {
    fn default() -> Self {
        Self {
            num_classes: default_num_classes(),
            confidence_threshold: default_confidence_threshold(),
            nms_threshold: default_nms_threshold(),
            max_detections: default_max_detections(),
            temporal_smoothing: TemporalSmoothingConfig::default(),
            tracking: TrackingConfig::default(),
            ema_smoothing: EmaSmoothingConfig::default(),
            verbose_stats: false,
        }
    }
}

impl Default for BBoxConfig {
    fn default() -> Self {
        Self {
            base_thickness: default_base_thickness(),
            corner_length_factor: default_corner_length_factor(),
            corner_min_length: default_corner_min_length(),
            corner_max_length: default_corner_max_length(),
            corner_thickness: default_corner_thickness(),
        }
    }
}

impl Default for LabelConfig {
    fn default() -> Self {
        Self {
            font_size: default_font_size(),
            show_confidence: default_show_confidence(),
            show_track_id: default_show_track_id(),
            label_offset_y: default_label_offset_y(),
        }
    }
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            enable_full_ui: default_enable_full_ui(),
            display_mode: DisplayMode::default(),
            show_speaker: default_show_speaker(),
            bbox: BBoxConfig::default(),
            label: LabelConfig::default(),
        }
    }
}

impl Default for RenderingConfig {
    fn default() -> Self {
        Self {
            font_path: None,
            text_antialiasing: default_text_antialiasing(),
            debug_rendering: default_debug_rendering(),
        }
    }
}

impl Default for KeyingConfig {
    fn default() -> Self {
        Self {
            keyer_level: default_keyer_level(),
            enable_internal_keying: default_enable_internal_keying(),
            config_path_1080p: None,
            config_path_4k: None,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            gpu_buffer_pool_size: default_gpu_buffer_pool_size(),
            gpu_buffer_reuse: default_gpu_buffer_reuse(),
            print_detailed_timings: default_print_detailed_timings(),
            print_per_frame_latency: default_print_per_frame_latency(),
        }
    }
}

pub fn load_config(path: &str) -> Result<PipelineConfig> {
    let content = fs::read_to_string(path)
        .map_err(|e| anyhow!("Failed to read config file '{}': {}", path, e))?;

    let config: PipelineConfig = toml::from_str(&content)
        .map_err(|e| anyhow!("Failed to parse config file '{}': {}", path, e))?;

    // Validate required paths
    if !std::path::Path::new(&config.inference.engine_path).exists() {
        return Err(anyhow!(
            "TensorRT engine not found: {}",
            config.inference.engine_path
        ));
    }
    if !std::path::Path::new(&config.inference.lib_path).exists() {
        return Err(anyhow!(
            "TRT shim library not found: {}",
            config.inference.lib_path
        ));
    }

    Ok(config)
}
