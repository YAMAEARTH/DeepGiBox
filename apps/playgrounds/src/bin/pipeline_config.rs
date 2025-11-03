/// Configuration structures for pipeline_capture_to_output_v5
/// 
/// This module provides type-safe configuration loading from TOML files.

use serde::{Deserialize, Serialize};
use anyhow::Result;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PipelineConfig {
    pub general: GeneralConfig,
    pub capture: CaptureConfig,
    pub preprocessing: PreprocessingConfig,
    pub inference: InferenceConfig,
    pub postprocessing: PostprocessingConfig,
    pub overlay: OverlayConfig,
    pub rendering: RenderingConfig,
    pub keying: KeyingConfig,
    pub performance: PerformanceConfig,
    pub classes: ClassesConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GeneralConfig {
    #[serde(default = "default_test_duration")]
    pub test_duration_seconds: u64,
    #[serde(default)]
    pub enable_debug_dumps: bool,
    #[serde(default = "default_debug_dump_count")]
    pub debug_dump_frame_count: usize,
    #[serde(default = "default_stats_interval")]
    pub stats_print_interval: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CaptureConfig {
    #[serde(default)]
    pub device_index: usize,
    #[serde(default = "default_resolution")]
    pub expected_resolution: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PreprocessingConfig {
    #[serde(default = "default_output_size")]
    pub output_width: u32,
    #[serde(default = "default_output_size")]
    pub output_height: u32,
    #[serde(default)]
    pub use_fp16: bool,
    #[serde(default)]
    pub cuda_device: i32,
    #[serde(default = "default_mean")]
    pub mean: [f32; 3],
    #[serde(default = "default_std")]
    pub std: [f32; 3],
    #[serde(default = "default_chroma_order")]
    pub chroma_order: String,
    #[serde(default = "default_crop_region")]
    pub crop_region: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InferenceConfig {
    pub engine_path: String,
    pub lib_path: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PostprocessingConfig {
    #[serde(default = "default_num_classes")]
    pub num_classes: usize,
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,
    #[serde(default = "default_nms_threshold")]
    pub nms_threshold: f32,
    #[serde(default = "default_max_detections")]
    pub max_detections: usize,
    pub temporal_smoothing: TemporalSmoothingConfig,
    pub tracking: TrackingConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TemporalSmoothingConfig {
    #[serde(default)]
    pub enable: bool,
    #[serde(default = "default_window_size")]
    pub window_size: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrackingConfig {
    #[serde(default)]
    pub enable: bool,
    #[serde(default = "default_max_age")]
    pub max_age: usize,
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f32,
    #[serde(default = "default_iou_threshold")]
    pub iou_threshold: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OverlayConfig {
    #[serde(default = "default_true")]
    pub enable_full_ui: bool,
    pub bbox: BBoxConfig,
    pub label: LabelConfig,
    pub colors: ColorConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BBoxConfig {
    #[serde(default = "default_bbox_thickness")]
    pub base_thickness: u8,
    #[serde(default = "default_corner_factor")]
    pub corner_length_factor: f32,
    #[serde(default = "default_corner_min")]
    pub corner_min_length: f32,
    #[serde(default = "default_corner_max")]
    pub corner_max_length: f32,
    #[serde(default = "default_corner_thickness")]
    pub corner_thickness: u8,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LabelConfig {
    #[serde(default = "default_font_size")]
    pub font_size: u16,
    #[serde(default = "default_true")]
    pub show_confidence: bool,
    #[serde(default = "default_true")]
    pub show_track_id: bool,
    #[serde(default = "default_label_offset")]
    pub label_offset_y: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ColorConfig {
    #[serde(default = "default_color_class0")]
    pub class_0: [u8; 4],
    #[serde(default = "default_color_class1")]
    pub class_1: [u8; 4],
    #[serde(default = "default_color_unknown")]
    pub class_unknown: [u8; 4],
    pub hud: HudColorConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HudColorConfig {
    #[serde(default = "default_color_white")]
    pub text: [u8; 4],
    #[serde(default = "default_color_green")]
    pub active: [u8; 4],
    #[serde(default = "default_color_gray")]
    pub inactive: [u8; 4],
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RenderingConfig {
    pub font_path: Option<String>,
    #[serde(default = "default_true")]
    pub text_antialiasing: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KeyingConfig {
    pub config_path_1080p: Option<String>,
    pub config_path_4k: Option<String>,
    #[serde(default = "default_keyer_level")]
    pub keyer_level: u8,
    #[serde(default = "default_true")]
    pub enable_internal_keying: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PerformanceConfig {
    #[serde(default = "default_pool_size")]
    pub gpu_buffer_pool_size: usize,
    #[serde(default = "default_true")]
    pub gpu_buffer_reuse: bool,
    #[serde(default = "default_true")]
    pub print_detailed_timings: bool,
    #[serde(default = "default_true")]
    pub print_per_frame_latency: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClassesConfig {
    pub labels: Vec<String>,
    pub modes: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Default value functions
// ═══════════════════════════════════════════════════════════════════════════

fn default_test_duration() -> u64 { 30 }
fn default_debug_dump_count() -> usize { 5 }
fn default_stats_interval() -> u64 { 60 }
fn default_resolution() -> String { "1080p60".to_string() }
fn default_output_size() -> u32 { 512 }
fn default_mean() -> [f32; 3] { [0.0, 0.0, 0.0] }
fn default_std() -> [f32; 3] { [1.0, 1.0, 1.0] }
fn default_chroma_order() -> String { "UYVY".to_string() }
fn default_crop_region() -> String { "Olympus".to_string() }
fn default_num_classes() -> usize { 2 }
fn default_confidence_threshold() -> f32 { 0.25 }
fn default_nms_threshold() -> f32 { 0.45 }
fn default_max_detections() -> usize { 100 }
fn default_window_size() -> usize { 4 }
fn default_max_age() -> usize { 30 }
fn default_min_confidence() -> f32 { 0.3 }
fn default_iou_threshold() -> f32 { 0.3 }
fn default_true() -> bool { true }
fn default_bbox_thickness() -> u8 { 2 }
fn default_corner_factor() -> f32 { 0.22 }
fn default_corner_min() -> f32 { 18.0 }
fn default_corner_max() -> f32 { 40.0 }
fn default_corner_thickness() -> u8 { 3 }
fn default_font_size() -> u16 { 16 }
fn default_label_offset() -> f32 { 10.0 }
fn default_color_class0() -> [u8; 4] { [255, 0, 255, 0] } // Green
fn default_color_class1() -> [u8; 4] { [255, 255, 0, 0] } // Red
fn default_color_unknown() -> [u8; 4] { [255, 128, 128, 128] } // Gray
fn default_color_white() -> [u8; 4] { [255, 255, 255, 255] }
fn default_color_green() -> [u8; 4] { [255, 0, 255, 0] }
fn default_color_gray() -> [u8; 4] { [128, 128, 128, 128] }
fn default_keyer_level() -> u8 { 255 }
fn default_pool_size() -> usize { 10 }

// ═══════════════════════════════════════════════════════════════════════════
// Implementation
// ═══════════════════════════════════════════════════════════════════════════

impl PipelineConfig {
    /// Load configuration from TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: PipelineConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Load default configuration
    pub fn default() -> Self {
        Self {
            general: GeneralConfig {
                test_duration_seconds: default_test_duration(),
                enable_debug_dumps: false,
                debug_dump_frame_count: default_debug_dump_count(),
                stats_print_interval: default_stats_interval(),
            },
            capture: CaptureConfig {
                device_index: 0,
                expected_resolution: default_resolution(),
            },
            preprocessing: PreprocessingConfig {
                output_width: default_output_size(),
                output_height: default_output_size(),
                use_fp16: false,
                cuda_device: 0,
                mean: default_mean(),
                std: default_std(),
                chroma_order: default_chroma_order(),
                crop_region: default_crop_region(),
            },
            inference: InferenceConfig {
                engine_path: "configs/model/v7_optimized_YOLOv5.engine".to_string(),
                lib_path: "trt-shim/build/libtrt_shim.so".to_string(),
            },
            postprocessing: PostprocessingConfig {
                num_classes: default_num_classes(),
                confidence_threshold: default_confidence_threshold(),
                nms_threshold: default_nms_threshold(),
                max_detections: default_max_detections(),
                temporal_smoothing: TemporalSmoothingConfig {
                    enable: true,
                    window_size: default_window_size(),
                },
                tracking: TrackingConfig {
                    enable: true,
                    max_age: default_max_age(),
                    min_confidence: default_min_confidence(),
                    iou_threshold: default_iou_threshold(),
                },
            },
            overlay: OverlayConfig {
                enable_full_ui: true,
                bbox: BBoxConfig {
                    base_thickness: default_bbox_thickness(),
                    corner_length_factor: default_corner_factor(),
                    corner_min_length: default_corner_min(),
                    corner_max_length: default_corner_max(),
                    corner_thickness: default_corner_thickness(),
                },
                label: LabelConfig {
                    font_size: default_font_size(),
                    show_confidence: true,
                    show_track_id: true,
                    label_offset_y: default_label_offset(),
                },
                colors: ColorConfig {
                    class_0: default_color_class0(),
                    class_1: default_color_class1(),
                    class_unknown: default_color_unknown(),
                    hud: HudColorConfig {
                        text: default_color_white(),
                        active: default_color_green(),
                        inactive: default_color_gray(),
                    },
                },
            },
            rendering: RenderingConfig {
                font_path: None,
                text_antialiasing: true,
            },
            keying: KeyingConfig {
                config_path_1080p: Some("configs/dev_1080p60_yuv422_fp16_trt.toml".to_string()),
                config_path_4k: Some("configs/dev_4k30_yuv422_fp16_trt.toml".to_string()),
                keyer_level: default_keyer_level(),
                enable_internal_keying: true,
            },
            performance: PerformanceConfig {
                gpu_buffer_pool_size: default_pool_size(),
                gpu_buffer_reuse: true,
                print_detailed_timings: true,
                print_per_frame_latency: true,
            },
            classes: ClassesConfig {
                labels: vec!["Hyper".to_string(), "Neo".to_string()],
                modes: vec!["COLON".to_string(), "EGD".to_string()],
            },
        }
    }

    /// Get DeckLink config path based on resolution
    pub fn get_decklink_config_path(&self, width: u32, height: u32) -> String {
        if width == 3840 && height == 2160 {
            self.keying.config_path_4k.clone()
                .unwrap_or_else(|| "configs/dev_4k30_yuv422_fp16_trt.toml".to_string())
        } else {
            self.keying.config_path_1080p.clone()
                .unwrap_or_else(|| "configs/dev_1080p60_yuv422_fp16_trt.toml".to_string())
        }
    }

    /// Get class label by ID
    pub fn get_class_label(&self, class_id: usize) -> String {
        self.classes.labels.get(class_id)
            .cloned()
            .unwrap_or_else(|| "Unknown".to_string())
    }

    /// Get class mode by ID (for full UI)
    pub fn get_class_mode(&self, class_id: usize) -> String {
        self.classes.modes.get(class_id)
            .cloned()
            .unwrap_or_else(|| "UNKNOWN".to_string())
    }

    /// Get class color by ID (ARGB format)
    pub fn get_class_color(&self, class_id: usize) -> (u8, u8, u8, u8) {
        let color = match class_id {
            0 => self.overlay.colors.class_0,
            1 => self.overlay.colors.class_1,
            _ => self.overlay.colors.class_unknown,
        };
        (color[0], color[1], color[2], color[3])
    }

    /// Parse crop region from string
    pub fn get_crop_region(&self) -> preprocess_cuda::CropRegion {
        match self.preprocessing.crop_region.as_str() {
            "Olympus" => preprocess_cuda::CropRegion::Olympus,
            "Pentax" => preprocess_cuda::CropRegion::Pentax,
            "Fuji" => preprocess_cuda::CropRegion::Fuji,
            "None" => preprocess_cuda::CropRegion::None,
            _ => {
                eprintln!("Unknown crop region: {}, using Olympus", self.preprocessing.crop_region);
                preprocess_cuda::CropRegion::Olympus
            }
        }
    }

    /// Parse chroma order from string
    pub fn get_chroma_order(&self) -> preprocess_cuda::ChromaOrder {
        match self.preprocessing.chroma_order.as_str() {
            "UYVY" => preprocess_cuda::ChromaOrder::UYVY,
            "YUYV" => preprocess_cuda::ChromaOrder::YUYV,
            _ => {
                eprintln!("Unknown chroma order: {}, using UYVY", self.preprocessing.chroma_order);
                preprocess_cuda::ChromaOrder::UYVY
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PipelineConfig::default();
        assert_eq!(config.general.test_duration_seconds, 30);
        assert_eq!(config.preprocessing.output_width, 512);
        assert_eq!(config.postprocessing.num_classes, 2);
    }

    #[test]
    fn test_load_config() {
        // This test requires the config file to exist
        if let Ok(config) = PipelineConfig::from_file("configs/pipeline_config.toml") {
            assert!(config.preprocessing.output_width > 0);
            assert!(config.postprocessing.confidence_threshold >= 0.0);
            assert!(config.postprocessing.confidence_threshold <= 1.0);
        }
    }
}
