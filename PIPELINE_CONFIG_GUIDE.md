# Pipeline Configuration Guide

## Overview

The pipeline configuration system allows you to customize all aspects of the DeepGiBox processing pipeline through a single TOML configuration file.

## Configuration File

The main configuration file is located at:
```
configs/pipeline_config.toml
```

## Configuration Sections

### 1. General Settings

```toml
[general]
test_duration_seconds = 30        # Test duration (0 = unlimited)
enable_debug_dumps = false        # Enable debug file dumps
debug_dump_frame_count = 5        # Dump first N frames
stats_print_interval = 60         # Print stats every N frames
```

### 2. Capture Settings

```toml
[capture]
device_index = 0                  # DeckLink device index
expected_resolution = "1080p60"   # Expected input resolution
```

### 3. Preprocessing Settings

```toml
[preprocessing]
output_width = 512                # Model input width
output_height = 512               # Model input height
use_fp16 = false                  # Use FP16 precision
cuda_device = 0                   # CUDA device ID
mean = [0.0, 0.0, 0.0]           # Normalization mean (RGB)
std = [1.0, 1.0, 1.0]            # Normalization std (RGB)
chroma_order = "UYVY"            # YUV422 chroma order
crop_region = "Olympus"          # Crop region preset
```

**Crop Region Options:**
- `"Olympus"` - Olympus endoscope crop
- `"Pentax"` - Pentax endoscope crop
- `"Fuji"` - Fuji endoscope crop
- `"None"` - No cropping

### 4. Inference Settings

```toml
[inference]
engine_path = "configs/model/v7_optimized_YOLOv5.engine"
lib_path = "trt-shim/build/libtrt_shim.so"
```

### 5. Postprocessing Settings

```toml
[postprocessing]
num_classes = 2                   # Number of classes
confidence_threshold = 0.25       # Confidence threshold (0.0-1.0)
nms_threshold = 0.45              # NMS IoU threshold
max_detections = 100              # Max detections (0 = unlimited)

[postprocessing.temporal_smoothing]
enable = true                     # Enable temporal smoothing
window_size = 4                   # Smoothing window size

[postprocessing.tracking]
enable = true                     # Enable SORT tracking
max_age = 30                      # Max track age (frames)
min_confidence = 0.3              # Min confidence for tracking
iou_threshold = 0.3               # IoU threshold for matching
```

### 6. Overlay Settings

```toml
[overlay]
enable_full_ui = true             # Enable full UI mode

[overlay.bbox]
base_thickness = 2                # Bbox thickness (1080p base)
corner_length_factor = 0.22       # Corner bracket length factor
corner_min_length = 18.0          # Min corner length (px)
corner_max_length = 40.0          # Max corner length (px)
corner_thickness = 3              # Corner bracket thickness

[overlay.label]
font_size = 16                    # Font size (1080p base)
show_confidence = true            # Show confidence score
show_track_id = true              # Show track ID
label_offset_y = 10.0             # Label Y offset (px)

[overlay.colors]
# ARGB format: [Alpha, Red, Green, Blue]
class_0 = [255, 0, 255, 0]       # Class 0: Green (Hyperplastic)
class_1 = [255, 255, 0, 0]       # Class 1: Red (Neoplastic)
class_unknown = [255, 128, 128, 128]  # Unknown: Gray

[overlay.colors.hud]
text = [255, 255, 255, 255]      # HUD text: White
active = [255, 0, 255, 0]        # Active elements: Green
inactive = [128, 128, 128, 128]  # Inactive elements: Gray
```

### 7. Rendering Settings

```toml
[rendering]
# font_path = "testsupport/DejaVuSans.ttf"  # Custom font (optional)
text_antialiasing = true          # Enable text anti-aliasing
```

### 8. Hardware Keying Settings

```toml
[keying]
# Auto-selected based on resolution if not specified
# config_path_1080p = "configs/dev_1080p60_yuv422_fp16_trt.toml"
# config_path_4k = "configs/dev_4k30_yuv422_fp16_trt.toml"
keyer_level = 255                 # Keyer level (0-255)
enable_internal_keying = true     # Enable internal keying
```

### 9. Performance Settings

```toml
[performance]
gpu_buffer_pool_size = 10         # GPU buffer pool size
gpu_buffer_reuse = true           # Enable buffer reuse
print_detailed_timings = true     # Print detailed timings
print_per_frame_latency = true    # Print per-frame latency
```

### 10. Class Labels

```toml
[classes]
labels = [
    "Hyper",    # Class 0: Hyperplastic
    "Neo",      # Class 1: Neoplastic
]

modes = [
    "COLON",    # Class 0 mode
    "EGD",      # Class 1 mode
]
```

## Usage Example

### Load Configuration in Rust

```rust
mod pipeline_config;
use pipeline_config::PipelineConfig;

fn main() -> Result<()> {
    // Load config from file
    let config = PipelineConfig::from_file("configs/pipeline_config.toml")?;
    
    // Or use default config
    // let config = PipelineConfig::default();
    
    // Access configuration values
    println!("Test duration: {}s", config.general.test_duration_seconds);
    println!("Confidence threshold: {}", config.postprocessing.confidence_threshold);
    
    // Use config values
    let crop_region = config.get_crop_region();
    let class_label = config.get_class_label(0);
    let class_color = config.get_class_color(1);
    
    Ok(())
}
```

## Tips

### Resolution Scaling

The pipeline automatically scales UI elements based on resolution:
- **1080p (1920×1080)**: Base scale = 1.0
- **4K (3840×2160)**: Base scale = 2.0

This means:
- Bounding box thickness: 2px @ 1080p → 4px @ 4K
- Font sizes scale proportionally
- Other UI elements scale automatically

### Performance Tuning

**For better FPS:**
- Increase `confidence_threshold` (fewer detections)
- Decrease `max_detections`
- Disable `temporal_smoothing` or reduce `window_size`
- Disable `tracking` if not needed
- Set `enable_full_ui = false` for simpler overlay

**For better accuracy:**
- Decrease `confidence_threshold` (more detections)
- Enable `temporal_smoothing` with larger `window_size`
- Enable `tracking` for stable detection IDs

### Debug Mode

Enable debug dumps to save intermediate results:
```toml
[general]
enable_debug_dumps = true
debug_dump_frame_count = 5
```

This will save to `output/test/`:
- Raw YUV422 frames
- Preprocessing outputs
- Inference outputs
- Detection results
- Overlay plans
- BGRA overlay buffers

## Color Format

All colors use **ARGB format**: `[Alpha, Red, Green, Blue]`

Examples:
- White: `[255, 255, 255, 255]`
- Red: `[255, 255, 0, 0]`
- Green: `[255, 0, 255, 0]`
- Blue: `[255, 0, 0, 255]`
- Semi-transparent white: `[128, 255, 255, 255]`

## Advanced Usage

### Multiple Configurations

You can create multiple config files for different scenarios:

```bash
configs/
  pipeline_config.toml          # Default
  pipeline_config_1080p.toml    # 1080p optimized
  pipeline_config_4k.toml       # 4K optimized
  pipeline_config_fast.toml     # Speed optimized
  pipeline_config_accurate.toml # Accuracy optimized
```

Load specific config:
```rust
let config = PipelineConfig::from_file("configs/pipeline_config_4k.toml")?;
```

### Runtime Override

You can override config values at runtime:
```rust
let mut config = PipelineConfig::from_file("configs/pipeline_config.toml")?;

// Override values
config.postprocessing.confidence_threshold = 0.35;
config.overlay.enable_full_ui = false;
```

## Troubleshooting

### Issue: Bounding boxes too thin/thick

Adjust:
```toml
[overlay.bbox]
base_thickness = 3  # Increase for thicker lines
```

### Issue: Too many false positives

Adjust:
```toml
[postprocessing]
confidence_threshold = 0.35  # Increase threshold
```

### Issue: Labels too small/large

Adjust:
```toml
[overlay.label]
font_size = 20  # Increase for larger text
```

### Issue: Pipeline too slow

Try:
```toml
[postprocessing]
confidence_threshold = 0.30  # Reduce detections
max_detections = 50          # Limit max detections

[postprocessing.temporal_smoothing]
enable = false               # Disable smoothing

[postprocessing.tracking]
enable = false               # Disable tracking

[overlay]
enable_full_ui = false       # Simpler overlay
```
