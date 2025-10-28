# Preprocessing Validation Guide

## Overview
Preprocessing stage validates input frames to handle **init frames** from video streams that may have incorrect dimensions.

## Production Video Specs
- **Expected Input**: `1920×1080` (1080p)
- **Output**: `1×3×512×512` (NCHW format)
- **Letterbox**: Maintains aspect ratio, pads to square

## Frame Validation

### Default Behavior
```rust
// Preprocessor validates frames by default
// Expected: 1920×1080, panics on mismatch
let preprocessor = Preprocessor::new((512, 512), true, 0)?;
```

### Flexible Validation
```rust
// Option 1: Accept any input size (for testing)
let preprocessor = Preprocessor::new((512, 512), true, 0)?
    .with_input_validation(None, false);

// Option 2: Custom expected size
let preprocessor = Preprocessor::new((512, 512), true, 0)?
    .with_input_validation(Some((1920, 1080)), false);
```

### Skip Invalid Frames (Future)
```rust
// TODO: Change Stage trait to return Result<TensorInputPacket>
// let preprocessor = Preprocessor::new((512, 512), true, 0)?
//     .with_input_validation(Some((1920, 1080)), true); // skip_invalid = true
```

## Letterbox Calculation

### Formula
```
scale = min(model_w / orig_w, model_h / orig_h)
new_w = orig_w * scale
new_h = orig_h * scale
pad_x = (model_w - new_w) / 2
pad_y = (model_h - new_h) / 2
```

### Examples

**1920×1080 → 512×512:**
```
scale = min(512/1920, 512/1080) = 0.4741
new_w = 1920 * 0.4741 = 910
new_h = 1080 * 0.4741 = 512
pad_x = (512 - 910) / 2 = -199  ❌ Invalid! Image wider than square!
pad_y = (512 - 512) / 2 = 0

Correction:
scale = 512 / 1080 = 0.4741 (limited by height)
new_w = 1920 * 0.4741 = 910  → clipped to 512
pad_x = 0
pad_y = 0
```

**Actually for 1920×1080:**
```
scale = min(512/1920, 512/1080) = min(0.2667, 0.4741) = 0.2667
new_w = 1920 * 0.2667 = 512
new_h = 1080 * 0.2667 = 288
pad_x = (512 - 512) / 2 = 0
pad_y = (512 - 288) / 2 = 112
```

**1267×954 → 512×512:**
```
scale = min(512/1267, 512/954) = min(0.404, 0.537) = 0.404
new_w = 1267 * 0.404 = 512
new_h = 954 * 0.404 = 385
pad_x = (512 - 512) / 2 = 0
pad_y = (512 - 385) / 2 = 63
```

## Postprocessing Integration

Postprocessing **automatically calculates** letterbox parameters from `FrameMeta`:

```rust
// In PostStage::process()
let orig_w = input.from.width as f32;  // From FrameMeta
let orig_h = input.from.height as f32;

let scale = (MODEL_W / orig_w).min(MODEL_H / orig_h);
let pad_x = (MODEL_W - new_w) / 2.0;
let pad_y = (MODEL_H - new_h) / 2.0;

config.letterbox_scale = scale;
config.letterbox_pad = (pad_x, pad_y);
config.original_size = (input.from.width, input.from.height);
```

**No manual configuration needed!** ✅

## Bbox Coordinate Space

### Model Output
```
Bbox in 512×512 space: (cx, cy, w, h)
```

### After Postprocessing
```
Bbox in original space: (x, y, w, h) in 1920×1080
```

### Conversion (done automatically in decode_predictions)
```rust
// Remove padding
let mut left = cx - w * 0.5 - pad_x;
let mut top = cy - h * 0.5 - pad_y;

// Scale back to original
left *= inv_scale;
top *= inv_scale;

// Clamp to original image bounds
left = left.clamp(0.0, orig_w);
top = top.clamp(0.0, orig_h);
```

## Testing

### Test with Sample Image (1267×954)
```bash
cargo run -p playgrounds --bin inference_v2
```

Expected output:
```
Image loaded: 1267x954
Letterbox [1267x954 → 512x512]: scale=0.4041, pad=(0.0,63.0)
bbox=(235.5,366.8,259.0,400.8)  # Coordinates in 1267×954 space ✅
```

### Test with Production Video (1920×1080)
```bash
# TODO: Add test binary for video stream
cargo run -p playgrounds --bin test_video_preprocessing
```

Expected output:
```
Frame #0: 1920x1080 ✅
Frame #1: 1920x1080 ✅
Letterbox [1920x1080 → 512x512]: scale=0.2667, pad=(0.0,112.0)
bbox=(450.2,234.5,180.3,220.8)  # Coordinates in 1920×1080 space ✅
```

## Common Issues

### ❌ Wrong bbox coordinates
**Problem:** Bbox coordinates don't match video
**Solution:** Check `letterbox_scale` and `letterbox_pad` are calculated from actual `FrameMeta`

### ❌ Init frame crashes
**Problem:** First 1-2 frames have wrong dimensions
**Solution:** Use `.with_input_validation(Some((1920, 1080)), true)` to skip invalid frames

### ❌ Aspect ratio distortion
**Problem:** Objects appear stretched/squashed
**Solution:** Verify letterbox maintains aspect ratio (padding on one axis only)

## Implementation Status

- ✅ Letterbox calculation from FrameMeta
- ✅ Bbox coordinate conversion (model → original space)
- ✅ Frame size validation (panic on mismatch)
- ⏳ Skip invalid frames (requires Result-based Stage trait)
- ⏳ Dynamic model input size (currently hardcoded to 512×512)
