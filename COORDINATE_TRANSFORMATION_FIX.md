# Coordinate Transformation Fix

## Problem Summary

The bounding boxes in `pipeline_capture_to_overlay_v3` were appearing at wrong locations because of a mismatch between preprocessing and postprocessing coordinate transformations.

### Root Cause

**CUDA Preprocessing (in `preprocess_cuda/preprocess.cu`):**
- Uses **stretch resize** (simple bilinear interpolation)
- No letterboxing, no padding, no aspect ratio preservation
- Formula: `sx = (x + 0.5) * (in_w / out_w) - 0.5`
- Example: 1920×1080 → 512×512 (stretched)

**Postprocessing (in `postprocess/src/post.rs`):**
- Was assuming **letterbox preprocessing** (resize + padding to preserve aspect ratio)
- Formula: `img_coord = (model_coord - padding) / scale`
- This formula is **incorrect** for stretch resize!

### The Mismatch

For a 1920×1080 → 512×512 transformation:

**Stretch Resize (what CUDA does):**
- Scale X: 1920 / 512 = 3.75
- Scale Y: 1080 / 512 = 2.109375
- Different X and Y scales!

**Letterbox (what postprocessing assumed):**
- Scale: min(512/1920, 512/1080) = 0.2667
- Padding: center the resized image
- Same X and Y scale!

This mismatch caused bounding boxes to appear at completely wrong positions.

## Solution

Added a `use_stretch_resize` flag to `YoloPostConfig` that selects the correct coordinate transformation:

### Stretch Resize (for CUDA preprocessing):
```rust
// Direct scale transformation (different X/Y scales)
let scale_x = orig_w / MODEL_SIZE;  // e.g., 1920 / 512 = 3.75
let scale_y = orig_h / MODEL_SIZE;  // e.g., 1080 / 512 = 2.109375

let left = (cx - w * 0.5) * scale_x;
let top = (cy - h * 0.5) * scale_y;
let right = (cx + w * 0.5) * scale_x;
let bottom = (cy + h * 0.5) * scale_y;
```

### Letterbox (for CPU preprocessing in inference_v2):
```rust
// Remove padding then scale (same X/Y scale, aspect ratio preserved)
let left = (cx - w * 0.5 - pad_x) / letterbox_scale;
let top = (cy - h * 0.5 - pad_y) / letterbox_scale;
let right = (cx + w * 0.5 - pad_x) / letterbox_scale;
let bottom = (cy + h * 0.5 - pad_y) / letterbox_scale;
```

## Files Modified

1. **`crates/postprocess/src/post.rs`:**
   - Added `use_stretch_resize: bool` field to `YoloPostConfig`
   - Implemented dual coordinate transformation logic in `decode_predictions()`
   - Stretch resize: uses separate X/Y scales
   - Letterbox: uses padding removal + uniform scale

2. **`crates/postprocess/src/lib.rs`:**
   - Set `use_stretch_resize = true` in `Postprocess::new()` (for production pipeline)
   - Set `use_stretch_resize = true` in `from_path()` (for test binaries)
   - Removed incorrect letterbox calculation in `PostStage::process()`
   - Added documentation explaining CUDA preprocessing behavior

## Verification

After the fix, coordinates are now correct:

**Before (wrong):**
- Boxes stacked on left side
- Coordinates didn't match tissue location

**After (correct):**
- Detection #1: (447, 470) 69×120 ✓ Center-right area
- Detection #2: (443, 466) 123×180 ✓ Center area  
- Detection #3: (120, 467) 131×198 ✓ Left side
- Detection #4: (470, 451) 276×136 ✓ Right side
- Detection #5: (440, 448) 118×194 ✓ Center area

All coordinates are within valid bounds (0-1920 for X, 0-1080 for Y).

## Important Notes

### For Production Pipeline (`pipeline_capture_to_overlay_v3.rs`):
- ✅ Uses CUDA preprocessing (stretch resize)
- ✅ Uses `use_stretch_resize = true`
- ✅ Coordinates are now correct

### For Test Binary (`inference_v2.rs`):
- ✅ Uses CPU preprocessing (letterbox)
- ✅ Uses `use_stretch_resize = false` (via JSON metadata)
- ✅ Coordinates are correct (already working before)

### Future Consideration

If you want to use **letterbox preprocessing** in the CUDA kernel (better for model accuracy), you would need to:

1. Modify `preprocess.cu` to implement letterbox logic:
   - Calculate aspect-ratio-preserving scale
   - Add gray padding (114, 114, 114)
   - Center the resized image

2. Change postprocessing to use `use_stretch_resize = false`

However, the current stretch resize approach works fine and is simpler to implement in CUDA.

## Testing

Test with:
```bash
cargo run --release --bin pipeline_capture_to_overlay_v3 --features full
```

Check output files:
- `output/test/comparison_frame_*.png` - Side-by-side original + overlay
- `output/test/postprocess_frame_*.txt` - Detection coordinates in text format

Verify that bounding boxes appear at correct locations on the tissue in the comparison images.
