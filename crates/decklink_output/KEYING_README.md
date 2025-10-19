# DeckLink Internal Keying Module

This module provides internal keying functionality for Blackmagic DeckLink devices, allowing you to composite a fill signal (main video) with a key signal (alpha channel overlay).

## Overview

Internal keying is a feature of professional video hardware that allows real-time compositing of two video signals:
- **Fill Frame**: The main video signal (background) - typically YUV8bit format
- **Key Frame**: The overlay with alpha channel (foreground) - typically BGRA8 format

The DeckLink hardware (or in this implementation, software compositing) combines these two signals using the alpha channel of the key frame to determine transparency.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Fill Frame    │     │   Key Frame     │
│  (YUV8 Video)   │     │  (BGRA Overlay) │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │ Internal Keying       │
         │ (Composite with       │
         │  alpha blending)      │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │  DeckLink Output      │
         │  (BGRA Video Signal)  │
         └───────────────────────┘
```

## Files

- `src/keying.rs` - Rust API for internal keying
- `output_shim.cpp` - C++ wrapper for DeckLink SDK output functions
- `build.rs` - Build script to compile C++ shim
- `src/bin/test_internal_keying.rs` - Test program demonstrating usage

## Usage

### Basic Example

```rust
use decklink_output::keying::InternalKeyingOutput;
use common_io::{RawFramePacket, OverlayFramePacket};

// Create output stage
let mut output = InternalKeyingOutput::new(
    0,      // device_index
    1920,   // width
    1080,   // height
    60,     // fps_num
    1       // fps_den
);

// Open device
output.open()?;

// Start playback
output.start_playback()?;

// Submit frames
loop {
    let fill_frame: RawFramePacket = get_fill_frame(); // Your YUV8 video
    let key_frame: OverlayFramePacket = get_key_frame(); // Your BGRA overlay
    
    output.submit_keying_frames(&fill_frame, &key_frame)?;
}

// Stop when done
output.stop_playback()?;
```

### Frame Format Requirements

#### Fill Frame (Main Video)
- **Format**: YUV422_8 (8-bit YUV, 2 bytes per pixel)
- **Stride**: width × 2
- **Memory**: CPU memory (MemLoc::Cpu)
- **Color Space**: BT.709 recommended

#### Key Frame (Overlay)
- **Format**: BGRA8 (8-bit BGRA with alpha, 4 bytes per pixel)
- **Stride**: width × 4
- **Memory**: CPU memory (MemLoc::Cpu)
- **Alpha Channel**: 
  - 0 = fully transparent (fill frame visible)
  - 255 = fully opaque (key frame visible)
  - 1-254 = semi-transparent (blended)

### Supported Resolutions

Common configurations:
- **1080p60**: 1920×1080 @ 60fps
- **1080p30**: 1920×1080 @ 30fps
- **4K30**: 3840×2160 @ 30fps
- **720p60**: 1280×720 @ 60fps

## Test Program

Run the test program to verify the internal keying functionality:

```bash
# Build and run
cargo run --bin test_internal_keying [device_index]

# Example
cargo run --bin test_internal_keying 0
```

### What the Test Does

1. Opens DeckLink output device at 1080p60
2. Generates:
   - **Fill**: SMPTE color bars in YUV422 format
   - **Key**: Moving semi-transparent box in BGRA format with alpha
3. Outputs 3 seconds of video
4. Shows a colored box moving horizontally over color bars

### Expected Output

If internal keying is working correctly, you should see:
- Color bars as the background (from fill frame)
- A semi-transparent colored box moving left to right (from key frame with alpha)
- Smooth compositing where the box partially reveals the background based on alpha values

## Technical Details

### Internal Keying vs External Keying

- **Internal Keying**: The DeckLink card composites fill and key internally
- **External Keying**: Separate hardware mixer composites the signals

This implementation uses software compositing to achieve internal keying effect, as standard DeckLink cards may not support true hardware internal keying (which typically requires specialized broadcast hardware).

### Pixel Format Notes

**YUV422_8 Format (UYVY)**:
```
Byte 0: U (Cb)
Byte 1: Y0 (Luma for pixel 0)
Byte 2: V (Cr)
Byte 3: Y1 (Luma for pixel 1)
```
Two pixels share the same chroma (U,V) values.

**BGRA8 Format**:
```
Byte 0: B (Blue)
Byte 1: G (Green)
Byte 2: R (Red)
Byte 3: A (Alpha)
```
Each pixel has independent color and alpha.

### Alpha Blending Formula

The compositing formula used:
```
output = key * (alpha/255) + fill * (1 - alpha/255)
```

Where:
- `key` = key frame RGB values
- `fill` = fill frame RGB values (converted from YUV)
- `alpha` = key frame alpha channel value (0-255)

## Performance Considerations

1. **Frame Buffering**: The DeckLink card buffers several frames. Monitor buffer count to avoid overflow:
   ```rust
   let buffered = output.buffered_frame_count();
   if buffered > 10 {
       // Too many buffered frames, slow down
   }
   ```

2. **Timing**: Submit frames at the exact frame rate to maintain sync:
   ```rust
   let frame_duration = Duration::from_secs_f64(1.0 / fps as f64);
   thread::sleep(frame_duration);
   ```

3. **Memory Location**: Both fill and key frames must be in CPU memory. GPU-to-CPU transfer may be needed if frames are on GPU.

## Integration with Pipeline

In the DeepGIBox pipeline:

```
DeckLinkInput → PreprocessCUDA → Inference → Postprocess
                                                  ↓
                                          OverlayPlan
                                                  ↓
                                          OverlayRender (BGRA)
                                                  ↓
                                          InternalKeyingOutput
                                          (Fill = Input YUV8)
                                          (Key = Overlay BGRA)
```

The original captured YUV8 frame is used as the fill, and the rendered BGRA overlay becomes the key.

## Troubleshooting

### "Failed to open DeckLink output device"
- Check device index (try 0, 1, 2...)
- Ensure DeckLink drivers are installed
- Verify device is not in use by another application
- Check cable connections

### "Failed to schedule frame"
- Too many frames buffered - reduce submission rate
- Check frame dimensions match configured resolution
- Verify frame data is valid and in CPU memory

### No output visible
- Check DeckLink output monitor/cable
- Verify correct output connector selected (SDI vs HDMI)
- Ensure playback is started
- Check that frames are being submitted continuously

### Artifacts or tearing
- Ensure frames are submitted at exact frame rate
- Check buffer count isn't growing unbounded
- Verify stride calculations are correct

## API Reference

### `InternalKeyingOutput`

#### Methods

- `new(device_index, width, height, fps_num, fps_den) -> Self`
  - Create new output stage
  
- `open(&mut self) -> Result<()>`
  - Open DeckLink device and initialize

- `submit_keying_frames(&mut self, fill: &RawFramePacket, key: &OverlayFramePacket) -> Result<()>`
  - Submit fill and key frames for output

- `start_playback(&mut self) -> Result<()>`
  - Begin scheduled frame playback

- `stop_playback(&mut self) -> Result<()>`
  - Stop playback

- `buffered_frame_count(&self) -> u32`
  - Get number of frames waiting in buffer

- `is_open(&self) -> bool`
  - Check if device is open

- `is_playing(&self) -> bool`
  - Check if playback is active

### Helper Functions

- `bgra_to_argb(bgra_data: &[u8], width: usize, height: usize) -> Vec<u8>`
  - Convert BGRA to ARGB format if needed

## Building

Dependencies:
- DeckLink SDK (headers in `../../include/`)
- C++14 compiler
- System libraries: dl, pthread, stdc++

The build is handled automatically by `build.rs` when you run `cargo build`.

## License

This code interfaces with the Blackmagic DeckLink SDK which has its own license terms. See the SDK documentation for details.

## See Also

- [Blackmagic DeckLink SDK Documentation](https://www.blackmagicdesign.com/developer)
- [INSTRUCTION.md](../../INSTRUCTION.md) - Overall project structure
- [common_io](../common_io/) - Frame packet definitions
