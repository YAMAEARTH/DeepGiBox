/// Internal Keying module for Blackmagic DeckLink
/// 
/// This module provides internal keying functionality for compositing
/// a fill signal (main video) with a key signal (alpha channel).
/// 
/// Input formats:
/// - Fill: YUV8bit (8-bit YUV422 from capture)
/// - Key: BGRA8 (overlay with alpha channel)
/// 
/// The DeckLink card will internally composite the fill and key signals
/// to produce the final keyed output.

use anyhow::{anyhow, bail, ensure, Result};
use common_io::{MemLoc, OverlayFramePacket, PixelFormat, RawFramePacket};
use std::ptr::NonNull;

// External C functions from the output shim
extern "C" {
    /// Initialize DeckLink output device for internal keying
    /// 
    /// # Arguments
    /// * `device_index` - Index of the DeckLink device to use
    /// * `width` - Video width
    /// * `height` - Video height
    /// * `fps_num` - Frame rate numerator (e.g., 60 for 60fps, 30 for 30fps)
    /// * `fps_den` - Frame rate denominator (typically 1)
    /// 
    /// # Returns
    /// * `true` if initialization succeeded, `false` otherwise
    fn decklink_output_open(
        device_index: i32,
        width: i32,
        height: i32,
        fps_num: i32,
        fps_den: i32,
    ) -> bool;

    /// Submit a pair of fill (YUV) and key (BGRA) frames for internal keying output
    /// 
    /// # Arguments
    /// * `fill_data` - Pointer to YUV8bit fill frame data
    /// * `fill_stride` - Row stride of fill frame in bytes
    /// * `key_data` - Pointer to BGRA key frame data (with alpha channel)
    /// * `key_stride` - Row stride of key frame in bytes
    /// 
    /// # Returns
    /// * `true` if frames were scheduled successfully, `false` otherwise
    fn decklink_output_submit_keying(
        fill_data: *const u8,
        fill_stride: i32,
        key_data: *const u8,
        key_stride: i32,
    ) -> bool;

    /// Start playback of scheduled frames
    fn decklink_output_start_playback() -> bool;

    /// Stop playback
    fn decklink_output_stop_playback() -> bool;

    /// Close DeckLink output device
    fn decklink_output_close();

    /// Get the number of buffered frames waiting to be displayed
    fn decklink_output_buffered_frame_count() -> u32;
}

/// Internal keying output stage
pub struct InternalKeyingOutput {
    device_index: i32,
    width: i32,
    height: i32,
    fps_num: i32,
    fps_den: i32,
    is_open: bool,
    is_playing: bool,
}

impl InternalKeyingOutput {
    /// Create a new internal keying output stage
    /// 
    /// # Arguments
    /// * `device_index` - Index of the DeckLink device (usually 0)
    /// * `width` - Video width (e.g., 1920, 3840)
    /// * `height` - Video height (e.g., 1080, 2160)
    /// * `fps_num` - Frame rate numerator (e.g., 60, 30)
    /// * `fps_den` - Frame rate denominator (typically 1)
    pub fn new(device_index: i32, width: i32, height: i32, fps_num: i32, fps_den: i32) -> Self {
        Self {
            device_index,
            width,
            height,
            fps_num,
            fps_den,
            is_open: false,
            is_playing: false,
        }
    }

    /// Open the DeckLink device and initialize for internal keying
    pub fn open(&mut self) -> Result<()> {
        if self.is_open {
            return Ok(());
        }

        let success = unsafe {
            decklink_output_open(
                self.device_index,
                self.width,
                self.height,
                self.fps_num,
                self.fps_den,
            )
        };

        if !success {
            bail!(
                "Failed to open DeckLink output device {} for internal keying ({}x{} @ {}/{}fps)",
                self.device_index,
                self.width,
                self.height,
                self.fps_num,
                self.fps_den
            );
        }

        self.is_open = true;
        Ok(())
    }

    /// Submit a fill (YUV) and key (BGRA with alpha) frame pair for internal keying
    /// 
    /// # Arguments
    /// * `fill` - The fill frame (main video in YUV8bit format)
    /// * `key` - The key frame (overlay in BGRA format with alpha channel)
    pub fn submit_keying_frames(
        &mut self,
        fill: &RawFramePacket,
        key: &OverlayFramePacket,
    ) -> Result<()> {
        ensure!(self.is_open, "DeckLink output device is not open");

        // Validate fill frame
        ensure!(
            fill.meta.width == self.width as u32,
            "Fill frame width {} does not match output width {}",
            fill.meta.width,
            self.width
        );
        ensure!(
            fill.meta.height == self.height as u32,
            "Fill frame height {} does not match output height {}",
            fill.meta.height,
            self.height
        );

        // Validate key frame
        ensure!(
            key.from.width == self.width as u32,
            "Key frame width {} does not match output width {}",
            key.from.width,
            self.width
        );
        ensure!(
            key.from.height == self.height as u32,
            "Key frame height {} does not match output height {}",
            key.from.height,
            self.height
        );

        // Get fill frame data (must be on CPU)
        let fill_ptr = match fill.data.loc {
            MemLoc::Cpu => {
                NonNull::new(fill.data.ptr).ok_or_else(|| anyhow!("Fill frame pointer is null"))?
            }
            MemLoc::Gpu { device } => {
                bail!("Fill frame is on GPU (device {}), must be on CPU for output", device);
            }
        };

        // Get key frame data (must be on CPU)
        let key_ptr = match key.argb.loc {
            MemLoc::Cpu => {
                NonNull::new(key.argb.ptr).ok_or_else(|| anyhow!("Key frame pointer is null"))?
            }
            MemLoc::Gpu { device } => {
                bail!("Key frame is on GPU (device {}), must be on CPU for output", device);
            }
        };

        // Ensure we have proper pixel formats
        // Fill should be YUV422_8 but we'll accept any YUV format
        // Key should have alpha channel
        let key_has_alpha = matches!(
            key.from.pixfmt,
            PixelFormat::ARGB8 | PixelFormat::BGRA8
        );
        ensure!(
            key_has_alpha,
            "Key frame must have alpha channel (ARGB8 or BGRA8), got {:?}",
            key.from.pixfmt
        );

        let fill_stride = fill.data.stride as i32;
        let key_stride = key.stride as i32;

        // Submit frames to DeckLink
        let success = unsafe {
            decklink_output_submit_keying(
                fill_ptr.as_ptr(),
                fill_stride,
                key_ptr.as_ptr(),
                key_stride,
            )
        };

        if !success {
            bail!("Failed to submit keying frames to DeckLink output");
        }

        Ok(())
    }

    /// Start playback of scheduled frames
    pub fn start_playback(&mut self) -> Result<()> {
        ensure!(self.is_open, "DeckLink output device is not open");

        if self.is_playing {
            return Ok(());
        }

        let success = unsafe { decklink_output_start_playback() };

        if !success {
            bail!("Failed to start DeckLink output playback");
        }

        self.is_playing = true;
        Ok(())
    }

    /// Stop playback
    pub fn stop_playback(&mut self) -> Result<()> {
        if !self.is_playing {
            return Ok(());
        }

        let success = unsafe { decklink_output_stop_playback() };

        if !success {
            bail!("Failed to stop DeckLink output playback");
        }

        self.is_playing = false;
        Ok(())
    }

    /// Get the number of frames buffered in the DeckLink output queue
    pub fn buffered_frame_count(&self) -> u32 {
        if !self.is_open {
            return 0;
        }
        unsafe { decklink_output_buffered_frame_count() }
    }

    /// Check if the output device is open
    pub fn is_open(&self) -> bool {
        self.is_open
    }

    /// Check if playback is active
    pub fn is_playing(&self) -> bool {
        self.is_playing
    }
}

impl Drop for InternalKeyingOutput {
    fn drop(&mut self) {
        if self.is_playing {
            let _ = self.stop_playback();
        }
        if self.is_open {
            unsafe {
                decklink_output_close();
            }
            self.is_open = false;
        }
    }
}

/// Helper function to convert BGRA to ARGB if needed
/// 
/// DeckLink internal keying typically expects ARGB format, but our overlay
/// is in BGRA. This function converts BGRA -> ARGB by swapping R and B channels.
pub fn bgra_to_argb(bgra_data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let pixel_count = width * height;
    let mut argb = vec![0u8; pixel_count * 4];

    for i in 0..pixel_count {
        let bgra_offset = i * 4;
        let argb_offset = i * 4;

        // BGRA: [B, G, R, A]
        // ARGB: [A, R, G, B]
        argb[argb_offset + 0] = bgra_data[bgra_offset + 3]; // A
        argb[argb_offset + 1] = bgra_data[bgra_offset + 2]; // R
        argb[argb_offset + 2] = bgra_data[bgra_offset + 1]; // G
        argb[argb_offset + 3] = bgra_data[bgra_offset + 0]; // B
    }

    argb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bgra_to_argb_conversion() {
        // Create a simple 2x2 BGRA image
        let bgra = vec![
            // Pixel 0: B=0, G=64, R=128, A=255
            0, 64, 128, 255,
            // Pixel 1: B=32, G=96, R=160, A=200
            32, 96, 160, 200,
            // Pixel 2: B=16, G=80, R=144, A=180
            16, 80, 144, 180,
            // Pixel 3: B=48, G=112, R=176, A=220
            48, 112, 176, 220,
        ];

        let argb = bgra_to_argb(&bgra, 2, 2);

        // Verify conversion
        assert_eq!(argb.len(), 16);
        
        // Pixel 0: A=255, R=128, G=64, B=0
        assert_eq!(argb[0..4], [255, 128, 64, 0]);
        
        // Pixel 1: A=200, R=160, G=96, B=32
        assert_eq!(argb[4..8], [200, 160, 96, 32]);
        
        // Pixel 2: A=180, R=144, G=80, B=16
        assert_eq!(argb[8..12], [180, 144, 80, 16]);
        
        // Pixel 3: A=220, R=176, G=112, B=48
        assert_eq!(argb[12..16], [220, 176, 112, 48]);
    }
}
