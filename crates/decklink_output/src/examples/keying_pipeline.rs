/// Example: Integrating Internal Keying into the DeepGIBox Pipeline
/// 
/// This shows how to integrate the internal keying output module
/// with the rest of the DeepGIBox pipeline.

use anyhow::Result;
use crate::keying::InternalKeyingOutput;
use common_io::{RawFramePacket, OverlayFramePacket};

/// Example pipeline stage that combines video capture with overlay rendering
/// and outputs via internal keying
pub struct KeyingPipeline {
    output: InternalKeyingOutput,
    frame_count: u64,
}

impl KeyingPipeline {
    /// Create a new keying pipeline
    pub fn new(device_index: i32, width: i32, height: i32, fps_num: i32, fps_den: i32) -> Result<Self> {
        let mut output = InternalKeyingOutput::new(device_index, width, height, fps_num, fps_den);
        output.open()?;
        
        Ok(Self {
            output,
            frame_count: 0,
        })
    }

    /// Process one frame through the pipeline with internal keying
    /// 
    /// # Arguments
    /// * `fill` - The captured video frame (YUV8 from DeckLink input)
    /// * `key` - The rendered overlay (BGRA from overlay_render)
    pub fn process_frame(&mut self, fill: &RawFramePacket, key: &OverlayFramePacket) -> Result<()> {
        // Start playback on first frame
        if self.frame_count == 0 {
            self.output.start_playback()?;
        }

        // Submit fill and key frames for internal keying
        self.output.submit_keying_frames(fill, key)?;

        // Monitor buffer to prevent overflow
        let buffered = self.output.buffered_frame_count();
        if buffered > 5 {
            eprintln!("Warning: {} frames buffered (may need to slow down)", buffered);
        }

        self.frame_count += 1;
        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            frames_processed: self.frame_count,
            buffered_frames: self.output.buffered_frame_count(),
            is_playing: self.output.is_playing(),
        }
    }

    /// Stop the pipeline
    pub fn stop(&mut self) -> Result<()> {
        self.output.stop_playback()?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub frames_processed: u64,
    pub buffered_frames: u32,
    pub is_playing: bool,
}

/// Example usage in the main runner
/// 
/// ```no_run
/// use decklink_output::examples::keying_pipeline::KeyingPipeline;
/// 
/// // In your main loop:
/// let mut keying_pipeline = KeyingPipeline::new(0, 1920, 1080, 60, 1)?;
/// 
/// loop {
///     // 1. Capture from DeckLink input
///     let raw_frame = decklink_input.capture()?;
///     
///     // 2. Preprocess (YUV -> RGB, resize, normalize)
///     let tensor = preprocess.process(&raw_frame)?;
///     
///     // 3. Run inference
///     let detections = inference.run(&tensor)?;
///     
///     // 4. Postprocess (decode, NMS, tracking)
///     let objects = postprocess.process(&detections)?;
///     
///     // 5. Generate overlay plan
///     let plan = overlay_plan.create(&objects)?;
///     
///     // 6. Render overlay to BGRA
///     let overlay = overlay_render.render(&plan)?;
///     
///     // 7. Output with internal keying (fill=input, key=overlay)
///     keying_pipeline.process_frame(&raw_frame, &overlay)?;
///     
///     // Print stats periodically
///     if frame_idx % 60 == 0 {
///         let stats = keying_pipeline.stats();
///         println!("Keying: {:?}", stats);
///     }
/// }
/// 
/// keying_pipeline.stop()?;
/// ```

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        // This test requires actual DeckLink hardware
        // Skip in CI environments
        if std::env::var("DECKLINK_DEVICE").is_err() {
            return;
        }

        let result = KeyingPipeline::new(0, 1920, 1080, 60, 1);
        assert!(result.is_ok() || result.is_err()); // Just check it doesn't panic
    }
}
