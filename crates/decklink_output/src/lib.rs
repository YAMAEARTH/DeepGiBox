use anyhow::{anyhow, bail, ensure, Context, Result};
use common_io::{FrameMeta, MemLoc, OverlayFramePacket, PixelFormat, RawFramePacket};
use config::AppConfig;
use std::ptr::NonNull;
use std::slice;

pub mod keying;
pub mod examples;

pub fn from_path(cfg: &str) -> Result<OutputStage> {
    let config = AppConfig::from_file(cfg)?;
    Ok(OutputStage {
        _config: config,
        last_frame: None,
    })
}

pub struct OutputStage {
    _config: AppConfig,
    last_frame: Option<OutputFrame>,
}

pub struct OutputFrame {
    pub video: Option<VideoFrame>,
    pub overlay: Option<OverlayArgbFrame>,
}

pub struct VideoFrame {
    pub meta: FrameMeta,
    pub bytes: Vec<u8>,
    pub stride: usize,
}

pub struct OverlayArgbFrame {
    pub meta: FrameMeta,
    pub argb: Vec<u8>,
    pub stride: usize,
}

pub struct OutputRequest<'a> {
    pub video: Option<&'a RawFramePacket>,
    pub overlay: Option<&'a OverlayFramePacket>,
}

impl OutputStage {
    pub fn submit(&mut self, request: OutputRequest<'_>) -> Result<()> {
        let video = if let Some(video_packet) = request.video {
            Some(Self::copy_video_frame(video_packet).context("failed to prepare input frame")?)
        } else {
            None
        };

        let overlay = if let Some(overlay_packet) = request.overlay {
            Some(
                Self::prepare_overlay_frame(overlay_packet)
                    .context("failed to prepare overlay frame")?,
            )
        } else {
            None
        };

        ensure!(
            video.is_some() || overlay.is_some(),
            "output request contains neither input video nor overlay frame"
        );

        self.last_frame = Some(OutputFrame { video, overlay });
        Ok(())
    }

    pub fn last_frame(&self) -> Option<&OutputFrame> {
        self.last_frame.as_ref()
    }

    pub fn take_last_frame(&mut self) -> Option<OutputFrame> {
        self.last_frame.take()
    }

    fn copy_video_frame(packet: &RawFramePacket) -> Result<VideoFrame> {
        ensure!(
            packet.meta.width > 0 && packet.meta.height > 0,
            "input frame has invalid dimensions {}x{}",
            packet.meta.width,
            packet.meta.height
        );

        ensure!(
            packet.data.stride > 0,
            "input frame stride is zero"
        );

        match packet.data.loc {
            MemLoc::Cpu => {}
            MemLoc::Gpu { device } => {
                bail!("GPU input frames (device {device}) are not supported yet");
            }
        }

        let ptr = NonNull::new(packet.data.ptr)
            .ok_or_else(|| anyhow!("input frame buffer pointer is null"))?;
        let height = packet.meta.height as usize;
        let stride = packet.data.stride;
        let required = stride
            .checked_mul(height)
            .ok_or_else(|| anyhow!("input frame size overflow"))?;
        ensure!(
            packet.data.len >= required,
            "input buffer length {} smaller than required {}",
            packet.data.len,
            required
        );

        let src = unsafe { slice::from_raw_parts(ptr.as_ptr(), required) };
        let mut bytes = vec![0u8; required];
        bytes.copy_from_slice(src);

        let mut meta = packet.meta.clone();
        meta.stride_bytes = stride as u32;

        Ok(VideoFrame {
            meta,
            bytes,
            stride,
        })
    }

    fn prepare_overlay_frame(frame: &OverlayFramePacket) -> Result<OverlayArgbFrame> {
        ensure!(
            frame.from.width > 0 && frame.from.height > 0,
            "overlay frame has invalid dimensions {}x{}",
            frame.from.width,
            frame.from.height
        );
        ensure!(frame.stride > 0, "overlay frame stride is zero");

        match frame.argb.loc {
            MemLoc::Cpu => {}
            MemLoc::Gpu { device } => {
                bail!("GPU overlay frames (device {device}) are not supported yet");
            }
        }

        let ptr = NonNull::new(frame.argb.ptr)
            .ok_or_else(|| anyhow!("overlay frame buffer pointer is null"))?;

        match frame.from.pixfmt {
            PixelFormat::ARGB8 => Self::copy_argb(frame, ptr),
            PixelFormat::RGB8 => Self::convert_rgb_to_argb(frame, ptr),
            other => Err(anyhow!(
                "unsupported overlay pixel format {:?} (expected ARGB8 or RGB8)",
                other
            )),
        }
    }

    fn copy_argb(frame: &OverlayFramePacket, ptr: NonNull<u8>) -> Result<OverlayArgbFrame> {
        let width = frame.from.width as usize;
        let height = frame.from.height as usize;
        let row_argb = width
            .checked_mul(4)
            .ok_or_else(|| anyhow!("overlay width too large"))?;
        ensure!(
            frame.stride >= row_argb,
            "overlay stride {} smaller than expected row bytes {}",
            frame.stride,
            row_argb
        );

        let required = frame
            .stride
            .checked_mul(height)
            .ok_or_else(|| anyhow!("overlay frame size overflow"))?;
        ensure!(
            frame.argb.len >= required,
            "overlay buffer length {} smaller than required {}",
            frame.argb.len,
            required
        );

        let src = unsafe { slice::from_raw_parts(ptr.as_ptr(), required) };
        let mut argb = vec![0u8; row_argb * height];
        for row in 0..height {
            let src_row = &src[row * frame.stride..row * frame.stride + row_argb];
            let dst_row = &mut argb[row * row_argb..(row + 1) * row_argb];
            dst_row.copy_from_slice(src_row);
        }

        Ok(OverlayArgbFrame {
            meta: updated_meta(frame.from.clone(), row_argb as u32),
            argb,
            stride: row_argb,
        })
    }

    fn convert_rgb_to_argb(
        frame: &OverlayFramePacket,
        ptr: NonNull<u8>,
    ) -> Result<OverlayArgbFrame> {
        let width = frame.from.width as usize;
        let height = frame.from.height as usize;
        let row_rgb = width
            .checked_mul(3)
            .ok_or_else(|| anyhow!("overlay width too large"))?;
        ensure!(
            frame.stride >= row_rgb,
            "RGB overlay stride {} smaller than row bytes {}",
            frame.stride,
            row_rgb
        );

        let required = frame
            .stride
            .checked_mul(height)
            .ok_or_else(|| anyhow!("RGB overlay frame size overflow"))?;
        ensure!(
            frame.argb.len >= required,
            "RGB overlay buffer length {} smaller than required {}",
            frame.argb.len,
            required
        );

        let src = unsafe { slice::from_raw_parts(ptr.as_ptr(), required) };
        let row_argb = width * 4;
        let mut argb = vec![0u8; row_argb * height];

        for row in 0..height {
            let src_row = &src[row * frame.stride..row * frame.stride + row_rgb];
            let mut dst_idx = row * row_argb;
            for px in 0..width {
                let base = px * 3;
                argb[dst_idx] = 0xFF;
                argb[dst_idx + 1] = src_row[base];
                argb[dst_idx + 2] = src_row[base + 1];
                argb[dst_idx + 3] = src_row[base + 2];
                dst_idx += 4;
            }
        }

        Ok(OverlayArgbFrame {
            meta: updated_meta(frame.from.clone(), row_argb as u32),
            argb,
            stride: row_argb,
        })
    }
}

fn updated_meta(mut meta: FrameMeta, stride_bytes: u32) -> FrameMeta {
    meta.pixfmt = PixelFormat::ARGB8;
    meta.stride_bytes = stride_bytes;
    meta
}
