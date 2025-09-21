use anyhow::{bail, Result};

extern "C" {
    fn decklink_preview_gl_create() -> bool;
    fn decklink_preview_gl_initialize_gl() -> bool;
    fn decklink_preview_gl_enable() -> bool;
    fn decklink_preview_gl_render() -> bool;
    fn decklink_preview_gl_disable();
    fn decklink_preview_gl_destroy();
    fn decklink_preview_gl_seq() -> u64;
    fn decklink_preview_gl_last_timestamp_ns() -> u64;
    fn decklink_preview_gl_last_latency_ns() -> u64;
}

/// Wrapper around the global DeckLink OpenGL preview helper.
pub struct PreviewGl {
    initialized: bool,
}

impl PreviewGl {
    pub fn create() -> Result<Self> {
        unsafe {
            if decklink_preview_gl_create() {
                Ok(Self { initialized: false })
            } else {
                bail!("CreateOpenGLScreenPreviewHelper failed");
            }
        }
    }

    pub fn initialize_gl(&mut self) -> Result<()> {
        unsafe {
            if decklink_preview_gl_initialize_gl() {
                self.initialized = true;
                Ok(())
            } else {
                bail!("InitializeGL failed");
            }
        }
    }

    pub fn enable(&self) -> Result<()> {
        if !self.initialized {
            bail!("Preview GL must be initialized before enabling");
        }
        unsafe {
            if decklink_preview_gl_enable() {
                Ok(())
            } else {
                bail!("Failed to enable GL preview callback");
            }
        }
    }

    pub fn render(&self) -> bool {
        unsafe { decklink_preview_gl_render() }
    }

    pub fn seq(&self) -> u64 {
        unsafe { decklink_preview_gl_seq() }
    }

    pub fn last_timestamp_ns(&self) -> u64 {
        unsafe { decklink_preview_gl_last_timestamp_ns() }
    }

    pub fn last_latency_ns(&self) -> u64 {
        unsafe { decklink_preview_gl_last_latency_ns() }
    }

    pub fn disable(&self) {
        unsafe {
            decklink_preview_gl_disable();
        }
    }
}

impl Drop for PreviewGl {
    fn drop(&mut self) {
        unsafe {
            decklink_preview_gl_destroy();
        }
    }
}
