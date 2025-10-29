// Rust FFI Bindings for DeckLink Extended API
// 
// This module provides safe Rust wrappers around the C API
// 
// Usage:
//   1. Copy this file to your Rust project
//   2. Build shim_extended.cpp into a shared library
//   3. Link against it in your build.rs

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

// ============================================================================
// Raw C FFI Declarations
// ============================================================================

#[repr(C)]
pub struct DLDeviceList {
    pub names: *mut *mut c_char,
    pub count: i32,
}

#[repr(C)]
pub struct DLDisplayMode {
    pub mode_id: u32,
    pub name: *mut c_char,
    pub width: i32,
    pub height: i32,
    pub frame_duration: i64,
    pub frame_scale: i64,
    pub field_dominance: u32,
    pub flags: u32,
}

#[repr(C)]
pub struct DLDisplayModeList {
    pub modes: *mut DLDisplayMode,
    pub count: i32,
}

#[repr(C)]
#[derive(Debug, Default, Clone)]
pub struct DLDeviceAttributes {
    pub supports_input_format_detection: bool,
    pub supports_internal_keying: bool,
    pub supports_external_keying: bool,
    pub supports_hd_keying: bool,
    pub supports_idle_output: bool,
    pub supports_smpte_level_a_output: bool,
    pub video_input_connections: i64,
    pub video_output_connections: i64,
    pub audio_input_connections: i64,
    pub audio_output_connections: i64,
    pub max_audio_channels: i32,
    pub duplex_mode: i32,
    pub persistent_id: i64,
    pub supports_quad_link_sdi: bool,
    pub supports_dual_link_sdi: bool,
}

#[repr(C)]
pub struct CaptureFrame {
    pub data: *const u8,
    pub width: i32,
    pub height: i32,
    pub row_bytes: i32,
    pub seq: u64,
    pub gpu_data: *const u8,
    pub gpu_row_bytes: i32,
    pub gpu_device: u32,
}

#[link(name = "decklink_shim")]
extern "C" {
    // Device Management
    pub fn decklink_list_devices() -> DLDeviceList;
    pub fn decklink_free_device_list(list: DLDeviceList);
    pub fn decklink_get_device_attributes(device_index: i32, out: *mut DLDeviceAttributes) -> bool;

    // Display Modes
    pub fn decklink_get_output_display_modes(device_index: i32) -> DLDisplayModeList;
    pub fn decklink_free_display_mode_list(list: DLDisplayModeList);

    // Configuration
    pub fn decklink_set_video_input_connection(device_index: i32, connection: i64) -> bool;
    pub fn decklink_set_audio_input_connection(device_index: i32, connection: i64) -> bool;
    pub fn decklink_set_sdi_output_link_configuration(device_index: i32, link_config: i64) -> bool;
    pub fn decklink_set_444_sdi_video_output(device_index: i32, enable: bool) -> bool;
    pub fn decklink_write_configuration_to_preferences(device_index: i32) -> bool;

    // Keyer Control
    pub fn decklink_keyer_enable_internal(device_index: i32) -> bool;
    pub fn decklink_keyer_enable_external(device_index: i32) -> bool;
    pub fn decklink_keyer_disable(device_index: i32) -> bool;
    pub fn decklink_keyer_set_level(device_index: i32, level: u8) -> bool;
    pub fn decklink_keyer_ramp_up(device_index: i32, number_of_frames: u32) -> bool;
    pub fn decklink_keyer_ramp_down(device_index: i32, number_of_frames: u32) -> bool;

    // Constants - Pixel Formats
    pub fn decklink_get_pixel_format_8bit_yuv() -> u32;
    pub fn decklink_get_pixel_format_10bit_yuv() -> u32;
    pub fn decklink_get_pixel_format_8bit_bgra() -> u32;
    pub fn decklink_get_pixel_format_10bit_rgb() -> u32;

    // Constants - Display Modes
    pub fn decklink_get_mode_hd1080p6000() -> u32;
    pub fn decklink_get_mode_hd1080p5994() -> u32;
    pub fn decklink_get_mode_hd1080p50() -> u32;
    pub fn decklink_get_mode_hd720p60() -> u32;
    pub fn decklink_get_mode_4k2160p60() -> u32;

    // Constants - Connections
    pub fn decklink_get_connection_sdi() -> i64;
    pub fn decklink_get_connection_hdmi() -> i64;
    pub fn decklink_get_audio_connection_embedded() -> i64;
    pub fn decklink_get_link_configuration_single_link() -> i64;
    pub fn decklink_get_link_configuration_quad_link() -> i64;

    // Capture/Output (from original shim)
    pub fn decklink_capture_open(device_index: i32) -> bool;
    pub fn decklink_capture_get_frame(out: *mut CaptureFrame) -> bool;
    pub fn decklink_capture_close();
    
    pub fn decklink_output_open(device_index: i32, width: i32, height: i32, fps: f64) -> bool;
    pub fn decklink_output_send_frame(bgra_data: *const u8, width: i32, height: i32) -> bool;
    pub fn decklink_output_start_scheduled_playback() -> bool;
    pub fn decklink_output_close();
}

// ============================================================================
// Safe Rust Wrappers
// ============================================================================

pub struct DeckLinkDevices {
    raw: DLDeviceList,
}

impl DeckLinkDevices {
    pub fn new() -> Self {
        unsafe {
            let raw = decklink_list_devices();
            DeckLinkDevices { raw }
        }
    }

    pub fn count(&self) -> usize {
        self.raw.count as usize
    }

    pub fn get_name(&self, index: usize) -> Option<String> {
        if index >= self.count() {
            return None;
        }
        unsafe {
            let name_ptr = *self.raw.names.add(index);
            if name_ptr.is_null() {
                return None;
            }
            CStr::from_ptr(name_ptr)
                .to_str()
                .ok()
                .map(|s| s.to_owned())
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = String> + '_ {
        (0..self.count()).filter_map(|i| self.get_name(i))
    }
}

impl Drop for DeckLinkDevices {
    fn drop(&mut self) {
        unsafe {
            decklink_free_device_list(self.raw);
        }
    }
}

pub struct DisplayModes {
    raw: DLDisplayModeList,
}

impl DisplayModes {
    pub fn new(device_index: i32) -> Self {
        unsafe {
            let raw = decklink_get_output_display_modes(device_index);
            DisplayModes { raw }
        }
    }

    pub fn count(&self) -> usize {
        self.raw.count as usize
    }

    pub fn get(&self, index: usize) -> Option<DisplayModeInfo> {
        if index >= self.count() {
            return None;
        }
        unsafe {
            let mode = &*self.raw.modes.add(index);
            let name = if mode.name.is_null() {
                "Unknown".to_string()
            } else {
                CStr::from_ptr(mode.name)
                    .to_str()
                    .unwrap_or("Unknown")
                    .to_string()
            };

            Some(DisplayModeInfo {
                mode_id: mode.mode_id,
                name,
                width: mode.width,
                height: mode.height,
                fps: mode.frame_scale as f64 / mode.frame_duration as f64,
                field_dominance: mode.field_dominance,
                flags: mode.flags,
            })
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = DisplayModeInfo> + '_ {
        (0..self.count()).filter_map(|i| self.get(i))
    }
}

impl Drop for DisplayModes {
    fn drop(&mut self) {
        unsafe {
            decklink_free_display_mode_list(self.raw);
        }
    }
}

#[derive(Debug, Clone)]
pub struct DisplayModeInfo {
    pub mode_id: u32,
    pub name: String,
    pub width: i32,
    pub height: i32,
    pub fps: f64,
    pub field_dominance: u32,
    pub flags: u32,
}

pub struct DeckLinkDevice {
    index: i32,
}

impl DeckLinkDevice {
    pub fn new(index: i32) -> Self {
        DeckLinkDevice { index }
    }

    pub fn get_attributes(&self) -> Result<DLDeviceAttributes, String> {
        unsafe {
            let mut attrs = DLDeviceAttributes::default();
            if decklink_get_device_attributes(self.index, &mut attrs) {
                Ok(attrs)
            } else {
                Err("Failed to get device attributes".to_string())
            }
        }
    }

    pub fn get_display_modes(&self) -> DisplayModes {
        DisplayModes::new(self.index)
    }

    // Configuration methods
    pub fn set_video_input(&self, connection: i64) -> Result<(), String> {
        unsafe {
            if decklink_set_video_input_connection(self.index, connection) {
                Ok(())
            } else {
                Err("Failed to set video input connection".to_string())
            }
        }
    }

    pub fn set_audio_input(&self, connection: i64) -> Result<(), String> {
        unsafe {
            if decklink_set_audio_input_connection(self.index, connection) {
                Ok(())
            } else {
                Err("Failed to set audio input connection".to_string())
            }
        }
    }

    pub fn set_sdi_link_config(&self, link_config: i64) -> Result<(), String> {
        unsafe {
            if decklink_set_sdi_output_link_configuration(self.index, link_config) {
                Ok(())
            } else {
                Err("Failed to set SDI link configuration".to_string())
            }
        }
    }

    pub fn set_444_output(&self, enable: bool) -> Result<(), String> {
        unsafe {
            if decklink_set_444_sdi_video_output(self.index, enable) {
                Ok(())
            } else {
                Err("Failed to set 444 output mode".to_string())
            }
        }
    }

    pub fn save_configuration(&self) -> Result<(), String> {
        unsafe {
            if decklink_write_configuration_to_preferences(self.index) {
                Ok(())
            } else {
                Err("Failed to write configuration".to_string())
            }
        }
    }

    // Keyer methods
    pub fn enable_internal_keying(&self) -> Result<(), String> {
        unsafe {
            if decklink_keyer_enable_internal(self.index) {
                Ok(())
            } else {
                Err("Failed to enable internal keying".to_string())
            }
        }
    }

    pub fn enable_external_keying(&self) -> Result<(), String> {
        unsafe {
            if decklink_keyer_enable_external(self.index) {
                Ok(())
            } else {
                Err("Failed to enable external keying".to_string())
            }
        }
    }

    pub fn disable_keying(&self) -> Result<(), String> {
        unsafe {
            if decklink_keyer_disable(self.index) {
                Ok(())
            } else {
                Err("Failed to disable keying".to_string())
            }
        }
    }

    pub fn set_keyer_level(&self, level: u8) -> Result<(), String> {
        unsafe {
            if decklink_keyer_set_level(self.index, level) {
                Ok(())
            } else {
                Err("Failed to set keyer level".to_string())
            }
        }
    }

    pub fn fade_in(&self, frames: u32) -> Result<(), String> {
        unsafe {
            if decklink_keyer_ramp_up(self.index, frames) {
                Ok(())
            } else {
                Err("Failed to fade in".to_string())
            }
        }
    }

    pub fn fade_out(&self, frames: u32) -> Result<(), String> {
        unsafe {
            if decklink_keyer_ramp_down(self.index, frames) {
                Ok(())
            } else {
                Err("Failed to fade out".to_string())
            }
        }
    }
}

// ============================================================================
// Example Usage
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_devices() {
        let devices = DeckLinkDevices::new();
        println!("Found {} DeckLink devices:", devices.count());
        for (i, name) in devices.iter().enumerate() {
            println!("  [{}] {}", i, name);
        }
    }

    #[test]
    fn test_device_attributes() {
        let device = DeckLinkDevice::new(0);
        match device.get_attributes() {
            Ok(attrs) => {
                println!("Device attributes:");
                println!("  Internal keying: {}", attrs.supports_internal_keying);
                println!("  External keying: {}", attrs.supports_external_keying);
                println!("  Max audio channels: {}", attrs.max_audio_channels);
            }
            Err(e) => println!("Error: {}", e),
        }
    }

    #[test]
    fn test_display_modes() {
        let modes = DisplayModes::new(0);
        println!("Available display modes:");
        for (i, mode) in modes.iter().enumerate() {
            println!("  [{}] {} - {}x{} @ {:.2} fps",
                i, mode.name, mode.width, mode.height, mode.fps);
        }
    }

    #[test]
    fn test_keyer_setup() {
        let device = DeckLinkDevice::new(0);
        
        // Enable internal keying
        if let Err(e) = device.enable_internal_keying() {
            println!("Failed to enable keying: {}", e);
            return;
        }
        
        // Set full opacity
        device.set_keyer_level(255).unwrap();
        
        println!("Internal keying enabled successfully");
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

pub mod constants {
    use super::*;

    pub fn pixel_format_8bit_yuv() -> u32 {
        unsafe { decklink_get_pixel_format_8bit_yuv() }
    }

    pub fn pixel_format_8bit_bgra() -> u32 {
        unsafe { decklink_get_pixel_format_8bit_bgra() }
    }

    pub fn mode_1080p60() -> u32 {
        unsafe { decklink_get_mode_hd1080p6000() }
    }

    pub fn mode_720p60() -> u32 {
        unsafe { decklink_get_mode_hd720p60() }
    }

    pub fn connection_hdmi() -> i64 {
        unsafe { decklink_get_connection_hdmi() }
    }

    pub fn connection_sdi() -> i64 {
        unsafe { decklink_get_connection_sdi() }
    }

    pub fn audio_embedded() -> i64 {
        unsafe { decklink_get_audio_connection_embedded() }
    }

    pub fn link_quad() -> i64 {
        unsafe { decklink_get_link_configuration_quad_link() }
    }
}
