// DeckLink Extended API - C Header for Rust FFI
// To use: Copy this to your Rust project and use with bindgen or manual FFI

#ifndef DECKLINK_EXTENDED_H
#define DECKLINK_EXTENDED_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Structures
// ============================================================================

typedef struct DLDeviceList {
    char** names;
    int32_t count;
} DLDeviceList;

typedef struct DLDisplayMode {
    uint32_t mode_id;
    char* name;
    int32_t width;
    int32_t height;
    int64_t frame_duration;
    int64_t frame_scale;
    uint32_t field_dominance;
    uint32_t flags;
} DLDisplayMode;

typedef struct DLDisplayModeList {
    DLDisplayMode* modes;
    int32_t count;
} DLDisplayModeList;

typedef struct DLDeviceAttributes {
    bool supports_input_format_detection;
    bool supports_internal_keying;
    bool supports_external_keying;
    bool supports_hd_keying;
    bool supports_idle_output;
    bool supports_smpte_level_a_output;
    int64_t video_input_connections;
    int64_t video_output_connections;
    int64_t audio_input_connections;
    int64_t audio_output_connections;
    int32_t max_audio_channels;
    int32_t duplex_mode;
    int64_t persistent_id;
    bool supports_quad_link_sdi;
    bool supports_dual_link_sdi;
} DLDeviceAttributes;

typedef struct CaptureFrame {
    const uint8_t* data;
    int32_t width;
    int32_t height;
    int32_t row_bytes;
    uint64_t seq;
    const uint8_t* gpu_data;
    int32_t gpu_row_bytes;
    uint32_t gpu_device;
} CaptureFrame;

// ============================================================================
// Device Discovery and Management
// ============================================================================

DLDeviceList decklink_list_devices(void);
void decklink_free_device_list(DLDeviceList list);
bool decklink_get_device_attributes(int32_t device_index, DLDeviceAttributes* out);

// ============================================================================
// Display Mode Management
// ============================================================================

DLDisplayModeList decklink_get_output_display_modes(int32_t device_index);
void decklink_free_display_mode_list(DLDisplayModeList list);

// ============================================================================
// Configuration Management
// ============================================================================

bool decklink_set_video_input_connection(int32_t device_index, int64_t connection);
bool decklink_set_audio_input_connection(int32_t device_index, int64_t connection);
bool decklink_set_sdi_output_link_configuration(int32_t device_index, int64_t link_config);
bool decklink_set_444_sdi_video_output(int32_t device_index, bool enable);
bool decklink_write_configuration_to_preferences(int32_t device_index);

// ============================================================================
// Keyer Control Functions
// ============================================================================

bool decklink_keyer_enable_internal(int32_t device_index);
bool decklink_keyer_enable_external(int32_t device_index);
bool decklink_keyer_disable(int32_t device_index);
bool decklink_keyer_set_level(int32_t device_index, uint8_t level);
bool decklink_keyer_ramp_up(int32_t device_index, uint32_t number_of_frames);
bool decklink_keyer_ramp_down(int32_t device_index, uint32_t number_of_frames);

// ============================================================================
// Pixel Format Constants
// ============================================================================

uint32_t decklink_get_pixel_format_8bit_yuv(void);
uint32_t decklink_get_pixel_format_10bit_yuv(void);
uint32_t decklink_get_pixel_format_8bit_argb(void);
uint32_t decklink_get_pixel_format_8bit_bgra(void);
uint32_t decklink_get_pixel_format_10bit_rgb(void);
uint32_t decklink_get_pixel_format_12bit_rgb(void);
uint32_t decklink_get_pixel_format_12bit_rgble(void);
uint32_t decklink_get_pixel_format_10bit_rgbxle(void);
uint32_t decklink_get_pixel_format_10bit_rgbx(void);

// ============================================================================
// Display Mode Constants
// ============================================================================

uint32_t decklink_get_mode_ntsc(void);
uint32_t decklink_get_mode_pal(void);
uint32_t decklink_get_mode_hd1080p2398(void);
uint32_t decklink_get_mode_hd1080p24(void);
uint32_t decklink_get_mode_hd1080p25(void);
uint32_t decklink_get_mode_hd1080p2997(void);
uint32_t decklink_get_mode_hd1080p30(void);
uint32_t decklink_get_mode_hd1080p50(void);
uint32_t decklink_get_mode_hd1080p5994(void);
uint32_t decklink_get_mode_hd1080p6000(void);
uint32_t decklink_get_mode_hd720p50(void);
uint32_t decklink_get_mode_hd720p5994(void);
uint32_t decklink_get_mode_hd720p60(void);
uint32_t decklink_get_mode_2k2398(void);
uint32_t decklink_get_mode_2k24(void);
uint32_t decklink_get_mode_2k25(void);
uint32_t decklink_get_mode_4k2160p2398(void);
uint32_t decklink_get_mode_4k2160p24(void);
uint32_t decklink_get_mode_4k2160p25(void);
uint32_t decklink_get_mode_4k2160p2997(void);
uint32_t decklink_get_mode_4k2160p30(void);
uint32_t decklink_get_mode_4k2160p50(void);
uint32_t decklink_get_mode_4k2160p5994(void);
uint32_t decklink_get_mode_4k2160p60(void);

// ============================================================================
// Video Connection Constants
// ============================================================================

int64_t decklink_get_connection_sdi(void);
int64_t decklink_get_connection_hdmi(void);
int64_t decklink_get_connection_optical_sdi(void);
int64_t decklink_get_connection_component(void);
int64_t decklink_get_connection_composite(void);
int64_t decklink_get_connection_svideo(void);

// ============================================================================
// Audio Connection Constants
// ============================================================================

int64_t decklink_get_audio_connection_embedded(void);
int64_t decklink_get_audio_connection_aesebu(void);
int64_t decklink_get_audio_connection_analog(void);
int64_t decklink_get_audio_connection_analog_xlr(void);
int64_t decklink_get_audio_connection_analog_rca(void);

// ============================================================================
// Link Configuration Constants
// ============================================================================

int64_t decklink_get_link_configuration_single_link(void);
int64_t decklink_get_link_configuration_dual_link(void);
int64_t decklink_get_link_configuration_quad_link(void);

// ============================================================================
// Capture API (from original shim.cpp)
// ============================================================================

bool decklink_capture_open(int32_t device_index);
bool decklink_capture_get_frame(CaptureFrame* out);
void decklink_capture_close(void);
bool decklink_capture_copy_host_region(size_t offset, size_t len, void* dst);

// ============================================================================
// Output API (from original shim.cpp)
// ============================================================================

bool decklink_output_open(int32_t device_index, int32_t width, int32_t height, double fps);
bool decklink_output_send_frame(const uint8_t* bgra_data, int32_t width, int32_t height);
bool decklink_output_send_frame_gpu(const uint8_t* gpu_bgra_data, int32_t gpu_pitch, int32_t width, int32_t height);
bool decklink_output_start_scheduled_playback(void);
void decklink_output_close(void);

// Note: Original output keyer functions are now replaced by device-based keyer functions above

// ============================================================================
// OpenGL Preview API (from original shim.cpp)
// ============================================================================

bool decklink_preview_gl_create(void);
bool decklink_preview_gl_initialize_gl(void);
bool decklink_preview_gl_enable(void);
bool decklink_preview_gl_render(void);
void decklink_preview_gl_disable(void);
void decklink_preview_gl_destroy(void);
uint64_t decklink_preview_gl_seq(void);
uint64_t decklink_preview_gl_last_timestamp_ns(void);
uint64_t decklink_preview_gl_last_latency_ns(void);

#ifdef __cplusplus
}
#endif

#endif // DECKLINK_EXTENDED_H
