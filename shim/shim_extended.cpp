// Extended DeckLink C++ Shim for Rust FFI
// Comprehensive API wrapper with all common DeckLink features

#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <cstdio>
#include <chrono>
#include <thread>
#include <cstdint>
#include <cuda_runtime_api.h>
#include "dvpapi_cuda.h"

// Include DeckLink SDK headers
#include "DeckLinkAPI.h"
#include "DeckLinkAPIModes.h"
#include "DeckLinkAPITypes.h"
#include "DeckLinkAPIConfiguration.h"
#include "DeckLinkAPIVideoFrame_v14_2_1.h"
#include "DeckLinkAPIVideoInput_v14_2_1.h"
#include "DeckLinkAPIVideoOutput_v14_2_1.h"
#include "DeckLinkAPIScreenPreviewCallback_v14_2_1.h"
#include "DeckLinkAPIMemoryAllocator_v14_2_1.h"

static const REFIID kIID_IUnknown = IID_IUnknown;

// ============================================================================
// Video Output Callback for Scheduled Playback
// ============================================================================

// Global timing configuration for scheduled playback
static std::atomic<int64_t> g_timeScale{60000};       // Default 60000 for 59.94fps
static std::atomic<int64_t> g_frameDuration{1001};    // Default 1001 for 59.94fps
static std::atomic<uint64_t> g_output_frame_count{0}; // Frame counter
static std::atomic<bool> g_playback_started{false};   // Track playback state
static const uint64_t PREROLL_FRAME_COUNT = 3;        // Minimum frames before starting playback

class VideoOutputCallback : public IDeckLinkVideoOutputCallback_v14_2_1 {
private:
    std::atomic<ULONG> m_refCount;
    
public:
    VideoOutputCallback() : m_refCount(1) {}
    virtual ~VideoOutputCallback() {}
    
    // IUnknown methods
    virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, LPVOID *ppv) override {
        if (!ppv) return E_INVALIDARG;
        
        CFUUIDBytes unknown = CFUUIDGetUUIDBytes(IID_IUnknown);
        CFUUIDBytes iid_bytes = CFUUIDGetUUIDBytes(iid);
        CFUUIDBytes callback_bytes = CFUUIDGetUUIDBytes(IID_IDeckLinkVideoOutputCallback_v14_2_1);
        
        if (memcmp(&iid_bytes, &unknown, sizeof(CFUUIDBytes)) == 0) {
            *ppv = static_cast<IUnknown*>(this);
            AddRef();
            return S_OK;
        }
        if (memcmp(&iid_bytes, &callback_bytes, sizeof(CFUUIDBytes)) == 0) {
            *ppv = static_cast<IDeckLinkVideoOutputCallback_v14_2_1*>(this);
            AddRef();
            return S_OK;
        }
        
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    
    virtual ULONG STDMETHODCALLTYPE AddRef() override {
        return ++m_refCount;
    }
    
    virtual ULONG STDMETHODCALLTYPE Release() override {
        ULONG newRefCount = --m_refCount;
        if (newRefCount == 0) {
            delete this;
        }
        return newRefCount;
    }
    
    // IDeckLinkVideoOutputCallback_v14_2_1 methods
    virtual HRESULT STDMETHODCALLTYPE ScheduledFrameCompleted(
        IDeckLinkVideoFrame_v14_2_1* /* completedFrame */,
        BMDOutputFrameCompletionResult /* result */) override 
    {
        // Frame completed - application can schedule next frame here
        return S_OK;
    }
    
    virtual HRESULT STDMETHODCALLTYPE ScheduledPlaybackHasStopped() override {
        fprintf(stderr, "[shim_extended] Scheduled playback has stopped\n");
        return S_OK;
    }
};

// Global callback instance
static VideoOutputCallback* g_outputCallback = nullptr;

extern "C" {

// ============================================================================
// Common Types and Structures
// ============================================================================

struct DLDeviceList {
    char** names;
    int32_t count;
};

struct DLDisplayMode {
    uint32_t mode_id;
    char* name;
    int32_t width;
    int32_t height;
    int64_t frame_duration;
    int64_t frame_scale;
    uint32_t field_dominance;
    uint32_t flags;
};

struct DLDisplayModeList {
    DLDisplayMode* modes;
    int32_t count;
};

struct DLDeviceAttributes {
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
};

struct CaptureFrame {
    const uint8_t* data;
    int32_t width;
    int32_t height;
    int32_t row_bytes;
    uint64_t seq;
    const uint8_t* gpu_data;
    int32_t gpu_row_bytes;
    uint32_t gpu_device;
};

// ============================================================================
// Helper Functions
// ============================================================================

static char* dup_utf8_from_bmd_str(const char* s) {
    if (!s) return strdup("(unknown)");
    return strdup(s);
}

static void release_bmd_str(const char* s) {
    if (s) std::free((void*)s);
}

// ============================================================================
// Device Discovery and Management
// ============================================================================

// Note: decklink_list_devices() and decklink_free_device_list() 
// are already defined in shim.cpp, so we don't re-define them here

/*
DLDeviceList decklink_list_devices() {
    DLDeviceList out{nullptr, 0};

    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) {
        fprintf(stderr, "[shim] Failed to create DeckLink iterator\n");
        return out;
    }

    std::vector<char*> names;
    IDeckLink* deckLink = nullptr;

    while (it->Next(&deckLink) == S_OK && deckLink) {
        const char* bname = nullptr;
        if (deckLink->GetDisplayName(&bname) == S_OK) {
            char* cstr = dup_utf8_from_bmd_str(bname);
            release_bmd_str(bname);
            if (cstr) names.push_back(cstr);
        }
        deckLink->Release();
    }
    it->Release();

    if (names.empty()) {
        return out;
    }

    out.count = static_cast<int32_t>(names.size());
    out.names = (char**)std::malloc(sizeof(char*) * (size_t)out.count);
    if (!out.names) {
        for (char* s : names) std::free(s);
        out.count = 0;
        return out;
    }
    for (int32_t i = 0; i < out.count; ++i) {
        out.names[i] = names[(size_t)i];
    }
    return out;
}

void decklink_free_device_list(DLDeviceList list) {
    if (list.names) {
        for (int32_t i = 0; i < list.count; ++i) {
            if (list.names[i]) std::free(list.names[i]);
        }
        std::free(list.names);
    }
}
*/

bool decklink_get_device_attributes(int32_t device_index, DLDeviceAttributes* out) {
    if (!out) return false;

    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    IDeckLinkProfileAttributes* attr = nullptr;
    if (targetDev->QueryInterface(IID_IDeckLinkProfileAttributes, (void**)&attr) != S_OK) {
        targetDev->Release();
        return false;
    }

    // Query all attributes
    bool bval;
    int64_t ival;

    out->supports_input_format_detection = (attr->GetFlag(BMDDeckLinkSupportsInputFormatDetection, &bval) == S_OK) && bval;
    out->supports_internal_keying = (attr->GetFlag(BMDDeckLinkSupportsInternalKeying, &bval) == S_OK) && bval;
    out->supports_external_keying = (attr->GetFlag(BMDDeckLinkSupportsExternalKeying, &bval) == S_OK) && bval;
    out->supports_hd_keying = false;  // BMDDeckLinkSupportsHDKeying not available in this SDK version
    out->supports_idle_output = (attr->GetFlag(BMDDeckLinkSupportsIdleOutput, &bval) == S_OK) && bval;
    out->supports_smpte_level_a_output = (attr->GetFlag(BMDDeckLinkSupportsSMPTELevelAOutput, &bval) == S_OK) && bval;
    out->supports_quad_link_sdi = (attr->GetFlag(BMDDeckLinkSupportsQuadLinkSDI, &bval) == S_OK) && bval;
    out->supports_dual_link_sdi = (attr->GetFlag(BMDDeckLinkSupportsDualLinkSDI, &bval) == S_OK) && bval;

    out->video_input_connections = (attr->GetInt(BMDDeckLinkVideoInputConnections, &ival) == S_OK) ? ival : 0;
    out->video_output_connections = (attr->GetInt(BMDDeckLinkVideoOutputConnections, &ival) == S_OK) ? ival : 0;
    out->audio_input_connections = (attr->GetInt(BMDDeckLinkAudioInputConnections, &ival) == S_OK) ? ival : 0;
    out->audio_output_connections = (attr->GetInt(BMDDeckLinkAudioOutputConnections, &ival) == S_OK) ? ival : 0;
    out->max_audio_channels = (attr->GetInt(BMDDeckLinkMaximumAudioChannels, &ival) == S_OK) ? (int32_t)ival : 0;
    out->duplex_mode = (attr->GetInt(BMDDeckLinkDuplex, &ival) == S_OK) ? (int32_t)ival : 0;
    out->persistent_id = (attr->GetInt(BMDDeckLinkPersistentID, &ival) == S_OK) ? ival : -1;

    attr->Release();
    targetDev->Release();
    return true;
}

// ============================================================================
// Display Mode Management
// ============================================================================

DLDisplayModeList decklink_get_output_display_modes(int32_t device_index) {
    DLDisplayModeList out{nullptr, 0};

    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return out;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return out;

    IDeckLinkOutput_v14_2_1* output = nullptr;
    if (targetDev->QueryInterface(IID_IDeckLinkOutput_v14_2_1, (void**)&output) != S_OK) {
        targetDev->Release();
        return out;
    }

    IDeckLinkDisplayModeIterator* mode_it = nullptr;
    if (output->GetDisplayModeIterator(&mode_it) != S_OK || !mode_it) {
        output->Release();
        targetDev->Release();
        return out;
    }

    std::vector<DLDisplayMode> modes;
    IDeckLinkDisplayMode* mode = nullptr;

    while (mode_it->Next(&mode) == S_OK && mode) {
        DLDisplayMode dm;
        dm.mode_id = mode->GetDisplayMode();
        
        const char* name_str = nullptr;
        if (mode->GetName(&name_str) == S_OK) {
            dm.name = dup_utf8_from_bmd_str(name_str);
            release_bmd_str(name_str);
        } else {
            dm.name = strdup("Unknown");
        }

        dm.width = mode->GetWidth();
        dm.height = mode->GetHeight();
        mode->GetFrameRate(&dm.frame_duration, &dm.frame_scale);
        dm.field_dominance = mode->GetFieldDominance();
        dm.flags = mode->GetFlags();

        modes.push_back(dm);
        mode->Release();
    }

    mode_it->Release();
    output->Release();
    targetDev->Release();

    if (modes.empty()) return out;

    out.count = static_cast<int32_t>(modes.size());
    out.modes = (DLDisplayMode*)std::malloc(sizeof(DLDisplayMode) * modes.size());
    if (!out.modes) {
        for (auto& m : modes) std::free(m.name);
        out.count = 0;
        return out;
    }

    for (size_t i = 0; i < modes.size(); ++i) {
        out.modes[i] = modes[i];
    }

    return out;
}

void decklink_free_display_mode_list(DLDisplayModeList list) {
    if (list.modes) {
        for (int32_t i = 0; i < list.count; ++i) {
            if (list.modes[i].name) std::free(list.modes[i].name);
        }
        std::free(list.modes);
    }
}

// ============================================================================
// Configuration Management
// ============================================================================

bool decklink_set_video_input_connection(int32_t device_index, int64_t connection) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    IDeckLinkConfiguration* cfg = nullptr;
    bool success = false;

    if (targetDev->QueryInterface(IID_IDeckLinkConfiguration, (void**)&cfg) == S_OK && cfg) {
        success = (cfg->SetInt(bmdDeckLinkConfigVideoInputConnection, connection) == S_OK);
        cfg->Release();
    }

    targetDev->Release();
    return success;
}

bool decklink_set_audio_input_connection(int32_t device_index, int64_t connection) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    IDeckLinkConfiguration* cfg = nullptr;
    bool success = false;

    if (targetDev->QueryInterface(IID_IDeckLinkConfiguration, (void**)&cfg) == S_OK && cfg) {
        success = (cfg->SetInt(bmdDeckLinkConfigAudioInputConnection, connection) == S_OK);
        cfg->Release();
    }

    targetDev->Release();
    return success;
}

bool decklink_set_sdi_output_link_configuration(int32_t device_index, int64_t link_config) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    IDeckLinkConfiguration* cfg = nullptr;
    bool success = false;

    if (targetDev->QueryInterface(IID_IDeckLinkConfiguration, (void**)&cfg) == S_OK && cfg) {
        success = (cfg->SetInt(bmdDeckLinkConfigSDIOutputLinkConfiguration, link_config) == S_OK);
        cfg->Release();
    }

    targetDev->Release();
    return success;
}

bool decklink_set_444_sdi_video_output(int32_t device_index, bool enable) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    IDeckLinkConfiguration* cfg = nullptr;
    bool success = false;

    if (targetDev->QueryInterface(IID_IDeckLinkConfiguration, (void**)&cfg) == S_OK && cfg) {
        success = (cfg->SetFlag(bmdDeckLinkConfig444SDIVideoOutput, enable) == S_OK);
        cfg->Release();
    }

    targetDev->Release();
    return success;
}

bool decklink_write_configuration_to_preferences(int32_t device_index) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    IDeckLinkConfiguration* cfg = nullptr;
    bool success = false;

    if (targetDev->QueryInterface(IID_IDeckLinkConfiguration, (void**)&cfg) == S_OK && cfg) {
        success = (cfg->WriteConfigurationToPreferences() == S_OK);
        cfg->Release();
    }

    targetDev->Release();
    return success;
}

// ============================================================================
// Keyer Control Functions
// ============================================================================

bool decklink_keyer_enable_internal(int32_t device_index) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    IDeckLinkKeyer* keyer = nullptr;
    bool success = false;

    if (targetDev->QueryInterface(IID_IDeckLinkKeyer, (void**)&keyer) == S_OK && keyer) {
        success = (keyer->Enable(true) == S_OK);
        if (success) {
            success = (keyer->SetLevel(255) == S_OK);
        }
        keyer->Release();
    }

    targetDev->Release();
    return success;
}

bool decklink_keyer_enable_external(int32_t device_index) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    IDeckLinkKeyer* keyer = nullptr;
    bool success = false;

    if (targetDev->QueryInterface(IID_IDeckLinkKeyer, (void**)&keyer) == S_OK && keyer) {
        success = (keyer->Enable(false) == S_OK);
        keyer->Release();
    }

    targetDev->Release();
    return success;
}

bool decklink_keyer_disable(int32_t device_index) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    IDeckLinkKeyer* keyer = nullptr;
    bool success = false;

    if (targetDev->QueryInterface(IID_IDeckLinkKeyer, (void**)&keyer) == S_OK && keyer) {
        success = (keyer->Disable() == S_OK);
        keyer->Release();
    }

    targetDev->Release();
    return success;
}

bool decklink_keyer_set_level(int32_t device_index, uint8_t level) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    IDeckLinkKeyer* keyer = nullptr;
    bool success = false;

    if (targetDev->QueryInterface(IID_IDeckLinkKeyer, (void**)&keyer) == S_OK && keyer) {
        success = (keyer->SetLevel(level) == S_OK);
        keyer->Release();
    }

    targetDev->Release();
    return success;
}

bool decklink_keyer_ramp_up(int32_t device_index, uint32_t number_of_frames) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    IDeckLinkKeyer* keyer = nullptr;
    bool success = false;

    if (targetDev->QueryInterface(IID_IDeckLinkKeyer, (void**)&keyer) == S_OK && keyer) {
        success = (keyer->RampUp(number_of_frames) == S_OK);
        keyer->Release();
    }

    targetDev->Release();
    return success;
}

bool decklink_keyer_ramp_down(int32_t device_index, uint32_t number_of_frames) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    IDeckLinkKeyer* keyer = nullptr;
    bool success = false;

    if (targetDev->QueryInterface(IID_IDeckLinkKeyer, (void**)&keyer) == S_OK && keyer) {
        success = (keyer->RampDown(number_of_frames) == S_OK);
        keyer->Release();
    }

    targetDev->Release();
    return success;
}

// ============================================================================
// Advanced Keyer Configuration
// ============================================================================

bool decklink_configure_keyer_inputs(int32_t device_index, int64_t fill_connection, int64_t key_connection) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    IDeckLinkConfiguration* cfg = nullptr;
    bool success = false;

    if (targetDev->QueryInterface(IID_IDeckLinkConfiguration, (void**)&cfg) == S_OK && cfg) {
        // Set fill input connection (primary video input)
        bool fill_ok = (cfg->SetInt(bmdDeckLinkConfigVideoInputConnection, fill_connection) == S_OK);
        
        // Note: Key input is typically on a paired SDI input
        // For cards like DeckLink Duo 2: SDI 1 = Fill, SDI 2 = Key
        // The key input is automatically paired when keyer is enabled
        
        success = fill_ok;
        cfg->Release();
    }

    targetDev->Release();
    return success;
}

struct DLKeyerStatus {
    bool is_enabled;
    bool is_internal_mode;
    uint8_t current_level;
};

bool decklink_get_keyer_status(int32_t device_index, DLKeyerStatus* out) {
    if (!out) return false;

    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    // Note: DeckLink API doesn't provide direct status query for keyer
    // We can only check if keyer interface is available
    IDeckLinkKeyer* keyer = nullptr;
    bool has_keyer = (targetDev->QueryInterface(IID_IDeckLinkKeyer, (void**)&keyer) == S_OK);
    
    if (has_keyer && keyer) {
        // Keyer interface exists, but we can't query current state directly
        // Set default values indicating keyer is available
        out->is_enabled = true;  // Assume enabled if interface exists
        out->is_internal_mode = true;  // Default to internal
        out->current_level = 255;  // Full opacity
        keyer->Release();
    } else {
        out->is_enabled = false;
        out->is_internal_mode = false;
        out->current_level = 0;
    }

    targetDev->Release();
    return has_keyer;
}

// ============================================================================
// Video Input/Output Control
// ============================================================================

bool decklink_enable_video_input(int32_t device_index, uint32_t display_mode, uint32_t pixel_format, uint32_t flags) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    IDeckLinkInput_v14_2_1* input = nullptr;
    bool success = false;

    if (targetDev->QueryInterface(IID_IDeckLinkInput_v14_2_1, (void**)&input) == S_OK && input) {
        success = (input->EnableVideoInput((BMDDisplayMode)display_mode, (BMDPixelFormat)pixel_format, (BMDVideoInputFlags)flags) == S_OK);
        input->Release();
    }

    targetDev->Release();
    return success;
}

bool decklink_disable_video_input(int32_t device_index) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    IDeckLinkInput_v14_2_1* input = nullptr;
    bool success = false;

    if (targetDev->QueryInterface(IID_IDeckLinkInput_v14_2_1, (void**)&input) == S_OK && input) {
        success = (input->DisableVideoInput() == S_OK);
        input->Release();
    }

    targetDev->Release();
    return success;
}

// ============================================================================
// Video Output Configuration
// ============================================================================

// Set timing parameters for scheduled playback
bool decklink_set_timing_config(int64_t time_scale, int64_t frame_duration) {
    g_timeScale.store(time_scale);
    g_frameDuration.store(frame_duration);
    fprintf(stderr, "[CFG ] timing: timeScale=%ld, frameDuration=%ld\n", time_scale, frame_duration);
    return true;
}

// Get hardware reference clock for scheduled playback
bool decklink_get_hardware_reference_clock(
    int32_t device_index,
    int64_t time_scale,
    int64_t* out_hardware_time,
    int64_t* out_time_in_frame,
    int64_t* out_ticks_per_frame
) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    IDeckLinkOutput_v14_2_1* output = nullptr;
    bool success = false;

    if (targetDev->QueryInterface(IID_IDeckLinkOutput_v14_2_1, (void**)&output) == S_OK && output) {
        BMDTimeValue hardwareTime = 0;
        BMDTimeValue timeInFrame = 0;
        BMDTimeValue ticksPerFrame = 0;
        
        HRESULT result = output->GetHardwareReferenceClock(
            (BMDTimeScale)time_scale,
            &hardwareTime,
            &timeInFrame,
            &ticksPerFrame
        );
        
        if (result == S_OK) {
            *out_hardware_time = hardwareTime;
            *out_time_in_frame = timeInFrame;
            *out_ticks_per_frame = ticksPerFrame;
            fprintf(stderr, "[HWCLK] hardwareTime=%ld, timeInFrame=%ld, ticksPerFrame=%ld\n",
                    hardwareTime, timeInFrame, ticksPerFrame);
            success = true;
        } else {
            fprintf(stderr, "[HWCLK] GetHardwareReferenceClock failed: 0x%08x\n", result);
        }
        
        output->Release();
    }

    targetDev->Release();
    return success;
}

bool decklink_set_output_connection_sdi(int32_t device_index) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) {
        fprintf(stderr, "[shim_extended] set_output_connection_sdi: Failed to create iterator\n");
        return false;
    }

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) {
        fprintf(stderr, "[shim_extended] set_output_connection_sdi: Device %d not found\n", device_index);
        return false;
    }

    IDeckLinkConfiguration* config = nullptr;
    bool success = false;

    if (targetDev->QueryInterface(IID_IDeckLinkConfiguration, (void**)&config) == S_OK && config) {
        // Set output connection to SDI
        HRESULT result = config->SetInt(bmdDeckLinkConfigVideoOutputConnection, bmdVideoConnectionSDI);
        
        if (result == S_OK) {
            fprintf(stderr, "[CFG ] outputConn=SDI set=OK\n");
            success = true;
        } else {
            fprintf(stderr, "[CFG ] outputConn=SDI set=FAIL (error: 0x%08x)\n", result);
        }
        
        config->Release();
    } else {
        fprintf(stderr, "[shim_extended] set_output_connection_sdi: Failed to get configuration interface\n");
    }

    targetDev->Release();
    return success;
}

bool decklink_enable_video_output(int32_t device_index, uint32_t display_mode, uint32_t flags) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    IDeckLinkOutput_v14_2_1* output = nullptr;
    bool success = false;

    HRESULT qiResult = targetDev->QueryInterface(IID_IDeckLinkOutput_v14_2_1, (void**)&output);
    fprintf(stderr, "[shim_extended] QueryInterface(IDeckLinkOutput) result: 0x%08x\n", qiResult);
    
    if (qiResult == S_OK && output) {
        // Check if display mode is supported
        IDeckLinkDisplayModeIterator* displayModeIterator = nullptr;
        if (output->GetDisplayModeIterator(&displayModeIterator) == S_OK) {
            IDeckLinkDisplayMode* displayModeObj = nullptr;
            bool modeFound = false;
            fprintf(stderr, "[shim_extended] Checking display mode support...\n");
            while (displayModeIterator->Next(&displayModeObj) == S_OK) {
                BMDDisplayMode mode = displayModeObj->GetDisplayMode();
                fprintf(stderr, "[shim_extended]   Available mode: 0x%08x\n", mode);
                if (mode == (BMDDisplayMode)display_mode) {
                    modeFound = true;
                    fprintf(stderr, "[shim_extended]   ✓ Requested mode 0x%08x is supported!\n", display_mode);
                }
                displayModeObj->Release();
            }
            displayModeIterator->Release();
            
            if (!modeFound) {
                fprintf(stderr, "[shim_extended]   ✗ Requested mode 0x%08x NOT supported!\n", display_mode);
            }
        }
        
        // Create and set callback if not already created
        if (!g_outputCallback) {
            g_outputCallback = new VideoOutputCallback();
            fprintf(stderr, "[shim_extended] Created new VideoOutputCallback\n");
        }
        
        // Set the callback
        HRESULT cbResult = output->SetScheduledFrameCompletionCallback(g_outputCallback);
        fprintf(stderr, "[shim_extended] SetScheduledFrameCompletionCallback result: 0x%08x\n", cbResult);
        if (cbResult != S_OK) {
            fprintf(stderr, "[shim_extended] Warning: Failed to set output callback\n");
        }
        
        // Enable video output
        fprintf(stderr, "[shim_extended] Calling EnableVideoOutput(mode=0x%08x, flags=0x%08x)...\n", display_mode, flags);
        HRESULT enableResult = output->EnableVideoOutput((BMDDisplayMode)display_mode, (BMDVideoOutputFlags)flags);
        fprintf(stderr, "[shim_extended] EnableVideoOutput result: 0x%08x\n", enableResult);
        success = (enableResult == S_OK);
        output->Release();
    } else {
        fprintf(stderr, "[shim_extended] Failed to get IDeckLinkOutput interface\n");
    }

    targetDev->Release();
    return success;
}

bool decklink_disable_video_output(int32_t device_index) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    IDeckLinkOutput_v14_2_1* output = nullptr;
    bool success = false;

    if (targetDev->QueryInterface(IID_IDeckLinkOutput_v14_2_1, (void**)&output) == S_OK && output) {
        success = (output->DisableVideoOutput() == S_OK);
        output->Release();
    }

    targetDev->Release();
    return success;
}

// ============================================================================
// Reference Lock (Genlock)
// ============================================================================

bool decklink_wait_reference_locked(int32_t device_index, uint32_t timeout_ms) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) {
        fprintf(stderr, "[REF ] Failed to create iterator\n");
        return false;
    }

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) {
        fprintf(stderr, "[REF ] Device %d not found\n", device_index);
        return false;
    }

    IDeckLinkStatus* status = nullptr;
    bool locked = false;

    if (targetDev->QueryInterface(IID_IDeckLinkStatus, (void**)&status) == S_OK && status) {
        // Poll for reference lock
        uint32_t elapsed_ms = 0;
        const uint32_t poll_interval_ms = 50;
        
        while (elapsed_ms < timeout_ms) {
            bool ref_locked = false;
            HRESULT result = status->GetFlag(bmdDeckLinkStatusReferenceSignalLocked, &ref_locked);
            
            if (result == S_OK && ref_locked) {
                fprintf(stderr, "[REF ] locked=1 (after %u ms)\n", elapsed_ms);
                locked = true;
                break;
            }
            
            // Sleep and retry
            std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
            elapsed_ms += poll_interval_ms;
        }
        
        if (!locked) {
            fprintf(stderr, "[REF ] locked=0 (timeout after %u ms)\n", timeout_ms);
        }
        
        status->Release();
    } else {
        // No status interface - might not support reference detection
        fprintf(stderr, "[REF ] skipped (status interface not available)\n");
        locked = true; // Don't fail if not supported
    }

    targetDev->Release();
    return locked;
}

bool decklink_start_streams(int32_t device_index) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    // Try output first (for keying output)
    IDeckLinkOutput_v14_2_1* output = nullptr;
    if (targetDev->QueryInterface(IID_IDeckLinkOutput_v14_2_1, (void**)&output) == S_OK && output) {
        // Start scheduled playback with proper timing from global config
        BMDTimeScale timeScale = g_timeScale.load();
        BMDTimeValue startTime = 0;
        double speed = 1.0;
        
        HRESULT result = output->StartScheduledPlayback(startTime, timeScale, speed);
        fprintf(stderr, "[PLAY] StartScheduledPlayback(tsStart=%ld, timeScale=%ld, speed=%.1f) -> 0x%08x\n",
                startTime, timeScale, speed, result);
        
        bool success = (result == S_OK);
        output->Release();
        
        if (success) {
            targetDev->Release();
            return true;
        }
    } else {
        fprintf(stderr, "[shim_extended] Failed to get output interface for start_streams\n");
    }

    // Try input if output failed
    IDeckLinkInput_v14_2_1* input = nullptr;
    if (targetDev->QueryInterface(IID_IDeckLinkInput_v14_2_1, (void**)&input) == S_OK && input) {
        bool success = (input->StartStreams() == S_OK);
        input->Release();
        targetDev->Release();
        return success;
    }

    targetDev->Release();
    return false;
}

bool decklink_stop_streams(int32_t device_index) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    // Try input first
    IDeckLinkInput_v14_2_1* input = nullptr;
    if (targetDev->QueryInterface(IID_IDeckLinkInput_v14_2_1, (void**)&input) == S_OK && input) {
        bool success = (input->StopStreams() == S_OK);
        input->Release();
        targetDev->Release();
        return success;
    }

    // Try output if input failed
    IDeckLinkOutput_v14_2_1* output = nullptr;
    if (targetDev->QueryInterface(IID_IDeckLinkOutput_v14_2_1, (void**)&output) == S_OK && output) {
        bool success = (output->StopScheduledPlayback(0, nullptr, 100) == S_OK);
        output->Release();
        targetDev->Release();
        return success;
    }

    targetDev->Release();
    return false;
}

bool decklink_flush_streams(int32_t device_index) {
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) return false;

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) return false;

    // Try input first
    IDeckLinkInput_v14_2_1* input = nullptr;
    if (targetDev->QueryInterface(IID_IDeckLinkInput_v14_2_1, (void**)&input) == S_OK && input) {
        bool success = (input->FlushStreams() == S_OK);
        input->Release();
        targetDev->Release();
        return success;
    }

    targetDev->Release();
    return false;
}

// ============================================================================
// Video Input Flags Constants (for Rust)
// ============================================================================

uint32_t decklink_get_video_input_flag_default() { return bmdVideoInputFlagDefault; }
uint32_t decklink_get_video_input_flag_enable_format_detection() { return bmdVideoInputEnableFormatDetection; }
uint32_t decklink_get_video_input_dual_stream_3d() { return bmdVideoInputDualStream3D; }

// ============================================================================
// Video Output Flags Constants (for Rust)
// ============================================================================

uint32_t decklink_get_video_output_flag_default() { return bmdVideoOutputFlagDefault; }
uint32_t decklink_get_video_output_vanc() { return bmdVideoOutputVANC; }
uint32_t decklink_get_video_output_vitc() { return bmdVideoOutputVITC; }
uint32_t decklink_get_video_output_dual_stream_3d() { return bmdVideoOutputDualStream3D; }
uint32_t decklink_get_video_output_rp188() { return bmdVideoOutputRP188; }
uint32_t decklink_get_video_output_synchronize_to_playback_group() { return bmdVideoOutputSynchronizeToPlaybackGroup; }

// ============================================================================
// Pixel Format Constants (for Rust)
// ============================================================================

uint32_t decklink_get_pixel_format_8bit_yuv() { return bmdFormat8BitYUV; }
uint32_t decklink_get_pixel_format_10bit_yuv() { return bmdFormat10BitYUV; }
uint32_t decklink_get_pixel_format_8bit_argb() { return bmdFormat8BitARGB; }
uint32_t decklink_get_pixel_format_8bit_bgra() { return bmdFormat8BitBGRA; }
uint32_t decklink_get_pixel_format_10bit_rgb() { return bmdFormat10BitRGB; }
uint32_t decklink_get_pixel_format_12bit_rgb() { return bmdFormat12BitRGB; }
uint32_t decklink_get_pixel_format_12bit_rgble() { return bmdFormat12BitRGBLE; }
uint32_t decklink_get_pixel_format_10bit_rgbxle() { return bmdFormat10BitRGBXLE; }
uint32_t decklink_get_pixel_format_10bit_rgbx() { return bmdFormat10BitRGBX; }

// ============================================================================
// Display Mode Constants (for Rust)
// ============================================================================

uint32_t decklink_get_mode_ntsc() { return bmdModeNTSC; }
uint32_t decklink_get_mode_pal() { return bmdModePAL; }
uint32_t decklink_get_mode_hd1080p2398() { return bmdModeHD1080p2398; }
uint32_t decklink_get_mode_hd1080p24() { return bmdModeHD1080p24; }
uint32_t decklink_get_mode_hd1080p25() { return bmdModeHD1080p25; }
uint32_t decklink_get_mode_hd1080p2997() { return bmdModeHD1080p2997; }
uint32_t decklink_get_mode_hd1080p30() { return bmdModeHD1080p30; }
uint32_t decklink_get_mode_hd1080p50() { return bmdModeHD1080p50; }
uint32_t decklink_get_mode_hd1080p5994() { return bmdModeHD1080p5994; }
uint32_t decklink_get_mode_hd1080p6000() { return bmdModeHD1080p6000; }
uint32_t decklink_get_mode_hd720p50() { return bmdModeHD720p50; }
uint32_t decklink_get_mode_hd720p5994() { return bmdModeHD720p5994; }
uint32_t decklink_get_mode_hd720p60() { return bmdModeHD720p60; }
uint32_t decklink_get_mode_2k2398() { return bmdMode2k2398; }
uint32_t decklink_get_mode_2k24() { return bmdMode2k24; }
uint32_t decklink_get_mode_2k25() { return bmdMode2k25; }
uint32_t decklink_get_mode_4k2160p2398() { return bmdMode4K2160p2398; }
uint32_t decklink_get_mode_4k2160p24() { return bmdMode4K2160p24; }
uint32_t decklink_get_mode_4k2160p25() { return bmdMode4K2160p25; }
uint32_t decklink_get_mode_4k2160p2997() { return bmdMode4K2160p2997; }
uint32_t decklink_get_mode_4k2160p30() { return bmdMode4K2160p30; }
uint32_t decklink_get_mode_4k2160p50() { return bmdMode4K2160p50; }
uint32_t decklink_get_mode_4k2160p5994() { return bmdMode4K2160p5994; }
uint32_t decklink_get_mode_4k2160p60() { return bmdMode4K2160p60; }

// ============================================================================
// Video Connection Constants (for Rust)
// ============================================================================

int64_t decklink_get_connection_sdi() { return bmdVideoConnectionSDI; }
int64_t decklink_get_connection_hdmi() { return bmdVideoConnectionHDMI; }
int64_t decklink_get_connection_optical_sdi() { return bmdVideoConnectionOpticalSDI; }
int64_t decklink_get_connection_component() { return bmdVideoConnectionComponent; }
int64_t decklink_get_connection_composite() { return bmdVideoConnectionComposite; }
int64_t decklink_get_connection_svideo() { return bmdVideoConnectionSVideo; }

// ============================================================================
// Audio Connection Constants (for Rust)
// ============================================================================

int64_t decklink_get_audio_connection_embedded() { return bmdAudioConnectionEmbedded; }
int64_t decklink_get_audio_connection_aesebu() { return bmdAudioConnectionAESEBU; }
int64_t decklink_get_audio_connection_analog() { return bmdAudioConnectionAnalog; }
int64_t decklink_get_audio_connection_analog_xlr() { return bmdAudioConnectionAnalogXLR; }
int64_t decklink_get_audio_connection_analog_rca() { return bmdAudioConnectionAnalogRCA; }

// ============================================================================
// Link Configuration Constants (for Rust)
// ============================================================================

int64_t decklink_get_link_configuration_single_link() { return bmdLinkConfigurationSingleLink; }
int64_t decklink_get_link_configuration_dual_link() { return bmdLinkConfigurationDualLink; }
int64_t decklink_get_link_configuration_quad_link() { return bmdLinkConfigurationQuadLink; }

// ============================================================================
// Output Frame Scheduling (for Internal Keying)
// ============================================================================

// Synchronous frame display (no scheduling, immediate display)
bool decklink_display_frame_sync(
    int32_t device_index,
    const uint8_t* bgra_data,
    int32_t width,
    int32_t height,
    int32_t pitch
) {
    if (!bgra_data) {
        fprintf(stderr, "[shim_extended] decklink_display_frame_sync: NULL data pointer\n");
        return false;
    }

    // Get device
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) {
        fprintf(stderr, "[shim_extended] decklink_display_frame_sync: Failed to create iterator\n");
        return false;
    }

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) {
        fprintf(stderr, "[shim_extended] decklink_display_frame_sync: Device %d not found\n", device_index);
        return false;
    }

    // Get output interface
    IDeckLinkOutput_v14_2_1* output = nullptr;
    HRESULT result = targetDev->QueryInterface(IID_IDeckLinkOutput_v14_2_1, (void**)&output);
    
    if (result != S_OK || !output) {
        fprintf(stderr, "[shim_extended] decklink_display_frame_sync: Failed to get output interface\n");
        targetDev->Release();
        return false;
    }

    // Create mutable video frame
    IDeckLinkMutableVideoFrame_v14_2_1* frame = nullptr;
    result = output->CreateVideoFrame(
        width,
        height,
        pitch,
        bmdFormat8BitBGRA,  // BGRA with alpha channel
        bmdFrameFlagDefault,
        &frame
    );

    if (result != S_OK || !frame) {
        fprintf(stderr, "[shim_extended] decklink_display_frame_sync: Failed to create video frame (error: 0x%08x)\n", result);
        output->Release();
        targetDev->Release();
        return false;
    }

    // Copy BGRA data into frame
    void* frameBuffer = nullptr;
    result = frame->GetBytes(&frameBuffer);
    
    if (result != S_OK || !frameBuffer) {
        fprintf(stderr, "[shim_extended] decklink_display_frame_sync: Failed to get frame buffer\n");
        frame->Release();
        output->Release();
        targetDev->Release();
        return false;
    }

    memcpy(frameBuffer, bgra_data, height * pitch);

    // Display frame synchronously (immediate output, no scheduling)
    result = output->DisplayVideoFrameSync(frame);
    
    if (result != S_OK) {
        fprintf(stderr, "[shim_extended] decklink_display_frame_sync: Failed to display frame (error: 0x%08x)\n", result);
        frame->Release();
        output->Release();
        targetDev->Release();
        return false;
    }

    // Cleanup
    frame->Release();
    output->Release();
    targetDev->Release();
    
    return true;
}

bool decklink_send_keyer_frame(
    int32_t device_index,
    const uint8_t* bgra_data,
    int32_t width,
    int32_t height,
    int32_t pitch
) {
    if (!bgra_data) {
        fprintf(stderr, "[shim_extended] decklink_send_keyer_frame: NULL data pointer\n");
        return false;
    }

    // Get device
    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) {
        fprintf(stderr, "[shim_extended] decklink_send_keyer_frame: Failed to create iterator\n");
        return false;
    }

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    IDeckLink* targetDev = nullptr;

    while (it->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            targetDev = dev;
            break;
        }
        dev->Release();
        idx++;
    }
    it->Release();

    if (!targetDev) {
        fprintf(stderr, "[shim_extended] decklink_send_keyer_frame: Device %d not found\n", device_index);
        return false;
    }

    // Get output interface
    IDeckLinkOutput_v14_2_1* output = nullptr;
    if (targetDev->QueryInterface(IID_IDeckLinkOutput_v14_2_1, (void**)&output) != S_OK || !output) {
        fprintf(stderr, "[shim_extended] decklink_send_keyer_frame: Failed to get output interface\n");
        targetDev->Release();
        return false;
    }

    // Create mutable video frame (BGRA format = 8-bit ARGB)
    IDeckLinkMutableVideoFrame_v14_2_1* frame = nullptr;
    HRESULT result = output->CreateVideoFrame(
        width,
        height,
        pitch,
        bmdFormat8BitARGB,  // BGRA in memory = ARGB in DeckLink SDK
        bmdFrameFlagDefault,
        &frame
    );

    if (result != S_OK || !frame) {
        fprintf(stderr, "[shim_extended] decklink_send_keyer_frame: Failed to create video frame (error: 0x%08x)\n", result);
        output->Release();
        targetDev->Release();
        return false;
    }

    // Copy BGRA data to frame
    void* frameBuffer = nullptr;
    result = frame->GetBytes(&frameBuffer);
    if (result != S_OK || !frameBuffer) {
        fprintf(stderr, "[shim_extended] decklink_send_keyer_frame: Failed to get frame buffer\n");
        frame->Release();
        output->Release();
        targetDev->Release();
        return false;
    }

    // Copy data row by row to handle pitch differences
    long framePitch = frame->GetRowBytes();
    
    const uint8_t* src = bgra_data;
    uint8_t* dst = (uint8_t*)frameBuffer;
    int32_t copyWidth = width * 4; // 4 bytes per pixel (BGRA)
    
    for (int32_t y = 0; y < height; y++) {
        memcpy(dst, src, copyWidth);
        src += pitch;
        dst += framePitch;
    }

    // Schedule frame for output with proper timing
    // Use global timing configuration
    BMDTimeScale timeScale = g_timeScale.load();
    BMDTimeValue frameDuration = g_frameDuration.load();
    
    // Check if playback has started by trying to get hardware time
    BMDTimeValue hardwareTime = 0;
    double playbackSpeed = 1.0;
    HRESULT timeResult = output->GetScheduledStreamTime(timeScale, &hardwareTime, &playbackSpeed);
    
    uint64_t frameNum = g_output_frame_count.fetch_add(1);
    BMDTimeValue displayTime;
    
    if (timeResult == S_OK && hardwareTime > 0) {
        // Playback has started - schedule relative to hardware time
        displayTime = hardwareTime + (frameNum * frameDuration);
        fprintf(stderr, "[SCHED] frame %lu at time %ld (hw=%ld)\n", 
                frameNum, displayTime, hardwareTime);
    } else {
        // Playback not started yet - use frame-based timing
        displayTime = frameNum * frameDuration;
        fprintf(stderr, "[PRER] frame %lu at time %ld (timeScale=%ld)\n",
                frameNum, displayTime, timeScale);
    }
    
    result = output->ScheduleVideoFrame(frame, displayTime, frameDuration, timeScale);
    
    if (result != S_OK) {
        fprintf(stderr, "[shim_extended] decklink_send_keyer_frame: Failed to schedule frame (error: 0x%08x)\n", result);
        fprintf(stderr, "[shim_extended]   displayTime=%ld, frameDuration=%ld, timeScale=%ld\n",
                displayTime, frameDuration, timeScale);
        frame->Release();
        output->Release();
        targetDev->Release();
        return false;
    }

    // Cleanup
    frame->Release();
    output->Release();
    targetDev->Release();

    return true;
}

} // extern "C"
