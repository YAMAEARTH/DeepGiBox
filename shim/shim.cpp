// /Users/yamaearth/Documents/3_1/Capstone/Blackmagic DeckLink SDK 14.4/rust/shim/shim.cpp
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <cstdio>
#include <chrono>
#include <cstdint>
#include <cuda_runtime_api.h>
#include "dvpapi_cuda.h"

// Include DeckLink SDK header (ensure include path points to where this header lives)
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

extern "C" {

struct DLDeviceList {
    char** names;
    int32_t count;
};

static char* dup_utf8_from_bmd_str(const char* s) {
    if (!s) return strdup("(unknown)");
    return strdup(s);
}

static void release_bmd_str(const char* s) {
    if (s) std::free((void*)s); // Linux SDK returns malloc'd char*
}

DLDeviceList decklink_list_devices() {
    DLDeviceList out{nullptr, 0};

    IDeckLinkIterator* it = CreateDeckLinkIteratorInstance();
    if (!it) {
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

// Capture API (C ABI)
struct CaptureFrame {
    const uint8_t* data;
    int32_t width;
    int32_t height;
    int32_t row_bytes; // bytes per row in data
    uint64_t seq;      // monotonically increasing frame sequence
    const uint8_t* gpu_data;
    int32_t gpu_row_bytes;
    uint32_t gpu_device;
};

bool decklink_capture_open(int32_t device_index);
bool decklink_capture_get_frame(CaptureFrame* out);
void decklink_capture_close();

// Output API (C ABI)
bool decklink_output_open(int32_t device_index, int32_t width, int32_t height, double fps);
bool decklink_output_send_frame(const uint8_t* bgra_data, int32_t width, int32_t height);
bool decklink_output_send_frame_gpu(const uint8_t* gpu_bgra_data, int32_t gpu_pitch, int32_t width, int32_t height);
void decklink_output_close();

} // extern "C"

// ---- C++ implementation for capture ----

namespace {

struct SharedFrame {
    std::vector<uint8_t> data; // CPU copy for safe access
    int32_t width = 0;
    int32_t height = 0;
    int32_t row_bytes = 0; // bytes per input row as reported by DeckLink
    std::mutex mtx;
    uint64_t seq = 0;
    uint8_t* gpu_ptr = nullptr;
    size_t gpu_pitch = 0;
    int32_t gpu_width = 0;
    int32_t gpu_height = 0;
    bool gpu_ready = false;
    bool cuda_initialized = false;
    bool cuda_available = false;
    uint32_t cuda_device = 0;
    cudaStream_t cuda_stream = nullptr;
    DVPBufferHandle dvp_gpu_handle = 0;
};

static IDeckLinkIterator* g_iterator = nullptr;
static IDeckLink* g_device = nullptr;
static IDeckLinkInput_v14_2_1* g_input = nullptr;
static SharedFrame g_shared;
class DvpAllocator;
static DvpAllocator* g_allocator = nullptr;

// Store detected input format for internal keying sync
static BMDDisplayMode g_detected_input_mode = bmdModeUnknown;
static double g_detected_input_fps = 0.0;

struct DvpContext {
    bool initialized = false;
    uint32_t buffer_alignment = 1;
    uint32_t gpu_stride_alignment = 1;
    uint32_t semaphore_alignment = 1;
    uint32_t semaphore_alloc_size = 0;
    uint32_t semaphore_payload_offset = 0;
    uint32_t semaphore_payload_size = 0;
};

static DvpContext g_dvp;
static bool get_dvp_host_handle(const void* host_ptr, DVPBufferHandle* out_handle);

static void log_cuda_error(const char* label, cudaError_t err) {
    if (!label) label = "(unknown)";
    fprintf(
        stderr,
        "[shim][cuda] %s failed: %s (%d)\n",
        label,
        cudaGetErrorString(err),
        static_cast<int>(err));
}

static const char* dvp_status_to_string(DVPStatus status) {
    switch (status) {
        case DVP_STATUS_OK: return "DVP_STATUS_OK";
        case DVP_STATUS_INVALID_PARAMETER: return "DVP_STATUS_INVALID_PARAMETER";
        case DVP_STATUS_UNSUPPORTED: return "DVP_STATUS_UNSUPPORTED";
        case DVP_STATUS_END_ENUMERATION: return "DVP_STATUS_END_ENUMERATION";
        case DVP_STATUS_INVALID_DEVICE: return "DVP_STATUS_INVALID_DEVICE";
        case DVP_STATUS_OUT_OF_MEMORY: return "DVP_STATUS_OUT_OF_MEMORY";
        case DVP_STATUS_INVALID_OPERATION: return "DVP_STATUS_INVALID_OPERATION";
        case DVP_STATUS_TIMEOUT: return "DVP_STATUS_TIMEOUT";
        case DVP_STATUS_INVALID_CONTEXT: return "DVP_STATUS_INVALID_CONTEXT";
        case DVP_STATUS_INVALID_RESOURCE_TYPE: return "DVP_STATUS_INVALID_RESOURCE_TYPE";
        case DVP_STATUS_INVALID_FORMAT_OR_TYPE: return "DVP_STATUS_INVALID_FORMAT_OR_TYPE";
        case DVP_STATUS_DEVICE_UNINITIALIZED: return "DVP_STATUS_DEVICE_UNINITIALIZED";
        case DVP_STATUS_UNSIGNALED: return "DVP_STATUS_UNSIGNALED";
        case DVP_STATUS_SYNC_ERROR: return "DVP_STATUS_SYNC_ERROR";
        case DVP_STATUS_SYNC_STILL_BOUND: return "DVP_STATUS_SYNC_STILL_BOUND";
        case DVP_STATUS_ERROR: return "DVP_STATUS_ERROR";
        default: return "DVP_STATUS_UNKNOWN";
    }
}

static void log_dvp_error(const char* label, DVPStatus status) {
    if (!label) label = "(unknown)";
    fprintf(
        stderr,
        "[shim][dvp] %s failed: %s (%d)\n",
        label,
        dvp_status_to_string(status),
        static_cast<int>(status));
}

static int pick_cuda_device() {
    const char* env = std::getenv("DECKLINK_CUDA_DEVICE");
    if (env && *env) {
        char* end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        if (end && *end == '\0' && parsed >= 0) {
            return static_cast<int>(parsed);
        }
    }
    return 0;
}

static bool ensure_cuda_stream_locked(SharedFrame& frame) {
    if (frame.cuda_available) {
        return true;
    }
    if (frame.cuda_initialized && !frame.cuda_available) {
        return false;
    }

    frame.cuda_initialized = true;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        log_cuda_error("cudaGetDeviceCount", err);
        frame.cuda_available = false;
        return false;
    }
    if (device_count <= 0) {
        fprintf(stderr, "[shim][cuda] No CUDA devices detected\n");
        frame.cuda_available = false;
        return false;
    }

    int target_device = pick_cuda_device();
    if (target_device >= device_count) {
        fprintf(
            stderr,
            "[shim][cuda] Requested device %d is out of range, using %d\n",
            target_device,
            device_count - 1);
        target_device = device_count - 1;
    }

    err = cudaSetDevice(target_device);
    if (err != cudaSuccess) {
        log_cuda_error("cudaSetDevice", err);
        frame.cuda_available = false;
        return false;
    }

    frame.cuda_device = static_cast<uint32_t>(target_device);
    err = cudaStreamCreateWithFlags(&frame.cuda_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        log_cuda_error("cudaStreamCreateWithFlags", err);
        frame.cuda_stream = nullptr;
        frame.cuda_available = false;
        return false;
    }

    frame.cuda_available = true;
    return true;
}

static bool ensure_dvp_initialized_locked(SharedFrame& frame) {
    if (g_dvp.initialized) {
        return true;
    }
    if (!ensure_cuda_stream_locked(frame)) {
        return false;
    }

    DVPStatus init_status = dvpInitCUDAContext(DVP_DEVICE_FLAGS_SHARE_APP_CONTEXT);
    if (init_status != DVP_STATUS_OK && init_status != DVP_STATUS_INVALID_OPERATION) {
        log_dvp_error("dvpInitCUDAContext", init_status);
        return false;
    }

    uint32_t buffer_align = 1;
    uint32_t stride_align = 1;
    uint32_t semaphore_align = 1;
    uint32_t semaphore_alloc = 0;
    uint32_t semaphore_payload_offset = 0;
    uint32_t semaphore_payload_size = 0;
    DVPStatus const_status = dvpGetRequiredConstantsCUDACtx(
        &buffer_align,
        &stride_align,
        &semaphore_align,
        &semaphore_alloc,
        &semaphore_payload_offset,
        &semaphore_payload_size);
    if (const_status != DVP_STATUS_OK) {
        log_dvp_error("dvpGetRequiredConstantsCUDACtx", const_status);
        if (init_status == DVP_STATUS_OK) {
            (void)dvpCloseCUDAContext();
        }
        return false;
    }

    g_dvp.buffer_alignment = buffer_align == 0 ? 1u : buffer_align;
    g_dvp.gpu_stride_alignment = stride_align == 0 ? 1u : stride_align;
    g_dvp.semaphore_alignment = semaphore_align == 0 ? 1u : semaphore_align;
    g_dvp.semaphore_alloc_size = semaphore_alloc;
    g_dvp.semaphore_payload_offset = semaphore_payload_offset;
    g_dvp.semaphore_payload_size = semaphore_payload_size;
    g_dvp.initialized = true;
    return true;
}

static bool ensure_cuda_buffer_locked(SharedFrame& frame, int32_t width, int32_t height, int32_t row_bytes) {
    if (!ensure_dvp_initialized_locked(frame)) {
        frame.gpu_ready = false;
        return false;
    }

    cudaError_t err = cudaSetDevice(static_cast<int>(frame.cuda_device));
    if (err != cudaSuccess) {
        log_cuda_error("cudaSetDevice", err);
        frame.gpu_ready = false;
        frame.cuda_available = false;
        return false;
    }

    const bool needs_alloc =
        frame.gpu_ptr == nullptr ||
        frame.gpu_width != width ||
        frame.gpu_height != height ||
        frame.gpu_pitch != static_cast<size_t>(row_bytes) ||
        frame.dvp_gpu_handle == 0;

    if (needs_alloc) {
        if (frame.dvp_gpu_handle != 0) {
            DVPStatus free_status = dvpFreeBuffer(frame.dvp_gpu_handle);
            if (free_status != DVP_STATUS_OK) {
                log_dvp_error("dvpFreeBuffer", free_status);
            }
            frame.dvp_gpu_handle = 0;
        }
        if (frame.gpu_ptr) {
            cudaError_t free_err = cudaFree(frame.gpu_ptr);
            if (free_err != cudaSuccess) {
                log_cuda_error("cudaFree", free_err);
            }
            frame.gpu_ptr = nullptr;
        }
        frame.gpu_pitch = 0;
        frame.gpu_width = 0;
        frame.gpu_height = 0;

        size_t total_bytes = static_cast<size_t>(row_bytes) * static_cast<size_t>(height);
        if (total_bytes == 0) {
            frame.gpu_ready = false;
            return false;
        }

        err = cudaMalloc(reinterpret_cast<void**>(&frame.gpu_ptr), total_bytes);
        if (err != cudaSuccess) {
            log_cuda_error("cudaMalloc", err);
            frame.gpu_ptr = nullptr;
            frame.gpu_ready = false;
            return false;
        }

        CUdeviceptr cu_ptr = reinterpret_cast<CUdeviceptr>(frame.gpu_ptr);
        DVPBufferHandle gpu_handle = 0;
        DVPStatus create_status = dvpCreateGPUCUDADevicePtr(cu_ptr, &gpu_handle);
        if (create_status != DVP_STATUS_OK) {
            log_dvp_error("dvpCreateGPUCUDADevicePtr", create_status);
            cudaError_t free_err = cudaFree(frame.gpu_ptr);
            if (free_err != cudaSuccess) {
                log_cuda_error("cudaFree", free_err);
            }
            frame.gpu_ptr = nullptr;
            frame.gpu_ready = false;
            return false;
        }

        frame.dvp_gpu_handle = gpu_handle;
        frame.gpu_pitch = static_cast<size_t>(row_bytes);
        frame.gpu_width = width;
        frame.gpu_height = height;
    }

    return frame.dvp_gpu_handle != 0;
}

static bool copy_frame_to_gpu_locked(SharedFrame& frame, const void* src, int32_t row_bytes, int32_t height) {
    if (!frame.cuda_available || !frame.gpu_ptr || frame.dvp_gpu_handle == 0) {
        frame.gpu_ready = false;
        return false;
    }
    if (!src) {
        frame.gpu_ready = false;
        return false;
    }
    DVPBufferHandle src_handle = 0;
    if (!get_dvp_host_handle(src, &src_handle) || src_handle == 0) {
        frame.gpu_ready = false;
        return false;
    }

    size_t total_bytes = static_cast<size_t>(row_bytes) * static_cast<size_t>(height);
    if (total_bytes == 0 || total_bytes > UINT32_MAX) {
        frame.gpu_ready = false;
        return false;
    }

    DVPStatus begin_status = dvpBegin();
    if (begin_status != DVP_STATUS_OK) {
        log_dvp_error("dvpBegin", begin_status);
        frame.gpu_ready = false;
        return false;
    }

    DVPStatus copy_status = dvpMemcpy(
        src_handle,
        0,
        0,
        DVP_TIMEOUT_IGNORED,
        frame.dvp_gpu_handle,
        0,
        0,
        0,
        0,
        static_cast<uint32_t>(total_bytes));
    if (copy_status != DVP_STATUS_OK) {
        log_dvp_error("dvpMemcpy", copy_status);
        (void)dvpEnd();
        frame.gpu_ready = false;
        return false;
    }

    DVPStatus end_status = dvpEnd();
    if (end_status != DVP_STATUS_OK) {
        log_dvp_error("dvpEnd", end_status);
        frame.gpu_ready = false;
        return false;
    }

    frame.gpu_ready = true;
    return true;
}

class DvpAllocator : public IDeckLinkMemoryAllocator_v14_2_1 {
public:
    DvpAllocator() : m_ref(1) {}
    virtual ~DvpAllocator() = default;

    HRESULT QueryInterface(REFIID iid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (memcmp(&iid, &kIID_IUnknown, sizeof(REFIID)) == 0 ||
            memcmp(&iid, &IID_IDeckLinkMemoryAllocator_v14_2_1, sizeof(REFIID)) == 0) {
            *ppv = static_cast<IDeckLinkMemoryAllocator_v14_2_1*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }

    ULONG AddRef() override { return ++m_ref; }

    ULONG Release() override {
        ULONG r = --m_ref;
        if (r == 0) {
            release_all();
            delete this;
        }
        return r;
    }

    HRESULT AllocateBuffer(uint32_t bufferSize, void** allocatedBuffer) override {
        if (!allocatedBuffer || bufferSize == 0) {
            return E_POINTER;
        }
        uint32_t target_device = 0;
        {
            std::lock_guard<std::mutex> lk(g_shared.mtx);
            if (!ensure_dvp_initialized_locked(g_shared)) {
                return E_FAIL;
            }
            target_device = g_shared.cuda_device;
        }

        cudaError_t set_err = cudaSetDevice(static_cast<int>(target_device));
        if (set_err != cudaSuccess) {
            log_cuda_error("cudaSetDevice", set_err);
            return E_FAIL;
        }

        size_t alignment = static_cast<size_t>(g_dvp.buffer_alignment);
        if (alignment == 0) {
            alignment = 64;
        }
        void* host_ptr = nullptr;
        int err = posix_memalign(&host_ptr, alignment, static_cast<size_t>(bufferSize));
        if (err != 0 || !host_ptr) {
            fprintf(stderr, "[shim][dvp] posix_memalign failed (err=%d)\n", err);
            return E_FAIL;
        }

        DVPSysmemBufferDesc desc{};
        desc.width = bufferSize;
        desc.height = 1;
        desc.stride = bufferSize;
        desc.size = bufferSize;
        desc.format = DVP_BUFFER;
        desc.type = DVP_UNSIGNED_BYTE;
        desc.bufAddr = host_ptr;

        DVPBufferHandle handle = 0;
        DVPStatus create_status = dvpCreateBuffer(&desc, &handle);
        if (create_status != DVP_STATUS_OK) {
            log_dvp_error("dvpCreateBuffer", create_status);
            std::free(host_ptr);
            return E_FAIL;
        }

        DVPStatus bind_status = dvpBindToCUDACtx(handle);
        if (bind_status != DVP_STATUS_OK) {
            log_dvp_error("dvpBindToCUDACtx", bind_status);
            (void)dvpDestroyBuffer(handle);
            std::free(host_ptr);
            return E_FAIL;
        }

        {
            std::lock_guard<std::mutex> lk(m_mutex);
            m_allocations.push_back(Allocation{
                host_ptr,
                static_cast<size_t>(bufferSize),
                handle,
            });
        }

        *allocatedBuffer = host_ptr;
        return S_OK;
    }

    HRESULT ReleaseBuffer(void* buffer) override {
        if (!buffer) {
            return E_POINTER;
        }

        Allocation alloc{};
        {
            std::lock_guard<std::mutex> lk(m_mutex);
            auto it = std::find_if(
                m_allocations.begin(),
                m_allocations.end(),
                [&](const Allocation& a) { return a.host == buffer; });
            if (it == m_allocations.end()) {
                return E_FAIL;
            }
            alloc = *it;
            m_allocations.erase(it);
        }

        DVPStatus unbind_status = dvpUnbindFromCUDACtx(alloc.handle);
        if (unbind_status != DVP_STATUS_OK) {
            log_dvp_error("dvpUnbindFromCUDACtx", unbind_status);
        }
        DVPStatus destroy_status = dvpDestroyBuffer(alloc.handle);
        if (destroy_status != DVP_STATUS_OK) {
            log_dvp_error("dvpDestroyBuffer", destroy_status);
        }
        std::free(alloc.host);
        return S_OK;
    }

    HRESULT Commit() override { return S_OK; }
    HRESULT Decommit() override { return S_OK; }

    bool get_buffer_handle(const void* host_ptr, DVPBufferHandle* out) {
        if (!host_ptr || !out) return false;
        std::lock_guard<std::mutex> lk(m_mutex);
        for (const auto& alloc : m_allocations) {
            if (alloc.host == host_ptr) {
                *out = alloc.handle;
                return true;
            }
        }
        return false;
    }

private:
    struct Allocation {
        void* host;
        size_t size;
        DVPBufferHandle handle;
    };

    void release_all() {
        std::lock_guard<std::mutex> lk(m_mutex);
        for (auto& alloc : m_allocations) {
            if (!alloc.host) continue;
            DVPStatus unbind_status = dvpUnbindFromCUDACtx(alloc.handle);
            if (unbind_status != DVP_STATUS_OK) {
                log_dvp_error("dvpUnbindFromCUDACtx", unbind_status);
            }
            DVPStatus destroy_status = dvpDestroyBuffer(alloc.handle);
            if (destroy_status != DVP_STATUS_OK) {
                log_dvp_error("dvpDestroyBuffer", destroy_status);
            }
            std::free(alloc.host);
        }
        m_allocations.clear();
    }

    std::atomic<ULONG> m_ref;
    std::mutex m_mutex;
    std::vector<Allocation> m_allocations;
};

static bool get_dvp_host_handle(const void* host_ptr, DVPBufferHandle* out_handle) {
    if (!host_ptr || !out_handle) {
        return false;
    }
    if (!g_allocator) {
        return false;
    }
    return g_allocator->get_buffer_handle(host_ptr, out_handle);
}

// OpenGL preview helper (cross-platform)
static IDeckLinkGLScreenPreviewHelper* g_glPreview = nullptr;
static IDeckLinkVideoFrame_v14_2_1* g_glFrame = nullptr;
static IDeckLinkScreenPreviewCallback_v14_2_1* g_glCallback = nullptr;
static std::mutex g_glMutex;
static std::mutex g_glFrameMutex;
static std::atomic<uint64_t> g_gl_seq{0};
static std::atomic<uint64_t> g_gl_last_arrival_ns{0};
static std::atomic<uint64_t> g_gl_last_latency_ns{0};
static std::atomic<uint32_t> g_last_getbytes_base{0};
static std::atomic<uint32_t> g_last_getbytes_fallback{0};
static std::atomic<uint32_t> g_last_getbytes_buffer{0};
static std::atomic<uint32_t> g_last_getbytes_pix{0};
static std::atomic<uint64_t> g_gl_last_render_seq{0};

extern "C" void decklink_preview_gl_disable();

class CaptureCallback : public IDeckLinkInputCallback_v14_2_1 {
public:
    CaptureCallback() : m_ref(1) {}

    // IUnknown
    HRESULT QueryInterface(REFIID iid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (memcmp(&iid, &kIID_IUnknown, sizeof(REFIID)) == 0) {
            *ppv = static_cast<IUnknown*>(this);
            AddRef();
            return S_OK;
        }
        if (memcmp(&iid, &IID_IDeckLinkInputCallback_v14_2_1, sizeof(REFIID)) == 0) {
            *ppv = static_cast<IDeckLinkInputCallback_v14_2_1*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    ULONG AddRef() override { return ++m_ref; }
    ULONG Release() override {
        ULONG r = --m_ref;
        if (r == 0) delete this;
        return r;
    }

    // IDeckLinkInputCallback
    HRESULT VideoInputFormatChanged(BMDVideoInputFormatChangedEvents /*events*/, IDeckLinkDisplayMode* newDisplayMode, BMDDetectedVideoInputFormatFlags /*detectedSignalFlags*/) override {
        if (!newDisplayMode || !g_input) return S_OK;
        BMDTimeValue fd = 0; BMDTimeScale ts = 0;
        newDisplayMode->GetFrameRate(&fd, &ts);
        long w = newDisplayMode->GetWidth();
        long h = newDisplayMode->GetHeight();
        double fps = (ts && fd) ? (double)ts/(double)fd : 0.0;
        fprintf(stderr, "[shim] FormatChanged: %ldx%ld, fps=%.3f\n", w, h, fps);

        // Store detected mode and fps for internal keying sync
        g_detected_input_mode = newDisplayMode->GetDisplayMode();
        g_detected_input_fps = fps;
        fprintf(stderr, "[shim] Stored input mode=0x%x, fps=%.3f for output sync\n", 
                (unsigned)g_detected_input_mode, g_detected_input_fps);

        // Avoid reconfiguring to the same mode/pixel format repeatedly
        static BMDDisplayMode s_lastMode = (BMDDisplayMode)0;
        if (s_lastMode == newDisplayMode->GetDisplayMode()) {
            return S_OK;
        }

        g_input->StopStreams();
        HRESULT en = g_input->EnableVideoInput(newDisplayMode->GetDisplayMode(), bmdFormat8BitYUV, bmdVideoInputEnableFormatDetection);
        if (en == S_OK) {
            fprintf(stderr, "[shim] FormatChanged: pix=0x%x\n", (unsigned)bmdFormat8BitYUV);
            g_input->StartStreams();
            s_lastMode = newDisplayMode->GetDisplayMode();
        } else {
            fprintf(stderr, "[shim] FormatChanged: EnableVideoInput failed hr=0x%08x\n", (unsigned)en);
            s_lastMode = (BMDDisplayMode)0;
        }
        return S_OK;
    }

    HRESULT VideoInputFrameArrived(IDeckLinkVideoInputFrame_v14_2_1* videoFrame, IDeckLinkAudioInputPacket* /*audioPacket*/) override {
        if (!videoFrame)
            return S_OK;

        long width = videoFrame->GetWidth();
        long height = videoFrame->GetHeight();
        long rowBytes = videoFrame->GetRowBytes();
        BMDPixelFormat pixfmt = videoFrame->GetPixelFormat();
        void* bytes = nullptr;
        IDeckLinkVideoBuffer* videoBuffer = nullptr;
        HRESULT buffer_qri = videoFrame->QueryInterface(IID_IDeckLinkVideoBuffer, (void**)&videoBuffer);
        HRESULT buffer_start = E_NOINTERFACE;
        HRESULT buffer_gb = E_NOINTERFACE;
        bool buffer_access = false;
        if (buffer_qri == S_OK && videoBuffer) {
            buffer_start = videoBuffer->StartAccess(bmdBufferAccessRead);
            if (buffer_start == S_OK) {
                buffer_access = true;
                buffer_gb = videoBuffer->GetBytes(&bytes);
                if (buffer_gb != S_OK || !bytes) {
                    bytes = nullptr;
                    videoBuffer->EndAccess(bmdBufferAccessRead);
                    buffer_access = false;
                }
            }
        }
        const REFIID try_iids[] = {
            IID_IDeckLinkVideoInputFrame_v14_2_1,
            IID_IDeckLinkVideoFrame_v14_2_1,
        };
        HRESULT qri_results[sizeof(try_iids) / sizeof(try_iids[0])] = {
            E_NOINTERFACE,
            E_NOINTERFACE,
        };
        HRESULT getbytes_results[sizeof(try_iids) / sizeof(try_iids[0])] = {
            E_NOINTERFACE,
            E_NOINTERFACE,
        };
        IDeckLinkVideoFrame_v14_2_1* vf = nullptr;
        for (size_t i = 0; i < sizeof(try_iids) / sizeof(try_iids[0]); ++i) {
            if (bytes) break;
            void* qptr = nullptr;
            HRESULT qri = videoFrame->QueryInterface(try_iids[i], &qptr);
            qri_results[i] = qri;
            if (qri != S_OK || !qptr) {
                continue;
            }
            auto* tmp = static_cast<IDeckLinkVideoFrame_v14_2_1*>(qptr);
            HRESULT gb = tmp->GetBytes(&bytes);
            getbytes_results[i] = gb;
            if (gb == S_OK && bytes != nullptr) {
                vf = tmp;
            } else {
                tmp->Release();
                bytes = nullptr;
            }
        }
        if (!bytes) {
            const uint32_t ubase_q = (uint32_t)qri_results[0];
            const uint32_t ubase_gb = (uint32_t)getbytes_results[0];
            const uint32_t ufb_q = (uint32_t)qri_results[1];
            const uint32_t ufb_gb = (uint32_t)getbytes_results[1];
            const uint32_t upix = (uint32_t)pixfmt;
            const uint32_t ubuf = (uint32_t)buffer_qri ^ (uint32_t)buffer_start ^ (uint32_t)buffer_gb;
            uint32_t last_base_q = g_last_getbytes_base.load(std::memory_order_relaxed);
            uint32_t last_fb_q = g_last_getbytes_fallback.load(std::memory_order_relaxed);
            uint32_t last_pix = g_last_getbytes_pix.load(std::memory_order_relaxed);
            uint32_t last_buf = g_last_getbytes_buffer.load(std::memory_order_relaxed);
            if (last_base_q != (ubase_q ^ ubase_gb) ||
                last_fb_q != (ufb_q ^ ufb_gb) ||
                last_pix != upix ||
                last_buf != ubuf) {
                fprintf(stderr,
                        "[shim] GetBytes failed (buffer qri=0x%08x start=0x%08x gb=0x%08x; input_v14 qri=0x%08x gb=0x%08x; frame_v14 qri=0x%08x gb=0x%08x; pix=0x%x)\n",
                        (unsigned)buffer_qri,
                        (unsigned)buffer_start,
                        (unsigned)buffer_gb,
                        (unsigned)ubase_q,
                        (unsigned)ubase_gb,
                        (unsigned)ufb_q,
                        (unsigned)ufb_gb,
                        (unsigned)pixfmt);
                g_last_getbytes_base.store(ubase_q ^ ubase_gb, std::memory_order_relaxed);
                g_last_getbytes_fallback.store(ufb_q ^ ufb_gb, std::memory_order_relaxed);
                g_last_getbytes_buffer.store(ubuf, std::memory_order_relaxed);
                g_last_getbytes_pix.store(upix, std::memory_order_relaxed);
            }
            if (buffer_access && videoBuffer) {
                videoBuffer->EndAccess(bmdBufferAccessRead);
                buffer_access = false;
            }
            if (videoBuffer) {
                videoBuffer->Release();
                videoBuffer = nullptr;
            }
            if (vf) vf->Release();
            return S_OK;
        }
        g_last_getbytes_base.store(0, std::memory_order_relaxed);
        g_last_getbytes_fallback.store(0, std::memory_order_relaxed);
        g_last_getbytes_buffer.store(0, std::memory_order_relaxed);
        g_last_getbytes_pix.store(0, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lk(g_shared.mtx);
            const int32_t w = (int32_t)width;
            const int32_t h = (int32_t)height;
            const int32_t srcStride = (int32_t)rowBytes;
            const size_t wantSize = (size_t)srcStride * (size_t)h;

            if (g_shared.data.size() != wantSize) {
                g_shared.data.resize(wantSize);
            }
            if (bytes && wantSize > 0) {
                std::memcpy(g_shared.data.data(), bytes, wantSize);
            } else if (!g_shared.data.empty()) {
                std::memset(g_shared.data.data(), 0, g_shared.data.size());
            }

            g_shared.width = w;
            g_shared.height = h;
            g_shared.row_bytes = srcStride;
            g_shared.gpu_ready = false;

            static bool s_logged = false;
            if (!s_logged) {
                fprintf(stderr, "[shim] First frame: %dx%d, rb=%ld, pix=0x%x\n", w, h, (long)rowBytes, (unsigned)pixfmt);
                s_logged = true;
            }
            g_shared.seq++;

            if (!g_shared.gpu_ready && bytes && g_allocator && ensure_cuda_buffer_locked(g_shared, w, h, srcStride)) {
                (void)copy_frame_to_gpu_locked(g_shared, bytes, srcStride, h);
            }
        }
        if (buffer_access && videoBuffer) {
            videoBuffer->EndAccess(bmdBufferAccessRead);
            buffer_access = false;
        }
        if (videoBuffer) {
            videoBuffer->Release();
            videoBuffer = nullptr;
        }
        if (vf) vf->Release();
        return S_OK;
    }

private:
    std::atomic<ULONG> m_ref;
};

static CaptureCallback* g_callback = nullptr;

class GLPreviewCallback : public IDeckLinkScreenPreviewCallback_v14_2_1 {
public:
    GLPreviewCallback() : m_ref(1) {}

    HRESULT QueryInterface(REFIID iid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (memcmp(&iid, &kIID_IUnknown, sizeof(REFIID)) == 0) {
            *ppv = static_cast<IUnknown*>(this);
            AddRef();
            return S_OK;
        }
        if (memcmp(&iid, &IID_IDeckLinkScreenPreviewCallback_v14_2_1, sizeof(REFIID)) == 0) {
            *ppv = static_cast<IDeckLinkScreenPreviewCallback_v14_2_1*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }

    ULONG AddRef() override { return ++m_ref; }

    ULONG Release() override {
        ULONG r = --m_ref;
        if (r == 0) delete this;
        return r;
    }

    HRESULT DrawFrame(IDeckLinkVideoFrame_v14_2_1* frame) override {
        std::lock_guard<std::mutex> lk(g_glFrameMutex);
        if (g_glFrame) {
            g_glFrame->Release();
            g_glFrame = nullptr;
        }
        if (frame) {
            frame->AddRef();
            g_glFrame = frame;
        }
        g_gl_seq.fetch_add(1, std::memory_order_relaxed);
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        g_gl_last_arrival_ns.store(
            (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now).count(),
            std::memory_order_relaxed);
        return S_OK;
    }

private:
    std::atomic<ULONG> m_ref;
};

static bool pick_supported_mode(IDeckLinkInput_v14_2_1* input, BMDVideoConnection connection, BMDDisplayMode& outMode, BMDPixelFormat& outPix) {
    if (!input) return false;
    IDeckLinkDisplayModeIterator* it = nullptr;
    if (input->GetDisplayModeIterator(&it) != S_OK || !it) return false;

    // Scan for the first display mode supported for the current connection using 8-bit YUV
    BMDDisplayMode chosenMode = (BMDDisplayMode)0;
    BMDPixelFormat chosenPix = bmdFormatUnspecified;

    IDeckLinkDisplayMode* mode = nullptr;
    while (it->Next(&mode) == S_OK && mode) {
        bool supported = false; BMDDisplayMode actualMode = (BMDDisplayMode)0;
        HRESULT hr = input->DoesSupportVideoMode(connection, mode->GetDisplayMode(), bmdFormat8BitYUV,
                            bmdNoVideoInputConversion, bmdSupportedVideoModeDefault, &actualMode, &supported);
        if (hr == S_OK && supported) {
            chosenMode = mode->GetDisplayMode();
            chosenPix = bmdFormat8BitYUV;
            mode->Release();
            break;
        }
        mode->Release();
    }
    it->Release();

    if (chosenMode != (BMDDisplayMode)0) { outMode = chosenMode; outPix = chosenPix; return true; }
    return false;
}

} // namespace

extern "C" bool decklink_capture_open(int32_t device_index) {
    if (g_input) return true; // already open

    g_iterator = CreateDeckLinkIteratorInstance();
    if (!g_iterator) {
        fprintf(stderr, "[shim] no iterator (DeckLink API not present)\n");
        return false;
    }

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    while (g_iterator->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            g_device = dev; // take ownership
            break;
        }
        dev->Release();
        idx++;
    }
    if (!g_device) {
        fprintf(stderr, "[shim] device index %d not found\n", device_index);
        g_iterator->Release(); g_iterator = nullptr;
        return false;
    }

    // Prefer HDMI input on UltraStudio Recorder; if device doesn't support HDMI, fall back to any supported connection
    {
        int64_t supportedConns = 0;
        IDeckLinkProfileAttributes* attr = nullptr;
        if (g_device && g_device->QueryInterface(IID_IDeckLinkProfileAttributes, (void**)&attr) == S_OK && attr) {
            (void)attr->GetInt(BMDDeckLinkVideoInputConnections, &supportedConns);
            attr->Release();
        }

        BMDVideoConnection desired = bmdVideoConnectionHDMI;
        if ((supportedConns & bmdVideoConnectionHDMI) == 0) {
            // Pick first available connection
            if (supportedConns & bmdVideoConnectionSDI) desired = bmdVideoConnectionSDI;
            else if (supportedConns & bmdVideoConnectionComponent) desired = bmdVideoConnectionComponent;
            else if (supportedConns & bmdVideoConnectionComposite) desired = bmdVideoConnectionComposite;
            else if (supportedConns & bmdVideoConnectionSVideo) desired = bmdVideoConnectionSVideo;
            else desired = bmdVideoConnectionUnspecified;
        }

        IDeckLinkConfiguration* cfg = nullptr;
        HRESULT r = g_device && g_device->QueryInterface(IID_IDeckLinkConfiguration, (void**)&cfg);
        if (r == S_OK && cfg) {
            if (desired != bmdVideoConnectionUnspecified)
                (void)cfg->SetInt(bmdDeckLinkConfigVideoInputConnection, (int64_t)desired);
            cfg->Release();
        } else {
            fprintf(stderr, "[shim] IDeckLinkConfiguration not available, hr=0x%08x\n", (unsigned)r);
        }
    }

    // Get input interface
    HRESULT qri = g_device->QueryInterface(IID_IDeckLinkInput_v14_2_1, (void**)&g_input);
    if (qri != S_OK || !g_input) {
        fprintf(stderr, "[shim] QueryInterface(IID_IDeckLinkInput_v14_2_1) failed hr=0x%08x\n", (unsigned)qri);
        g_device->Release(); g_device = nullptr;
        g_iterator->Release(); g_iterator = nullptr;
        return false;
    }

    if (!g_allocator) {
        DvpAllocator* alloc = new DvpAllocator();
        if (g_input->SetVideoInputFrameMemoryAllocator(alloc) == S_OK) {
            g_allocator = alloc;
            g_allocator->AddRef(); // keep global reference alive
        } else {
            fprintf(stderr, "[shim] SetVideoInputFrameMemoryAllocator failed, falling back to internal buffers\n");
            g_input->SetVideoInputFrameMemoryAllocator(nullptr);
            alloc->Release();
            g_allocator = nullptr;
        }
    }

    // Try enabling with format detection first (unknown mode, YUV)
    g_callback = new CaptureCallback();
    if (g_input->SetCallback(g_callback) != S_OK) {
        fprintf(stderr, "[shim] SetCallback failed\n");
        g_callback->Release(); g_callback = nullptr;
        if (g_input && g_allocator) {
            g_input->SetVideoInputFrameMemoryAllocator(nullptr);
            g_allocator->Release();
            g_allocator = nullptr;
        }
        g_input->Release(); g_input = nullptr;
        g_device->Release(); g_device = nullptr;
        g_iterator->Release(); g_iterator = nullptr;
        return false;
    }

    // Determine current input connection (we set HDMI above, but read the active one)
    int64_t currConn = bmdVideoConnectionUnspecified;
    {
        IDeckLinkConfiguration* cfg = nullptr;
        if (g_device && g_device->QueryInterface(IID_IDeckLinkConfiguration, (void**)&cfg) == S_OK && cfg) {
            (void)cfg->GetInt(bmdDeckLinkConfigVideoInputConnection, &currConn);
            cfg->Release();
        }
    }

    // Determine supported input connections and whether format detection is supported
    int64_t supportedConns = 0;
    bool supportsDetect = false;
    {
        IDeckLinkProfileAttributes* attr = nullptr;
        if (g_device && g_device->QueryInterface(IID_IDeckLinkProfileAttributes, (void**)&attr) == S_OK && attr) {
            (void)attr->GetInt(BMDDeckLinkVideoInputConnections, &supportedConns);
            bool flag = false;
            if (attr->GetFlag(BMDDeckLinkSupportsInputFormatDetection, &flag) == S_OK) supportsDetect = flag;
            attr->Release();
        }
    }
    fprintf(stderr, "[shim] supportedConns=0x%llx, currConn=0x%llx, detect=%d\n",
            (unsigned long long)supportedConns, (unsigned long long)currConn, supportsDetect ? 1 : 0);

    BMDVideoInputFlags flags = supportsDetect ? bmdVideoInputEnableFormatDetection : bmdVideoInputFlagDefault;

    auto try_enable = [&](BMDVideoConnection conn) -> bool {
        // Switch active input connector to the one we are trying
        {
            IDeckLinkConfiguration* cfg = nullptr;
            if (g_device && g_device->QueryInterface(IID_IDeckLinkConfiguration, (void**)&cfg) == S_OK && cfg) {
                (void)cfg->SetInt(bmdDeckLinkConfigVideoInputConnection, (int64_t)conn);
                cfg->Release();
            }
        }
        // First attempt: format detection with unknown mode (if supported)
        if (supportsDetect) {
            HRESULT en = g_input->EnableVideoInput(bmdModeUnknown, bmdFormat8BitYUV, flags);
            if (en == S_OK) return true;
        }

        // Fallback: choose explicit mode and pixel format
        BMDDisplayMode mode = (BMDDisplayMode)0;
        BMDPixelFormat pix = bmdFormatUnspecified;
        if (!pick_supported_mode(g_input, conn, mode, pix)) {
            fprintf(stderr, "[shim] No supported display mode for conn=0x%x\n", (unsigned)conn);
            return false;
        }
        HRESULT en = g_input->EnableVideoInput(mode, pix, flags);
        if (en == S_OK) return true;
        fprintf(stderr, "[shim] EnableVideoInput failed hr=0x%08x (mode=%u, pix=0x%x)\n", (unsigned)en, (unsigned)mode, (unsigned)pix);
        return false;
    };

    // Try current connection first, then fall back through supported connections and finally unspecified
    bool ok = false;
    if (currConn != bmdVideoConnectionUnspecified)
        ok = try_enable((BMDVideoConnection)currConn);
    if (!ok && (supportedConns & bmdVideoConnectionHDMI)) ok = try_enable(bmdVideoConnectionHDMI);
    if (!ok && (supportedConns & bmdVideoConnectionSDI)) ok = try_enable(bmdVideoConnectionSDI);
    if (!ok && (supportedConns & bmdVideoConnectionComponent)) ok = try_enable(bmdVideoConnectionComponent);
    if (!ok && (supportedConns & bmdVideoConnectionComposite)) ok = try_enable(bmdVideoConnectionComposite);
    if (!ok && (supportedConns & bmdVideoConnectionSVideo)) ok = try_enable(bmdVideoConnectionSVideo);
    if (!ok) ok = try_enable(bmdVideoConnectionUnspecified);

    if (!ok) {
        fprintf(stderr, "[shim] Could not enable any mode on available connections\n");
        g_input->SetCallback(nullptr);
        g_callback->Release(); g_callback = nullptr;
        if (g_input && g_allocator) {
            g_input->SetVideoInputFrameMemoryAllocator(nullptr);
            g_allocator->Release();
            g_allocator = nullptr;
        }
        g_input->Release(); g_input = nullptr;
        g_device->Release(); g_device = nullptr;
        g_iterator->Release(); g_iterator = nullptr;
        return false;
    }

    if (g_input->StartStreams() != S_OK) {
        fprintf(stderr, "[shim] StartStreams failed\n");
        g_input->DisableVideoInput();
        g_input->SetCallback(nullptr);
        g_callback->Release(); g_callback = nullptr;
        if (g_input && g_allocator) {
            g_input->SetVideoInputFrameMemoryAllocator(nullptr);
            g_allocator->Release();
            g_allocator = nullptr;
        }
        g_input->Release(); g_input = nullptr;
        g_device->Release(); g_device = nullptr;
        g_iterator->Release(); g_iterator = nullptr;
        return false;
    }

    return true;
}

extern "C" bool decklink_capture_get_frame(CaptureFrame* out) {
    if (!out) return false;
    std::lock_guard<std::mutex> lk(g_shared.mtx);
    if (g_shared.data.empty() || g_shared.width <= 0 || g_shared.height <= 0) {
        out->data = nullptr;
        out->width = 0;
        out->height = 0;
        out->row_bytes = 0;
        out->seq = 0;
        out->gpu_data = nullptr;
        out->gpu_row_bytes = 0;
        out->gpu_device = 0;
        return false;
    }
    out->data = g_shared.data.data();
    out->width = g_shared.width;
    out->height = g_shared.height;
    out->row_bytes = g_shared.row_bytes;
    out->seq = g_shared.seq;
    if (g_shared.gpu_ready && g_shared.gpu_ptr) {
        out->gpu_data = g_shared.gpu_ptr;
        out->gpu_row_bytes = static_cast<int32_t>(g_shared.gpu_pitch);
        out->gpu_device = g_shared.cuda_device;
    } else {
        out->gpu_data = nullptr;
        out->gpu_row_bytes = 0;
        out->gpu_device = 0;
    }
    return true;
}

extern "C" void decklink_capture_close() {
    decklink_preview_gl_disable();
    if (g_input) {
        g_input->StopStreams();
        g_input->DisableVideoInput();
        g_input->SetCallback(nullptr);
        if (g_allocator) {
            g_input->SetVideoInputFrameMemoryAllocator(nullptr);
        }
    }
    if (g_callback) { g_callback->Release(); g_callback = nullptr; }
    if (g_input) { g_input->Release(); g_input = nullptr; }
    if (g_device) { g_device->Release(); g_device = nullptr; }
    if (g_iterator) { g_iterator->Release(); g_iterator = nullptr; }

    // Reset shared frame
    {
        std::lock_guard<std::mutex> lk(g_shared.mtx);
        g_shared.data.clear();
        g_shared.width = 0;
        g_shared.height = 0;
        g_shared.row_bytes = 0;
        g_shared.seq = 0;
        if (g_shared.cuda_stream || g_shared.gpu_ptr || g_shared.cuda_available) {
            cudaError_t err = cudaSetDevice(static_cast<int>(g_shared.cuda_device));
            if (err != cudaSuccess) {
                log_cuda_error("cudaSetDevice", err);
            }
        }
        if (g_shared.dvp_gpu_handle != 0) {
            DVPStatus free_status = dvpFreeBuffer(g_shared.dvp_gpu_handle);
            if (free_status != DVP_STATUS_OK) {
                log_dvp_error("dvpFreeBuffer", free_status);
            }
            g_shared.dvp_gpu_handle = 0;
        }
        if (g_shared.gpu_ptr) {
            cudaError_t err = cudaFree(g_shared.gpu_ptr);
            if (err != cudaSuccess) {
                log_cuda_error("cudaFree", err);
            }
            g_shared.gpu_ptr = nullptr;
        }
        if (g_shared.cuda_stream) {
            cudaError_t err = cudaStreamDestroy(g_shared.cuda_stream);
            if (err != cudaSuccess) {
                log_cuda_error("cudaStreamDestroy", err);
            }
            g_shared.cuda_stream = nullptr;
        }
        g_shared.gpu_pitch = 0;
        g_shared.gpu_width = 0;
        g_shared.gpu_height = 0;
        g_shared.gpu_ready = false;
        g_shared.cuda_initialized = false;
        g_shared.cuda_available = false;
        g_shared.cuda_device = 0;
    }

    if (g_allocator) {
        g_allocator->Release();
        g_allocator = nullptr;
    }

    if (g_dvp.initialized) {
        DVPStatus close_status = dvpCloseCUDAContext();
        if (close_status != DVP_STATUS_OK) {
            log_dvp_error("dvpCloseCUDAContext", close_status);
        }
        g_dvp = DvpContext{};
    }
}

extern "C" bool decklink_capture_copy_host_region(size_t offset, size_t len, void* dst) {
    if (!dst || len == 0) {
        return true;
    }
    std::lock_guard<std::mutex> lk(g_shared.mtx);
    if (g_shared.data.empty() || offset > g_shared.data.size() ||
        len > g_shared.data.size() || offset + len > g_shared.data.size()) {
        return false;
    }
    std::memcpy(dst, g_shared.data.data() + offset, len);
    return true;
}

extern "C" bool decklink_preview_gl_create() {
    std::lock_guard<std::mutex> lk(g_glMutex);
    if (g_glPreview)
        return true;
    IDeckLinkGLScreenPreviewHelper* helper = CreateOpenGLScreenPreviewHelper();
    if (!helper) {
        fprintf(stderr, "[shim] CreateOpenGLScreenPreviewHelper failed\n");
        return false;
    }
    g_glPreview = helper;
    g_gl_seq.store(0, std::memory_order_relaxed);
    g_gl_last_arrival_ns.store(0, std::memory_order_relaxed);
    g_gl_last_latency_ns.store(0, std::memory_order_relaxed);
    g_gl_last_render_seq.store(0, std::memory_order_relaxed);
    return true;
}

extern "C" bool decklink_preview_gl_initialize_gl() {
    std::lock_guard<std::mutex> lk(g_glMutex);
    if (!g_glPreview)
        return false;
    HRESULT hr = g_glPreview->InitializeGL();
    if (hr != S_OK) {
        fprintf(stderr, "[shim] IDeckLinkGLScreenPreviewHelper::InitializeGL failed hr=0x%08x\n", (unsigned)hr);
        return false;
    }
    return true;
}

extern "C" bool decklink_preview_gl_enable() {
    std::lock_guard<std::mutex> lk(g_glMutex);
    if (!g_glPreview) {
        if (!decklink_preview_gl_create())
            return false;
    }
    if (!g_glCallback)
        g_glCallback = new GLPreviewCallback();
    if (!g_input) {
        fprintf(stderr, "[shim] decklink_preview_gl_enable: input not initialized\n");
        return false;
    }
    HRESULT hr = g_input->SetScreenPreviewCallback(g_glCallback);
    if (hr != S_OK) {
        fprintf(stderr, "[shim] SetScreenPreviewCallback(GL) failed hr=0x%08x\n", (unsigned)hr);
        return false;
    }
    return true;
}

extern "C" bool decklink_preview_gl_render() {
    IDeckLinkVideoFrame_v14_2_1* frame = nullptr;
    uint64_t arrival = 0;
    {
        std::lock_guard<std::mutex> lk(g_glFrameMutex);
        if (g_glFrame) {
            g_glFrame->AddRef();
            frame = g_glFrame;
            arrival = g_gl_last_arrival_ns.load(std::memory_order_relaxed);
        }
    }
    if (!frame || !g_glPreview)
        return false;

    IDeckLinkVideoFrame* baseFrame = nullptr;
    HRESULT hr = frame->QueryInterface(IID_IDeckLinkVideoFrame, (void**)&baseFrame);
    if (hr == S_OK && baseFrame) {
        hr = g_glPreview->SetFrame(baseFrame);
        baseFrame->Release();
    } else {
        fprintf(stderr, "[shim] QueryInterface(IDeckLinkVideoFrame) failed hr=0x%08x\n", (unsigned)hr);
        frame->Release();
        return false;
    }
    frame->Release();
    if (hr != S_OK) {
        fprintf(stderr, "[shim] SetFrame failed hr=0x%08x\n", (unsigned)hr);
        return false;
    }
    hr = g_glPreview->PaintGL();
    if (hr != S_OK) {
        fprintf(stderr, "[shim] PaintGL failed hr=0x%08x\n", (unsigned)hr);
        return false;
    }
    uint64_t seq = g_gl_seq.load(std::memory_order_relaxed);
    uint64_t last_render = g_gl_last_render_seq.load(std::memory_order_relaxed);
    if (seq != last_render) {
        auto now_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        if (arrival != 0 && now_ns >= arrival)
            g_gl_last_latency_ns.store(now_ns - arrival, std::memory_order_relaxed);
        g_gl_last_render_seq.store(seq, std::memory_order_relaxed);
    }
    return true;
}

extern "C" void decklink_preview_gl_disable() {
    if (g_input)
        (void)g_input->SetScreenPreviewCallback(nullptr);
    {
        std::lock_guard<std::mutex> lk(g_glFrameMutex);
        if (g_glFrame) {
            g_glFrame->Release();
            g_glFrame = nullptr;
        }
    }
    std::lock_guard<std::mutex> lk(g_glMutex);
    if (g_glCallback) {
        g_glCallback->Release();
        g_glCallback = nullptr;
    }
    g_gl_seq.store(0, std::memory_order_relaxed);
    g_gl_last_arrival_ns.store(0, std::memory_order_relaxed);
    g_gl_last_latency_ns.store(0, std::memory_order_relaxed);
    g_gl_last_render_seq.store(0, std::memory_order_relaxed);
}

extern "C" void decklink_preview_gl_destroy() {
    decklink_preview_gl_disable();
    std::lock_guard<std::mutex> lk(g_glMutex);
    if (g_glPreview) {
        g_glPreview->Release();
        g_glPreview = nullptr;
    }
}

extern "C" uint64_t decklink_preview_gl_seq() {
    return g_gl_seq.load(std::memory_order_relaxed);
}

extern "C" uint64_t decklink_preview_gl_last_timestamp_ns() {
    return g_gl_last_arrival_ns.load(std::memory_order_relaxed);
}

extern "C" uint64_t decklink_preview_gl_last_latency_ns() {
    return g_gl_last_latency_ns.load(std::memory_order_relaxed);
}

// ---- Get detected input format (for internal keying sync) ----

extern "C" bool decklink_get_detected_input_format(int32_t* out_width, int32_t* out_height, double* out_fps, uint32_t* out_mode) {
    if (g_detected_input_mode == bmdModeUnknown || g_detected_input_fps == 0.0) {
        return false;
    }

    // Get mode details
    IDeckLinkDisplayMode* mode = nullptr;
    if (g_device) {
        IDeckLinkInput_v14_2_1* input_tmp = nullptr;
        if (g_device->QueryInterface(IID_IDeckLinkInput_v14_2_1, (void**)&input_tmp) == S_OK && input_tmp) {
            IDeckLinkDisplayModeIterator* iter = nullptr;
            if (input_tmp->GetDisplayModeIterator(&iter) == S_OK && iter) {
                while (iter->Next(&mode) == S_OK && mode) {
                    if (mode->GetDisplayMode() == g_detected_input_mode) {
                        break;
                    }
                    mode->Release();
                    mode = nullptr;
                }
                iter->Release();
            }
            input_tmp->Release();
        }
    }

    if (mode) {
        if (out_width) *out_width = mode->GetWidth();
        if (out_height) *out_height = mode->GetHeight();
        if (out_fps) *out_fps = g_detected_input_fps;
        if (out_mode) *out_mode = (uint32_t)g_detected_input_mode;
        mode->Release();
        return true;
    }

    return false;
}

// ---- Output implementation ----

namespace {
    static IDeckLinkIterator* g_output_iterator = nullptr;
    static IDeckLink* g_output_device = nullptr;
    static IDeckLinkOutput_v14_2_1* g_output = nullptr;
    static BMDDisplayMode g_output_display_mode = bmdModeHD1080p6000;
    static int32_t g_output_width = 0;
    static int32_t g_output_height = 0;
    static int32_t g_output_fps_num = 60;
    static int32_t g_output_fps_den = 1;
}

extern "C" bool decklink_output_open(int32_t device_index, int32_t width, int32_t height, double fps) {
    if (g_output) {
        fprintf(stderr, "[shim][output] Already opened\n");
        return true;
    }

    g_output_iterator = CreateDeckLinkIteratorInstance();
    if (!g_output_iterator) {
        fprintf(stderr, "[shim][output] Failed to create DeckLink iterator\n");
        return false;
    }

    IDeckLink* dev = nullptr;
    int32_t idx = 0;
    while (g_output_iterator->Next(&dev) == S_OK && dev) {
        if (idx == device_index) {
            g_output_device = dev;
            break;
        }
        dev->Release();
        idx++;
    }

    if (!g_output_device) {
        fprintf(stderr, "[shim][output] Device index %d not found\n", device_index);
        g_output_iterator->Release();
        g_output_iterator = nullptr;
        return false;
    }

    HRESULT qri = g_output_device->QueryInterface(IID_IDeckLinkOutput_v14_2_1, (void**)&g_output);
    if (qri != S_OK || !g_output) {
        fprintf(stderr, "[shim][output] QueryInterface(IDeckLinkOutput_v14_2_1) failed hr=0x%08x\n", (unsigned)qri);
        g_output_device->Release();
        g_output_device = nullptr;
        g_output_iterator->Release();
        g_output_iterator = nullptr;
        return false;
    }

    BMDDisplayMode chosen_mode = bmdModeUnknown;
    
    // CRITICAL FOR INTERNAL KEYING: Use exact input mode if available
    if (g_detected_input_mode != bmdModeUnknown && g_detected_input_fps > 0.0) {
        fprintf(stderr, "[shim][output] Using detected input mode=0x%x (fps=%.3f) for internal keying sync\n",
                (unsigned)g_detected_input_mode, g_detected_input_fps);
        chosen_mode = g_detected_input_mode;
        
        // Get framerate from the mode
        IDeckLinkDisplayModeIterator* mode_it = nullptr;
        if (g_output->GetDisplayModeIterator(&mode_it) == S_OK && mode_it) {
            IDeckLinkDisplayMode* mode = nullptr;
            while (mode_it->Next(&mode) == S_OK && mode) {
                if (mode->GetDisplayMode() == g_detected_input_mode) {
                    BMDTimeValue frame_duration;
                    BMDTimeScale time_scale;
                    mode->GetFrameRate(&frame_duration, &time_scale);
                    g_output_fps_num = (int32_t)time_scale;
                    g_output_fps_den = (int32_t)frame_duration;
                    mode->Release();
                    break;
                }
                mode->Release();
            }
            mode_it->Release();
        }
    } else {
        // Fallback: Find matching display mode by resolution and fps
        fprintf(stderr, "[shim][output] No detected input mode, searching for %dx%d@%.2ffps\n", 
                width, height, fps);
        
        IDeckLinkDisplayModeIterator* mode_it = nullptr;
        if (g_output->GetDisplayModeIterator(&mode_it) != S_OK || !mode_it) {
            fprintf(stderr, "[shim][output] Failed to get display mode iterator\n");
            g_output->Release();
            g_output = nullptr;
            g_output_device->Release();
            g_output_device = nullptr;
            g_output_iterator->Release();
            g_output_iterator = nullptr;
            return false;
        }

        IDeckLinkDisplayMode* mode = nullptr;
        while (mode_it->Next(&mode) == S_OK && mode) {
            long mode_width = mode->GetWidth();
            long mode_height = mode->GetHeight();
            BMDTimeValue frame_duration;
            BMDTimeScale time_scale;
            mode->GetFrameRate(&frame_duration, &time_scale);
            double mode_fps = (double)time_scale / (double)frame_duration;

            // Match resolution and framerate (with tolerance)
            if (mode_width == width && mode_height == height && std::abs(mode_fps - fps) < 1.0) {
                chosen_mode = mode->GetDisplayMode();
                g_output_fps_num = (int32_t)time_scale;
                g_output_fps_den = (int32_t)frame_duration;
                mode->Release();
                break;
            }
            mode->Release();
        }
        mode_it->Release();
    }

    if (chosen_mode == bmdModeUnknown) {
        // Fallback: try 1080p60 for 1920x1080@60fps
        if (width == 1920 && height == 1080 && std::abs(fps - 60.0) < 1.0) {
            chosen_mode = bmdModeHD1080p6000;
            g_output_fps_num = 60000;
            g_output_fps_den = 1000;
        } else {
            fprintf(stderr, "[shim][output] No matching display mode found for %dx%d@%.2ffps\n", width, height, fps);
            g_output->Release();
            g_output = nullptr;
            g_output_device->Release();
            g_output_device = nullptr;
            g_output_iterator->Release();
            g_output_iterator = nullptr;
            return false;
        }
    }

    g_output_display_mode = chosen_mode;
    g_output_width = width;
    g_output_height = height;

    HRESULT enable_hr = g_output->EnableVideoOutput(chosen_mode, bmdVideoOutputFlagDefault);
    if (enable_hr != S_OK) {
        fprintf(stderr, "[shim][output] EnableVideoOutput failed hr=0x%08x\n", (unsigned)enable_hr);
        g_output->Release();
        g_output = nullptr;
        g_output_device->Release();
        g_output_device = nullptr;
        g_output_iterator->Release();
        g_output_iterator = nullptr;
        return false;
    }

    fprintf(stderr, "[shim][output] Opened device %d: %dx%d@%.2ffps (mode=%u)\n",
            device_index, width, height, fps, (unsigned)chosen_mode);
    return true;
}

extern "C" bool decklink_output_send_frame(const uint8_t* bgra_data, int32_t width, int32_t height) {
    if (!g_output || !bgra_data) {
        return false;
    }

    if (width != g_output_width || height != g_output_height) {
        fprintf(stderr, "[shim][output] Frame dimension mismatch: expected %dx%d, got %dx%d\n",
                g_output_width, g_output_height, width, height);
        return false;
    }

    // Create video frame (BGRA format: bmdFormat8BitBGRA)
    IDeckLinkMutableVideoFrame_v14_2_1* frame = nullptr;
    int32_t row_bytes = width * 4; // BGRA = 4 bytes per pixel
    HRESULT create_hr = g_output->CreateVideoFrame(width, height, row_bytes, bmdFormat8BitBGRA, bmdFrameFlagDefault, &frame);
    if (create_hr != S_OK || !frame) {
        fprintf(stderr, "[shim][output] CreateVideoFrame failed hr=0x%08x\n", (unsigned)create_hr);
        return false;
    }

    // Copy BGRA data to frame
    void* frame_bytes = nullptr;
    HRESULT getbytes_hr = frame->GetBytes(&frame_bytes);
    if (getbytes_hr != S_OK || !frame_bytes) {
        fprintf(stderr, "[shim][output] GetBytes failed hr=0x%08x\n", (unsigned)getbytes_hr);
        frame->Release();
        return false;
    }

    std::memcpy(frame_bytes, bgra_data, (size_t)(row_bytes * height));

    // Display frame synchronously
    HRESULT display_hr = g_output->DisplayVideoFrameSync(frame);
    frame->Release();

    if (display_hr != S_OK) {
        fprintf(stderr, "[shim][output] DisplayVideoFrameSync failed hr=0x%08x\n", (unsigned)display_hr);
        return false;
    }

    return true;
}

extern "C" bool decklink_output_send_frame_gpu(const uint8_t* gpu_bgra_data, int32_t gpu_pitch, int32_t width, int32_t height) {
    if (!g_output || !gpu_bgra_data) {
        return false;
    }

    if (width != g_output_width || height != g_output_height) {
        fprintf(stderr, "[shim][output] Frame dimension mismatch: expected %dx%d, got %dx%d\n",
                g_output_width, g_output_height, width, height);
        return false;
    }

    // Create video frame (BGRA format: bmdFormat8BitBGRA)
    IDeckLinkMutableVideoFrame_v14_2_1* frame = nullptr;
    int32_t row_bytes = width * 4; // BGRA = 4 bytes per pixel
    HRESULT create_hr = g_output->CreateVideoFrame(width, height, row_bytes, bmdFormat8BitBGRA, bmdFrameFlagDefault, &frame);
    if (create_hr != S_OK || !frame) {
        fprintf(stderr, "[shim][output] CreateVideoFrame failed hr=0x%08x\n", (unsigned)create_hr);
        return false;
    }

    // Get frame buffer pointer (CPU-side)
    void* frame_bytes = nullptr;
    HRESULT getbytes_hr = frame->GetBytes(&frame_bytes);
    if (getbytes_hr != S_OK || !frame_bytes) {
        fprintf(stderr, "[shim][output] GetBytes failed hr=0x%08x\n", (unsigned)getbytes_hr);
        frame->Release();
        return false;
    }

    // Copy directly from GPU to DeckLink frame buffer (GPU → CPU)
    cudaError_t copy_err = cudaMemcpy2D(
        frame_bytes,           // dst (CPU/DeckLink buffer)
        row_bytes,             // dst pitch
        gpu_bgra_data,         // src (GPU)
        gpu_pitch,             // src pitch
        row_bytes,             // width in bytes
        height,                // height
        cudaMemcpyDeviceToHost
    );

    if (copy_err != cudaSuccess) {
        fprintf(stderr, "[shim][output] cudaMemcpy2D failed: %s\n", cudaGetErrorString(copy_err));
        frame->Release();
        return false;
    }

    // Display frame synchronously
    HRESULT display_hr = g_output->DisplayVideoFrameSync(frame);
    frame->Release();

    if (display_hr != S_OK) {
        fprintf(stderr, "[shim][output] DisplayVideoFrameSync failed hr=0x%08x\n", (unsigned)display_hr);
        return false;
    }

    return true;
}

extern "C" void decklink_output_close() {
    if (g_output) {
        g_output->DisableVideoOutput();
        g_output->Release();
        g_output = nullptr;
    }
    if (g_output_device) {
        g_output_device->Release();
        g_output_device = nullptr;
    }
    if (g_output_iterator) {
        g_output_iterator->Release();
        g_output_iterator = nullptr;
    }
    g_output_width = 0;
    g_output_height = 0;
    g_output_display_mode = bmdModeUnknown;
}

// ============================================================================
// Keyer Control Functions
// ============================================================================

extern "C" bool decklink_keyer_enable_internal() {
    if (!g_output_device) {
        fprintf(stderr, "[shim] keyer_enable_internal: Output device not initialized\n");
        return false;
    }

    IDeckLinkKeyer* keyer = nullptr;
    HRESULT result = g_output_device->QueryInterface(IID_IDeckLinkKeyer, (void**)&keyer);
    
    if (result != S_OK || !keyer) {
        fprintf(stderr, "[shim] keyer_enable_internal: Device does not support keying\n");
        return false;
    }

    // Enable internal keying (false = internal, true = external)
    result = keyer->Enable(false);
    keyer->Release();
    
    if (result != S_OK) {
        fprintf(stderr, "[shim] keyer_enable_internal: Failed to enable internal keyer (0x%08x)\n", result);
        return false;
    }

    fprintf(stderr, "[shim] Internal keyer enabled\n");
    return true;
}

extern "C" bool decklink_keyer_set_level(uint8_t level) {
    if (!g_output_device) {
        fprintf(stderr, "[shim] keyer_set_level: Output device not initialized\n");
        return false;
    }

    IDeckLinkKeyer* keyer = nullptr;
    HRESULT result = g_output_device->QueryInterface(IID_IDeckLinkKeyer, (void**)&keyer);
    
    if (result != S_OK || !keyer) {
        fprintf(stderr, "[shim] keyer_set_level: Device does not support keying\n");
        return false;
    }

    result = keyer->SetLevel(level);
    keyer->Release();
    
    if (result != S_OK) {
        fprintf(stderr, "[shim] keyer_set_level: Failed to set level (0x%08x)\n", result);
        return false;
    }

    return true;
}

extern "C" bool decklink_keyer_disable() {
    if (!g_output_device) {
        fprintf(stderr, "[shim] keyer_disable: Output device not initialized\n");
        return false;
    }

    IDeckLinkKeyer* keyer = nullptr;
    HRESULT result = g_output_device->QueryInterface(IID_IDeckLinkKeyer, (void**)&keyer);
    
    if (result != S_OK || !keyer) {
        fprintf(stderr, "[shim] keyer_disable: Device does not support keying\n");
        return false;
    }

    result = keyer->Disable();
    keyer->Release();
    
    if (result != S_OK) {
        fprintf(stderr, "[shim] keyer_disable: Failed to disable keyer (0x%08x)\n", result);
        return false;
    }

    fprintf(stderr, "[shim] Keyer disabled\n");
    return true;
}

extern "C" int64_t decklink_get_connection_sdi() {
    return bmdVideoConnectionSDI;
}

extern "C" bool decklink_set_video_output_connection(int64_t connection) {
    if (!g_output_device) {
        fprintf(stderr, "[shim] set_video_output_connection: Output device not initialized\n");
        return false;
    }

    IDeckLinkConfiguration* config = nullptr;
    HRESULT result = g_output_device->QueryInterface(IID_IDeckLinkConfiguration, (void**)&config);
    
    if (result != S_OK || !config) {
        fprintf(stderr, "[shim] set_video_output_connection: Failed to get configuration interface\n");
        return false;
    }

    result = config->SetInt(bmdDeckLinkConfigVideoOutputConnection, connection);
    config->Release();
    
    if (result != S_OK) {
        fprintf(stderr, "[shim] set_video_output_connection: Failed to set connection (0x%08x)\n", result);
        return false;
    }

    return true;
}
