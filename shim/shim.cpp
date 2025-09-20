// /Users/yamaearth/Documents/3_1/Capstone/Blackmagic DeckLink SDK 14.4/rust/shim/shim.cpp
#include <vector>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <cstdio>
#include <chrono>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <cerrno>

#if defined(__linux__)
#include <sys/mman.h>
#include <sys/resource.h>
#endif

#include "DVPAPI.h"
#include "dvpapi_gl.h"

#if defined(_WIN32)
  #include <windows.h>
#elif defined(__APPLE__)
  #include <CoreFoundation/CoreFoundation.h>
  #include <CoreVideo/CoreVideo.h>
#endif

// Include DeckLink SDK header (ensure include path points to where this header lives)
#include "DeckLinkAPI.h"
#include "DeckLinkAPIModes.h"
#include "DeckLinkAPITypes.h"
#include "DeckLinkAPIConfiguration.h"
#include "DeckLinkAPIMemoryAllocator_v14_2_1.h"
#include "DeckLinkAPIVideoInput_v14_2_1.h"
#include "DeckLinkAPIVideoFrame_v14_2_1.h"
#include "DeckLinkAPIScreenPreviewCallback_v14_2_1.h"
#include "DeckLinkAPI_v14_2_1.h"

#if !defined(__APPLE__)
static const REFIID kIID_IUnknown = IID_IUnknown;
#endif

extern "C" {

struct DLDeviceList {
    char** names;
    int32_t count;
};

static char* dup_utf8_from_bmd_str(
#if defined(_WIN32)
    BSTR s
#elif defined(__APPLE__)
    CFStringRef s
#else
    const char* s
#endif
) {
#if defined(_WIN32)
    if (!s) return strdup("(unknown)");
    int size = WideCharToMultiByte(CP_UTF8, 0, s, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 0) return strdup("(unknown)");
    char* out = (char*)std::malloc((size_t)size);
    if (!out) return nullptr;
    WideCharToMultiByte(CP_UTF8, 0, s, -1, out, size, nullptr, nullptr);
    return out;
#elif defined(__APPLE__)
    if (!s) return strdup("(unknown)");
    CFIndex len = CFStringGetLength(s);
    CFIndex maxSize = CFStringGetMaximumSizeForEncoding(len, kCFStringEncodingUTF8) + 1;
    char* out = (char*)std::malloc((size_t)maxSize);
    if (!out) return nullptr;
    if (!CFStringGetCString(s, out, maxSize, kCFStringEncodingUTF8)) {
        std::free(out);
        return strdup("(unknown)");
    }
    return out;
#else
    if (!s) return strdup("(unknown)");
    return strdup(s);
#endif
}

static void release_bmd_str(
#if defined(_WIN32)
    BSTR s
#elif defined(__APPLE__)
    CFStringRef s
#else
    const char* s
#endif
) {
#if defined(_WIN32)
    if (s) SysFreeString(s);
#elif defined(__APPLE__)
    if (s) CFRelease(s);
#else
    if (s) std::free((void*)s); // Linux SDK returns malloc'd char*
#endif
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
#if defined(_WIN32)
        BSTR bname = nullptr;
#elif defined(__APPLE__)
        CFStringRef bname = nullptr;
#else
        const char* bname = nullptr;
#endif
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
};

struct DeckLinkDVPFrame {
    uint64_t seq;
    uint64_t timestamp_ns;
    uint64_t sysmem_handle;
    uint64_t sync_handle;
    uint64_t semaphore_addr;
    const uint8_t* cpu_ptr;
    uint64_t buffer_size;
    uint32_t width;
    uint32_t height;
    uint32_t row_bytes;
    uint32_t pixel_format;
    uint32_t release_value;
    uint32_t reserved;
};

bool decklink_capture_open(int32_t device_index);
bool decklink_capture_get_frame(CaptureFrame* out);
void decklink_capture_close();
bool decklink_capture_latest_dvp_frame(DeckLinkDVPFrame* out);
void decklink_capture_reset_dvp_fence();

} // extern "C"

// ---- C++ implementation for capture ----

namespace {

struct DVPRequirements {
    uint32_t bufferAddrAlignment = 4096;
    uint32_t bufferStrideAlignment = 256;
    uint32_t semaphoreAddrAlignment = 64;
    uint32_t semaphoreAllocSize = 0;
    uint32_t semaphorePayloadOffset = 0;
    uint32_t semaphorePayloadSize = 0;
};

struct DVPBufferRecord {
    void* basePtr = nullptr;
    size_t size = 0;
    DVPBufferHandle sysmemHandle = 0;
    DVPSyncObjectHandle syncHandle = 0;
    uint32_t* semaphore = nullptr;
    void* semaphoreBacking = nullptr;
    uint32_t releaseValue = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t rowBytes = 0;
    BMDPixelFormat pixelFormat = bmdFormatUnspecified;
    uint64_t seq = 0;
    uint64_t timestampNs = 0;
};

struct LatestFrameState {
    DVPBufferRecord* buffer = nullptr;
    uint64_t seq = 0;
};

static std::mutex g_dvpInitMutex;
static bool g_dvpInitialized = false;
static DVPRequirements g_dvpRequirements{};

class GPUDirectAllocator : public IDeckLinkMemoryAllocator_v14_2_1 {
public:
    GPUDirectAllocator();
    virtual ~GPUDirectAllocator();

    // IUnknown
    HRESULT QueryInterface(REFIID iid, void** ppv) override;
    ULONG AddRef() override;
    ULONG Release() override;

    // IDeckLinkMemoryAllocator_v14_2_1
    HRESULT AllocateBuffer(uint32_t bufferSize, void** allocatedBuffer) override;
    HRESULT ReleaseBuffer(void* buffer) override;
    HRESULT Commit() override;
    HRESULT Decommit() override;

    DVPBufferRecord* resolve(void* ptr);
    void evacuateAll();

private:
    struct BufferEntry {
        std::unique_ptr<DVPBufferRecord> record;
    };

    std::mutex m_mutex;
    std::unordered_map<void*, BufferEntry> m_buffers;
    std::atomic<ULONG> m_refCount{1};
    bool m_committed = false;
};

static IDeckLinkIterator* g_iterator = nullptr;
static IDeckLink* g_device = nullptr;
static IDeckLinkInput_v14_2_1* g_input = nullptr;
static GPUDirectAllocator* g_allocator = nullptr;
static LatestFrameState g_latestFrame{};
static std::mutex g_latestMutex;
static std::atomic<uint64_t> g_frameSeq{0};
static IDeckLinkScreenPreviewCallback_v14_2_1* g_screenPreview = nullptr; // Cocoa Screen Preview (NSView)
static std::atomic<uint64_t> g_preview_seq{0};

// OpenGL preview helper (cross-platform)
static IDeckLinkGLScreenPreviewHelper* g_glPreview = nullptr;
static IDeckLinkVideoFrame_v14_2_1* g_glFrame = nullptr;
static IDeckLinkScreenPreviewCallback_v14_2_1* g_glCallback = nullptr;
static std::mutex g_glMutex;
static std::mutex g_glFrameMutex;
static std::atomic<uint64_t> g_gl_seq{0};
static std::atomic<uint64_t> g_gl_last_arrival_ns{0};
static std::atomic<uint64_t> g_gl_last_latency_ns{0};
static std::atomic<uint32_t> g_last_getbytes_qrv{0};
static std::atomic<uint32_t> g_last_getbytes_pix{0};
static std::atomic<uint64_t> g_gl_last_render_seq{0};

#if defined(__linux__)
static void ensure_memlock_limit(size_t bytes)
{
    struct rlimit limit;
    if (getrlimit(RLIMIT_MEMLOCK, &limit) != 0) {
        fprintf(stderr, "[shim] getrlimit(RLIMIT_MEMLOCK) failed: %s\n", strerror(errno));
        return;
    }

    rlim_t desired = static_cast<rlim_t>(bytes);
    if (limit.rlim_cur >= desired && (limit.rlim_max == RLIM_INFINITY || limit.rlim_max >= desired))
        return;

    struct rlimit newLimit = limit;
    if (newLimit.rlim_max < desired && newLimit.rlim_max != RLIM_INFINITY)
        newLimit.rlim_max = desired;
    newLimit.rlim_cur = desired;
    if (setrlimit(RLIMIT_MEMLOCK, &newLimit) != 0) {
        fprintf(stderr, "[shim] setrlimit(RLIMIT_MEMLOCK) failed: %s\n", strerror(errno));
    }
}
#else
static void ensure_memlock_limit(size_t) {}
#endif

static bool ensure_gpudirect_initialized()
{
    std::lock_guard<std::mutex> lk(g_dvpInitMutex);
    if (g_dvpInitialized)
        return true;

    ensure_memlock_limit(512ull * 1024ull * 1024ull);

    DVPStatus status = dvpInitGLContext(DVP_DEVICE_FLAGS_SHARE_APP_CONTEXT);
    if (status != DVP_STATUS_OK) {
        fprintf(stderr, "[shim] dvpInitGLContext failed (status=%d)\n", (int)status);
        return false;
    }

    status = dvpGetRequiredConstantsGLCtx(&g_dvpRequirements.bufferAddrAlignment,
                                          &g_dvpRequirements.bufferStrideAlignment,
                                          &g_dvpRequirements.semaphoreAddrAlignment,
                                          &g_dvpRequirements.semaphoreAllocSize,
                                          &g_dvpRequirements.semaphorePayloadOffset,
                                          &g_dvpRequirements.semaphorePayloadSize);
    if (status != DVP_STATUS_OK) {
        fprintf(stderr, "[shim] dvpGetRequiredConstantsGLCtx failed (status=%d)\n", (int)status);
        dvpCloseGLContext();
        return false;
    }

    if (g_dvpRequirements.bufferAddrAlignment == 0)
        g_dvpRequirements.bufferAddrAlignment = 4096;
    if (g_dvpRequirements.semaphoreAddrAlignment == 0)
        g_dvpRequirements.semaphoreAddrAlignment = 64;
    if (g_dvpRequirements.semaphoreAllocSize == 0)
        g_dvpRequirements.semaphoreAllocSize = 4096;
    if (g_dvpRequirements.semaphorePayloadSize == 0)
        g_dvpRequirements.semaphorePayloadSize = sizeof(uint32_t);

    g_dvpInitialized = true;
    return true;
}

static void shutdown_gpudirect()
{
    std::lock_guard<std::mutex> lk(g_dvpInitMutex);
    if (!g_dvpInitialized)
        return;
    dvpCloseGLContext();
    g_dvpInitialized = false;
    g_dvpRequirements = DVPRequirements{};
}

static void reset_latest_frame()
{
    std::lock_guard<std::mutex> lk(g_latestMutex);
    g_latestFrame = LatestFrameState{};
}

GPUDirectAllocator::GPUDirectAllocator() = default;

GPUDirectAllocator::~GPUDirectAllocator()
{
    evacuateAll();
}

HRESULT GPUDirectAllocator::QueryInterface(REFIID iid, void** ppv)
{
    if (!ppv)
        return E_POINTER;

#if defined(__APPLE__)
    CFUUIDBytes iunknown = CFUUIDGetUUIDBytes(IUnknownUUID);
    if (memcmp(&iid, &iunknown, sizeof(CFUUIDBytes)) == 0) {
        *ppv = static_cast<IUnknown*>(this);
        AddRef();
        return S_OK;
    }
    if (memcmp(&iid, &IID_IDeckLinkMemoryAllocator_v14_2_1, sizeof(CFUUIDBytes)) == 0) {
        *ppv = static_cast<IDeckLinkMemoryAllocator_v14_2_1*>(this);
        AddRef();
        return S_OK;
    }
#else
    if (memcmp(&iid, &kIID_IUnknown, sizeof(REFIID)) == 0) {
        *ppv = static_cast<IUnknown*>(this);
        AddRef();
        return S_OK;
    }
    if (memcmp(&iid, &IID_IDeckLinkMemoryAllocator_v14_2_1, sizeof(REFIID)) == 0) {
        *ppv = static_cast<IDeckLinkMemoryAllocator_v14_2_1*>(this);
        AddRef();
        return S_OK;
    }
#endif

    *ppv = nullptr;
    return E_NOINTERFACE;
}

ULONG GPUDirectAllocator::AddRef()
{
    return ++m_refCount;
}

ULONG GPUDirectAllocator::Release()
{
    ULONG v = --m_refCount;
    if (v == 0)
        delete this;
    return v;
}

HRESULT GPUDirectAllocator::AllocateBuffer(uint32_t bufferSize, void** allocatedBuffer)
{
    if (!allocatedBuffer || bufferSize == 0)
        return E_INVALIDARG;

    if (!ensure_gpudirect_initialized())
        return E_FAIL;

    size_t requested = static_cast<size_t>(bufferSize);
    size_t alignment = std::max<size_t>(g_dvpRequirements.bufferAddrAlignment, 4096);
    void* base = nullptr;
    int err = posix_memalign(&base, alignment, requested);
    if (err != 0 || !base) {
        fprintf(stderr, "[shim] posix_memalign failed (err=%d)\n", err);
        return E_OUTOFMEMORY;
    }

#if defined(__linux__)
    if (mlock(base, requested) != 0) {
        fprintf(stderr, "[shim] mlock failed: %s\n", strerror(errno));
    }
#endif

    DVPSysmemBufferDesc desc{};
    desc.width = 0;
    desc.height = 0;
    desc.stride = 0;
    desc.size = bufferSize;
    desc.format = DVP_BUFFER;
    desc.type = DVP_UNSIGNED_BYTE;
    desc.bufAddr = base;

    DVPBufferHandle sysmemHandle = 0;
    DVPStatus status = dvpCreateBuffer(&desc, &sysmemHandle);
    if (status != DVP_STATUS_OK) {
        fprintf(stderr, "[shim] dvpCreateBuffer failed (status=%d)\n", (int)status);
#if defined(__linux__)
        munlock(base, requested);
#endif
        std::free(base);
        return E_FAIL;
    }

    const size_t semaphoreAllocSize = std::max<size_t>(g_dvpRequirements.semaphoreAllocSize, sizeof(uint32_t));
    const size_t semaphoreAlignment = std::max<size_t>(g_dvpRequirements.semaphoreAddrAlignment, alignof(uint32_t));
    void* semBacking = std::malloc(semaphoreAllocSize + semaphoreAlignment);
    if (!semBacking) {
        fprintf(stderr, "[shim] semaphore allocation failed\n");
        dvpDestroyBuffer(sysmemHandle);
#if defined(__linux__)
        munlock(base, requested);
#endif
        std::free(base);
        return E_OUTOFMEMORY;
    }
    uintptr_t semAddr = reinterpret_cast<uintptr_t>(semBacking);
    semAddr = (semAddr + semaphoreAlignment - 1) & ~(static_cast<uintptr_t>(semaphoreAlignment) - 1);
    uint32_t* semaphore = reinterpret_cast<uint32_t*>(semAddr);
    std::memset(semaphore, 0, g_dvpRequirements.semaphorePayloadSize);

    DVPSyncObjectDesc syncDesc{};
    syncDesc.sem = semaphore;
    syncDesc.flags = 0;
    syncDesc.externalClientWaitFunc = nullptr;

    DVPSyncObjectHandle syncHandle = 0;
    status = dvpImportSyncObject(&syncDesc, &syncHandle);
    if (status != DVP_STATUS_OK) {
        fprintf(stderr, "[shim] dvpImportSyncObject failed (status=%d)\n", (int)status);
        std::free(semBacking);
        dvpDestroyBuffer(sysmemHandle);
#if defined(__linux__)
        munlock(base, requested);
#endif
        std::free(base);
        return E_FAIL;
    }

    auto record = std::make_unique<DVPBufferRecord>();
    record->basePtr = base;
    record->size = requested;
    record->sysmemHandle = sysmemHandle;
    record->syncHandle = syncHandle;
    record->semaphore = semaphore;
    record->semaphoreBacking = semBacking;

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_buffers.emplace(base, BufferEntry{std::move(record)});
    }

    *allocatedBuffer = base;
    return S_OK;
}

HRESULT GPUDirectAllocator::ReleaseBuffer(void* buffer)
{
    if (!buffer)
        return S_OK;

    std::unique_ptr<DVPBufferRecord> record;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_buffers.find(buffer);
        if (it == m_buffers.end())
            return S_FALSE;
        record = std::move(it->second.record);
        m_buffers.erase(it);
    }

    if (record) {
        if (record->syncHandle)
            dvpFreeSyncObject(record->syncHandle);
        if (record->sysmemHandle)
            dvpDestroyBuffer(record->sysmemHandle);
        if (record->semaphoreBacking)
            std::free(record->semaphoreBacking);
#if defined(__linux__)
        if (record->basePtr && record->size)
            munlock(record->basePtr, record->size);
#endif
        if (record->basePtr)
            std::free(record->basePtr);
    }

    return S_OK;
}

HRESULT GPUDirectAllocator::Commit()
{
    m_committed = true;
    return S_OK;
}

HRESULT GPUDirectAllocator::Decommit()
{
    m_committed = false;
    evacuateAll();
    return S_OK;
}

DVPBufferRecord* GPUDirectAllocator::resolve(void* ptr)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_buffers.find(ptr);
    if (it == m_buffers.end())
        return nullptr;
    return it->second.record.get();
}

void GPUDirectAllocator::evacuateAll()
{
    std::vector<void*> keys;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        keys.reserve(m_buffers.size());
        for (auto& kv : m_buffers)
            keys.push_back(kv.first);
    }
    for (void* ptr : keys)
        ReleaseBuffer(ptr);
}

extern "C" void decklink_preview_gl_disable();

class CaptureCallback : public IDeckLinkInputCallback_v14_2_1 {
public:
    CaptureCallback() : m_ref(1) {}

    // IUnknown
    HRESULT QueryInterface(REFIID iid, void** ppv) override {
        if (!ppv) return E_POINTER;
#if defined(__APPLE__)
        // macOS style IUnknown UUID เปรียบเทียบผ่าน CoreFoundation
        CFUUIDBytes iunknown = CFUUIDGetUUIDBytes(IUnknownUUID);
        if (memcmp(&iid, &iunknown, sizeof(CFUUIDBytes)) == 0) {
            *ppv = static_cast<IUnknown*>(this);
            AddRef();
            return S_OK;
        }
        if (memcmp(&iid, &IID_IDeckLinkInputCallback_v14_2_1, sizeof(CFUUIDBytes)) == 0) {
            *ppv = static_cast<IDeckLinkInputCallback_v14_2_1*>(this);
            AddRef();
            return S_OK;
        }
#else
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
#endif
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
    HRESULT VideoInputFormatChanged(BMDVideoInputFormatChangedEvents /*events*/, IDeckLinkDisplayMode* newDisplayMode, BMDDetectedVideoInputFormatFlags detectedSignalFlags) override {
        if (!newDisplayMode || !g_input) return S_OK;
        BMDPixelFormat pix = bmdFormat10BitYUV;
        if (detectedSignalFlags & bmdDetectedVideoInputRGB444)
            pix = bmdFormat8BitBGRA;
        else if (detectedSignalFlags & bmdDetectedVideoInputYCbCr422)
            pix = bmdFormat10BitYUV;
        BMDTimeValue fd = 0; BMDTimeScale ts = 0;
        newDisplayMode->GetFrameRate(&fd, &ts);
        long w = newDisplayMode->GetWidth();
        long h = newDisplayMode->GetHeight();
        fprintf(stderr, "[shim] FormatChanged: %ldx%ld, fps=%.3f, pix=0x%x\n", w, h, (ts && fd) ? (double)ts/(double)fd : 0.0, (unsigned)pix);

        // Avoid reconfiguring to the same mode/pixel format repeatedly
        static BMDDisplayMode s_lastMode = (BMDDisplayMode)0;
        static BMDPixelFormat s_lastPix = (BMDPixelFormat)0;
        if (s_lastMode == newDisplayMode->GetDisplayMode() && s_lastPix == pix) {
            return S_OK;
        }

        g_input->StopStreams();
        HRESULT en = g_input->EnableVideoInput(newDisplayMode->GetDisplayMode(), pix, bmdVideoInputEnableFormatDetection);
        if (en != S_OK && (pix == bmdFormat10BitYUV)) {
            // Fallback to 8-bit UYVY if 10-bit fails
            en = g_input->EnableVideoInput(newDisplayMode->GetDisplayMode(), bmdFormat8BitYUV, bmdVideoInputEnableFormatDetection);
        }
        g_input->StartStreams();

        s_lastMode = newDisplayMode->GetDisplayMode();
        s_lastPix = pix;
        return S_OK;
    }

    HRESULT VideoInputFrameArrived(IDeckLinkVideoInputFrame_v14_2_1* videoFrame, IDeckLinkAudioInputPacket* /*audioPacket*/) override {
        if (!videoFrame)
            return S_OK;

        // If a screen preview is attached, let DeckLink render directly into NSView.
        // Avoid any CPU-side GetBytes/copy/convert to minimize latency.
        if (g_screenPreview != nullptr) {
            g_preview_seq.fetch_add(1, std::memory_order_relaxed);
            return S_OK;
        }

        long width = videoFrame->GetWidth();
        long height = videoFrame->GetHeight();
        long rowBytes = videoFrame->GetRowBytes();
        BMDPixelFormat pixfmt = videoFrame->GetPixelFormat();
        void* bytes = nullptr;
        IDeckLinkVideoFrame_v14_2_1* vf = nullptr;
        HRESULT qrv = videoFrame->QueryInterface(IID_IDeckLinkVideoFrame_v14_2_1, (void**)&vf);
        if (qrv == S_OK && vf) {
            if (vf->GetBytes(&bytes) != S_OK || bytes == nullptr) {
                vf->Release();
                bytes = nullptr;
            }
        }
#if defined(__APPLE__)
        else {
            // เส้นทางเฉพาะ macOS: ดึง CVPixelBuffer แล้วอ่านตำแหน่งข้อมูลโดยตรง
            IDeckLinkMacVideoBuffer* macBuf = nullptr;
            if (videoFrame->QueryInterface(IID_IDeckLinkMacVideoBuffer, (void**)&macBuf) == S_OK && macBuf) {
                CVPixelBufferRef pb = nullptr;
                if (macBuf->CreateCVPixelBufferRef((void**)&pb) == S_OK && pb) {
                    CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                    bytes = CVPixelBufferGetBaseAddress(pb);
                    if (bytes) {
                        rowBytes = (long)CVPixelBufferGetBytesPerRow(pb);
                        OSType cvfmt = CVPixelBufferGetPixelFormatType(pb);
                        // If DeckLink pixfmt is Unspecified, derive from CVPixelBuffer format
                        if (pixfmt == bmdFormatUnspecified) {
                            if (cvfmt == kCVPixelFormatType_422YpCbCr8) pixfmt = bmdFormat8BitYUV; // '2vuy' (UYVY)
                            else if (cvfmt == kCVPixelFormatType_422YpCbCr8_yuvs) pixfmt = bmdFormat8BitYUV; // YUYV
                            else if (cvfmt == kCVPixelFormatType_422YpCbCr10) pixfmt = bmdFormat10BitYUV; // v210
                        }
                    }
                    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                    CFRelease(pb);
                }
                macBuf->Release();
            }
        }
#endif
        if (!bytes) {
            const uint32_t uqrv = (uint32_t)qrv;
            const uint32_t upix = (uint32_t)pixfmt;
            uint32_t last_q = g_last_getbytes_qrv.load(std::memory_order_relaxed);
            uint32_t last_p = g_last_getbytes_pix.load(std::memory_order_relaxed);
            if (last_q != uqrv || last_p != upix) {
                fprintf(stderr, "[shim] GetBytes failed (qrv=0x%08x, pix=0x%x)\n", (unsigned)qrv, (unsigned)pixfmt);
                g_last_getbytes_qrv.store(uqrv, std::memory_order_relaxed);
                g_last_getbytes_pix.store(upix, std::memory_order_relaxed);
            }
            if (vf) vf->Release();
            return S_OK;
        }
        g_last_getbytes_qrv.store(0, std::memory_order_relaxed);
        g_last_getbytes_pix.store(0, std::memory_order_relaxed);

        DVPBufferRecord* record = (g_allocator && bytes) ? g_allocator->resolve(bytes) : nullptr;
        if (!record) {
            static bool s_warned = false;
            if (!s_warned) {
                fprintf(stderr, "[shim] GPUDirect buffer lookup failed; frame pointer not registered with allocator\n");
                s_warned = true;
            }
            if (vf) vf->Release();
            return S_OK;
        }

        record->width = static_cast<uint32_t>(std::max<long>(width, 0));
        record->height = static_cast<uint32_t>(std::max<long>(height, 0));
        record->rowBytes = static_cast<uint32_t>(std::max<long>(rowBytes, 0));
        record->pixelFormat = pixfmt;

        uint64_t seq = g_frameSeq.fetch_add(1, std::memory_order_relaxed) + 1;
        record->seq = seq;
        record->timestampNs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        record->releaseValue += 1;
        if (record->semaphore) {
            record->semaphore[0] = record->releaseValue;
        }

        {
            std::lock_guard<std::mutex> lk(g_latestMutex);
            g_latestFrame.buffer = record;
            g_latestFrame.seq = seq;
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
#if defined(__APPLE__)
        CFUUIDBytes iunknown = CFUUIDGetUUIDBytes(IUnknownUUID);
        if (memcmp(&iid, &iunknown, sizeof(CFUUIDBytes)) == 0) {
            *ppv = static_cast<IUnknown*>(this);
            AddRef();
            return S_OK;
        }
        if (memcmp(&iid, &IID_IDeckLinkScreenPreviewCallback_v14_2_1, sizeof(CFUUIDBytes)) == 0) {
            *ppv = static_cast<IDeckLinkScreenPreviewCallback_v14_2_1*>(this);
            AddRef();
            return S_OK;
        }
#else
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
#endif
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

    // Scan for the first display mode supported for the current connection, preferring v210 then UYVY then BGRA/ARGB
    BMDDisplayMode chosenMode = (BMDDisplayMode)0;
    BMDPixelFormat chosenPix = bmdFormatUnspecified;

    IDeckLinkDisplayMode* mode = nullptr;
    while (it->Next(&mode) == S_OK && mode) {
        bool supported = false; BMDDisplayMode actualMode = (BMDDisplayMode)0;

        // Prefer 8-bit UYVY for lower CPU conversion cost, then v210, then BGRA/ARGB
        const BMDPixelFormat prefs[] = { bmdFormat8BitYUV, bmdFormat10BitYUV, bmdFormat8BitBGRA, bmdFormat8BitARGB };
        for (BMDPixelFormat pf : prefs) {
            supported = false; actualMode = (BMDDisplayMode)0;
            HRESULT hr = input->DoesSupportVideoMode(connection, mode->GetDisplayMode(), pf,
                                bmdNoVideoInputConversion, bmdSupportedVideoModeDefault, &actualMode, &supported);
            if (hr == S_OK && supported) {
                chosenMode = mode->GetDisplayMode();
                chosenPix = pf;
                break;
            }
        }
        if (chosenPix != bmdFormatUnspecified) {
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

    if (!ensure_gpudirect_initialized()) {
        fprintf(stderr, "[shim] GPUDirect initialization failed\n");
        return false;
    }

    reset_latest_frame();
    g_frameSeq.store(0, std::memory_order_relaxed);

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
        fprintf(stderr, "[shim] QueryInterface(IID_IDeckLinkInput) failed hr=0x%08x\n", (unsigned)qri);
        g_device->Release(); g_device = nullptr;
        g_iterator->Release(); g_iterator = nullptr;
        return false;
    }

    g_allocator = new GPUDirectAllocator();
    if (g_input->SetVideoInputFrameMemoryAllocator(g_allocator) != S_OK) {
        fprintf(stderr, "[shim] SetVideoInputFrameMemoryAllocator failed\n");
        g_allocator->Release(); g_allocator = nullptr;
        g_input->Release(); g_input = nullptr;
        g_device->Release(); g_device = nullptr;
        g_iterator->Release(); g_iterator = nullptr;
        return false;
    }

    // Try enabling with format detection first (unknown mode, YUV)
    g_callback = new CaptureCallback();
    if (g_input->SetCallback(g_callback) != S_OK) {
        fprintf(stderr, "[shim] SetCallback failed\n");
        g_callback->Release(); g_callback = nullptr;
        g_input->SetVideoInputFrameMemoryAllocator(nullptr);
        if (g_allocator) { g_allocator->Release(); g_allocator = nullptr; }
        g_input->Release(); g_input = nullptr;
        g_device->Release(); g_device = nullptr;
        g_iterator->Release(); g_iterator = nullptr;
        return false;
    }

    // If a screen preview is already attached (NSView provided earlier), hook it up now.
    if (g_screenPreview) {
        (void)g_input->SetScreenPreviewCallback(g_screenPreview);
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
            // Try 8-bit UYVY first for lower CPU conversion
            HRESULT en = g_input->EnableVideoInput(bmdModeUnknown, bmdFormat8BitYUV, flags);
            if (en == S_OK) return true;
            en = g_input->EnableVideoInput(bmdModeUnknown, bmdFormat10BitYUV, flags);
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
        g_input->SetVideoInputFrameMemoryAllocator(nullptr);
        if (g_allocator) { g_allocator->Release(); g_allocator = nullptr; }
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
        g_input->SetVideoInputFrameMemoryAllocator(nullptr);
        if (g_allocator) { g_allocator->Release(); g_allocator = nullptr; }
        g_input->Release(); g_input = nullptr;
        g_device->Release(); g_device = nullptr;
        g_iterator->Release(); g_iterator = nullptr;
        return false;
    }

    return true;
}

extern "C" bool decklink_capture_get_frame(CaptureFrame* out) {
    if (out) {
        out->data = nullptr;
        out->width = 0;
        out->height = 0;
        out->row_bytes = 0;
        out->seq = 0;
    }
    fprintf(stderr, "[shim] decklink_capture_get_frame is disabled; use decklink_capture_latest_dvp_frame() for zero-copy access\n");
    return false;
}

extern "C" bool decklink_capture_latest_dvp_frame(DeckLinkDVPFrame* out) {
    if (!out)
        return false;

    std::lock_guard<std::mutex> lk(g_latestMutex);
    if (!g_latestFrame.buffer)
        return false;

    const DVPBufferRecord* record = g_latestFrame.buffer;
    out->seq = record->seq;
    out->timestamp_ns = record->timestampNs;
    out->sysmem_handle = record->sysmemHandle;
    out->sync_handle = record->syncHandle;
    out->semaphore_addr = reinterpret_cast<uint64_t>(record->semaphore);
    out->cpu_ptr = static_cast<const uint8_t*>(record->basePtr);
    out->buffer_size = record->size;
    out->width = record->width;
    out->height = record->height;
    out->row_bytes = record->rowBytes;
    out->pixel_format = static_cast<uint32_t>(record->pixelFormat);
    out->release_value = record->releaseValue;
    out->reserved = 0;
    return true;
}

extern "C" void decklink_capture_reset_dvp_fence() {
    reset_latest_frame();
    g_frameSeq.store(0, std::memory_order_relaxed);
}

extern "C" void decklink_capture_close() {
    decklink_preview_gl_disable();
    if (g_input) {
        g_input->StopStreams();
        g_input->DisableVideoInput();
        g_input->SetCallback(nullptr);
        if (g_screenPreview) {
            g_input->SetScreenPreviewCallback(nullptr);
        }
        g_input->SetVideoInputFrameMemoryAllocator(nullptr);
    }
    if (g_callback) { g_callback->Release(); g_callback = nullptr; }
    if (g_allocator) { g_allocator->Release(); g_allocator = nullptr; }
    if (g_input) { g_input->Release(); g_input = nullptr; }
    if (g_device) { g_device->Release(); g_device = nullptr; }
    if (g_iterator) { g_iterator->Release(); g_iterator = nullptr; }
    reset_latest_frame();
    g_frameSeq.store(0, std::memory_order_relaxed);
    if (g_screenPreview) { g_screenPreview->Release(); g_screenPreview = nullptr; }
    g_preview_seq.store(0, std::memory_order_relaxed);
    shutdown_gpudirect();
}

extern "C" bool decklink_preview_attach_nsview(void* nsview_ptr) {
#if defined(__APPLE__)
    if (g_screenPreview) { g_screenPreview->Release(); g_screenPreview = nullptr; }
    IDeckLinkScreenPreviewCallback* previewBase = CreateCocoaScreenPreview(nsview_ptr);
    if (!previewBase) {
        fprintf(stderr, "[shim] CreateCocoaScreenPreview failed\n");
        return false;
    }
    IDeckLinkScreenPreviewCallback_v14_2_1* preview = nullptr;
    if (previewBase->QueryInterface(IID_IDeckLinkScreenPreviewCallback_v14_2_1, (void**)&preview) != S_OK || !preview) {
        fprintf(stderr, "[shim] QueryInterface(IDeckLinkScreenPreviewCallback_v14_2_1) failed\n");
        previewBase->Release();
        return false;
    }
    previewBase->Release();
    g_screenPreview = preview;
    if (g_input) {
        if (g_input->SetScreenPreviewCallback(g_screenPreview) != S_OK) {
            fprintf(stderr, "[shim] SetScreenPreviewCallback failed\n");
            return false;
        }
    }
    return true;
#else
    (void)nsview_ptr;
    fprintf(stderr, "[shim] decklink_preview_attach_nsview is only supported on macOS\n");
    return false;
#endif
}

extern "C" void decklink_preview_detach() {
    if (g_input) {
        g_input->SetScreenPreviewCallback(nullptr);
    }
    if (g_screenPreview) { g_screenPreview->Release(); g_screenPreview = nullptr; }
}

extern "C" uint64_t decklink_preview_seq() {
    return g_preview_seq.load(std::memory_order_relaxed);
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
    HRESULT qri = frame->QueryInterface(IID_IDeckLinkVideoFrame, (void**)&baseFrame);
    if (qri != S_OK || !baseFrame) {
        fprintf(stderr, "[shim] QueryInterface(IDeckLinkVideoFrame) failed hr=0x%08x\n", (unsigned)qri);
        frame->Release();
        return false;
    }

    HRESULT hr = g_glPreview->SetFrame(baseFrame);
    baseFrame->Release();
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
