// /Users/yamaearth/Documents/3_1/Capstone/Blackmagic DeckLink SDK 14.4/rust/shim/shim.cpp
#include <vector>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <cstdio>
#include <chrono>

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
#include "DeckLinkAPIVideoFrame_v14_2_1.h"
#include "DeckLinkAPIScreenPreviewCallback_v14_2_1.h"

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

bool decklink_capture_open(int32_t device_index);
bool decklink_capture_get_frame(CaptureFrame* out);
void decklink_capture_close();

} // extern "C"

// ---- C++ implementation for capture ----

namespace {

struct SharedFrame {
    std::vector<uint8_t> data; // BGRA
    int32_t width = 0;
    int32_t height = 0;
    int32_t row_bytes = 0; // width * 4
    std::mutex mtx;
    uint64_t seq = 0;
};

static IDeckLinkIterator* g_iterator = nullptr;
static IDeckLink* g_device = nullptr;
static IDeckLinkInput* g_input = nullptr;
static SharedFrame g_shared;
static IDeckLinkScreenPreviewCallback* g_screenPreview = nullptr; // Cocoa Screen Preview (NSView)
static std::atomic<uint64_t> g_preview_seq{0};

// OpenGL preview helper (cross-platform)
static IDeckLinkGLScreenPreviewHelper* g_glPreview = nullptr;
static IDeckLinkVideoFrame* g_glFrame = nullptr;
static IDeckLinkScreenPreviewCallback* g_glCallback = nullptr;
static std::mutex g_glMutex;
static std::mutex g_glFrameMutex;
static std::atomic<uint64_t> g_gl_seq{0};
static std::atomic<uint64_t> g_gl_last_arrival_ns{0};
static std::atomic<uint64_t> g_gl_last_latency_ns{0};
static std::atomic<uint32_t> g_last_getbytes_qrv{0};
static std::atomic<uint32_t> g_last_getbytes_pix{0};
static std::atomic<uint64_t> g_gl_last_render_seq{0};

extern "C" void decklink_preview_gl_disable();

class CaptureCallback : public IDeckLinkInputCallback {
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
        if (memcmp(&iid, &IID_IDeckLinkInputCallback, sizeof(CFUUIDBytes)) == 0) {
            *ppv = static_cast<IDeckLinkInputCallback*>(this);
            AddRef();
            return S_OK;
        }
#else
        if (memcmp(&iid, &kIID_IUnknown, sizeof(REFIID)) == 0) {
            *ppv = static_cast<IUnknown*>(this);
            AddRef();
            return S_OK;
        }
        if (memcmp(&iid, &IID_IDeckLinkInputCallback, sizeof(REFIID)) == 0) {
            *ppv = static_cast<IDeckLinkInputCallback*>(this);
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

    static inline uint8_t clamp_u8(int v) {
        return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }

    static void convert_uyvy_to_bgra(const uint8_t* src, int width, int height, int srcStride, uint8_t* dst, int dstStride) {
        for (int y = 0; y < height; ++y) {
            const uint8_t* s = src + (size_t)y * (size_t)srcStride;
            uint8_t* d = dst + (size_t)y * (size_t)dstStride;
            for (int x = 0; x < width; x += 2) {
                int U = s[0];
                int Y0 = s[1];
                int V = s[2];
                int Y1 = s[3];
                s += 4;

                int C0 = Y0 - 16;
                int C1 = Y1 - 16;
                int D = U - 128;
                int E = V - 128;

                // Pixel 0
                int R0 = (298 * C0 + 409 * E + 128) >> 8;
                int G0 = (298 * C0 - 100 * D - 208 * E + 128) >> 8;
                int B0 = (298 * C0 + 516 * D + 128) >> 8;
                d[0] = clamp_u8(B0);
                d[1] = clamp_u8(G0);
                d[2] = clamp_u8(R0);
                d[3] = 255;

                // Pixel 1
                int R1 = (298 * C1 + 409 * E + 128) >> 8;
                int G1 = (298 * C1 - 100 * D - 208 * E + 128) >> 8;
                int B1 = (298 * C1 + 516 * D + 128) >> 8;
                d[4] = clamp_u8(B1);
                d[5] = clamp_u8(G1);
                d[6] = clamp_u8(R1);
                d[7] = 255;
                d += 8;
            }
        }
    }

    static void convert_yuyv_to_bgra(const uint8_t* src, int width, int height, int srcStride, uint8_t* dst, int dstStride) {
        for (int y = 0; y < height; ++y) {
            const uint8_t* s = src + (size_t)y * (size_t)srcStride;
            uint8_t* d = dst + (size_t)y * (size_t)dstStride;
            for (int x = 0; x < width; x += 2) {
                int Y0 = s[0];
                int U  = s[1];
                int Y1 = s[2];
                int V  = s[3];
                s += 4;

                int C0 = Y0 - 16;
                int C1 = Y1 - 16;
                int D = U - 128;
                int E = V - 128;

                int R0 = (298 * C0 + 409 * E + 128) >> 8;
                int G0 = (298 * C0 - 100 * D - 208 * E + 128) >> 8;
                int B0 = (298 * C0 + 516 * D + 128) >> 8;
                d[0] = clamp_u8(B0);
                d[1] = clamp_u8(G0);
                d[2] = clamp_u8(R0);
                d[3] = 255;

                int R1 = (298 * C1 + 409 * E + 128) >> 8;
                int G1 = (298 * C1 - 100 * D - 208 * E + 128) >> 8;
                int B1 = (298 * C1 + 516 * D + 128) >> 8;
                d[4] = clamp_u8(B1);
                d[5] = clamp_u8(G1);
                d[6] = clamp_u8(R1);
                d[7] = 255;
                d += 8;
            }
        }
    }

    static void convert_v210_to_bgra(const uint8_t* src, int width, int height, int srcStride, uint8_t* dst, int dstStride) {
        auto next_sample = [&](const uint8_t*& ps, uint32_t& word, int& left) -> uint16_t {
            if (left == 0) {
                word = *(const uint32_t*)ps;
                ps += 4;
                left = 3;
            }
            uint16_t s = (uint16_t)(word & 0x3FF);
            word >>= 10;
            left--;
            return s;
        };

        for (int y = 0; y < height; ++y) {
            const uint8_t* ps = src + (size_t)y * (size_t)srcStride;
            uint8_t* d = dst + (size_t)y * (size_t)dstStride;
            uint32_t word = 0; int left = 0;

            for (int x = 0; x < width; x += 2) {
                // Sample stream pattern in v210: U, Y, V, Y, repeating
                uint16_t U10 = next_sample(ps, word, left);
                uint16_t Y0_10 = next_sample(ps, word, left);
                uint16_t V10 = next_sample(ps, word, left);
                uint16_t Y1_10 = next_sample(ps, word, left);

                // Downscale to 8-bit (approx.)
                int U = (int)(U10 >> 2);
                int V = (int)(V10 >> 2);
                int Y0 = (int)(Y0_10 >> 2);
                int Y1 = (int)(Y1_10 >> 2);

                int C0 = Y0 - 16;
                int C1 = Y1 - 16;
                int D = U - 128;
                int E = V - 128;

                // Pixel 0
                int R0 = (298 * C0 + 409 * E + 128) >> 8;
                int G0 = (298 * C0 - 100 * D - 208 * E + 128) >> 8;
                int B0 = (298 * C0 + 516 * D + 128) >> 8;
                d[0] = clamp_u8(B0);
                d[1] = clamp_u8(G0);
                d[2] = clamp_u8(R0);
                d[3] = 255;

                // Pixel 1
                int R1 = (298 * C1 + 409 * E + 128) >> 8;
                int G1 = (298 * C1 - 100 * D - 208 * E + 128) >> 8;
                int B1 = (298 * C1 + 516 * D + 128) >> 8;
                d[4] = clamp_u8(B1);
                d[5] = clamp_u8(G1);
                d[6] = clamp_u8(R1);
                d[7] = 255;

                d += 8;
            }
        }
    }

    HRESULT VideoInputFrameArrived(IDeckLinkVideoInputFrame* videoFrame, IDeckLinkAudioInputPacket* /*audioPacket*/) override {
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

        {
            std::lock_guard<std::mutex> lk(g_shared.mtx);
            // Ensure buffer size
            const int32_t w = (int32_t)width;
            const int32_t h = (int32_t)height;
            const int32_t dstStride = w * 4;
            const size_t wantSize = (size_t)dstStride * (size_t)h;
            if (g_shared.data.size() != wantSize) g_shared.data.resize(wantSize);
            g_shared.width = w;
            g_shared.height = h;
            g_shared.row_bytes = dstStride;

            if (pixfmt == bmdFormat8BitBGRA) {
                // Copy with row strides
                const uint8_t* src = (const uint8_t*)bytes;
                for (int32_t y = 0; y < h; ++y) {
                    const uint8_t* s = src + (size_t)y * (size_t)rowBytes;
                    uint8_t* d = g_shared.data.data() + (size_t)y * (size_t)dstStride;
                    std::memcpy(d, s, (size_t)dstStride);
                }
            } else if (pixfmt == bmdFormat8BitARGB) {
                // Convert ARGB -> BGRA by channel swap per pixel
                const uint8_t* src = (const uint8_t*)bytes;
                for (int32_t y = 0; y < h; ++y) {
                    const uint8_t* s = src + (size_t)y * (size_t)rowBytes;
                    uint8_t* d = g_shared.data.data() + (size_t)y * (size_t)dstStride;
                    for (int32_t x = 0; x < w; ++x) {
                        uint8_t a = s[0];
                        uint8_t r = s[1];
                        uint8_t g = s[2];
                        uint8_t b = s[3];
                        d[0] = b; d[1] = g; d[2] = r; d[3] = a;
                        s += 4; d += 4;
                    }
                }
            } else if (pixfmt == bmdFormat8BitYUV) {
                // Convert UYVY to BGRA
                const uint8_t* src = (const uint8_t*)bytes;
                // Heuristic: if the first plane looks like YUYV, handle; else default UYVY
                // We don't know exact CVPF here; try both depending on alignment of U and Y
                convert_uyvy_to_bgra(src, w, h, (int)rowBytes, g_shared.data.data(), dstStride);
            } else if (pixfmt == bmdFormat10BitYUV) {
                const uint8_t* src = (const uint8_t*)bytes;
                convert_v210_to_bgra(src, w, h, (int)rowBytes, g_shared.data.data(), dstStride);
            } else {
                // Unsupported format: fill black
                std::memset(g_shared.data.data(), 0, g_shared.data.size());
            }
            static bool s_logged = false;
            if (!s_logged) {
                fprintf(stderr, "[shim] First frame: %dx%d, rb=%ld, pix=0x%x\n", w, h, (long)rowBytes, (unsigned)pixfmt);
                s_logged = true;
            }
            g_shared.seq++;
        }
        if (vf) vf->Release();
        return S_OK;
    }

private:
    std::atomic<ULONG> m_ref;
};

static CaptureCallback* g_callback = nullptr;

class GLPreviewCallback : public IDeckLinkScreenPreviewCallback {
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
        if (memcmp(&iid, &IID_IDeckLinkScreenPreviewCallback, sizeof(CFUUIDBytes)) == 0) {
            *ppv = static_cast<IDeckLinkScreenPreviewCallback*>(this);
            AddRef();
            return S_OK;
        }
#else
        if (memcmp(&iid, &kIID_IUnknown, sizeof(REFIID)) == 0) {
            *ppv = static_cast<IUnknown*>(this);
            AddRef();
            return S_OK;
        }
        if (memcmp(&iid, &IID_IDeckLinkScreenPreviewCallback, sizeof(REFIID)) == 0) {
            *ppv = static_cast<IDeckLinkScreenPreviewCallback*>(this);
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

    HRESULT DrawFrame(IDeckLinkVideoFrame* frame) override {
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

static bool pick_supported_mode(IDeckLinkInput* input, BMDVideoConnection connection, BMDDisplayMode& outMode, BMDPixelFormat& outPix) {
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
    HRESULT qri = g_device->QueryInterface(IID_IDeckLinkInput, (void**)&g_input);
    if (qri != S_OK || !g_input) {
        fprintf(stderr, "[shim] QueryInterface(IID_IDeckLinkInput) failed hr=0x%08x\n", (unsigned)qri);
        g_device->Release(); g_device = nullptr;
        g_iterator->Release(); g_iterator = nullptr;
        return false;
    }

    // Try enabling with format detection first (unknown mode, YUV)
    g_callback = new CaptureCallback();
    if (g_input->SetCallback(g_callback) != S_OK) {
        fprintf(stderr, "[shim] SetCallback failed\n");
        g_callback->Release(); g_callback = nullptr;
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
        return false;
    }
    out->data = g_shared.data.data();
    out->width = g_shared.width;
    out->height = g_shared.height;
    out->row_bytes = g_shared.row_bytes;
    out->seq = g_shared.seq;
    return true;
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
    }
    if (g_screenPreview) { g_screenPreview->Release(); g_screenPreview = nullptr; }
    g_preview_seq.store(0, std::memory_order_relaxed);
}

extern "C" bool decklink_preview_attach_nsview(void* nsview_ptr) {
#if defined(__APPLE__)
    if (g_screenPreview) { g_screenPreview->Release(); g_screenPreview = nullptr; }
    g_screenPreview = CreateCocoaScreenPreview(nsview_ptr);
    if (!g_screenPreview) {
        fprintf(stderr, "[shim] CreateCocoaScreenPreview failed\n");
        return false;
    }
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
    IDeckLinkVideoFrame* frame = nullptr;
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

    HRESULT hr = g_glPreview->SetFrame(frame);
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
