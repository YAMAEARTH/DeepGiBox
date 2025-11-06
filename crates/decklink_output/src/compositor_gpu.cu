// GPU Composite: DeckLink UYVY + Overlay ARGB (GPU) → BGRA สำหรับ Internal Keying
// ไม่มีการ copy จาก CPU!

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

// Helper functions (prefixed to avoid conflicts)
__device__ inline float compositor_clamp_f(float v, float min_val, float max_val) {
    return fminf(fmaxf(v, min_val), max_val);
}

__device__ inline uint8_t compositor_float_to_u8(float v) {
    return (uint8_t)compositor_clamp_f(v * 255.0f, 0.0f, 255.0f);
}

// UYVY to RGB conversion (prefixed to avoid conflicts with keying.cu)
__device__ void compositor_uyvy_to_rgb(uint8_t y, uint8_t u, uint8_t v, uint8_t* r, uint8_t* g, uint8_t* b) {
    float fy = (y - 16.0f) / 219.0f;
    float fu = (u - 128.0f) / 224.0f;
    float fv = (v - 128.0f) / 224.0f;
    
    float fr = fy + 1.402f * fv;
    float fg = fy - 0.344f * fu - 0.714f * fv;
    float fb = fy + 1.772f * fu;
    
    *r = compositor_float_to_u8(compositor_clamp_f(fr, 0.0f, 1.0f));
    *g = compositor_float_to_u8(compositor_clamp_f(fg, 0.0f, 1.0f));
    *b = compositor_float_to_u8(compositor_clamp_f(fb, 0.0f, 1.0f));
}

/**
 * Kernel: Composite Overlay (ARGB GPU) over DeckLink (UYVY GPU) → BGRA GPU
 * 
 * Pipeline (All on GPU - Zero CPU Copy):
 * 1. Read DeckLink UYVY from GPU → Convert to RGB
 * 2. Read Overlay ARGB from GPU (already rendered by overlay_render)
 * 3. Alpha blend: Overlay over Video
 * 4. Write BGRA to output GPU buffer
 * 
 * Performance: ~0.3ms @ 1080p60
 */
__global__ void composite_argb_overlay_kernel(
    const uint8_t* __restrict__ decklink_uyvy,   // DeckLink video (UYVY, GPU)
    const uint8_t* __restrict__ overlay_argb,     // Overlay from overlay_render (ARGB, GPU)
    uint8_t* __restrict__ output_bgra,            // Output (BGRA, GPU)
    int width, int height,
    int decklink_pitch,
    int overlay_pitch,
    int output_pitch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // === 1. Convert DeckLink UYVY to RGB (Background) ===
    int pixel_pair = x / 2;
    int is_second = x % 2;
    int uyvy_idx = y * decklink_pitch + pixel_pair * 4;
    
    uint8_t u_bg = decklink_uyvy[uyvy_idx + 0];
    uint8_t y_bg = decklink_uyvy[uyvy_idx + 1 + is_second * 2];
    uint8_t v_bg = decklink_uyvy[uyvy_idx + 2];
    
    uint8_t bg_r, bg_g, bg_b;
    compositor_uyvy_to_rgb(y_bg, u_bg, v_bg, &bg_r, &bg_g, &bg_b);
    
    // === 2. Read Overlay ARGB (Foreground) ===
    int overlay_idx = y * overlay_pitch + x * 4;
    uint8_t fg_a = overlay_argb[overlay_idx + 0];  // A
    uint8_t fg_r = overlay_argb[overlay_idx + 1];  // R
    uint8_t fg_g = overlay_argb[overlay_idx + 2];  // G
    uint8_t fg_b = overlay_argb[overlay_idx + 3];  // B
    float alpha = fg_a / 255.0f;
    
    // === 3. Alpha Blend: Overlay over DeckLink ===
    float inv_alpha = 1.0f - alpha;
    
    uint8_t out_b = (uint8_t)(fg_b * alpha + bg_b * inv_alpha);
    uint8_t out_g = (uint8_t)(fg_g * alpha + bg_g * inv_alpha);
    uint8_t out_r = (uint8_t)(fg_r * alpha + bg_r * inv_alpha);
    
    // === 4. Write BGRA Output ===
    int out_idx = y * output_pitch + x * 4;
    output_bgra[out_idx + 0] = out_b;  // B
    output_bgra[out_idx + 1] = out_g;  // G
    output_bgra[out_idx + 2] = out_r;  // R
    output_bgra[out_idx + 3] = 255;    // A (full opacity)
}

/**
 * Launcher: Composite ARGB overlay (GPU) with DeckLink (GPU)
 */
extern "C" void launch_composite_argb_overlay(
    const uint8_t* decklink_uyvy_gpu,
    const uint8_t* overlay_argb_gpu,
    uint8_t* output_bgra_gpu,
    int width, int height,
    int decklink_pitch,
    int overlay_pitch,
    int output_pitch,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    composite_argb_overlay_kernel<<<grid, block, 0, stream>>>(
        decklink_uyvy_gpu,
        overlay_argb_gpu,
        output_bgra_gpu,
        width, height,
        decklink_pitch,
        overlay_pitch,
        output_pitch
    );
}
