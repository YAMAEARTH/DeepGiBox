#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>

// Helper functions for color space conversion
__device__ inline float clamp_f(float v, float min_val, float max_val) {
    return fminf(fmaxf(v, min_val), max_val);
}

__device__ inline uint8_t float_to_u8(float v) {
    return (uint8_t)clamp_f(v * 255.0f, 0.0f, 255.0f);
}

// UYVY to RGB conversion
__device__ void uyvy_to_rgb(uint8_t y, uint8_t u, uint8_t v, uint8_t* r, uint8_t* g, uint8_t* b) {
    float fy = (y - 16.0f) / 219.0f;
    float fu = (u - 128.0f) / 224.0f;
    float fv = (v - 128.0f) / 224.0f;
    
    float fr = fy + 1.402f * fv;
    float fg = fy - 0.344f * fu - 0.714f * fv;
    float fb = fy + 1.772f * fu;
    
    *r = float_to_u8(clamp_f(fr, 0.0f, 1.0f));
    *g = float_to_u8(clamp_f(fg, 0.0f, 1.0f));
    *b = float_to_u8(clamp_f(fb, 0.0f, 1.0f));
}

// RGB to YUV conversion (for chroma keying)
__device__ void rgb_to_yuv(uint8_t r, uint8_t g, uint8_t b, float* y, float* u, float* v) {
    float fr = r / 255.0f;
    float fg = g / 255.0f;
    float fb = b / 255.0f;
    
    *y = 16.0f + 65.738f * fr + 129.057f * fg + 25.064f * fb;
    *u = 128.0f - 37.945f * fr - 74.494f * fg + 112.439f * fb;
    *v = 128.0f + 112.439f * fr - 94.154f * fg - 18.285f * fb;
}

/**
 * CUDA Kernel: Composite PNG (BGRA) over DeckLink (UYVY) with Chroma Keying
 * 
 * Pipeline:
 * 1. Convert DeckLink UYVY → RGB (background)
 * 2. Read PNG BGRA (foreground)
 * 3. Apply chroma key to PNG using RGB distance
 * 4. Composite: PNG over DeckLink
 * 5. Output as BGRA
 */
__global__ void composite_png_over_decklink_kernel(
    const uint8_t* __restrict__ decklink_uyvy,   // Background from DeckLink (UYVY)
    const uint8_t* __restrict__ png_bgra,         // Foreground from PNG (BGRA)
    uint8_t* __restrict__ output_bgra,            // Output (BGRA)
    int width, int height,
    int decklink_pitch, int png_pitch, int output_pitch,
    uint8_t key_r, uint8_t key_g, uint8_t key_b,  // Chroma key color (RGB)
    float threshold                                // Keying threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // === 1. Convert DeckLink UYVY to RGB (Background) ===
    // UYVY format: [U0 Y0 V0 Y1] for 2 pixels
    int pixel_pair = x / 2;
    int is_second = x % 2;
    int uyvy_idx = y * decklink_pitch + pixel_pair * 4;
    
    uint8_t u_bg = decklink_uyvy[uyvy_idx + 0];
    uint8_t y_bg = decklink_uyvy[uyvy_idx + 1 + is_second * 2];
    uint8_t v_bg = decklink_uyvy[uyvy_idx + 2];
    
    uint8_t bg_r, bg_g, bg_b;
    uyvy_to_rgb(y_bg, u_bg, v_bg, &bg_r, &bg_g, &bg_b);
    
    // === 2. Read PNG Foreground (BGRA) ===
    int png_idx = y * png_pitch + x * 4;
    uint8_t fg_b = png_bgra[png_idx + 0];
    uint8_t fg_g = png_bgra[png_idx + 1];
    uint8_t fg_r = png_bgra[png_idx + 2];
    uint8_t fg_a = png_bgra[png_idx + 3];
    
    // === 3. Chroma Key Detection (RGB distance) ===
    float dr = (float)fg_r - (float)key_r;
    float dg = (float)fg_g - (float)key_g;
    float db = (float)fg_b - (float)key_b;
    float distance = sqrtf(dr*dr + dg*dg + db*db) / 441.67f; // Normalize (max distance = sqrt(255^2 * 3))
    
    // Combine original alpha with chroma key
    float alpha = (distance < threshold) ? 0.0f : (fg_a / 255.0f);
    
    // Optional: Soft edge (feathering)
    float feather = 0.05f;
    if (distance >= threshold && distance < threshold + feather) {
        float t = (distance - threshold) / feather;
        alpha *= t;
    }
    
    // === 4. Composite: PNG over DeckLink ===
    float inv_alpha = 1.0f - alpha;
    
    int out_idx = y * output_pitch + x * 4;
    output_bgra[out_idx + 0] = (uint8_t)(fg_b * alpha + bg_b * inv_alpha); // B
    output_bgra[out_idx + 1] = (uint8_t)(fg_g * alpha + bg_g * inv_alpha); // G
    output_bgra[out_idx + 2] = (uint8_t)(fg_r * alpha + bg_r * inv_alpha); // R
    output_bgra[out_idx + 3] = 255; // Full opacity
}

/**
 * Launcher function (callable from C++)
 */
extern "C" void launch_composite_png_over_decklink(
    const uint8_t* decklink_uyvy,
    const uint8_t* png_bgra,
    uint8_t* output_bgra,
    int width, int height,
    int decklink_pitch, int png_pitch, int output_pitch,
    uint8_t key_r, uint8_t key_g, uint8_t key_b,
    float threshold,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    composite_png_over_decklink_kernel<<<grid, block, 0, stream>>>(
        decklink_uyvy, png_bgra, output_bgra,
        width, height,
        decklink_pitch, png_pitch, output_pitch,
        key_r, key_g, key_b,
        threshold
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[keying] Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
