// GPU-based overlay rendering - ไม่มีการ copy จาก CPU
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

// Helper functions
__device__ inline float clamp_f(float v, float min_val, float max_val) {
    return fminf(fmaxf(v, min_val), max_val);
}

__device__ inline void put_pixel(
    uint8_t* buf, int stride, int width, int height,
    int x, int y, uint8_t a, uint8_t r, uint8_t g, uint8_t b
) {
    if (x < 0 || y < 0 || x >= width || y >= height) return;
    int idx = y * stride + x * 4;
    // Write BGRA format (for DeckLink internal keying)
    buf[idx + 0] = b;  // Blue
    buf[idx + 1] = g;  // Green
    buf[idx + 2] = r;  // Red
    buf[idx + 3] = a;  // Alpha
}

/**
 * Kernel: วาดเส้นตรง (Bresenham algorithm)
 * แต่ละ thread วาด 1 pixel segment
 */
__global__ void draw_line_kernel(
    uint8_t* buf, int stride, int width, int height,
    int x0, int y0, int x1, int y1, int thickness,
    uint8_t a, uint8_t r, uint8_t g, uint8_t b
) {
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int steps = max(dx, dy);
    
    if (steps == 0) return;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= steps) return;
    
    float t = (float)idx / steps;
    int x = (int)(x0 + t * (x1 - x0));
    int y = (int)(y0 + t * (y1 - y0));
    
    int rad = thickness / 2;
    for (int dy = -rad; dy <= rad; dy++) {
        for (int dx = -rad; dx <= rad; dx++) {
            put_pixel(buf, stride, width, height, x + dx, y + dy, a, r, g, b);
        }
    }
}

/**
 * Kernel: วาดสี่เหลี่ยม (outline)
 */
__global__ void draw_rect_kernel(
    uint8_t* buf, int stride, int width, int height,
    int x, int y, int w, int h, int thickness,
    uint8_t a, uint8_t r, uint8_t g, uint8_t b
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // คำนวณ perimeter ทั้งหมด
    int perimeter = 2 * (w + h);
    if (idx >= perimeter) return;
    
    int px, py;
    if (idx < w) {
        // บน
        px = x + idx;
        py = y;
    } else if (idx < w + h) {
        // ขวา
        px = x + w;
        py = y + (idx - w);
    } else if (idx < 2 * w + h) {
        // ล่าง
        px = x + w - (idx - w - h);
        py = y + h;
    } else {
        // ซ้าย
        px = x;
        py = y + h - (idx - 2 * w - h);
    }
    
    int rad = thickness / 2;
    for (int dy = -rad; dy <= rad; dy++) {
        for (int dx = -rad; dx <= rad; dx++) {
            put_pixel(buf, stride, width, height, px + dx, py + dy, a, r, g, b);
        }
    }
}

/**
 * Kernel: เติมสี่เหลี่ยม
 */
__global__ void fill_rect_kernel(
    uint8_t* buf, int stride, int width, int height,
    int x, int y, int w, int h,
    uint8_t a, uint8_t r, uint8_t g, uint8_t b
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= w || py >= h) return;
    
    put_pixel(buf, stride, width, height, x + px, y + py, a, r, g, b);
}

/**
 * Kernel: ล้าง buffer (clear to transparent)
 */
__global__ void clear_buffer_kernel(
    uint8_t* buf, int stride, int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * stride + x * 4;
    // BGRA format
    buf[idx + 0] = 0; // B
    buf[idx + 1] = 0; // G
    buf[idx + 2] = 0; // R
    buf[idx + 3] = 0; // A (transparent)
}

// ==================== C API สำหรับเรียกจาก Rust ====================

extern "C" {

/**
 * ล้าง buffer
 */
void launch_clear_buffer(
    uint8_t* gpu_buf, int stride, int width, int height,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    clear_buffer_kernel<<<grid, block, 0, stream>>>(gpu_buf, stride, width, height);
}

/**
 * วาดเส้นตรง
 */
void launch_draw_line(
    uint8_t* gpu_buf, int stride, int width, int height,
    int x0, int y0, int x1, int y1, int thickness,
    uint8_t a, uint8_t r, uint8_t g, uint8_t b,
    cudaStream_t stream
) {
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int steps = max(dx, dy) + 1;
    
    int threads = 256;
    int blocks = (steps + threads - 1) / threads;
    
    draw_line_kernel<<<blocks, threads, 0, stream>>>(
        gpu_buf, stride, width, height,
        x0, y0, x1, y1, thickness,
        a, r, g, b
    );
}

/**
 * วาดสี่เหลี่ยม (outline)
 */
void launch_draw_rect(
    uint8_t* gpu_buf, int stride, int width, int height,
    int x, int y, int w, int h, int thickness,
    uint8_t a, uint8_t r, uint8_t g, uint8_t b,
    cudaStream_t stream
) {
    int perimeter = 2 * (w + h);
    int threads = 256;
    int blocks = (perimeter + threads - 1) / threads;
    
    draw_rect_kernel<<<blocks, threads, 0, stream>>>(
        gpu_buf, stride, width, height,
        x, y, w, h, thickness,
        a, r, g, b
    );
}

/**
 * เติมสี่เหลี่ยม
 */
void launch_fill_rect(
    uint8_t* gpu_buf, int stride, int width, int height,
    int x, int y, int w, int h,
    uint8_t a, uint8_t r, uint8_t g, uint8_t b,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    
    fill_rect_kernel<<<grid, block, 0, stream>>>(
        gpu_buf, stride, width, height,
        x, y, w, h,
        a, r, g, b
    );
}

} // extern "C"
