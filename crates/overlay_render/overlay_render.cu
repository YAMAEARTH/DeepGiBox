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

// GPU draw command structures (match overlay_plan)
struct GpuDrawRect {
    float x, y, w, h;
    uint8_t thickness;
    uint8_t r, g, b, a;
};

struct GpuDrawLabel {
    float x, y;
    uint16_t font_size;
    uint8_t r, g, b, a;
    char text[64];
};

struct GpuDrawCommand {
    int32_t cmd_type;  // 0=RECT, 1=FILL_RECT, 2=LABEL
    union {
        GpuDrawRect rect;
        GpuDrawLabel label;
    } data;
};

/**
 * Kernel: Execute RECT commands in parallel
 * Each command gets its own block, threads cooperate to draw outline
 */
__global__ void execute_rect_commands_kernel(
    uint8_t* buf, int stride, int width, int height,
    const GpuDrawCommand* commands, const int* rect_indices, int num_rect_commands
) {
    int cmd_idx = blockIdx.x;
    if (cmd_idx >= num_rect_commands) return;
    
    int actual_idx = rect_indices[cmd_idx];
    const GpuDrawCommand& cmd = commands[actual_idx];
    const GpuDrawRect& rect = cmd.data.rect;
    
    int x = (int)rect.x;
    int y = (int)rect.y;
    int w = (int)rect.w;
    int h = (int)rect.h;
    int thick = rect.thickness;
    int rad = thick / 2;
    
    // Each thread draws multiple pixels along edges
    int tid = threadIdx.x;
    int perimeter = 2 * (w + h);
    
    // Draw outline with thickness
    for (int i = tid; i < perimeter; i += blockDim.x) {
        int px, py;
        
        if (i < w) {
            // Top edge
            px = x + i;
            py = y;
        } else if (i < w + h) {
            // Right edge
            px = x + w;
            py = y + (i - w);
        } else if (i < 2 * w + h) {
            // Bottom edge
            px = x + (2 * w + h - i);
            py = y + h;
        } else {
            // Left edge
            px = x;
            py = y + (perimeter - i);
        }
        
        // Draw with thickness
        for (int t = -rad; t <= rad; t++) {
            if (i < w || i >= w + h) {
                // Horizontal edges - vary y
                put_pixel(buf, stride, width, height, px, py + t,
                         rect.a, rect.r, rect.g, rect.b);
            } else {
                // Vertical edges - vary x
                put_pixel(buf, stride, width, height, px + t, py,
                         rect.a, rect.r, rect.g, rect.b);
            }
        }
    }
}

/**
 * Kernel: Execute FILL_RECT commands in parallel
 * Each command gets its own grid of blocks for parallel pixel filling
 */
__global__ void execute_fill_commands_kernel(
    uint8_t* buf, int stride, int width, int height,
    const GpuDrawCommand* commands, const int* fill_indices, int num_fill_commands
) {
    // This kernel will be launched multiple times, once per fill command
    // Use blockIdx.z to identify which command (since we can't have huge X dimension)
    int cmd_idx = blockIdx.z;
    if (cmd_idx >= num_fill_commands) return;
    
    int actual_idx = fill_indices[cmd_idx];
    const GpuDrawCommand& cmd = commands[actual_idx];
    const GpuDrawRect& rect = cmd.data.rect;
    
    int x = (int)rect.x;
    int y = (int)rect.y;
    int w = (int)rect.w;
    int h = (int)rect.h;
    
    // Each thread handles one pixel
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px < w && py < h) {
        put_pixel(buf, stride, width, height, x + px, y + py,
                 rect.a, rect.r, rect.g, rect.b);
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
 * Kernel: Build index arrays for different command types
 */
__global__ void build_command_indices_kernel(
    const GpuDrawCommand* commands, int num_commands,
    int* rect_indices, int* fill_indices,
    int* num_rect, int* num_fill
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_commands) return;
    
    const GpuDrawCommand& cmd = commands[idx];
    
    if (cmd.cmd_type == 0) {
        // RECT
        int pos = atomicAdd(num_rect, 1);
        rect_indices[pos] = idx;
    } else if (cmd.cmd_type == 1) {
        // FILL_RECT
        int pos = atomicAdd(num_fill, 1);
        fill_indices[pos] = idx;
    }
    // cmd_type == 2 (LABEL) ignored - requires CPU text rendering
}

/**
 * Execute GPU commands (zero-copy rendering!)
 * Commands are already on GPU - no CPU transfers needed
 * 
 * Strategy: First pass separates commands by type, then parallel execution
 */
void launch_execute_commands(
    uint8_t* gpu_buf, int stride, int width, int height,
    const void* gpu_commands, int num_commands,
    cudaStream_t stream
) {
    if (num_commands <= 0) return;
    
    const GpuDrawCommand* cmds = (const GpuDrawCommand*)gpu_commands;
    
    // Allocate index buffers on GPU
    int* rect_indices;
    int* fill_indices;
    int* num_rect;
    int* num_fill;
    
    cudaMalloc(&rect_indices, num_commands * sizeof(int));
    cudaMalloc(&fill_indices, num_commands * sizeof(int));
    cudaMalloc(&num_rect, sizeof(int));
    cudaMalloc(&num_fill, sizeof(int));
    
    cudaMemsetAsync(num_rect, 0, sizeof(int), stream);
    cudaMemsetAsync(num_fill, 0, sizeof(int), stream);
    
    // First pass: build index arrays
    int threads = 256;
    int blocks = (num_commands + threads - 1) / threads;
    build_command_indices_kernel<<<blocks, threads, 0, stream>>>(
        cmds, num_commands,
        rect_indices, fill_indices,
        num_rect, num_fill
    );
    
    // Copy counts to CPU (only 2 × 4 bytes!)
    int h_num_rect = 0, h_num_fill = 0;
    cudaMemcpyAsync(&h_num_rect, num_rect, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_num_fill, num_fill, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Execute RECT commands
    if (h_num_rect > 0) {
        dim3 block(256, 1, 1);
        dim3 grid(h_num_rect, 1, 1);
        execute_rect_commands_kernel<<<grid, block, 0, stream>>>(
            gpu_buf, stride, width, height,
            cmds, rect_indices, h_num_rect
        );
    }
    
    // Execute FILL_RECT commands
    if (h_num_fill > 0) {
        // Launch one kernel per fill command (GPU-driven!)
        // Max grid size is typically 65535, so we batch if needed
        const int MAX_BATCH = 256;  // Process up to 256 fills at once
        
        for (int batch_start = 0; batch_start < h_num_fill; batch_start += MAX_BATCH) {
            int batch_size = (batch_start + MAX_BATCH < h_num_fill) ? MAX_BATCH : (h_num_fill - batch_start);
            
            // Launch kernel with Z dimension for command index
            // Each command gets 16x16 blocks in XY plane
            dim3 block(16, 16, 1);
            dim3 grid(8, 8, batch_size);  // 8x8 blocks = 128x128 pixels max per command
            
            execute_fill_commands_kernel<<<grid, block, 0, stream>>>(
                gpu_buf, stride, width, height,
                cmds, fill_indices + batch_start, batch_size
            );
        }
    }
    
    // Cleanup
    cudaFree(rect_indices);
    cudaFree(fill_indices);
    cudaFree(num_rect);
    cudaFree(num_fill);
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
