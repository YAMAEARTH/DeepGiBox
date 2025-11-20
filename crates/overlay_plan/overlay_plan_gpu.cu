// GPU Overlay Planning
// Generates draw commands directly on GPU, eliminating CPU→GPU transfer

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// Match Rust Detection struct from common_io
struct Detection {
    // BBox
    float bbox_x;
    float bbox_y;
    float bbox_w;
    float bbox_h;
    // Detection fields
    float score;
    int32_t class_id;
    int32_t track_id;  // -1 if None
    int32_t _padding;  // Alignment
};

// Kernel to unpack Detection structs → separate arrays (zero-copy optimization)
__global__ void unpack_detections_kernel(
    const Detection* detections,  // Input: array of Detection structs on GPU
    float* boxes,                 // Output: [N, 4] (x, y, w, h)
    float* scores,                // Output: [N]
    int32_t* classes,             // Output: [N]
    int num_detections
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_detections) return;
    
    const Detection& det = detections[idx];
    
    // Unpack bbox (4 floats)
    boxes[idx * 4 + 0] = det.bbox_x;
    boxes[idx * 4 + 1] = det.bbox_y;
    boxes[idx * 4 + 2] = det.bbox_w;
    boxes[idx * 4 + 3] = det.bbox_h;
    
    // Unpack score and class
    scores[idx] = det.score;
    classes[idx] = det.class_id;
}

// GPU-side draw command structure (matches Rust DrawOp)
struct GpuDrawRect {
    float x, y, w, h;
    uint8_t thickness;
    uint8_t r, g, b, a;
};

struct GpuDrawLabel {
    float x, y;
    uint16_t font_size;
    uint8_t r, g, b, a;
    char text[64];  // Fixed-size string for GPU
};

// Compact command buffer on GPU
struct GpuDrawCommand {
    int cmd_type;  // 0=RECT, 1=FILL_RECT, 2=LABEL
    
    union {
        GpuDrawRect rect;
        GpuDrawLabel label;
    } data;
};

// Kernel to generate overlay commands from detections
__global__ void generate_overlay_kernel(
    const float* boxes,           // [N, 4] (x, y, w, h normalized)
    const float* scores,          // [N] confidence scores
    const int* classes,           // [N] class IDs
    GpuDrawCommand* commands,     // Output: draw commands
    int* command_count,           // Output: number of commands
    int num_detections,
    int img_width,
    int img_height,
    float conf_threshold,
    int max_commands
) {
    int det_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (det_idx >= num_detections) return;
    
    float score = scores[det_idx];
    if (score < conf_threshold) return;
    
    // Get detection box (normalized 0-1)
    float x_norm = boxes[det_idx * 4 + 0];
    float y_norm = boxes[det_idx * 4 + 1];
    float w_norm = boxes[det_idx * 4 + 2];
    float h_norm = boxes[det_idx * 4 + 3];
    int class_id = classes[det_idx];
    
    // Convert to pixel coordinates
    float x_px = x_norm * img_width;
    float y_px = y_norm * img_height;
    float w_px = w_norm * img_width;
    float h_px = h_norm * img_height;
    
    // Calculate bounding box corners
    float left = x_px - w_px * 0.5f;
    float top = y_px - h_px * 0.5f;
    
    // Allocate command slots (each detection = 2 commands: box + label)
    int base_slot = atomicAdd(command_count, 2);
    if (base_slot + 1 >= max_commands) return;
    
    // Command 1: Draw bounding box
    GpuDrawCommand& box_cmd = commands[base_slot];
    box_cmd.cmd_type = 0;  // RECT
    box_cmd.data.rect.x = left;
    box_cmd.data.rect.y = top;
    box_cmd.data.rect.w = w_px;
    box_cmd.data.rect.h = h_px;
    box_cmd.data.rect.thickness = 3;
    
    // Color based on class (green for polyps, etc.)
    if (class_id == 0) {
        // Polyp - green
        box_cmd.data.rect.r = 0;
        box_cmd.data.rect.g = 255;
        box_cmd.data.rect.b = 0;
        box_cmd.data.rect.a = 255;
    } else {
        // Other - yellow
        box_cmd.data.rect.r = 255;
        box_cmd.data.rect.g = 255;
        box_cmd.data.rect.b = 0;
        box_cmd.data.rect.a = 255;
    }
    
    // Command 2: Draw label
    GpuDrawCommand& label_cmd = commands[base_slot + 1];
    label_cmd.cmd_type = 2;  // LABEL
    label_cmd.data.label.x = left;
    label_cmd.data.label.y = top - 5.0f;  // Above box
    label_cmd.data.label.font_size = 18;
    label_cmd.data.label.r = 255;
    label_cmd.data.label.g = 255;
    label_cmd.data.label.b = 255;
    label_cmd.data.label.a = 255;
    
    // Format text manually (snprintf not available in device code)
    // Result: "Polyp 95%" or "Obj 95%"
    int conf_pct = (int)(score * 100.0f);
    char* text = label_cmd.data.label.text;
    int pos = 0;
    
    if (class_id == 0) {
        // "Polyp "
        text[pos++] = 'P';
        text[pos++] = 'o';
        text[pos++] = 'l';
        text[pos++] = 'y';
        text[pos++] = 'p';
        text[pos++] = ' ';
    } else {
        // "Obj "
        text[pos++] = 'O';
        text[pos++] = 'b';
        text[pos++] = 'j';
        text[pos++] = ' ';
    }
    
    // Convert percentage to string (0-100)
    if (conf_pct >= 100) {
        text[pos++] = '1';
        text[pos++] = '0';
        text[pos++] = '0';
    } else if (conf_pct >= 10) {
        text[pos++] = '0' + (conf_pct / 10);
        text[pos++] = '0' + (conf_pct % 10);
    } else {
        text[pos++] = '0' + conf_pct;
    }
    
    text[pos++] = '%';
    text[pos] = '\0';
}

// Kernel to draw confidence bar (if needed)
__global__ void generate_confidence_bar_kernel(
    const float* scores,
    int num_detections,
    GpuDrawCommand* commands,
    int* command_count,
    int img_width,
    int img_height,
    int max_commands
) {
    // Find highest confidence detection
    __shared__ float max_conf;
    __shared__ int max_idx;
    
    if (threadIdx.x == 0) {
        max_conf = 0.0f;
        max_idx = -1;
    }
    __syncthreads();
    
    // Parallel max reduction
    int tid = threadIdx.x;
    if (tid < num_detections) {
        float conf = scores[tid];
        atomicMax((int*)&max_conf, __float_as_int(conf));
    }
    __syncthreads();
    
    if (threadIdx.x == 0 && max_conf > 0.0f) {
        // Draw confidence bar at bottom-right
        int bar_segments = 20;
        float segment_height = 5.0f;
        float segment_width = 30.0f;
        float bar_x = img_width - segment_width - 20.0f;
        float bar_y_start = img_height - (bar_segments * segment_height) - 20.0f;
        
        int filled_segments = (int)(max_conf * bar_segments);
        
        for (int i = 0; i < filled_segments; i++) {
            int slot = atomicAdd(command_count, 1);
            if (slot >= max_commands) break;
            
            GpuDrawCommand& cmd = commands[slot];
            cmd.cmd_type = 1;  // FILL_RECT
            cmd.data.rect.x = bar_x;
            cmd.data.rect.y = bar_y_start + i * segment_height;
            cmd.data.rect.w = segment_width;
            cmd.data.rect.h = segment_height;
            cmd.data.rect.thickness = 0;
            
            // Gradient: red → yellow → green
            float ratio = (float)i / bar_segments;
            if (ratio < 0.5f) {
                // Red to yellow
                cmd.data.rect.r = 255;
                cmd.data.rect.g = (uint8_t)(ratio * 2.0f * 255);
                cmd.data.rect.b = 0;
            } else {
                // Yellow to green
                cmd.data.rect.r = (uint8_t)((1.0f - ratio) * 2.0f * 255);
                cmd.data.rect.g = 255;
                cmd.data.rect.b = 0;
            }
            cmd.data.rect.a = 200;  // Semi-transparent
        }
    }
}

extern "C" {

// Unpack Detection structs to separate arrays (zero-copy optimization)
cudaError_t gpu_unpack_detections(
    const void* d_detections,  // Input: Detection array on GPU
    float* d_boxes,            // Output: [N, 4]
    float* d_scores,           // Output: [N]
    int32_t* d_classes,        // Output: [N]
    int num_detections,
    cudaStream_t stream
) {
    if (num_detections <= 0) return cudaSuccess;
    
    dim3 block(256);
    dim3 grid((num_detections + block.x - 1) / block.x);
    
    unpack_detections_kernel<<<grid, block, 0, stream>>>(
        (const Detection*)d_detections,
        d_boxes,
        d_scores,
        d_classes,
        num_detections
    );
    
    return cudaGetLastError();
}

// Generate overlay plan on GPU
cudaError_t gpu_generate_overlay(
    const float* d_boxes,
    const float* d_scores,
    const int* d_classes,
    int num_detections,
    GpuDrawCommand* d_commands,
    int* d_command_count,
    int img_width,
    int img_height,
    float conf_threshold,
    int max_commands,
    bool draw_confidence_bar,
    cudaStream_t stream
) {
    // Reset command count
    cudaMemsetAsync(d_command_count, 0, sizeof(int), stream);
    
    // Generate bounding boxes and labels
    dim3 block(256);
    dim3 grid((num_detections + block.x - 1) / block.x);
    
    generate_overlay_kernel<<<grid, block, 0, stream>>>(
        d_boxes,
        d_scores,
        d_classes,
        d_commands,
        d_command_count,
        num_detections,
        img_width,
        img_height,
        conf_threshold,
        max_commands
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    
    // Optionally add confidence bar
    if (draw_confidence_bar && num_detections > 0) {
        generate_confidence_bar_kernel<<<1, 256, 0, stream>>>(
            d_scores,
            num_detections,
            d_commands,
            d_command_count,
            img_width,
            img_height,
            max_commands
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
    }
    
    return cudaSuccess;
}

} // extern "C"
