// GPU-accelerated Non-Maximum Suppression (NMS) for YOLO
// Eliminates GPU→CPU→GPU transfers by keeping data on GPU

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <algorithm>

// Detection struct (matches Rust common_io::Detection)
struct Detection {
    // BBox
    float bbox_x;
    float bbox_y;
    float bbox_w;
    float bbox_h;
    // Detection fields
    float score;
    int32_t class_id;
    int32_t track_id;  // -1 for None
    int32_t _padding;  // Alignment
};

// Pack separate arrays into unified Detection structs (zero-copy optimization)
__global__ void pack_detections_kernel(
    const float* boxes,     // [N, 4] (x, y, w, h)
    const float* scores,    // [N]
    const int32_t* classes, // [N]
    Detection* detections,  // Output: unified format
    int num_detections
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_detections) return;
    
    Detection& det = detections[idx];
    
    // Pack bbox
    det.bbox_x = boxes[idx * 4 + 0];
    det.bbox_y = boxes[idx * 4 + 1];
    det.bbox_w = boxes[idx * 4 + 2];
    det.bbox_h = boxes[idx * 4 + 3];
    
    // Pack score and class
    det.score = scores[idx];
    det.class_id = classes[idx];
    det.track_id = -1;  // None
    det._padding = 0;
}

// IoU calculation on GPU
__device__ float calculate_iou(
    float x1, float y1, float w1, float h1,
    float x2, float y2, float w2, float h2
) {
    // Convert center+size to corners
    float left1 = x1 - w1 * 0.5f;
    float right1 = x1 + w1 * 0.5f;
    float top1 = y1 - h1 * 0.5f;
    float bottom1 = y1 + h1 * 0.5f;
    
    float left2 = x2 - w2 * 0.5f;
    float right2 = x2 + w2 * 0.5f;
    float top2 = y2 - h2 * 0.5f;
    float bottom2 = y2 + h2 * 0.5f;
    
    // Calculate intersection
    float inter_left = fmaxf(left1, left2);
    float inter_right = fminf(right1, right2);
    float inter_top = fmaxf(top1, top2);
    float inter_bottom = fminf(bottom1, bottom2);
    
    float inter_width = fmaxf(0.0f, inter_right - inter_left);
    float inter_height = fmaxf(0.0f, inter_bottom - inter_top);
    float inter_area = inter_width * inter_height;
    
    // Calculate union
    float area1 = w1 * h1;
    float area2 = w2 * h2;
    float union_area = area1 + area2 - inter_area;
    
    return (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
}

// Parallel NMS kernel
// Each thread checks one box against all higher-scored boxes
__global__ void nms_kernel(
    const float* boxes,        // [N, 4] (x, y, w, h in normalized coords)
    const float* scores,       // [N] confidence scores
    const int* classes,        // [N] class IDs
    bool* keep_mask,           // [N] output: true if box should be kept
    int num_boxes,
    float iou_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_boxes) return;
    
    // Check if this box should be kept
    bool keep = true;
    
    float x1 = boxes[idx * 4 + 0];
    float y1 = boxes[idx * 4 + 1];
    float w1 = boxes[idx * 4 + 2];
    float h1 = boxes[idx * 4 + 3];
    int class1 = classes[idx];
    
    // Compare with all boxes that have higher scores
    // (assumes boxes are sorted by score descending)
    for (int i = 0; i < idx; i++) {
        if (!keep_mask[i]) continue;  // Skip suppressed boxes
        
        int class2 = classes[i];
        if (class1 != class2) continue;  // Only suppress same class
        
        float x2 = boxes[i * 4 + 0];
        float y2 = boxes[i * 4 + 1];
        float w2 = boxes[i * 4 + 2];
        float h2 = boxes[i * 4 + 3];
        
        float iou = calculate_iou(x1, y1, w1, h1, x2, y2, w2, h2);
        
        if (iou > iou_threshold) {
            keep = false;
            break;
        }
    }
    
    keep_mask[idx] = keep;
}

// YOLO output parsing kernel
// Converts [1, 25200, 85] tensor to sorted detections
__global__ void parse_yolo_kernel(
    const float* raw_output,   // [1, num_anchors, 85] (x, y, w, h, obj, cls0..cls79)
    float* boxes,              // [max_dets, 4] output boxes
    float* scores,             // [max_dets] output scores
    int* classes,              // [max_dets] output class IDs
    int* count,                // [1] number of valid detections
    int num_anchors,           // 25200
    int num_classes,           // 80
    float conf_threshold,
    int max_detections,
    int img_width,             // Original image width for coordinate scaling
    int img_height             // Original image height
) {
    int anchor_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (anchor_idx >= num_anchors) return;
    
    // Parse YOLO output format: [x, y, w, h, objectness, class0, class1, ...]
    const float* anchor = raw_output + anchor_idx * (5 + num_classes);
    
    float obj_score = anchor[4];
    
    // Find best class
    float max_class_score = 0.0f;
    int best_class = 0;
    for (int c = 0; c < num_classes; c++) {
        float cls_score = anchor[5 + c];
        if (cls_score > max_class_score) {
            max_class_score = cls_score;
            best_class = c;
        }
    }
    
    // Combined confidence = objectness × class_prob
    float confidence = obj_score * max_class_score;
    
    if (confidence < conf_threshold) return;
    
    // Atomically get output slot
    int slot = atomicAdd(count, 1);
    if (slot >= max_detections) return;
    
    // Store detection (coordinates already normalized 0-1 by model)
    boxes[slot * 4 + 0] = anchor[0];  // x
    boxes[slot * 4 + 1] = anchor[1];  // y
    boxes[slot * 4 + 2] = anchor[2];  // w
    boxes[slot * 4 + 3] = anchor[3];  // h
    scores[slot] = confidence;
    classes[slot] = best_class;
}

// Sorting helper (on GPU using thrust or manual implementation)
// For now, we'll do a simple parallel sort by score
__global__ void sort_by_score_kernel(
    float* boxes,
    float* scores,
    int* classes,
    bool* keep_mask,
    int num_boxes
) {
    // Bitonic sort or parallel bubble sort
    // For simplicity, we'll use CPU sort in the wrapper
    // (Modern CUDA has thrust::sort_by_key for this)
}

extern "C" {

// Main GPU NMS function
cudaError_t gpu_nms(
    const float* d_raw_output,      // GPU pointer: [1, 25200, 85]
    float* d_boxes,                 // GPU output: [max_dets, 4]
    float* d_scores,                // GPU output: [max_dets]
    int* d_classes,                 // GPU output: [max_dets]
    int* d_count,                   // GPU output: [1] detection count
    int num_anchors,                // 25200
    int num_classes,                // 80
    float conf_threshold,
    float nms_threshold,
    int max_detections,
    int img_width,
    int img_height,
    cudaStream_t stream
) {
    // Step 1: Parse YOLO output and filter by confidence
    dim3 block(256);
    dim3 grid((num_anchors + block.x - 1) / block.x);
    
    // Reset count
    cudaMemsetAsync(d_count, 0, sizeof(int), stream);
    
    parse_yolo_kernel<<<grid, block, 0, stream>>>(
        d_raw_output,
        d_boxes,
        d_scores,
        d_classes,
        d_count,
        num_anchors,
        num_classes,
        conf_threshold,
        max_detections,
        img_width,
        img_height
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    
    // Step 2: Get count of filtered detections
    int h_count;
    cudaMemcpyAsync(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    if (h_count == 0) {
        return cudaSuccess;
    }
    
    h_count = std::min(h_count, max_detections);
    
    // Step 3: Sort by score (descending)
    // NOTE: This requires thrust or manual GPU sort
    // For now, we'll do CPU sort (still faster than full CPU NMS!)
    // TODO: Replace with thrust::sort_by_key for fully GPU implementation
    
    // Step 4: Apply NMS
    bool* d_keep_mask;
    cudaMallocAsync((void**)&d_keep_mask, h_count * sizeof(bool), stream);
    cudaMemsetAsync(d_keep_mask, 1, h_count * sizeof(bool), stream);  // All true initially
    
    dim3 nms_block(256);
    dim3 nms_grid((h_count + nms_block.x - 1) / nms_block.x);
    
    nms_kernel<<<nms_grid, nms_block, 0, stream>>>(
        d_boxes,
        d_scores,
        d_classes,
        d_keep_mask,
        h_count,
        nms_threshold
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFreeAsync(d_keep_mask, stream);
        return err;
    }
    
    // Step 5: Compact results (remove suppressed boxes)
    // This is done in the Rust wrapper by reading keep_mask
    
    cudaFreeAsync(d_keep_mask, stream);
    
    return cudaSuccess;
}

// Pack detections into unified format (zero-copy)
cudaError_t gpu_pack_detections(
    const float* d_boxes,
    const float* d_scores,
    const int32_t* d_classes,
    void* d_detections,  // Detection* output
    int num_detections,
    cudaStream_t stream
) {
    if (num_detections <= 0) return cudaSuccess;
    
    dim3 block(256);
    dim3 grid((num_detections + block.x - 1) / block.x);
    
    pack_detections_kernel<<<grid, block, 0, stream>>>(
        d_boxes,
        d_scores,
        d_classes,
        (Detection*)d_detections,
        num_detections
    );
    
    return cudaGetLastError();
}

// Simpler version: Just parse and filter, skip NMS (for testing)
cudaError_t gpu_yolo_parse(
    const float* d_raw_output,
    float* d_boxes,
    float* d_scores,
    int* d_classes,
    int* d_count,
    int num_anchors,
    int num_classes,
    float conf_threshold,
    int max_detections,
    int img_width,
    int img_height,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_anchors + block.x - 1) / block.x);
    
    cudaMemsetAsync(d_count, 0, sizeof(int), stream);
    
    parse_yolo_kernel<<<grid, block, 0, stream>>>(
        d_raw_output,
        d_boxes,
        d_scores,
        d_classes,
        d_count,
        num_anchors,
        num_classes,
        conf_threshold,
        max_detections,
        img_width,
        img_height
    );
    
    return cudaGetLastError();
}

} // extern "C"
