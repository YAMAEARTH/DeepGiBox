// include/trt_shim.h
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Build TensorRT engine from an ONNX file and save it
void build_engine(const char* onnx_path, const char* engine_path);

// Run inference on a serialized engine file (SLOW - loads engine each time)
void infer(const char* engine_path, const float* input_data, float* output_data, int input_size, int output_size);

// Fast inference API - keeps engine loaded in memory
typedef void* InferenceSession;

// Create a persistent inference session (loads engine once)
InferenceSession create_session(const char* engine_path);

// Run fast inference using a loaded session
void run_inference(InferenceSession session, const float* input_data, float* output_data, int input_size, int output_size);

// Destroy inference session and free resources
void destroy_session(InferenceSession session);

// ============================================================================
// ZERO-COPY API - For data already on GPU (ultimate performance!)
// ============================================================================

// Run inference with GPU pointers directly (NO CPU↔GPU transfer!)
// Input and output must be valid GPU device pointers
void run_inference_device(
    InferenceSession session, 
    const float* d_input,      // GPU pointer to input data
    float* d_output,           // GPU pointer to output data
    int input_size, 
    int output_size
);

// Get internal GPU buffer pointers (for external GPU operations)
// Returns device pointers that can be used by CUDA kernels
typedef struct {
    void* d_input;   // GPU input buffer pointer
    void* d_output;  // GPU output buffer pointer
    int input_size;
    int output_size;
} DeviceBuffers;

DeviceBuffers* get_device_buffers(InferenceSession session);

// Free DeviceBuffers structure allocated by get_device_buffers
// MUST be called after you're done with the buffers to prevent memory leak
void free_device_buffers(DeviceBuffers* buffers);

// Copy output data from GPU buffer to CPU (for postprocessing/visualization)
// Use this after run_inference_device() to retrieve results
void copy_output_to_cpu(
    InferenceSession session,
    float* output_cpu,         // CPU buffer to receive data
    int output_size            // Number of floats to copy
);

// Copy input data from CPU to GPU buffer (alternative to using your own GPU buffer)
// Use this before run_inference_device() if you want to prepare data on CPU first
void copy_input_to_gpu(
    InferenceSession session,
    const float* input_cpu,    // CPU buffer with input data
    int input_size             // Number of floats to copy
);

// Run inference with multiple inputs (image + thresholds)
// Use this for models like GIM that need threshold parameters
void run_inference_device_multiple_input(
    InferenceSession session, 
    const float* d_input,      // GPU pointer to main input (image)
    float* d_output,           // GPU pointer to output (unused, uses internal buffers)
    int input_size, 
    int output_size,
    float threshold_nbi,       // NBI threshold value
    float threshold_wle,       // WLE threshold value
    float c_threshold          // Classification threshold value
);

#ifdef __cplusplus
}
#endif
