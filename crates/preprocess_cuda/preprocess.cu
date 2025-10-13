// preprocess.cu - Fused preprocessing kernel for DeepGIBox
// Handles YUV422_8 (UYVY/YUY2), NV12, and BGRA8 → RGB conversion
// with BT.709 limited range, bilinear resize, normalization, and NCHW packing

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Pixel format enum (must match Rust)
#define PIXFMT_BGRA8     0
#define PIXFMT_NV12      1
#define PIXFMT_YUV422_8  2

// Chroma order for YUV422_8
#define CHROMA_UYVY      0
#define CHROMA_YUY2      1

// BT.709 limited range coefficients
__constant__ float BT709_C = 1.164f;
__constant__ float BT709_D_U = 2.112f;
__constant__ float BT709_D_V = 1.793f;
__constant__ float BT709_E_U = 0.213f;
__constant__ float BT709_E_V = 0.534f;

// Clamp helper
__device__ inline float clamp_f(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

// Sample YUV422_8 with bilinear interpolation
__device__ void sample_yuv422(
    const unsigned char* __restrict__ in_ptr,
    int in_w, int in_h, int in_stride,
    int chroma_order,
    float sx, float sy,
    float* Y_out, float* U_out, float* V_out
) {
    // Clamp coordinates
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    int x1 = min(x0 + 1, in_w - 1);
    int y1 = min(y0 + 1, in_h - 1);
    x0 = max(x0, 0);
    y0 = max(y0, 0);

    float fx = sx - floorf(sx);
    float fy = sy - floorf(sy);

    // YUV422: 2 bytes per pixel (4 bytes per 2 pixels)
    // UYVY: U0 Y0 V0 Y1
    // YUY2: Y0 U0 Y1 V0
    
    auto sample_pixel = [&](int x, int y) -> float4 {
        int byte_x = x * 2;  // 2 bytes per pixel
        const unsigned char* row = in_ptr + y * in_stride;
        
        // Find the pair position
        int pair_x = (x / 2) * 4;
        const unsigned char* pair = row + pair_x;
        
        float Y, U, V;
        if (chroma_order == CHROMA_UYVY) {
            // UYVY: U0 Y0 V0 Y1
            U = (float)pair[0];
            V = (float)pair[2];
            Y = (x & 1) ? (float)pair[3] : (float)pair[1];
        } else {
            // YUY2: Y0 U0 Y1 V0
            U = (float)pair[1];
            V = (float)pair[3];
            Y = (x & 1) ? (float)pair[2] : (float)pair[0];
        }
        
        return make_float4(Y, U, V, 0.0f);
    };

    // Bilinear interpolation
    float4 p00 = sample_pixel(x0, y0);
    float4 p10 = sample_pixel(x1, y0);
    float4 p01 = sample_pixel(x0, y1);
    float4 p11 = sample_pixel(x1, y1);

    float4 p0 = make_float4(
        p00.x * (1.0f - fx) + p10.x * fx,
        p00.y * (1.0f - fx) + p10.y * fx,
        p00.z * (1.0f - fx) + p10.z * fx,
        0.0f
    );
    float4 p1 = make_float4(
        p01.x * (1.0f - fx) + p11.x * fx,
        p01.y * (1.0f - fx) + p11.y * fx,
        p01.z * (1.0f - fx) + p11.z * fx,
        0.0f
    );
    float4 result = make_float4(
        p0.x * (1.0f - fy) + p1.x * fy,
        p0.y * (1.0f - fy) + p1.y * fy,
        p0.z * (1.0f - fy) + p1.z * fy,
        0.0f
    );

    *Y_out = result.x;
    *U_out = result.y;
    *V_out = result.z;
}

// Sample NV12 with bilinear interpolation
__device__ void sample_nv12(
    const unsigned char* __restrict__ y_plane,
    const unsigned char* __restrict__ uv_plane,
    int in_w, int in_h, int y_stride, int uv_stride,
    float sx, float sy,
    float* Y_out, float* U_out, float* V_out
) {
    // Y plane (full resolution)
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    int x1 = min(x0 + 1, in_w - 1);
    int y1 = min(y0 + 1, in_h - 1);
    x0 = max(x0, 0);
    y0 = max(y0, 0);

    float fx = sx - floorf(sx);
    float fy = sy - floorf(sy);

    // Sample Y
    float y00 = (float)y_plane[y0 * y_stride + x0];
    float y10 = (float)y_plane[y0 * y_stride + x1];
    float y01 = (float)y_plane[y1 * y_stride + x0];
    float y11 = (float)y_plane[y1 * y_stride + x1];
    *Y_out = (y00 * (1.0f - fx) + y10 * fx) * (1.0f - fy) +
             (y01 * (1.0f - fx) + y11 * fx) * fy;

    // UV plane (half resolution)
    float uv_sx = sx * 0.5f;
    float uv_sy = sy * 0.5f;
    int uv_x0 = (int)floorf(uv_sx);
    int uv_y0 = (int)floorf(uv_sy);
    int uv_x1 = min(uv_x0 + 1, (in_w / 2) - 1);
    int uv_y1 = min(uv_y0 + 1, (in_h / 2) - 1);
    uv_x0 = max(uv_x0, 0);
    uv_y0 = max(uv_y0, 0);

    float uv_fx = uv_sx - floorf(uv_sx);
    float uv_fy = uv_sy - floorf(uv_sy);

    // NV12: interleaved UV
    auto sample_uv = [&](int ux, int uy) -> float2 {
        const unsigned char* uv_pos = uv_plane + uy * uv_stride + ux * 2;
        return make_float2((float)uv_pos[0], (float)uv_pos[1]);
    };

    float2 uv00 = sample_uv(uv_x0, uv_y0);
    float2 uv10 = sample_uv(uv_x1, uv_y0);
    float2 uv01 = sample_uv(uv_x0, uv_y1);
    float2 uv11 = sample_uv(uv_x1, uv_y1);

    float2 uv_interp;
    uv_interp.x = (uv00.x * (1.0f - uv_fx) + uv10.x * uv_fx) * (1.0f - uv_fy) +
                  (uv01.x * (1.0f - uv_fx) + uv11.x * uv_fx) * uv_fy;
    uv_interp.y = (uv00.y * (1.0f - uv_fx) + uv10.y * uv_fx) * (1.0f - uv_fy) +
                  (uv01.y * (1.0f - uv_fx) + uv11.y * uv_fx) * uv_fy;

    *U_out = uv_interp.x;
    *V_out = uv_interp.y;
}

// Sample BGRA8 with bilinear interpolation
__device__ void sample_bgra8(
    const unsigned char* __restrict__ in_ptr,
    int in_w, int in_h, int in_stride,
    float sx, float sy,
    float* R_out, float* G_out, float* B_out
) {
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    int x1 = min(x0 + 1, in_w - 1);
    int y1 = min(y0 + 1, in_h - 1);
    x0 = max(x0, 0);
    y0 = max(y0, 0);

    float fx = sx - floorf(sx);
    float fy = sy - floorf(sy);

    auto sample_pixel = [&](int x, int y) -> float3 {
        const unsigned char* pixel = in_ptr + y * in_stride + x * 4;
        return make_float3((float)pixel[2], (float)pixel[1], (float)pixel[0]); // BGR -> RGB
    };

    float3 p00 = sample_pixel(x0, y0);
    float3 p10 = sample_pixel(x1, y0);
    float3 p01 = sample_pixel(x0, y1);
    float3 p11 = sample_pixel(x1, y1);

    float3 p0 = make_float3(
        p00.x * (1.0f - fx) + p10.x * fx,
        p00.y * (1.0f - fx) + p10.y * fx,
        p00.z * (1.0f - fx) + p10.z * fx
    );
    float3 p1 = make_float3(
        p01.x * (1.0f - fx) + p11.x * fx,
        p01.y * (1.0f - fx) + p11.y * fx,
        p01.z * (1.0f - fx) + p11.z * fx
    );
    float3 result = make_float3(
        p0.x * (1.0f - fy) + p1.x * fy,
        p0.y * (1.0f - fy) + p1.y * fy,
        p0.z * (1.0f - fy) + p1.z * fy
    );

    *R_out = result.x;
    *G_out = result.y;
    *B_out = result.z;
}

// BT.709 YUV to RGB conversion (limited range)
__device__ void yuv_to_rgb_bt709(float Y, float U, float V, float* R, float* G, float* B) {
    float C = Y - 16.0f;
    float D = U - 128.0f;
    float E = V - 128.0f;

    *R = BT709_C * C + BT709_D_V * E;
    *G = BT709_C * C - BT709_E_U * D - BT709_E_V * E;
    *B = BT709_C * C + BT709_D_U * D;

    *R = clamp_f(*R, 0.0f, 255.0f);
    *G = clamp_f(*G, 0.0f, 255.0f);
    *B = clamp_f(*B, 0.0f, 255.0f);
}

// Main fused kernel
extern "C" __global__ void fused_preprocess_kernel(
    const unsigned char* __restrict__ in_ptr,
    void* __restrict__ out_ptr,
    int in_w, int in_h, int in_stride,
    int out_w, int out_h,
    int pixfmt, int chroma_order,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b,
    bool fp16_output,
    const unsigned char* uv_plane,  // For NV12 only
    int uv_stride                    // For NV12 only
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= out_w || y >= out_h) return;

    // 1) Map output coords to input coords (bilinear)
    float sx = (x + 0.5f) * ((float)in_w / (float)out_w) - 0.5f;
    float sy = (y + 0.5f) * ((float)in_h / (float)out_h) - 0.5f;

    float R, G, B;

    // 2) Sample and convert based on pixel format
    if (pixfmt == PIXFMT_YUV422_8) {
        float Y, U, V;
        sample_yuv422(in_ptr, in_w, in_h, in_stride, chroma_order, sx, sy, &Y, &U, &V);
        yuv_to_rgb_bt709(Y, U, V, &R, &G, &B);
    } else if (pixfmt == PIXFMT_NV12) {
        float Y, U, V;
        sample_nv12(in_ptr, uv_plane, in_w, in_h, in_stride, uv_stride, sx, sy, &Y, &U, &V);
        yuv_to_rgb_bt709(Y, U, V, &R, &G, &B);
    } else if (pixfmt == PIXFMT_BGRA8) {
        sample_bgra8(in_ptr, in_w, in_h, in_stride, sx, sy, &R, &G, &B);
    }

    // 3) Normalize to [0,1]
    R /= 255.0f;
    G /= 255.0f;
    B /= 255.0f;

    // 4) Apply mean/std normalization
    R = (R - mean_r) / std_r;
    G = (G - mean_g) / std_g;
    B = (B - mean_b) / std_b;

    // 5) Write to NCHW layout
    size_t plane_size = out_w * out_h;
    size_t idx_r = 0 * plane_size + y * out_w + x;
    size_t idx_g = 1 * plane_size + y * out_w + x;
    size_t idx_b = 2 * plane_size + y * out_w + x;

    if (fp16_output) {
        half* out_half = (half*)out_ptr;
        out_half[idx_r] = __float2half(R);
        out_half[idx_g] = __float2half(G);
        out_half[idx_b] = __float2half(B);
    } else {
        float* out_float = (float*)out_ptr;
        out_float[idx_r] = R;
        out_float[idx_g] = G;
        out_float[idx_b] = B;
    }
}
