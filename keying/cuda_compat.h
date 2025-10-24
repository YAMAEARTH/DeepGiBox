#ifndef CUDA_COMPAT_H
#define CUDA_COMPAT_H

// Workaround for CUDA 12.0 + GCC 13.3 + glibc 2.39 (Ubuntu 24.04) compatibility issues
// The newer glibc defines _Float32, _Float64, _Float128 types that nvcc doesn't understand

// Prevent glibc from defining these GNU extension types
#define __GLIBC_USE_IEC_60559_TYPES_EXT 0
#define __HAVE_FLOAT128 0
#define __HAVE_DISTINCT_FLOAT128 0
#define __HAVE_FLOAT64X 0
#define __HAVE_FLOAT64X_LONG_DOUBLE 0
#define __HAVE_FLOAT32 0
#define __HAVE_FLOAT64 0
#define __HAVE_FLOAT32X 0

#endif // CUDA_COMPAT_H
