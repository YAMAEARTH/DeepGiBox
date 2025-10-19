cmake_minimum_required(VERSION 3.18)
project(trt_shim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Change these to match your TensorRT installation
set(TENSORRT_ROOT /usr/local/TensorRT-8.6.1.6)
include_directories(${TENSORRT_ROOT}/include include)
link_directories(${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib)

add_library(trt_shim SHARED src/trt_shim.cpp)
target_link_libraries(trt_shim nvinfer nvonnxparser cudart)
