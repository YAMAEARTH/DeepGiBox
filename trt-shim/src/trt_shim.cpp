// src/trt_shim.cpp
#include "trt_shim.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

void build_engine(const char* onnx_path, const char* engine_path) {
    std::cout << "Building engine from: " << onnx_path << std::endl;

    IBuilder* builder = createInferBuilder(gLogger);
    if (!builder) {
        std::cerr << "Failed to create builder!" << std::endl;
        return;
    }

    INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    if (!network) {
        std::cerr << "Failed to create network!" << std::endl;
        delete builder;
        return;
    }

    IBuilderConfig* config = builder->createBuilderConfig();
    if (!config) {
        std::cerr << "Failed to create builder config!" << std::endl;
        delete network;
        delete builder;
        return;
    }
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30);  // 1GB
    
    // Enable FP16 mode if supported
    if (builder->platformHasFastFp16()) {
        config->setFlag(BuilderFlag::kFP16);
        std::cout << "✅ FP16 mode enabled" << std::endl;
    } else {
        std::cout << "⚠️  FP16 not supported on this platform, using FP32" << std::endl;
    }

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser) {
        std::cerr << "Failed to create parser!" << std::endl;
        delete config;
        delete network;
        delete builder;
        return;
    }

    if (!parser->parseFromFile(onnx_path, static_cast<int>(ILogger::Severity::kINFO))) {
        std::cerr << "Failed to parse ONNX file: " << onnx_path << std::endl;
        delete parser;
        delete config;
        delete network;
        delete builder;
        return;
    }

    // Print network output information
    std::cout << "\n📊 Network Structure:" << std::endl;
    std::cout << "Inputs: " << network->getNbInputs() << std::endl;
    for (int i = 0; i < network->getNbInputs(); i++) {
        auto input = network->getInput(i);
        auto dims = input->getDimensions();
        std::cout << "  Input " << i << " '" << input->getName() << "': [";
        for (int j = 0; j < dims.nbDims; j++) {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "Outputs: " << network->getNbOutputs() << std::endl;
    for (int i = 0; i < network->getNbOutputs(); i++) {
        auto output = network->getOutput(i);
        auto dims = output->getDimensions();
        std::cout << "  Output " << i << " '" << output->getName() << "': [";
        for (int j = 0; j < dims.nbDims; j++) {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;

    IHostMemory* serializedEngine = builder->buildSerializedNetwork(*network, *config);
    if (!serializedEngine) {
        std::cerr << "Failed to build TensorRT engine!" << std::endl;
        delete parser;
        delete config;
        delete network;
        delete builder;
        return;
    }

    std::ofstream engineFile(engine_path, std::ios::binary);
    if (!engineFile.is_open()) {
        std::cerr << "Failed to open output file: " << engine_path << std::endl;
        delete serializedEngine;
        delete parser;
        delete config;
        delete network;
        delete builder;
        return;
    }

    engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();
    std::cout << "✅ Engine saved to: " << engine_path << std::endl;
    
    // Clean up in reverse order of creation
    delete serializedEngine;
    delete parser;
    delete config;
    delete network;
    delete builder;
}

void infer(const char* engine_path, const float* input_data, float* output_data, int input_size, int output_size) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Engine file not found: " << engine_path << std::endl;
        return;
    }

    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Failed to create runtime!" << std::endl;
        return;
    }

    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size);
    if (!engine) {
        std::cerr << "Failed to deserialize engine!" << std::endl;
        delete runtime;
        return;
    }

    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context!" << std::endl;
        delete engine;
        delete runtime;
        return;
    }

    // Get tensor names to properly bind them
    int32_t numIO = engine->getNbIOTensors();
    std::cout << "Number of I/O tensors: " << numIO << std::endl;
    
    // Print ALL tensor dimensions from the ENGINE (not ONNX)
    std::cout << "\n🔍 ACTUAL ENGINE TENSOR DIMENSIONS:" << std::endl;
    for (int32_t i = 0; i < numIO; i++) {
        const char* tensorName = engine->getIOTensorName(i);
        auto dims = engine->getTensorShape(tensorName);
        TensorIOMode mode = engine->getTensorIOMode(tensorName);
        
        std::cout << "  " << (mode == TensorIOMode::kINPUT ? "INPUT " : "OUTPUT") 
                  << " '" << tensorName << "': [";
        for (int32_t j = 0; j < dims.nbDims; j++) {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1) std::cout << ", ";
        }
        std::cout << "]";
        
        // Calculate total size
        int64_t total = 1;
        for (int32_t j = 0; j < dims.nbDims; j++) {
            total *= dims.d[j];
        }
        std::cout << " = " << total << " elements" << std::endl;
    }
    std::cout << std::endl;
    
    // Find all input and output tensors
    const char* inputName = nullptr;
    std::vector<const char*> outputNames;
    std::vector<void*> outputBuffers;
    
    for (int32_t i = 0; i < numIO; ++i) {
        const char* tensorName = engine->getIOTensorName(i);
        TensorIOMode mode = engine->getTensorIOMode(tensorName);
        
        if (mode == TensorIOMode::kINPUT && !inputName) {
            inputName = tensorName;
            std::cout << "Input tensor: " << tensorName << std::endl;
        } else if (mode == TensorIOMode::kOUTPUT) {
            outputNames.push_back(tensorName);
            std::cout << "Output tensor " << outputNames.size() << ": " << tensorName << std::endl;
        }
    }
    
    if (!inputName) {
        std::cerr << "Failed to find input tensor!" << std::endl;
        delete context;
        delete engine;
        delete runtime;
        return;
    }

    // Allocate GPU memory for input
    void* d_input = nullptr;
    cudaMalloc(&d_input, input_size * sizeof(float));
    
    // Allocate GPU memory for all outputs (we'll use the first one for simplicity)
    // In a real implementation, you'd query the actual output sizes
    for (size_t i = 0; i < outputNames.size(); ++i) {
        void* d_output = nullptr;
        // Use provided output_size for first output, allocate minimal for others
        size_t alloc_size = (i == 0) ? output_size : 1024;  // Minimal allocation for extra outputs
        cudaMalloc(&d_output, alloc_size * sizeof(float));
        outputBuffers.push_back(d_output);
    }

    // Set tensor addresses
    context->setTensorAddress(inputName, d_input);
    for (size_t i = 0; i < outputNames.size(); ++i) {
        context->setTensorAddress(outputNames[i], outputBuffers[i]);
    }

    // Copy input to GPU
    cudaMemcpy(d_input, input_data, input_size * sizeof(float), cudaMemcpyHostToDevice);

    // Execute inference
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    bool success = context->enqueueV3(stream);
    if (!success) {
        std::cerr << "Inference execution failed!" << std::endl;
        cudaStreamDestroy(stream);
        cudaFree(d_input);
        for (auto buf : outputBuffers) cudaFree(buf);
        delete context;
        delete engine;
        delete runtime;
        return;
    }

    // Wait for completion
    cudaStreamSynchronize(stream);
    std::cout << "✅ Inference completed" << std::endl;

    // Copy first output back to CPU
    if (!outputBuffers.empty()) {
        cudaMemcpy(output_data, outputBuffers[0], output_size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    for (auto buf : outputBuffers) {
        cudaFree(buf);
    }
    delete context;
    delete engine;
    delete runtime;
}

// Fast inference session structure
struct InferenceSessionImpl {
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    cudaStream_t stream;
    
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<void*> inputBuffers;
    std::vector<void*> outputBuffers;
    
    ~InferenceSessionImpl() {
        // Cleanup GPU buffers
        for (auto buf : inputBuffers) cudaFree(buf);
        for (auto buf : outputBuffers) cudaFree(buf);
        
        // Cleanup CUDA stream
        if (stream) cudaStreamDestroy(stream);
        
        // Cleanup TensorRT objects
        if (context) delete context;
        if (engine) delete engine;
        if (runtime) delete runtime;
    }
};

InferenceSession create_session(const char* engine_path) {
    std::cout << "Creating inference session from: " << engine_path << std::endl;
    
    // Load engine file
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Engine file not found: " << engine_path << std::endl;
        return nullptr;
    }

    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    // Create session
    auto session = new InferenceSessionImpl();
    
    // Create runtime and deserialize engine
    session->runtime = createInferRuntime(gLogger);
    if (!session->runtime) {
        std::cerr << "Failed to create runtime!" << std::endl;
        delete session;
        return nullptr;
    }

    session->engine = session->runtime->deserializeCudaEngine(engineData.data(), size);
    if (!session->engine) {
        std::cerr << "Failed to deserialize engine!" << std::endl;
        delete session;
        return nullptr;
    }

    session->context = session->engine->createExecutionContext();
    if (!session->context) {
        std::cerr << "Failed to create execution context!" << std::endl;
        delete session;
        return nullptr;
    }

    // Create CUDA stream
    cudaStreamCreate(&session->stream);

    // Discover all I/O tensors
    int32_t numIO = session->engine->getNbIOTensors();
    std::cout << "Session created with " << numIO << " I/O tensors" << std::endl;
    
    // Print ACTUAL engine tensor dimensions
    std::cout << "\n🔍 ACTUAL ENGINE TENSOR DIMENSIONS:" << std::endl;
    
    for (int32_t i = 0; i < numIO; ++i) {
        const char* tensorName = session->engine->getIOTensorName(i);
        TensorIOMode mode = session->engine->getTensorIOMode(tensorName);
        auto dims = session->engine->getTensorShape(tensorName);
        
        std::cout << "  " << (mode == TensorIOMode::kINPUT ? "INPUT " : "OUTPUT") 
                  << " '" << tensorName << "': [";
        for (int32_t j = 0; j < dims.nbDims; j++) {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1) std::cout << ", ";
        }
        std::cout << "]";
        
        // Calculate total size
        int64_t total = 1;
        for (int32_t j = 0; j < dims.nbDims; j++) {
            total *= dims.d[j];
        }
        std::cout << " = " << total << " elements" << std::endl;
        
        if (mode == TensorIOMode::kINPUT) {
            session->inputNames.push_back(tensorName);
        } else if (mode == TensorIOMode::kOUTPUT) {
            session->outputNames.push_back(tensorName);
        }
    }
    std::cout << std::endl;
    
    // Allocate GPU buffers for all inputs
    for (const auto& name : session->inputNames) {
        auto dims = session->engine->getTensorShape(name);
        int64_t size = 1;
        for (int j = 0; j < dims.nbDims; j++) {
            size *= dims.d[j];
        }
        void* buffer;
        cudaMalloc(&buffer, size * sizeof(float));
        session->inputBuffers.push_back(buffer);
    }
    
    // Allocate GPU buffers for all outputs
    for (const auto& name : session->outputNames) {
        auto dims = session->engine->getTensorShape(name);
        int64_t size = 1;
        for (int j = 0; j < dims.nbDims; j++) {
            size *= dims.d[j];
        }
        void* buffer;
        cudaMalloc(&buffer, size * sizeof(float));
        session->outputBuffers.push_back(buffer);
    }
    
    std::cout << "✅ Session ready (inputs: " << session->inputNames.size() 
              << ", outputs: " << session->outputNames.size() << ")" << std::endl;
    
    return static_cast<InferenceSession>(session);
}

void run_inference(InferenceSession session_ptr, const float* input_data, float* output_data, int input_size, int output_size) {
    if (!session_ptr) {
        std::cerr << "Invalid session!" << std::endl;
        return;
    }
    
    auto session = static_cast<InferenceSessionImpl*>(session_ptr);
    
    // Set tensor addresses (buffers were pre-allocated in create_session)
    for (size_t i = 0; i < session->inputNames.size(); ++i) {
        session->context->setTensorAddress(session->inputNames[i], session->inputBuffers[i]);
    }
    
    for (size_t i = 0; i < session->outputNames.size(); ++i) {
        session->context->setTensorAddress(session->outputNames[i], session->outputBuffers[i]);
    }
    
    // Copy input to GPU
    cudaMemcpyAsync(session->inputBuffers[0], input_data, input_size * sizeof(float), 
                    cudaMemcpyHostToDevice, session->stream);
    
    // Execute inference
    session->context->enqueueV3(session->stream);
    
    // Copy output from GPU
    cudaMemcpyAsync(output_data, session->outputBuffers[0], output_size * sizeof(float), 
                    cudaMemcpyDeviceToHost, session->stream);
    
    // Wait for completion
    cudaStreamSynchronize(session->stream);
}

// ============================================================================
// ZERO-COPY API - Direct GPU pointer operations
// ============================================================================

void run_inference_device(InferenceSession session_ptr, const float* d_input, float* d_output, int input_size, int output_size) {
    if (!session_ptr) {
        std::cerr << "Invalid session!" << std::endl;
        return;
    }
    
    auto session = static_cast<InferenceSessionImpl*>(session_ptr);
    
    // Set tensor addresses for inference
    // Input: use user-provided GPU buffer
    // Outputs: use pre-allocated TensorRT managed buffers from session->outputBuffers
    int32_t numIO = session->engine->getNbIOTensors();
    int outputIdx = 0;
    
    for (int32_t i = 0; i < numIO; ++i) {
        const char* name = session->engine->getIOTensorName(i);
        TensorIOMode mode = session->engine->getTensorIOMode(name);
        
        if (mode == TensorIOMode::kINPUT) {
            // Point to user's GPU input buffer (zero-copy input)
            session->context->setTensorAddress(name, const_cast<float*>(d_input));
        } else if (mode == TensorIOMode::kOUTPUT) {
            // Use TensorRT's internal managed buffers for ALL outputs
            // This avoids illegal memory access from trying to write multiple outputs
            // into a single user-provided buffer
            session->context->setTensorAddress(name, session->outputBuffers[outputIdx]);
            outputIdx++;
        }
    }
    
    // Execute inference directly on GPU - NO MEMORY COPIES!
    bool success = session->context->enqueueV3(session->stream);
    if (!success) {
        std::cerr << "Inference execution failed!" << std::endl;
        return;
    }
    
    // Wait for GPU to finish
    cudaStreamSynchronize(session->stream);
}

DeviceBuffers* get_device_buffers(InferenceSession session_ptr) {
    if (!session_ptr) {
        return nullptr;
    }
    
    auto session = static_cast<InferenceSessionImpl*>(session_ptr);
    
    // Return the pre-allocated GPU buffers for external use
    DeviceBuffers* buffers = new DeviceBuffers();
    buffers->d_input = session->inputBuffers.empty() ? nullptr : session->inputBuffers[0];
    
    // Use LAST output buffer (final detection output, not intermediate outputs)
    buffers->d_output = session->outputBuffers.empty() ? nullptr : session->outputBuffers.back();
    
    // Calculate sizes from tensor dimensions
    if (!session->inputNames.empty()) {
        auto dims = session->engine->getTensorShape(session->inputNames[0]);
        int64_t size = 1;
        for (int j = 0; j < dims.nbDims; j++) {
            size *= dims.d[j];
        }
        buffers->input_size = static_cast<int>(size);
    } else {
        buffers->input_size = 0;
    }
    
    if (!session->outputNames.empty()) {
        // Use LAST output (final detection output)
        auto dims = session->engine->getTensorShape(session->outputNames.back());
        int64_t size = 1;
        for (int j = 0; j < dims.nbDims; j++) {
            size *= dims.d[j];
        }
        buffers->output_size = static_cast<int>(size);
    } else {
        buffers->output_size = 0;
    }
    
    return buffers;
}

// Copy output data from GPU to CPU after zero-copy inference
void copy_output_to_cpu(InferenceSession session_ptr, float* output_cpu, int output_size) {
    if (!session_ptr || !output_cpu) {
        std::cerr << "Invalid session or output buffer!" << std::endl;
        return;
    }
    
    auto session = static_cast<InferenceSessionImpl*>(session_ptr);
    
    if (session->outputBuffers.empty()) {
        std::cerr << "No output buffers allocated!" << std::endl;
        return;
    }
    
    // Calculate actual size of LAST output tensor (final detection output)
    // Use .back() to get the final concatenated output, not intermediate outputs
    auto dims = session->engine->getTensorShape(session->outputNames.back());
    int64_t actual_size = 1;
    for (int j = 0; j < dims.nbDims; j++) {
        actual_size *= dims.d[j];
    }
    
    // Use the smaller of the two to avoid buffer overflow
    int64_t copy_size = std::min((int64_t)output_size, actual_size);
    
    // Copy from GPU output buffer to CPU (use .back() for final output)
    cudaError_t err = cudaMemcpy(output_cpu, session->outputBuffers.back(), copy_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    }
}

// Copy input data from CPU to GPU buffer before zero-copy inference
void copy_input_to_gpu(InferenceSession session_ptr, const float* input_cpu, int input_size) {
    if (!session_ptr || !input_cpu) {
        std::cerr << "Invalid session or input buffer!" << std::endl;
        return;
    }
    
    auto session = static_cast<InferenceSessionImpl*>(session_ptr);
    
    if (session->inputBuffers.empty()) {
        std::cerr << "No input buffers allocated!" << std::endl;
        return;
    }
    
    // Copy from CPU input to GPU buffer
    cudaMemcpy(session->inputBuffers[0], input_cpu, input_size * sizeof(float), 
               cudaMemcpyHostToDevice);
}

void destroy_session(InferenceSession session_ptr) {
    if (session_ptr) {
        auto session = static_cast<InferenceSessionImpl*>(session_ptr);
        delete session;
        std::cout << "Session destroyed" << std::endl;
    }
}
