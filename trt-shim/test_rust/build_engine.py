#!/usr/bin/env python3
"""
Simple script to build TensorRT engine from ONNX model
"""
import tensorrt as trt
import sys
import os

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_file_path, engine_file_path):
    """Build TensorRT engine from ONNX model."""
    
    print(f"Building TensorRT engine from: {onnx_file_path}")
    print(f"Output engine file: {engine_file_path}")
    
    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    print("Parsing ONNX file...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    print("ONNX file parsed successfully!")
    
    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # Build engine
    print("Building TensorRT engine (this may take a while)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build engine!")
        return False
    
    # Save engine to file
    print(f"Saving engine to {engine_file_path}")
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
    
    file_size = os.path.getsize(engine_file_path) / (1024 * 1024)
    print(f"✅ Engine built successfully! Size: {file_size:.2f} MB")
    return True

if __name__ == "__main__":
    onnx_path = "assets/YOLOv5.onnx"
    engine_path = "assets/optimized_YOLOv5.engine"
    
    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX file not found: {onnx_path}")
        sys.exit(1)
    
    success = build_engine(onnx_path, engine_path)
    sys.exit(0 if success else 1)
