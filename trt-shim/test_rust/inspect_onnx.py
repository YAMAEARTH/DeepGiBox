#!/usr/bin/env python3
"""
Inspect ONNX model structure and output shapes
"""
import onnx
import sys

def inspect_onnx(onnx_path):
    print(f"Loading ONNX model: {onnx_path}\n")
    model = onnx.load(onnx_path)
    
    print("=" * 70)
    print("MODEL INPUTS:")
    print("=" * 70)
    for input_tensor in model.graph.input:
        print(f"  Name: {input_tensor.name}")
        dims = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"  Shape: {dims}")
        print()
    
    print("=" * 70)
    print("MODEL OUTPUTS:")
    print("=" * 70)
    for output_tensor in model.graph.output:
        print(f"  Name: {output_tensor.name}")
        dims = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"  Shape: {dims}")
        
        # For YOLOv5, calculate num_classes
        if len(dims) == 3:
            num_detections = dims[1]
            values_per_detection = dims[2]
            num_classes = values_per_detection - 5  # 4 bbox + 1 objectness + classes
            print(f"  → {num_detections} detection anchors")
            print(f"  → {values_per_detection} values per detection")
            print(f"  → {num_classes} classes (bbox:4 + obj:1 + classes:{num_classes})")
        print()

if __name__ == "__main__":
    onnx_path = "assets/YOLOv5.onnx"
    inspect_onnx(onnx_path)
