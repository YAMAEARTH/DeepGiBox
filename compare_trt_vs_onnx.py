#!/usr/bin/env python3
"""
Compare TensorRT output vs ONNX Runtime output for the same image
"""
import json
import numpy as np
import onnxruntime as ort
from PIL import Image

# Load TensorRT result
print("📂 Loading TensorRT output...")
with open("output/raw_segmentation.json", 'r') as f:
    trt_data = json.load(f)

trt_image_path = trt_data['image']['path']
trt_mask = np.array(trt_data['segmentation_mask'], dtype=np.int32).reshape(512, 640)
trt_class_dist = trt_data['class_distribution']

print(f"   Image: {trt_image_path}")
print(f"   TensorRT Class 0: {trt_class_dist.get('0', 0)} pixels")
print(f"   TensorRT Class 255: {trt_class_dist.get('255', 0)} pixels")

# Run ONNX Runtime on same image
print(f"\n🔄 Running ONNX Runtime on same image...")
session = ort.InferenceSession("configs/model/gim_model_decrypted.onnx")

img = Image.open(trt_image_path).convert('RGB')
img_resized = img.resize((640, 512))
img_array = np.array(img_resized, dtype=np.float32) / 255.0

# Use same thresholds as TensorRT
threshold_nbi = 0.15
threshold_wle = 0.15
c_threshold = 0.25

inputs = {
    'input': img_array.astype(np.float32),
    'threshold_NBI': np.array([threshold_nbi], dtype=np.float32),
    'threshold_WLE': np.array([threshold_wle], dtype=np.float32),
    'c_threshold': np.array([c_threshold], dtype=np.float32)
}

outputs = session.run(None, inputs)
onnx_mask = outputs[0].astype(np.int32)

# Compare
print(f"\n📊 Comparison:")
print(f"   TensorRT shape: {trt_mask.shape}")
print(f"   ONNX shape: {onnx_mask.shape}")

# Flatten for comparison
trt_flat = trt_mask.flatten()
onnx_flat = onnx_mask.flatten()

# Check if identical
if np.array_equal(trt_flat, onnx_flat):
    print(f"   ✅ IDENTICAL: TensorRT and ONNX produce exactly the same output!")
else:
    diff_pixels = np.sum(trt_flat != onnx_flat)
    diff_pct = diff_pixels / len(trt_flat) * 100
    print(f"   ⚠️  DIFFERENT: {diff_pixels} pixels differ ({diff_pct:.2f}%)")
    
    # Show class distributions
    trt_unique, trt_counts = np.unique(trt_flat, return_counts=True)
    onnx_unique, onnx_counts = np.unique(onnx_flat, return_counts=True)
    
    print(f"\n   TensorRT distribution:")
    for val, count in zip(trt_unique, trt_counts):
        pct = count / len(trt_flat) * 100
        print(f"     Class {val}: {count} ({pct:.2f}%)")
    
    print(f"\n   ONNX Runtime distribution:")
    for val, count in zip(onnx_unique, onnx_counts):
        pct = count / len(onnx_flat) * 100
        print(f"     Class {val}: {count} ({pct:.2f}%)")

# Sample comparison
print(f"\n   First 20 pixels:")
print(f"   TensorRT: {trt_flat[:20]}")
print(f"   ONNX:     {onnx_flat[:20]}")

print(f"\n   Middle 20 pixels:")
mid = len(trt_flat) // 2
print(f"   TensorRT: {trt_flat[mid:mid+20]}")
print(f"   ONNX:     {onnx_flat[mid:mid+20]}")
