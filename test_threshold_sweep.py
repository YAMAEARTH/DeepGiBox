#!/usr/bin/env python3
"""
Systematic threshold sweep to find values that produce both classes
"""
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load model
model_path = "configs/model/gim_model_decrypted.onnx"
session = ort.InferenceSession(model_path)

# Load and preprocess image
img = Image.open("apps/playgrounds/sample_img_3.png").convert('RGB')
img_resized = img.resize((640, 512))
img_array = np.array(img_resized, dtype=np.float32) / 255.0

print("🔍 Systematic Threshold Sweep")
print("=" * 70)

# Test a grid of threshold values
nbi_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
wle_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
c_values = [0.1, 0.2, 0.3, 0.4, 0.5]

best_diversity = 0
best_config = None

for c in c_values:
    print(f"\nc_threshold = {c:.2f}")
    print("-" * 70)
    
    for nbi in nbi_values:
        for wle in wle_values:
            inputs = {
                'input': img_array.astype(np.float32),
                'threshold_NBI': np.array([nbi], dtype=np.float32),
                'threshold_WLE': np.array([wle], dtype=np.float32),
                'c_threshold': np.array([c], dtype=np.float32)
            }
            
            outputs = session.run(None, inputs)
            seg_mask = outputs[0]
            classification = outputs[1][0]
            
            unique, counts = np.unique(seg_mask, return_counts=True)
            num_classes = len(unique)
            
            # Calculate diversity score (how many pixels are in minority class)
            if num_classes > 1:
                minority_percent = min(counts) / seg_mask.size * 100
                diversity = minority_percent
                
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_config = (nbi, wle, c, unique, counts, classification)
                
                if minority_percent >= 1.0:  # At least 1% of each class
                    print(f"  NBI={nbi:.2f}, WLE={wle:.2f} → ", end="")
                    for val, count in zip(unique, counts):
                        percent = count / seg_mask.size * 100
                        print(f"Class {int(val)}: {percent:5.2f}% | ", end="")
                    print(f"Classif: {classification:.2f}")

print("\n" + "=" * 70)
print("🏆 Best Configuration Found:")
print("=" * 70)

if best_config:
    nbi, wle, c, unique, counts, classification = best_config
    print(f"threshold_NBI: {nbi:.2f}")
    print(f"threshold_WLE: {wle:.2f}")
    print(f"c_threshold: {c:.2f}")
    print(f"Classification output: {classification:.4f}")
    print(f"\nClass distribution:")
    for val, count in zip(unique, counts):
        percent = count / seg_mask.size * 100
        print(f"  Class {int(val):3d}: {count:7d} pixels ({percent:6.2f}%)")
    print(f"\nBest diversity: {best_diversity:.2f}% (minority class)")
else:
    print("❌ No configuration found with multiple classes!")
    print("The model may not be producing varied output for this image.")
