#!/usr/bin/env python3
"""
Test GIM model with ONNX Runtime to understand expected preprocessing
"""
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load the ONNX model
model_path = "configs/model/gim_model_decrypted.onnx"
print(f"Loading model from {model_path}...")
session = ort.InferenceSession(model_path)

# Print model info
print("\n📊 Model Inputs:")
for input_meta in session.get_inputs():
    print(f"  {input_meta.name}: {input_meta.shape} ({input_meta.type})")

print("\n📊 Model Outputs:")
for output_meta in session.get_outputs():
    print(f"  {output_meta.name}: {output_meta.shape} ({output_meta.type})")

# Load and preprocess image
img_path = "apps/playgrounds/sample_img_3.png"
print(f"\n🖼️  Loading image from {img_path}...")
img = Image.open(img_path).convert('RGB')
print(f"  Original size: {img.size}")

# Resize to model input size (640x512 -> width x height in PIL)
img_resized = img.resize((640, 512))
print(f"  Resized to: {img_resized.size}")

# Test different preprocessing approaches
print("\n🧪 Testing different preprocessing approaches...")

preprocessing_methods = [
    ("Raw [0,255]", lambda img: np.array(img, dtype=np.float32)),
    ("Normalized [0,1]", lambda img: np.array(img, dtype=np.float32) / 255.0),
    ("ImageNet (mean/std)", lambda img: (np.array(img, dtype=np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])),
    ("ImageNet (CHW)", lambda img: (np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0 - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)),
]

for method_name, preprocess_fn in preprocessing_methods:
    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print(f"{'='*60}")
    
    try:
        # Preprocess
        img_array = preprocess_fn(img_resized)
        print(f"  Preprocessed shape: {img_array.shape}")
        print(f"  Value range: [{img_array.min():.4f}, {img_array.max():.4f}]")
        print(f"  Mean: {img_array.mean():.4f}, Std: {img_array.std():.4f}")
        
        # Ensure HWC format for model input
        if img_array.shape == (3, 512, 640):  # CHW -> HWC
            img_array = img_array.transpose(1, 2, 0)
            print(f"  Converted CHW to HWC: {img_array.shape}")
        
        # Prepare inputs (matching model signature)
        threshold_nbi = np.array([0.3], dtype=np.float32)
        threshold_wle = np.array([0.3], dtype=np.float32)
        c_threshold = np.array([0.5], dtype=np.float32)
        
        inputs = {
            'input': img_array.astype(np.float32),
            'threshold_NBI': threshold_nbi,
            'threshold_WLE': threshold_wle,
            'c_threshold': c_threshold
        }
        
        # Run inference
        outputs = session.run(None, inputs)
        
        # Analyze outputs
        # Output order from model: [0] GIM segmentation, [1] NBI&WLE classification
        seg_mask = outputs[0]  # GIM segmentation output
        classification = outputs[1]  # NBI&WLE classification
        
        print(f"\n  📊 Segmentation Output:")
        print(f"    Shape: {seg_mask.shape}")
        if seg_mask.ndim >= 2:
            print(f"    Value range: [{seg_mask.min():.4f}, {seg_mask.max():.4f}]")
            # Get first 20x20 region for analysis
            sample_region = seg_mask if seg_mask.ndim == 2 else seg_mask.squeeze()
            if sample_region.ndim == 2 and sample_region.shape[0] >= 20 and sample_region.shape[1] >= 20:
                unique_vals = np.unique(sample_region[:20, :20])
                print(f"    Unique values in first 20x20: {unique_vals[:10]}")
            
            # Count class distribution
            flat_mask = seg_mask.flatten()
            unique, counts = np.unique(flat_mask, return_counts=True)
            print(f"    Class distribution:")
            for val, count in zip(unique, counts):
                percent = 100.0 * count / flat_mask.size
                if percent > 0.1:  # Only show classes with > 0.1%
                    print(f"      Class {int(val):3d}: {count:6d} pixels ({percent:5.2f}%)")
        else:
            print(f"    WARNING: Unexpected shape (expected 2D mask)")
            print(f"    Value: {seg_mask}")
        
        print(f"\n  📊 Classification Output:")
        print(f"    Shape: {classification.shape}")
        print(f"    Value: {classification}")
        
        # Check if we got meaningful output
        if seg_mask.ndim >= 2:
            flat_mask = seg_mask.flatten()
            unique = np.unique(flat_mask)
            if len(unique) > 1:
                print(f"\n  ✅ SUCCESS! Got {len(unique)} unique classes")
            else:
                print(f"\n  ⚠️  Only one class detected - may need different preprocessing")
        else:
            print(f"\n  ❌ ERROR: Segmentation output has wrong dimensions")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "="*60)
print("Testing with different threshold values...")
print("="*60)

# Test with different thresholds using the best preprocessing
img_array = np.array(img_resized, dtype=np.float32) / 255.0

threshold_combinations = [
    (0.1, 0.1, 0.2),
    (0.3, 0.3, 0.5),
    (0.5, 0.5, 0.7),
    (0.7, 0.7, 0.9),
]

for nbi, wle, c in threshold_combinations:
    print(f"\nThresholds: NBI={nbi}, WLE={wle}, C={c}")
    
    inputs = {
        'input': img_array.astype(np.float32),
        'threshold_NBI': np.array([nbi], dtype=np.float32),
        'threshold_WLE': np.array([wle], dtype=np.float32),
        'c_threshold': np.array([c], dtype=np.float32)
    }
    
    outputs = session.run(None, inputs)
    seg_mask = outputs[0]  # First output is segmentation
    classification = outputs[1]  # Second output is classification
    
    print(f"  Segmentation shape: {seg_mask.shape}")
    print(f"  Classification: {classification}")
    
    if seg_mask.ndim >= 2:
        flat_mask = seg_mask.flatten()
        unique, counts = np.unique(flat_mask, return_counts=True)
        print(f"  Unique classes: {len(unique)}")
        for val, count in zip(unique, counts):
            percent = 100.0 * count / flat_mask.size
            if percent > 0.1:
                print(f"    Class {int(val):3d}: {percent:5.2f}%")
    else:
        print(f"  WARNING: Unexpected segmentation shape: {seg_mask.shape}")
