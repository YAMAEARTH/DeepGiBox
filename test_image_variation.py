#!/usr/bin/env python3
"""
Test if different images produce different segmentation outputs
"""
import onnxruntime as ort
import numpy as np
from PIL import Image
import glob

# Load model
model_path = "configs/model/gim_model_decrypted.onnx"
session = ort.InferenceSession(model_path)

# Test images
image_paths = glob.glob("apps/playgrounds/sample_img*.png") + glob.glob("apps/playgrounds/sample_img*.jpg")

print("🔍 Testing Image Variation Detection")
print("=" * 70)

# Optimal thresholds
threshold_nbi = 0.15
threshold_wle = 0.15
c_threshold = 0.25

results = []

for img_path in sorted(image_paths)[:4]:  # Test first 4 images
    print(f"\n📷 Image: {img_path}")
    
    # Load and preprocess
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((640, 512))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Check input statistics
    print(f"  Input stats: min={img_array.min():.4f}, max={img_array.max():.4f}, "
          f"mean={img_array.mean():.4f}, std={img_array.std():.4f}")
    print(f"  First 15 values: {img_array.flatten()[:15]}")
    
    # Run inference
    inputs = {
        'input': img_array.astype(np.float32),
        'threshold_NBI': np.array([threshold_nbi], dtype=np.float32),
        'threshold_WLE': np.array([threshold_wle], dtype=np.float32),
        'c_threshold': np.array([c_threshold], dtype=np.float32)
    }
    
    outputs = session.run(None, inputs)
    seg_mask = outputs[0]
    classification = outputs[1][0]
    
    # Analyze output
    unique, counts = np.unique(seg_mask, return_counts=True)
    class_0_pct = counts[0] / seg_mask.size * 100 if 0 in unique else 0
    class_255_pct = counts[1] / seg_mask.size * 100 if len(counts) > 1 else 0
    
    print(f"  Output: Class 0: {class_0_pct:.1f}%, Class 255: {class_255_pct:.1f}%")
    print(f"  Classification: {classification:.4f}")
    
    # Store a hash of the segmentation to compare
    seg_hash = hash(seg_mask.tobytes())
    results.append({
        'path': img_path,
        'seg_hash': seg_hash,
        'class_0_pct': class_0_pct,
        'class_255_pct': class_255_pct,
        'classification': classification
    })

print("\n" + "=" * 70)
print("🔍 Comparison Analysis:")
print("=" * 70)

# Check if all outputs are the same
seg_hashes = [r['seg_hash'] for r in results]
if len(set(seg_hashes)) == 1:
    print("⚠️  WARNING: All images produce IDENTICAL segmentation output!")
    print("   This suggests the model is not processing the input image.")
    print("   Possible causes:")
    print("   1. Input tensor is not being passed correctly to the model")
    print("   2. Model always uses some default/cached input")
    print("   3. Thresholds are completely overriding the image input")
elif len(set(seg_hashes)) == len(results):
    print("✅ SUCCESS: Each image produces unique segmentation output!")
else:
    print(f"⚠️  PARTIAL: {len(set(seg_hashes))} unique outputs from {len(results)} images")

print("\nDetailed comparison:")
for i, r in enumerate(results):
    print(f"  {i+1}. {r['path']}")
    print(f"     Class distribution: {r['class_0_pct']:.1f}% / {r['class_255_pct']:.1f}%")
    print(f"     Seg hash: {r['seg_hash']}")
