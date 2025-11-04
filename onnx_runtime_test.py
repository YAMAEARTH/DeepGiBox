#!/usr/bin/env python3
"""
Quick ONNX Runtime test for GIM model.
- Loads ONNX model
- Prints model inputs/outputs
- Prepares a dummy image and scalar thresholds
- Runs inference and prints output shapes and statistics
"""
import os
import numpy as np
import onnxruntime as ort
import onnx

MODEL_PATH = os.path.expanduser("/home/earth/Documents/pun/DeepGiBox/configs/model/gim_model_decrypted.onnx")

print("Loading ONNX model:", MODEL_PATH)
if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"ONNX model not found: {MODEL_PATH}")

# Inspect via ONNX
onnx_model = onnx.load(MODEL_PATH)
print("ONNX model loaded. Opset: ", onnx_model.opset_import[0].version if onnx_model.opset_import else 'unknown')

# Create ORT session (CPU)
so = ort.SessionOptions()
so.log_severity_level = 0
sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"]) 

print("\n--- Model inputs ---")
inputs = sess.get_inputs()
for i in inputs:
    print(f" {i.name}: shape={i.shape}, dtype={i.type}")

print("\n--- Model outputs ---")
outputs = sess.get_outputs()
for o in outputs:
    print(f" {o.name}: shape={o.shape}, dtype={o.type}")

# Build input feed
feed = {}
for inp in inputs:
    name = inp.name
    shape = []
    for d in inp.shape:
        if isinstance(d, str) or d is None:
            # symbolic/dynamic dimension: pick 1
            shape.append(1)
        else:
            shape.append(int(d))

    # If this is the image input and has 3 dims (H,W,C) add batch dim
    if len(shape) == 3:
        # the reported model uses HWC without batch; we'll prepend batch=1
        img_shape = (1, shape[0], shape[1], shape[2])
        print(f"Preparing dummy image for '{name}' with shape {img_shape}")
        # Random image in [0,1]
        feed[name] = np.random.rand(*img_shape).astype(np.float32)
    else:
        # treat as scalar or 1D
        if len(shape) == 0:
            print(f"Preparing scalar 0.5 for '{name}'")
            feed[name] = np.array(0.5, dtype=np.float32)
        else:
            # 1-D or other shape: fill with 0.5
            arr_shape = tuple(shape)
            print(f"Preparing array for '{name}' with shape {arr_shape}")
            feed[name] = np.full(arr_shape, 0.5, dtype=np.float32)

print("\nRunning inference...")
outs = sess.run(None, feed)

print("\n--- Outputs ---")
for name, out in zip([o.name for o in outputs], outs):
    out = np.array(out)
    print(f"Output '{name}': shape={out.shape}, dtype={out.dtype}")
    print(f"  min={out.min()}, max={out.max()}, mean={out.mean():.6f}")
    flat = out.flatten()
    print("  first 40 values:", flat[:40])

# Save outputs for inspection
np.savez_compressed("onnx_test_outputs.npz", **{o.name: np.array(v) for o, v in zip(outputs, outs)})
print("\nSaved outputs to onnx_test_outputs.npz")
