#!/usr/bin/env python3
"""
Interactive real-time threshold adjustment for GIM segmentation.
Adjust NBI, WLE, and C thresholds with sliders and see results instantly.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import subprocess
import json
from pathlib import Path
from PIL import Image
import time

class InteractiveThresholdViewer:
    def __init__(self, image_path, engine_path, lib_path):
        self.image_path = image_path
        self.engine_path = engine_path
        self.lib_path = lib_path
        
        # Load and resize original image
        self.original_img = Image.open(image_path).convert('RGB')
        self.original_img = self.original_img.resize((640, 512), Image.Resampling.LANCZOS)
        self.original_img = np.array(self.original_img)
        
        # Initialize with balanced thresholds
        self.current_nbi = 0.145
        self.current_wle = 0.145
        self.current_c = 0.260
        
        # Create figure and axes
        self.fig = plt.figure(figsize=(16, 9))
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Main overlay display (top-left, larger)
        self.ax_overlay = self.fig.add_subplot(gs[0, 0])
        self.ax_overlay.set_title('Segmentation Overlay (Red = Foreground)', fontsize=14, fontweight='bold')
        self.ax_overlay.axis('off')
        
        # Segmentation mask only (top-right)
        self.ax_mask = self.fig.add_subplot(gs[0, 1])
        self.ax_mask.set_title('Segmentation Mask', fontsize=12, fontweight='bold')
        self.ax_mask.axis('off')
        
        # Original image (bottom-left)
        self.ax_original = self.fig.add_subplot(gs[1, 0])
        self.ax_original.set_title('Original Image', fontsize=12)
        self.ax_original.imshow(self.original_img)
        self.ax_original.axis('off')
        
        # Statistics panel (bottom-right)
        self.ax_stats = self.fig.add_subplot(gs[1, 1])
        self.ax_stats.axis('off')
        
        # Create sliders at the bottom
        slider_height = 0.03
        slider_bottom = 0.02
        slider_spacing = 0.04
        
        ax_nbi = plt.axes([0.15, slider_bottom + 2*slider_spacing, 0.7, slider_height])
        ax_wle = plt.axes([0.15, slider_bottom + 1*slider_spacing, 0.7, slider_height])
        ax_c = plt.axes([0.15, slider_bottom, 0.7, slider_height])
        
        self.slider_nbi = Slider(ax_nbi, 'NBI Threshold', 0.10, 0.20, valinit=self.current_nbi, valstep=0.001)
        self.slider_wle = Slider(ax_wle, 'WLE Threshold', 0.10, 0.20, valinit=self.current_wle, valstep=0.001)
        self.slider_c = Slider(ax_c, 'C Threshold', 0.20, 0.30, valinit=self.current_c, valstep=0.001)
        
        # Connect slider events
        self.slider_nbi.on_changed(self.on_slider_change)
        self.slider_wle.on_changed(self.on_slider_change)
        self.slider_c.on_changed(self.on_slider_change)
        
        # Initial render
        self.update_display()
        
    def run_inference(self, nbi, wle, c):
        """Run Rust inference binary with given thresholds"""
        # Create a temporary config file with single threshold
        config = [(nbi, wle, c, "interactive")]
        
        # Write to a temp Rust file
        rust_code = f'''
use anyhow::Result;
use common_io::{{ColorSpace, DType, FrameMeta, MemLoc, MemRef, PixelFormat, TensorDesc, TensorInputPacket, RawDetectionsPacket}};
use cudarc::driver::{{CudaDevice, CudaSlice, DevicePtr}};
use image::io::Reader as ImageReader;
use inference_v3::TrtInferenceStage;

fn main() -> Result<()> {{
    let cuda_device = CudaDevice::new(0)?;
    let img = ImageReader::open("{self.image_path}")?.decode()?.to_rgb8();
    let (orig_w, orig_h) = img.dimensions();
    
    let preprocessed = preprocess_image_hwc(&img, 640, 512)?;
    let gpu_buffer: CudaSlice<f32> = cuda_device.htod_sync_copy(&preprocessed)?;
    let gpu_ptr = *gpu_buffer.device_ptr() as *mut u8;
    let gpu_len = preprocessed.len() * std::mem::size_of::<f32>();
    
    let frame_meta = FrameMeta {{
        source_id: 1, width: orig_w, height: orig_h, pixfmt: PixelFormat::RGB8,
        colorspace: ColorSpace::BT709, frame_idx: 1, pts_ns: 0,
        t_capture_ns: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos() as u64,
        stride_bytes: orig_w * 3, crop_region: None,
    }};
    
    let tensor_desc = TensorDesc {{ n: 1, c: 3, h: 512, w: 640, dtype: DType::Fp32, device: 0 }};
    let mem_ref = MemRef {{ ptr: gpu_ptr, len: gpu_len, stride: (640 * 3 * 4) as usize, loc: MemLoc::Gpu {{ device: 0 }} }};
    let tensor_packet = TensorInputPacket {{ from: frame_meta, desc: tensor_desc, data: mem_ref }};
    
    let mut inference_stage = TrtInferenceStage::new("{self.engine_path}", "{self.lib_path}")?;
    let segmentation: RawDetectionsPacket = inference_stage.infer_with_thresholds(tensor_packet, {nbi}, {wle}, {c})?;
    
    let seg_output = &segmentation.raw_output[0..327680];
    let segmentation_mask: Vec<i32> = seg_output.iter().map(|&v| v.round() as i32).collect();
    
    let mut class_counts = std::collections::HashMap::new();
    for &val in seg_output.iter() {{
        let class_id = val.round() as i32;
        *class_counts.entry(class_id).or_insert(0) += 1;
    }}
    
    let json_data = serde_json::json!({{
        "segmentation_mask": segmentation_mask,
        "class_distribution": class_counts,
    }});
    
    std::fs::write("output/seg_interactive.json", serde_json::to_string(&json_data)?)?;
    std::mem::forget(gpu_buffer);
    Ok(())
}}

fn preprocess_image_hwc(img: &image::RgbImage, target_w: u32, target_h: u32) -> Result<Vec<f32>> {{
    let resized = image::imageops::resize(img, target_w, target_h, image::imageops::FilterType::Lanczos3);
    let mut tensor = vec![0.0f32; (target_h * target_w * 3) as usize];
    for y in 0..target_h {{
        for x in 0..target_w {{
            let pixel = resized.get_pixel(x, y);
            let base_idx = ((y * target_w + x) * 3) as usize;
            tensor[base_idx] = pixel[0] as f32 / 255.0;
            tensor[base_idx + 1] = pixel[1] as f32 / 255.0;
            tensor[base_idx + 2] = pixel[2] as f32 / 255.0;
        }}
    }}
    Ok(tensor)
}}
'''
        
        # For now, just call the existing binary with a temp config
        # This is a simplified approach - in production you'd want a proper IPC mechanism
        import tempfile
        import os
        
        # Just call existing inference_v3 - we'll use the last generated config
        # For real-time, we need a different approach...
        
        # Let's use ONNX Runtime for real-time instead (much faster to iterate)
        return self.run_onnx_inference(nbi, wle, c)
    
    def run_onnx_inference(self, nbi, wle, c):
        """Run ONNX Runtime inference for faster iteration"""
        try:
            import onnxruntime as ort
            
            # Load ONNX model if not cached
            if not hasattr(self, 'ort_session'):
                model_path = "configs/model/gim_model_decrypted.onnx"
                if not Path(model_path).exists():
                    print(f"⚠️  ONNX model not found at {model_path}")
                    return None
                print(f"Loading ONNX model from {model_path}...")
                self.ort_session = ort.InferenceSession(model_path)
                
                # Preprocess and cache image (HWC format: [512, 640, 3])
                img = Image.open(self.image_path).convert('RGB')
                img = img.resize((640, 512), Image.Resampling.LANCZOS)
                img_array = np.array(img, dtype=np.float32) / 255.0
                # GIM model expects [H, W, C] format, not [N, H, W, C]
                self.img_tensor = img_array.astype(np.float32)
            
            # Prepare inputs (no batch dimension - GIM uses HWC directly)
            inputs = {
                'input': self.img_tensor,  # [512, 640, 3] - HWC format
                'threshold_NBI': np.array([nbi], dtype=np.float32),
                'threshold_WLE': np.array([wle], dtype=np.float32),
                'c_threshold': np.array([c], dtype=np.float32),
            }
            
            # Run inference
            outputs = self.ort_session.run(None, inputs)
            
            # Debug: print output shapes
            print(f"Number of outputs: {len(outputs)}")
            for i, out in enumerate(outputs):
                print(f"  Output {i} shape: {out.shape}")
            
            # GIM outputs: [0] = classification (scalar), [1] = segmentation [512, 640]
            # But we need to check which is which
            if len(outputs) >= 2:
                # Find the segmentation output (should be 2D with shape [512, 640])
                if outputs[0].size > outputs[1].size:
                    seg_mask = outputs[0]
                    classification = outputs[1]
                else:
                    seg_mask = outputs[1]
                    classification = outputs[0]
                
                # Ensure seg_mask is 2D [512, 640]
                while seg_mask.ndim > 2:
                    seg_mask = seg_mask.squeeze()
                
                if seg_mask.ndim != 2:
                    print(f"⚠️  Unexpected segmentation shape: {seg_mask.shape}")
                    return None
            else:
                print(f"⚠️  Expected 2 outputs, got {len(outputs)}")
                return None
            
            # Calculate class distribution
            unique, counts = np.unique(seg_mask.round().astype(np.int32), return_counts=True)
            class_dist = dict(zip(unique.tolist(), counts.tolist()))
            
            return {
                'mask': seg_mask,
                'class_distribution': class_dist
            }
            
        except ImportError:
            print("⚠️  onnxruntime not installed. Install with: pip install onnxruntime")
            return None
        except Exception as e:
            print(f"❌ Error running inference: {e}")
            return None
    
    def update_display(self):
        """Update all displays with current thresholds"""
        start_time = time.time()
        
        # Run inference
        result = self.run_inference(self.current_nbi, self.current_wle, self.current_c)
        
        if result is None:
            return
        
        mask = result['mask']
        class_dist = result['class_distribution']
        
        # Create overlay
        overlay = self.original_img.copy().astype(np.float32)
        red_mask = (mask.round() == 255)
        overlay[red_mask, 0] = overlay[red_mask, 0] * 0.3 + 255 * 0.7  # Red
        overlay[red_mask, 1] = overlay[red_mask, 1] * 0.3
        overlay[red_mask, 2] = overlay[red_mask, 2] * 0.3
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # Update overlay display
        self.ax_overlay.clear()
        self.ax_overlay.imshow(overlay)
        self.ax_overlay.set_title(f'Segmentation Overlay (Red = Foreground)\nNBI={self.current_nbi:.3f}, WLE={self.current_wle:.3f}, C={self.current_c:.3f}',
                                 fontsize=14, fontweight='bold')
        self.ax_overlay.axis('off')
        
        # Update mask display
        self.ax_mask.clear()
        self.ax_mask.imshow(mask, cmap='gray', vmin=0, vmax=255)
        self.ax_mask.set_title('Segmentation Mask', fontsize=12, fontweight='bold')
        self.ax_mask.axis('off')
        
        # Update statistics
        total_pixels = mask.shape[0] * mask.shape[1]
        fg_count = class_dist.get(255, 0)
        bg_count = class_dist.get(0, 0)
        fg_pct = (fg_count / total_pixels) * 100
        bg_pct = (bg_count / total_pixels) * 100
        
        inference_time = (time.time() - start_time) * 1000
        
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        stats_text = f"""
╔══════════════════════════════════════╗
║          STATISTICS                  ║
╚══════════════════════════════════════╝

Thresholds:
  NBI:  {self.current_nbi:.3f}
  WLE:  {self.current_wle:.3f}
  C:    {self.current_c:.3f}

Class Distribution:
  Background (0):  {bg_pct:6.2f}%  ({bg_count:,} px)
  Foreground (255): {fg_pct:6.2f}%  ({fg_count:,} px)

Balance: {'✅ BALANCED' if 40 <= fg_pct <= 60 else '✓ Good' if 25 <= fg_pct <= 75 else '⚠ Imbalanced'}

Performance:
  Inference: {inference_time:.1f} ms
  FPS: {1000/inference_time:.1f}
        """
        
        self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        self.fig.canvas.draw_idle()
    
    def on_slider_change(self, val):
        """Called when any slider changes"""
        self.current_nbi = self.slider_nbi.val
        self.current_wle = self.slider_wle.val
        self.current_c = self.slider_c.val
        self.update_display()

def main():
    print("🚀 Starting Interactive Threshold Viewer...")
    print("📝 Adjust sliders to change thresholds in real-time")
    print("⚠️  Note: Requires ONNX Runtime for real-time performance")
    print("   Install with: pip install onnxruntime\n")
    
    image_path = "apps/playgrounds/sample_img_4.png"
    engine_path = "configs/model/gim_model.engine"
    lib_path = "trt-shim/build/libtrt_shim.so"
    
    viewer = InteractiveThresholdViewer(image_path, engine_path, lib_path)
    plt.show()

if __name__ == '__main__':
    main()
