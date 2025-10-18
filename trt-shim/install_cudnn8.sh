#!/bin/bash
# Script to install cuDNN 8.9.x for TensorRT 8.6

echo "=== cuDNN 8 Installation Script ==="
echo ""
echo "This script will install cuDNN 8.9.x alongside your existing cuDNN 9"
echo ""

# Check if file exists
if [ ! -f ~/Downloads/cudnn-linux-x86_64-8.9*.tar.xz ]; then
    echo "❌ ERROR: cuDNN 8.9.x tar.xz file not found in ~/Downloads/"
    echo ""
    echo "Please download it from:"
    echo "https://developer.nvidia.com/rdp/cudnn-archive"
    echo ""
    echo "Download: cuDNN v8.9.7 for CUDA 11.x (TAR file)"
    echo "File should be: cudnn-linux-x86_64-8.9.7.*_cuda11-archive.tar.xz"
    exit 1
fi

cd ~/Downloads

echo "📦 Extracting cuDNN 8.9..."
tar -xvf cudnn-linux-x86_64-8.9*.tar.xz

echo ""
echo "📁 Installing to /usr/local/cudnn-8.9..."
sudo mv cudnn-linux-x86_64-8.9*-archive /usr/local/cudnn-8.9

echo ""
echo "✅ cuDNN 8.9 installed successfully!"
echo ""
echo "Now run your program with:"
echo "cd /home/earth/Documents/pun/trt-shim/test_rust"
echo "LD_LIBRARY_PATH=/usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib:/usr/local/cudnn-8.9/lib:\$LD_LIBRARY_PATH ./target/release/test_rust"
