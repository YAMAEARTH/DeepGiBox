#!/bin/bash

# คำสั่งสำหรับรัน Internal Keying Demo
# ใช้สำหรับ composite PNG overlay บน DeckLink video

echo "🎬 Internal Keying - Quick Start Guide"
echo "========================================"
echo ""

# ตรวจสอบว่า build แล้วหรือยัง
if [ ! -f "target/release/internal_keying_demo" ]; then
    echo "❌ Binary not found. Building..."
    echo ""
    CC=gcc-12 CXX=g++-12 cargo build --release
    echo ""
fi

echo "📋 Available PNG files:"
ls -lh *.png 2>/dev/null || echo "   No PNG files found"
echo ""

echo "🎯 Usage Options:"
echo ""
echo "1️⃣  Fast Mode (Alpha-Only) - Recommended for PNG with transparency:"
echo "   ./target/release/internal_keying_demo foreground.png"
echo ""
echo "2️⃣  Slow Mode (Chroma Key) - For green screen:"
echo "   Edit internal_keying_demo/src/main.rs:"
echo "   Change: let use_alpha_only = false;  (line 58)"
echo "   Then rebuild and run"
echo ""

echo "🚀 Running Internal Keying Demo..."
echo "   (Ctrl+C to stop)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# เลือกภาพ (ใช้ argument แรก หรือ foreground.png)
PNG_FILE="${1:-foreground.png}"

if [ ! -f "$PNG_FILE" ]; then
    echo "❌ Error: $PNG_FILE not found!"
    echo ""
    echo "Available files:"
    ls -1 *.png 2>/dev/null || echo "   No PNG files"
    exit 1
fi

echo "📸 Using overlay: $PNG_FILE"
echo ""

# รันโปรแกรม
./target/release/internal_keying_demo "$PNG_FILE"
