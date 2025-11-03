#!/bin/bash
# Test script to dump all pipeline stages with mock data

echo "🧪 Testing Pipeline Dumps with Mock Data"
echo "=========================================="
echo ""

# Check if binary exists
if [ ! -f "target/release/pipeline_capture_to_output_v5" ]; then
    echo "❌ Binary not found. Please build first:"
    echo "   cargo build --release --bin pipeline_capture_to_output_v5"
    exit 1
fi

# Clear old debug files
echo "🧹 Cleaning old debug files..."
rm -f output/test/debug_frame_*.txt output/test/debug_frame_*.bin
echo ""

# Check for DeckLink cards
echo "🔍 Checking for DeckLink hardware..."
if lspci | grep -i blackmagic > /dev/null 2>&1; then
    echo "✅ DeckLink hardware detected"
    echo ""
    echo "⚠️  Note: This will attempt to capture from DeckLink device"
    echo "   Make sure SDI input is connected!"
    echo ""
    read -p "Press Enter to continue or Ctrl+C to cancel..."
    echo ""
    
    # Run with timeout to capture just a few frames
    echo "🎬 Running pipeline (will stop after 5-10 seconds)..."
    timeout 10s cargo run --release --bin pipeline_capture_to_output_v5 2>&1 | tee /tmp/pipeline_output.log
    
else
    echo "❌ No DeckLink hardware found"
    echo ""
    echo "   This pipeline requires DeckLink capture device."
    echo "   Debug files from previous runs may still exist in output/test/"
    exit 1
fi

echo ""
echo "📊 Checking generated files..."
echo ""

# List all debug files
ls -lh output/test/debug_frame_* 2>/dev/null | while read line; do
    echo "  $line"
done

echo ""
echo "📝 Summary:"
echo "=========================================="

# Count files by type
PREPROCESS_COUNT=$(ls output/test/debug_frame_*_preprocessing.txt 2>/dev/null | wc -l)
INFERENCE_COUNT=$(ls output/test/debug_frame_*_inference.txt 2>/dev/null | wc -l)
DETECTION_COUNT=$(ls output/test/debug_frame_*_detections.txt 2>/dev/null | wc -l)
PLAN_COUNT=$(ls output/test/debug_frame_*_overlay_plan.txt 2>/dev/null | wc -l)
BGRA_COUNT=$(ls output/test/debug_frame_*_overlay_bgra.bin 2>/dev/null | wc -l)

echo "  Preprocessing outputs: $PREPROCESS_COUNT"
echo "  Inference outputs:     $INFERENCE_COUNT"
echo "  Detection outputs:     $DETECTION_COUNT"
echo "  Overlay plans:         $PLAN_COUNT"
echo "  BGRA buffers:          $BGRA_COUNT"
echo ""

if [ $PREPROCESS_COUNT -gt 0 ]; then
    echo "✅ Pipeline dumps generated successfully!"
    echo ""
    echo "📂 Files are in: output/test/"
    echo ""
    echo "Example commands to view:"
    echo "  cat output/test/debug_frame_0000_preprocessing.txt"
    echo "  cat output/test/debug_frame_0000_inference.txt"
    echo "  cat output/test/debug_frame_0000_detections.txt"
    echo "  cat output/test/debug_frame_0000_overlay_plan.txt"
    echo "  hexdump -C output/test/debug_frame_0000_overlay_bgra.bin | head -50"
else
    echo "⚠️  No dump files generated"
    echo "   Check pipeline output above for errors"
fi

echo ""
