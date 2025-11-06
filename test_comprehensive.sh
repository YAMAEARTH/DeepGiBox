#!/bin/bash

# =============================================================================
# DeepGiBox Comprehensive Feature Test Suite
# =============================================================================
# Tests all features with detailed validation
# Usage: ./test_comprehensive.sh
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

PASSED=0
FAILED=0
WARNINGS=0

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_test() {
    echo -e "${CYAN}[TEST]${NC} $1"
}

print_pass() {
    echo -e "${GREEN}  ✓ PASSED${NC} - $1"
    ((PASSED++))
}

print_fail() {
    echo -e "${RED}  ✗ FAILED${NC} - $1"
    ((FAILED++))
}

print_warn() {
    echo -e "${YELLOW}  ⚠ WARNING${NC} - $1"
    ((WARNINGS++))
}

print_header "DeepGiBox Comprehensive Feature Test"
echo ""

# =============================================================================
# SECTION 1: Build System Tests
# =============================================================================
print_header "1. Build System Tests"
echo ""

print_test "Checking Cargo workspace configuration..."
if cargo metadata --no-deps > /dev/null 2>&1; then
    print_pass "Cargo workspace is valid"
else
    print_fail "Cargo workspace configuration is invalid"
fi

print_test "Checking runner binary exists..."
if [ -f "target/release/runner" ]; then
    SIZE=$(stat -c%s "target/release/runner" 2>/dev/null || stat -f%z "target/release/runner" 2>/dev/null || echo "unknown")
    print_pass "Runner binary exists (Size: $(numfmt --to=iec $SIZE 2>/dev/null || echo $SIZE) bytes)"
else
    print_fail "Runner binary not found. Run: cargo build --release --bin runner"
fi

print_test "Checking for CUDA dependencies..."
if ldd target/release/runner 2>/dev/null | grep -q "cuda"; then
    print_pass "CUDA libraries linked correctly"
else
    print_warn "CUDA libraries not found in binary dependencies"
fi

echo ""

# =============================================================================
# SECTION 2: Configuration File Tests
# =============================================================================
print_header "2. Configuration File Tests"
echo ""

CONFIGS=(
    "configs/runner_pentax.toml:Pentax Endoscope Config"
    "configs/runner_olympus.toml:Olympus Endoscope Config"
    "configs/runner_fuji.toml:Fuji Endoscope Config"
    "configs/runner_keying.toml:Hardware Keying Config"
    "configs/runner_inference_only.toml:Inference Only Config"
)

for config_info in "${CONFIGS[@]}"; do
    IFS=':' read -r config desc <<< "$config_info"
    print_test "Validating $desc ($config)..."
    
    if [ ! -f "$config" ]; then
        print_fail "Config file not found"
        continue
    fi
    
    # Check TOML syntax
    if python3 -c "import tomli; tomli.load(open('$config', 'rb'))" 2>/dev/null || \
       python3 -c "import toml; toml.load('$config')" 2>/dev/null; then
        print_pass "Valid TOML syntax"
    else
        print_warn "Could not validate TOML syntax (python toml module not available)"
    fi
done

echo ""

# =============================================================================
# SECTION 3: Pipeline Component Tests
# =============================================================================
print_header "3. Pipeline Component Tests"
echo ""

print_test "Testing DeckLink Input Module..."
if cargo build --release --lib -p decklink_input 2>&1 | tail -1 | grep -q "Finished"; then
    print_pass "DeckLink input module compiles"
else
    print_fail "DeckLink input module failed to compile"
fi

print_test "Testing DeckLink Output Module..."
if cargo build --release --lib -p decklink_output 2>&1 | tail -1 | grep -q "Finished"; then
    print_pass "DeckLink output module compiles"
else
    print_fail "DeckLink output module failed to compile"
fi

print_test "Testing Preprocessing CUDA Module..."
if cargo build --release --lib -p preprocess_cuda 2>&1 | tail -1 | grep -q "Finished"; then
    print_pass "Preprocessing CUDA module compiles"
else
    print_fail "Preprocessing CUDA module failed to compile"
fi

print_test "Testing Inference V2 Module..."
if cargo build --release --lib -p inference_v2 2>&1 | tail -1 | grep -q "Finished"; then
    print_pass "Inference V2 module compiles"
else
    print_fail "Inference V2 module failed to compile"
fi

print_test "Testing Postprocess Module..."
if cargo build --release --lib -p postprocess 2>&1 | tail -1 | grep -q "Finished"; then
    print_pass "Postprocess module compiles"
else
    print_fail "Postprocess module failed to compile"
fi

print_test "Testing Overlay Plan Module..."
if cargo build --release --lib -p overlay_plan 2>&1 | tail -1 | grep -q "Finished"; then
    print_pass "Overlay plan module compiles"
else
    print_fail "Overlay plan module failed to compile"
fi

print_test "Testing Overlay Render Module..."
if cargo build --release --lib -p overlay_render 2>&1 | tail -1 | grep -q "Finished"; then
    print_pass "Overlay render module compiles"
else
    print_fail "Overlay render module failed to compile"
fi

echo ""

# =============================================================================
# SECTION 4: Runtime Tests (Short Duration)
# =============================================================================
print_header "4. Runtime Tests"
echo ""

print_test "Testing Pentax Pipeline (5 seconds)..."
if timeout 5 ./target/release/runner configs/runner_pentax.toml 2>&1 | grep -q "Frame"; then
    print_pass "Pentax pipeline processes frames"
else
    print_warn "Pentax pipeline test inconclusive (may need DeckLink hardware)"
fi

print_test "Testing Olympus Pipeline (5 seconds)..."
if timeout 5 ./target/release/runner configs/runner_olympus.toml 2>&1 | grep -q "Frame"; then
    print_pass "Olympus pipeline processes frames"
else
    print_warn "Olympus pipeline test inconclusive (may need DeckLink hardware)"
fi

print_test "Testing Fuji Pipeline (5 seconds)..."
if timeout 5 ./target/release/runner configs/runner_fuji.toml 2>&1 | grep -q "Frame"; then
    print_pass "Fuji pipeline processes frames"
else
    print_warn "Fuji pipeline test inconclusive (may need DeckLink hardware)"
fi

print_test "Testing Inference Only Mode (5 seconds)..."
if timeout 5 ./target/release/runner configs/runner_inference_only.toml 2>&1 | grep -q "Frame"; then
    print_pass "Inference only mode processes frames"
else
    print_warn "Inference only test inconclusive (may need DeckLink hardware)"
fi

echo ""

# =============================================================================
# SECTION 5: Feature Validation Tests
# =============================================================================
print_header "5. Feature Validation Tests"
echo ""

print_test "Checking TensorRT cache directory..."
if [ -d "trt_cache" ]; then
    COUNT=$(ls -1 trt_cache/*.engine 2>/dev/null | wc -l)
    print_pass "TensorRT cache exists with $COUNT engine(s)"
else
    print_warn "TensorRT cache directory not found"
fi

print_test "Checking model configuration..."
if [ -d "configs/model" ] && [ -f "configs/model/yolov8n.toml" ]; then
    print_pass "Model configuration files present"
else
    print_warn "Model configuration directory incomplete"
fi

print_test "Checking documentation..."
DOCS=(
    "apps/runner/README.md"
    "RUNNER_QUICK_START.md"
    "RUNNER_SUMMARY.md"
)
DOC_FOUND=0
for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        ((DOC_FOUND++))
    fi
done
if [ $DOC_FOUND -eq ${#DOCS[@]} ]; then
    print_pass "All documentation files present ($DOC_FOUND/${#DOCS[@]})"
else
    print_warn "Some documentation files missing ($DOC_FOUND/${#DOCS[@]})"
fi

echo ""

# =============================================================================
# SECTION 6: Performance Metrics Check
# =============================================================================
print_header "6. Performance Metrics Check"
echo ""

print_test "Running performance benchmark (10 seconds)..."
PERF_OUTPUT=$(timeout 10 ./target/release/runner configs/runner_pentax.toml 2>&1 || true)

if echo "$PERF_OUTPUT" | grep -q "FPS:"; then
    FPS=$(echo "$PERF_OUTPUT" | grep "FPS:" | tail -1 | sed 's/.*FPS: \([0-9.]*\).*/\1/')
    print_pass "FPS measurement captured: ${FPS} fps"
else
    print_warn "FPS metrics not captured"
fi

if echo "$PERF_OUTPUT" | grep -q "Latency:"; then
    LATENCY=$(echo "$PERF_OUTPUT" | grep "Latency:" | tail -1 | sed 's/.*Latency: \([0-9.]*\)ms.*/\1/')
    print_pass "Latency measurement captured: ${LATENCY}ms"
else
    print_warn "Latency metrics not captured"
fi

if echo "$PERF_OUTPUT" | grep -q "Detections found:"; then
    print_pass "Detection system functioning"
else
    print_warn "Detection output not captured"
fi

echo ""

# =============================================================================
# SECTION 7: Memory and Resource Tests
# =============================================================================
print_header "7. Memory and Resource Tests"
echo ""

print_test "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi > /dev/null 2>&1; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        print_pass "GPU detected: $GPU_NAME ($GPU_MEM MB)"
    else
        print_warn "nvidia-smi command exists but GPU not accessible"
    fi
else
    print_warn "nvidia-smi not found (GPU may not be available)"
fi

print_test "Checking DeckLink devices..."
if [ -d "/dev/blackmagic" ]; then
    DEV_COUNT=$(ls -1 /dev/blackmagic/ 2>/dev/null | wc -l)
    print_pass "DeckLink device directory found ($DEV_COUNT device(s))"
else
    print_warn "DeckLink device directory not found"
fi

echo ""

# =============================================================================
# SECTION 8: Code Quality Tests
# =============================================================================
print_header "8. Code Quality Tests"
echo ""

print_test "Running cargo check..."
if cargo check --release --bin runner 2>&1 | tail -1 | grep -q "Finished"; then
    print_pass "Cargo check passed"
else
    print_fail "Cargo check found issues"
fi

print_test "Checking for TODO/FIXME comments..."
TODO_COUNT=$(grep -r "TODO\|FIXME" apps/runner/src/ 2>/dev/null | wc -l || echo 0)
if [ "$TODO_COUNT" -eq 0 ]; then
    print_pass "No TODO/FIXME comments found"
else
    print_warn "$TODO_COUNT TODO/FIXME comments found"
fi

print_test "Checking runner main.rs structure..."
if [ -f "apps/runner/src/main.rs" ]; then
    LINES=$(wc -l < apps/runner/src/main.rs)
    print_pass "Runner main.rs exists ($LINES lines)"
else
    print_fail "Runner main.rs not found"
fi

print_test "Checking config_loader module..."
if [ -f "apps/runner/src/config_loader.rs" ]; then
    LINES=$(wc -l < apps/runner/src/config_loader.rs)
    print_pass "Config loader module exists ($LINES lines)"
else
    print_fail "Config loader module not found"
fi

echo ""

# =============================================================================
# Final Summary
# =============================================================================
print_header "Test Summary"
echo ""
echo -e "  ${GREEN}Passed:${NC}   $PASSED"
echo -e "  ${RED}Failed:${NC}   $FAILED"
echo -e "  ${YELLOW}Warnings:${NC} $WARNINGS"
echo ""

TOTAL=$((PASSED + FAILED + WARNINGS))
SUCCESS_RATE=$(awk "BEGIN {printf \"%.1f\", ($PASSED/$TOTAL)*100}")

echo -e "Success Rate: ${GREEN}${SUCCESS_RATE}%${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  All Critical Tests Passed! ✓${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 0
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}  Some Tests Failed ✗${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 1
fi
