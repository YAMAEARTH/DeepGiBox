# Postprocessing Optimization Plan

## Current Performance (from test_detectionspacket.rs)

```
Postprocessing Breakdown (5 frames avg):
├─ Decode:  6.62ms   (9%)   ← YOLO anchor decoding
├─ NMS:    63.65ms  (90%)   ← Bottleneck!
└─ Track:   3.42ms   (1%)   ← SORT tracking
───────────────────────────
Total:     74.40ms  (100%)
```

**Issue:** NMS takes 63.65ms on ~1300 candidates → Too slow!

---

## 🎯 Optimization Strategies

### 1. **Top-K Prefilter** (Already in code but needs tuning)
**Current:** 5000 threshold  
**Problem:** Still too many candidates (1328 anchors)  
**Solution:** Lower to 300-500 before NMS

**Expected gain:** 60-70% faster NMS (63ms → 20-25ms)

### 2. **Replace similari NMS with fast implementation**
**Current:** similari::utils::nms (general-purpose, safe)  
**Problem:** Not optimized for YOLO batch processing  
**Solution:** Custom vectorized NMS with SIMD

**Expected gain:** 2-3x faster (20ms → 7-10ms)

### 3. **Skip NMS for low-density frames**
**Strategy:** If candidates < 100, skip class-wise NMS  
**Rationale:** Few boxes = unlikely overlaps

**Expected gain:** 5-10ms on sparse frames

### 4. **Parallel class-wise NMS**
**Current:** Sequential per-class processing  
**Solution:** Use rayon to parallelize across classes  
**Expected gain:** 30-40% on multi-class frames

### 5. **Remove temporal smoothing overhead**
**Current:** Vec allocations + median calculation every frame  
**Problem:** Adds 1-2ms with little benefit (scores already stable)  
**Solution:** Make it truly optional (no-op when disabled)

**Expected gain:** 1-2ms

### 6. **Optimize tracking**
**Current:** Converting formats multiple times  
**Solution:** Work with common format throughout  
**Expected gain:** 1-2ms

---

## 🚀 Implementation Plan

### Phase 1: Quick Wins (Target: 74ms → 30ms)

1. **Aggressive Top-K prefilter** (300 candidates)
2. **Skip temporal smoothing** (disable by default)
3. **Fast NMS implementation**

### Phase 2: Advanced (Target: 30ms → 15ms)

4. **Parallel class-wise NMS**
5. **SIMD-optimized IoU calculation**

### Phase 3: GPU Acceleration (Target: 15ms → 5ms)

6. **CUDA NMS kernel** (if needed for 4K)

---

## 📊 Expected Results

| Optimization | Current | After | Gain |
|--------------|---------|-------|------|
| Baseline | 74.40ms | - | - |
| + Top-K (300) | 74.40ms | 35ms | 53% |
| + Fast NMS | 35ms | 25ms | 29% |
| + Disable smoothing | 25ms | 23ms | 8% |
| + Parallel NMS | 23ms | 18ms | 22% |
| **Total** | **74.40ms** | **~18ms** | **76%** |

**Target:** **< 20ms** postprocessing (from 74ms)

---

## Implementation Priority

1. ✅ **Top-K prefilter tuning** (5 min, huge gain)
2. ✅ **Custom fast NMS** (30 min, major gain)
3. ✅ **Disable smoothing default** (2 min, small gain)
4. 🔄 **Parallel NMS** (15 min, medium gain)
5. ⏸️ **GPU NMS** (later if needed)

Let's start!
