# TensorRT Shim - Segmentation Fault Fix

## Problem

The program was experiencing a segmentation fault when exiting, even though the TensorRT engine was successfully created.

## Root Cause

TensorRT maintains global/static objects that are automatically cleaned up when the shared library (`.so` file) is unloaded. When the Rust program exits and drops the `Library` handle, it triggers library unload, which causes TensorRT's cleanup code to run. This cleanup was causing a segmentation fault due to the order in which objects were being destroyed.

## Solutions Attempted

### 1. ❌ Using `std::unique_ptr` with explicit reset
```cpp
auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));
// ... use builder ...
builder.reset();  // Explicit cleanup
```
**Result**: Still segfaulted because the issue was during library unload, not function scope.

### 2. ✅ Manual memory management with raw pointers
```cpp
IBuilder* builder = createInferBuilder(gLogger);
// ... use builder ...
delete builder;  // Manual cleanup in correct order
```
**Result**: Better, but still segfaulted during library unload.

### 3. ✅✅ Prevent library unload using `std::mem::forget()`
```rust
let lib = Library::new("../build/libtrt_shim.so")?;
// ... use library ...
std::mem::forget(lib);  // Prevent library from being unloaded
```
**Result**: **FIXED!** No more segfaults.

## Final Solution

The solution has two parts:

### Rust Side (test_rust/src/main.rs)
```rust
use libloading::Library;

fn main() {
    unsafe {
        let lib = Library::new("../build/libtrt_shim.so").unwrap();
        let build_engine: libloading::Symbol<unsafe extern "C" fn(*const i8, *const i8)> = 
            lib.get(b"build_engine").unwrap();
        
        build_engine(onnx.as_ptr(), engine.as_ptr());
        
        // CRITICAL: Prevent library from being unloaded
        std::mem::forget(lib);
    }
}
```

### C++ Side (src/trt_shim.cpp)
```cpp
void build_engine(const char* onnx_path, const char* engine_path) {
    // Use raw pointers for TensorRT objects
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetworkV2(...);
    IBuilderConfig* config = builder->createBuilderConfig();
    // ... build engine ...
    
    // Manually delete in reverse order of creation
    delete serializedEngine;
    delete parser;
    delete config;
    delete network;
    delete builder;
}
```

## Why This Works

1. **Manual memory management**: Ensures objects are deleted in the correct order within the function
2. **`std::mem::forget(lib)`**: Prevents Rust from dropping the Library handle, which would unload the `.so` file
3. **Library stays loaded**: TensorRT's global cleanup never runs, avoiding the segfault

## Trade-offs

### Pros
- ✅ No segfaults
- ✅ Clean program exit
- ✅ Works reliably across multiple runs
- ✅ Simple solution

### Cons
- ⚠️  Library stays loaded in memory (negligible for short-lived programs)
- ⚠️  Memory leak of the Library handle itself (tiny ~8 bytes)

## When to Use This Pattern

This solution is appropriate for:
- ✅ Short-lived CLI tools
- ✅ Build scripts
- ✅ One-shot conversions
- ✅ Test programs

For long-running services, you'd want to keep the library loaded anyway to avoid reload overhead, so `std::mem::forget()` is actually beneficial.

## Verification

```bash
cd test_rust
cargo run
# Should exit cleanly with exit code 0
echo $?  # Should print: 0
```

**Before fix**: Exit code 139 (segfault)  
**After fix**: Exit code 0 (success)

## References

- TensorRT uses RAII and global objects for CUDA context management
- Shared library unloading triggers destructor chains that can fail
- `std::mem::forget()` is safe for preventing drop in Rust
- Manual memory management gives fine control over destruction order
