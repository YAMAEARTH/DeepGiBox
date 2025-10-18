# TensorRT Shim Rust Test

This test demonstrates loading and using the TensorRT shim library from Rust using `libloading`.

## Status

✅ **Working**: The `build_engine` function successfully converts ONNX models to TensorRT engines  
✅ **Fixed**: Segfault issue resolved using `std::mem::forget()` to prevent premature library unload  
🚧 **In Progress**: The `infer` function needs proper TensorRT execution context setup

## Prerequisites

1. **Build the C++ shim library**:
   ```bash
   cd ../build
   cmake ..
   make
   ```

2. **ONNX model**: Place your ONNX model at `assets/YOLOv5.onnx`  
   (or modify the path in `src/main.rs`)

## Run the test

```bash
cargo run
```

The program will:
1. Load the shared library (`libtrt_shim.so`)
2. Call `build_engine` to convert ONNX → TensorRT engine
3. Verify the engine file was created
4. Display the file size
5. Exit cleanly without segfaults ✅

**Expected output**:
```
Building TensorRT engine...
[TRT logs...]
✅ Engine saved to: assets/optimized_YOLOv5.engine
✅ Success! Engine file created at: assets/optimized_YOLOv5.engine
   File size: 117084620 bytes (111.66 MB)
```

## How it works

The shim provides a C FFI interface to TensorRT:

```rust
// Load library
let lib = Library::new("../build/libtrt_shim.so")?;

// Get C function pointer
let build_engine: Symbol<unsafe extern "C" fn(*const i8, *const i8)> = 
    lib.get(b"build_engine")?;

// Call the function
let onnx = CString::new("assets/model.onnx")?;
let engine = CString::new("assets/model.engine")?;
build_engine(onnx.as_ptr(), engine.as_ptr());

// Prevent library unload to avoid TensorRT cleanup issues
std::mem::forget(lib);
```

### Why `std::mem::forget()`?

TensorRT has static/global objects that get cleaned up when the shared library is unloaded. By using `std::mem::forget(lib)`, we prevent the library from being unloaded, which avoids segfaults during cleanup. This is a safe workaround for short-lived programs.

For long-running services, you'd want to keep the library handle around anyway to avoid reloading overhead.

## Functions Available

### ✅ `build_engine(onnx_path, engine_path)`
Converts an ONNX model to optimized TensorRT engine format.

**Status**: Fully working, tested with YOLOv5 models

### 🚧 `infer(engine_path, input_data, output_data, input_size, output_size)`  
Runs inference on a TensorRT engine (needs proper buffer management).

**Status**: Implementation needs updates for proper tensor binding

## Next Steps

To use inference, the C++ `infer` function needs updates to properly:
- Query the engine for input/output tensor names and dimensions
- Set up execution context with proper tensor bindings
- Handle dynamic shapes if needed

## Files

- `src/main.rs` - Rust test code
- `../src/trt_shim.cpp` - C++ implementation
- `../include/trt_shim.h` - C header with function declarations
- `../build/libtrt_shim.so` - Compiled shared library
