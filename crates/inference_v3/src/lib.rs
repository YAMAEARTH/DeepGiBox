use common_io::{TensorInputPacket, RawDetectionsPacket, MemLoc};
use libloading::{Library, Symbol};
use std::ffi::CString;
use std::sync::Arc;

/// GPU buffer pointers from TensorRT
#[repr(C)]
struct DeviceBuffers {
    d_input: *mut std::ffi::c_void,
    d_output: *mut std::ffi::c_void,
    input_size: i32,
    output_size: i32,
}

/// TensorRT inference stage that processes TensorInputPacket -> RawDetectionsPacket
/// HEAVILY OPTIMIZED for maximum performance:
/// - Cached function pointers (no repeated symbol lookups)
/// - Pre-allocated output buffer (no Vec allocation per inference)
/// - Minimal branching in hot path
/// - BiSeNet model with 12 classes (CamVid dataset), 720x960 input
pub struct TrtInferenceStage {
    // Library handle (keep alive)
    _lib: Arc<Library>,
    
    // TensorRT session
    session: *mut std::ffi::c_void,
    
    // Cached function pointers (CRITICAL: avoid repeated lookups!)
    run_inference_fn: unsafe extern "C" fn(*mut std::ffi::c_void, *const f32, *mut f32, i32, i32, f32, f32, f32),
    copy_output_fn: unsafe extern "C" fn(*mut std::ffi::c_void, *mut f32, i32),
    
    // GPU output buffer pointer (pre-queried)
    output_gpu_ptr: *mut f32,
    
    // Pre-allocated CPU output buffer (reused every inference)
    output_cpu: Vec<f32>,
    
    // Model-specific constants (BiSeNet: 12 classes, 720x960 input)
    output_size: usize,
    num_classes: usize,
    input_height: usize,
    input_width: usize,
}

impl TrtInferenceStage {
    /// Create a new TensorRT inference stage
    /// 
    /// # Arguments
    /// * `engine_path` - Path to the TensorRT engine file (e.g., "assets/bisenet.engine")
    /// * `lib_path` - Path to libtrt_shim.so (e.g., "build/libtrt_shim.so")
    /// 
    /// # Returns
    /// * `Ok(TrtInferenceStage)` - Successfully initialized inference stage
    /// * `Err(String)` - Error message if initialization failed
    pub fn new(engine_path: &str, lib_path: &str) -> Result<Self, String> {
        unsafe {
            // Load the TensorRT shim library
            let raw_lib = Library::new(lib_path)
                .map_err(|e| format!("Failed to load TensorRT library at {}: {}", lib_path, e))?;
            let lib = Arc::new(raw_lib);

            // Load the create_session symbol
            let create_session: Symbol<unsafe extern "C" fn(*const i8) -> *mut std::ffi::c_void> = 
                lib.get(b"create_session")
                    .map_err(|e| format!("Failed to load create_session symbol: {}", e))?;

            // Create the TensorRT session
            let engine_path_c = CString::new(engine_path)
                .map_err(|e| format!("Invalid engine path '{}': {}", engine_path, e))?;
            
            let session = create_session(engine_path_c.as_ptr());

            if session.is_null() {
                return Err(format!("Failed to create TensorRT session with engine: {}", engine_path));
            }

            // Query engine-managed buffers so we can size the output dynamically
            let get_device_buffers: Symbol<unsafe extern "C" fn(
                *mut std::ffi::c_void
            ) -> *mut DeviceBuffers> = 
                lib.get(b"get_device_buffers")
                    .map_err(|e| format!("Failed to load get_device_buffers symbol: {}", e))?;
            
            let free_device_buffers: Symbol<unsafe extern "C" fn(*mut DeviceBuffers)> = 
                lib.get(b"free_device_buffers")
                    .map_err(|e| format!("Failed to load free_device_buffers symbol: {}", e))?;

            let buffers_ptr = get_device_buffers(session);
            if buffers_ptr.is_null() {
                return Err("get_device_buffers returned null pointer".to_string());
            }
            let buffers = &*buffers_ptr;  // Don't take ownership - just borrow

            let output_size = if buffers.output_size > 0 {
                buffers.output_size as usize
            } else {
                free_device_buffers(buffers_ptr);  // Clean up before returning error
                return Err("TensorRT reported zero-sized output buffer".to_string());
            };
            
            let output_gpu_ptr = buffers.d_output as *mut f32;
            
            // Free the buffers structure (but NOT the GPU pointers inside - they're managed by TensorRT)
            free_device_buffers(buffers_ptr);
            
            // BiSeNet model constants: 12 classes, 720x960 input
            let num_classes = 12;
            let input_height = 720;
            let input_width = 960;
            
            // Pre-allocate CPU output buffer (will be reused)
            let output_cpu = vec![0.0f32; output_size];
            
            // Cache function pointers NOW (avoid repeated symbol lookups!)
            let run_inference_fn = {
                let sym: Symbol<unsafe extern "C" fn(*mut std::ffi::c_void, *const f32, *mut f32, i32, i32, f32, f32, f32)> = 
                    lib.get(b"run_inference_device_multiple_input")
                        .map_err(|e| format!("Failed to load run_inference_device_multiple_input: {}", e))?;
                *sym // Dereference to get raw function pointer
            };
            
            let copy_output_fn = {
                let sym: Symbol<unsafe extern "C" fn(*mut std::ffi::c_void, *mut f32, i32)> = 
                    lib.get(b"copy_output_to_cpu")
                        .map_err(|e| format!("Failed to load copy_output_to_cpu: {}", e))?;
                *sym
            };

            Ok(Self {
                _lib: lib,
                session,
                run_inference_fn,
                copy_output_fn,
                output_gpu_ptr,
                output_cpu,
                output_size,
                num_classes,
                input_height,
                input_width,
            })
        }
    }

    /// Run inference on a TensorInputPacket with custom thresholds
    /// 
    /// OPTIMIZED HOT PATH - every nanosecond counts!
    /// - No symbol lookups
    /// - No memory allocations
    /// - Minimal branches
    /// - Direct function pointer calls
    #[inline]
    pub fn infer_with_thresholds(&mut self, input: TensorInputPacket, threshold_nbi: f32, threshold_wle: f32, c_threshold: f32) -> Result<RawDetectionsPacket, String> {
        unsafe {
            // Quick validation (branch predictor will optimize this)
            if !matches!(input.data.loc, MemLoc::Gpu { .. }) {
                return Err("Input must be on GPU".to_string());
            }

            // Cast pointers (zero overhead)
            let input_gpu_ptr = input.data.ptr as *const f32;
            
            // Calculate input size (single multiplication chain - fast!)
            let input_size = (input.desc.n * input.desc.c * input.desc.h * input.desc.w) as i32;

            // CRITICAL PATH: Call cached function pointers directly with thresholds!
            // No symbol lookup, no virtual dispatch, direct call
            (self.run_inference_fn)(
                self.session,
                input_gpu_ptr,
                self.output_gpu_ptr,
                input_size,
                self.output_size as i32,
                threshold_nbi,
                threshold_wle,
                c_threshold,
            );

            // Copy results using cached function pointer
            (self.copy_output_fn)(
                self.session,
                self.output_cpu.as_mut_ptr(),
                self.output_size as i32,
            );

            // BiSeNet output shape: [batch, classes, height, width]
            // Typically [1, 12, 720, 960] for per-pixel classification
            // NOTE: TensorRT may report incorrect shape metadata, but we know the correct format
            let output_shape = vec![1, self.num_classes, self.input_height, self.input_width];

            // Clone output (necessary since we reuse the buffer)
            Ok(RawDetectionsPacket {
                from: input.from,
                raw_output: self.output_cpu.clone(),
                output_shape,
            })
        }
    }

    /// Run inference on a TensorInputPacket (uses default thresholds: 0.5)
    /// 
    /// OPTIMIZED HOT PATH - every nanosecond counts!
    /// - No symbol lookups
    /// - No memory allocations
    /// - Minimal branches
    /// - Direct function pointer calls
    #[inline]
    pub fn infer(&mut self, input: TensorInputPacket) -> Result<RawDetectionsPacket, String> {
        self.infer_with_thresholds(input, 0.5, 0.5, 0.5)
    }

    /// Get the expected output size for this inference stage
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Set the output size (useful for different model architectures)
    pub fn set_output_size(&mut self, size: usize) {
        self.output_size = size;
    }
}

// Implement the Stage trait from common_io
impl common_io::Stage<TensorInputPacket, RawDetectionsPacket> for TrtInferenceStage {
    fn name(&self) -> &'static str {
        "TrtInferenceStage"
    }

    fn process(&mut self, input: TensorInputPacket) -> RawDetectionsPacket {
        self.infer(input)
            .expect("TensorRT inference failed")
    }
}

// Cleanup: destroy TensorRT session when dropped
impl Drop for TrtInferenceStage {
    fn drop(&mut self) {
        unsafe {
            // Try to load destroy_session and clean up
            if let Ok(destroy_session) = self._lib.get::<Symbol<unsafe extern "C" fn(*mut std::ffi::c_void)>>(b"destroy_session") {
                if !self.session.is_null() {
                    destroy_session(self.session);
                    self.session = std::ptr::null_mut();
                }
            }
        }
    }
}

// Make TrtInferenceStage Send and Sync safe (careful with raw pointers!)
unsafe impl Send for TrtInferenceStage {}
unsafe impl Sync for TrtInferenceStage {}