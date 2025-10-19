use libloading::Library;
use std::path::Path;

fn main() {
    unsafe {
        // Load the TRT shim library
        let lib = Library::new("../build/libtrt_shim.so").expect("Failed to load library");
        
        let build_engine: libloading::Symbol<unsafe extern "C" fn(*const i8, *const i8)> = 
            lib.get(b"build_engine").expect("Failed to get build_engine function");

        // Build v7 engine
        let onnx = std::ffi::CString::new("assets/YOLOv5.onnx").unwrap();
        let engine = std::ffi::CString::new("assets/v7_optimized_YOLOv5.engine").unwrap();

        println!("Building TensorRT engine v7...");
        println!("ONNX:   assets/YOLOv5.onnx");
        println!("Engine: assets/v7_optimized_YOLOv5.engine");
        println!("\nThis may take a few minutes...\n");
        
        build_engine(onnx.as_ptr(), engine.as_ptr());
        
        if Path::new("assets/v7_optimized_YOLOv5.engine").exists() {
            println!("\n✅ Engine file created successfully!");
            let metadata = std::fs::metadata("assets/v7_optimized_YOLOv5.engine").unwrap();
            println!("   Size: {:.2} MB", metadata.len() as f64 / 1024.0 / 1024.0);
        } else {
            println!("\n❌ Engine file was not created!");
        }
        
        std::mem::forget(lib);
    }
}
