// Build script for GPU-accelerated overlay planning

use std::env;
use std::path::PathBuf;

fn main() {
    // Only build CUDA if "gpu" feature is enabled
    if !cfg!(feature = "gpu") {
        return;
    }
    
    println!("cargo:rerun-if-changed=overlay_plan_gpu.cu");
    println!("cargo:rerun-if-changed=build.rs");
    
    // Check for CUDA toolkit
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    
    println!("cargo:warning=Using CUDA at: {}", cuda_path);
    
    // Compile CUDA kernel
    let mut build = cc::Build::new();
    
    build
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode=arch=compute_86,code=sm_86")  // RTX 30x0 series (Ampere)
        .flag("-gencode=arch=compute_75,code=sm_75")  // RTX 20x0 series (Turing)
        .flag("-gencode=arch=compute_70,code=sm_70")  // V100 (Volta)
        .flag("-gencode=arch=compute_61,code=sm_61")  // Quadro P4000, GTX 10x0 (Pascal)
        .flag("--std=c++14")
        .flag("-O3")
        .file("overlay_plan_gpu.cu");
    
    // Add CUDA include path
    let cuda_include = PathBuf::from(&cuda_path).join("include");
    build.include(cuda_include);
    
    build.compile("overlay_plan_gpu");
    
    // Link CUDA runtime
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    
    println!("cargo:warning=GPU overlay planning CUDA kernel compiled successfully!");
}
