use std::env;
use std::path::PathBuf;

fn main() {
    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let out_dir = env::var("OUT_DIR").unwrap();
    
    // Compile CUDA kernel
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_61,code=sm_61") // Quadro P4000, GTX 1080
        .flag("-gencode")
        .flag("arch=compute_75,code=sm_75") // RTX 2060+
        .flag("-gencode")
        .flag("arch=compute_86,code=sm_86") // RTX 3060+
        .include(format!("{}/include", cuda_path))
        .file("overlay_render.cu")
        .compile("overlay_render_cuda");

    // Link CUDA runtime
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    
    // Make sure the static library is linked
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=overlay_render_cuda");
    
    // Re-run if CUDA code changes
    println!("cargo:rerun-if-changed=overlay_render.cu");
}
