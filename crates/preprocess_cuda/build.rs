// build.rs - Compile CUDA kernels for preprocessing
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=preprocess.cu");
    
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cu_file = manifest_dir.join("preprocess.cu");
    
    // Try to find nvcc in multiple locations
    let nvcc_path = if PathBuf::from("/usr/bin/nvcc").exists() {
        PathBuf::from("/usr/bin/nvcc")
    } else if let Ok(path) = env::var("CUDA_PATH") {
        let candidate = PathBuf::from(path).join("bin/nvcc");
        if candidate.exists() {
            candidate
        } else {
            PathBuf::from("nvcc")
        }
    } else if let Ok(path) = env::var("CUDA_HOME") {
        let candidate = PathBuf::from(path).join("bin/nvcc");
        if candidate.exists() {
            candidate
        } else {
            PathBuf::from("nvcc")
        }
    } else if PathBuf::from("/usr/local/cuda/bin/nvcc").exists() {
        PathBuf::from("/usr/local/cuda/bin/nvcc")
    } else {
        println!("cargo:warning=nvcc not found. Trying 'nvcc' from PATH");
        PathBuf::from("nvcc")
    };
    
    println!("cargo:warning=Using nvcc at: {:?}", nvcc_path);
    println!("cargo:warning=CUDA source file: {:?}", cu_file);
    
    // Determine CUDA include directory
    let cuda_include = if PathBuf::from("/usr/include/cuda").exists() {
        PathBuf::from("/usr/include")
    } else if let Ok(path) = env::var("CUDA_PATH") {
        PathBuf::from(path).join("include")
    } else if let Ok(path) = env::var("CUDA_HOME") {
        PathBuf::from(path).join("include")
    } else if PathBuf::from("/usr/local/cuda/include").exists() {
        PathBuf::from("/usr/local/cuda/include")
    } else {
        PathBuf::from("/usr/include")
    };
    
    println!("cargo:warning=Using CUDA include directory: {:?}", cuda_include);
    
    // Compile CUDA kernel to PTX
    // Use sm_61 for Pascal (Quadro P4000), sm_75 for Turing, sm_86 for Ampere
    // Can also use lower version for wider compatibility
    let status = std::process::Command::new(&nvcc_path)
        .args(&[
            cu_file.to_str().unwrap(),
            "--ptx",
            "-o", out_dir.join("preprocess.ptx").to_str().unwrap(),
            "--gpu-architecture=sm_61",  // Pascal architecture (Quadro P4000)
            "-I", cuda_include.to_str().unwrap(),
            "--use_fast_math",
            "-O3",
            "-diag-suppress=177",  // Suppress unused variable warnings
        ])
        .status();
    
    match status {
        Ok(s) if s.success() => {
            println!("cargo:warning=Successfully compiled CUDA kernel to PTX");
        }
        Ok(s) => {
            println!("cargo:warning=nvcc failed with status: {}", s);
            println!("cargo:warning=Attempting fallback without fast_math...");
            
            // Fallback without fast_math
            let fallback_status = std::process::Command::new(&nvcc_path)
                .args(&[
                    cu_file.to_str().unwrap(),
                    "--ptx",
                    "-o", out_dir.join("preprocess.ptx").to_str().unwrap(),
                    "--gpu-architecture=sm_61",  // Pascal architecture
                    "-I", cuda_include.to_str().unwrap(),
                    "-O3",
                    "-diag-suppress=177",
                ])
                .status();
            
            if let Ok(s) = fallback_status {
                if !s.success() {
                    panic!("CUDA kernel compilation failed even with fallback");
                }
            }
        }
        Err(e) => {
            println!("cargo:warning=Failed to run nvcc: {}", e);
            println!("cargo:warning=Make sure CUDA toolkit is installed and nvcc is in PATH");
            println!("cargo:warning=You can set CUDA_PATH or CUDA_HOME environment variable");
        }
    }
}
