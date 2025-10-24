use std::{env, path::PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("crate directory must have workspace root as grandparent");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| String::from(""));

    let keying_cu = workspace_root.join("keying").join("keying.cu");
    let keying_dir = workspace_root.join("keying");
    let cuda_compat_h = keying_dir.join("cuda_compat.h");
    
    let mut build = cc::Build::new();
    build
        .cuda(true)
        .flag("-std=c++14")
        .flag("--allow-unsupported-compiler")
        .flag("--expt-relaxed-constexpr")
        .flag("-D_GLIBCXX_USE_CXX11_ABI=1")
        .warnings(false); // Disable warnings for CUDA compatibility
    
    // Add keying directory to include path
    build.include(&keying_dir);
    
    // Force include cuda_compat.h BEFORE any other headers (including system headers)
    // Use -include flag which works with both nvcc and gcc
    if cuda_compat_h.exists() {
        build.flag("-include");
        build.flag(cuda_compat_h.to_str().unwrap());
    }

    let mut extra_link_search: Vec<PathBuf> = Vec::new();
    let mut extra_link_libs: Vec<&'static str> = Vec::new();

    match target_os.as_str() {
        "linux" => {
            // CUDA toolkit paths
            let cuda_root = env::var("CUDA_HOME")
                .or_else(|_| env::var("CUDA_PATH"))
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("/usr/local/cuda"));
            let cuda_include = cuda_root.join("include");
            let cuda_lib64 = cuda_root.join("lib64");
            
            if cuda_include.exists() {
                build.include(&cuda_include);
                println!("cargo:warning=CUDA include: {}", cuda_include.display());
            }
            if cuda_lib64.exists() {
                extra_link_search.push(cuda_lib64);
            }

            extra_link_libs.extend_from_slice(&["cuda", "cudart"]);
        }
        other => {
            panic!("Unsupported target OS for keying module: {}", other);
        }
    }

    let build_kernel = keying_cu.exists() && env::var("SKIP_CUDA_BUILD").is_err();
    
    if !build_kernel {
        println!("cargo:warning=Skipping CUDA kernel build (SKIP_CUDA_BUILD is set or keying.cu not found)");
        println!("cargo:warning=Note: decklink_output will not have GPU compositor functionality");
    }
    
    if build_kernel {
        println!("cargo:warning=Building CUDA kernel: {}", keying_cu.display());
        build.file(&keying_cu);
    }

    for dir in extra_link_search.iter().filter(|p| p.exists()) {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    for lib in extra_link_libs {
        println!("cargo:rustc-link-lib=dylib={}", lib);
    }

    println!("cargo:rerun-if-changed={}", keying_cu.display());
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if build_kernel {
        build.compile("keying");
    }
}
