use std::{env, path::PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir
        .parent()
        .expect("crate directory must have a parent (workspace root)");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| String::from(""));

    let keying_dir = workspace_root.join("keying");
    let keying_cpp = keying_dir.join("keying.cu");
    
    let mut build = cc::Build::new();
    build
        .cuda(true)
        .flag("-std=c++14")
        .flag("--allow-unsupported-compiler")
        .flag("--expt-relaxed-constexpr")
        .flag("-D_GLIBCXX_USE_CXX11_ABI=1")
        .warnings(false); // Disable warnings for CUDA compatibility

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

    let build_kernel = keying_cpp.exists();
    
    if build_kernel {
        println!("cargo:warning=Building CUDA kernel: {}", keying_cpp.display());
        build.file(&keying_cpp);
    } else {
        println!("cargo:warning=keying.cu not found, skipping CUDA kernel build");
    }

    for dir in extra_link_search.iter().filter(|p| p.exists()) {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    for lib in extra_link_libs {
        println!("cargo:rustc-link-lib=dylib={}", lib);
    }

    println!("cargo:rerun-if-changed={}", keying_cpp.display());
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if build_kernel {
        build.compile("keying");
    }
}
