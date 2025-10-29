use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_root = PathBuf::from(&manifest_dir).parent().unwrap().to_path_buf();
    
    let include_dir = project_root.join("include");
    let shim_dir = project_root.join("shim");
    let dvp_include = "/opt/NVIDIA_GPUDirect/gpudirect/sdk/linux/include";
    let cuda_include = "/usr/lib/cuda/include";
    
    println!("cargo:rerun-if-changed={}/shim.cpp", shim_dir.display());
    
    // Build shim.cpp (with new keyer functions)
    cc::Build::new()
        .cpp(true)
        .file(shim_dir.join("shim.cpp"))
        .file(include_dir.join("DeckLinkAPIDispatch.cpp"))
        .include(&include_dir)
        .include(&shim_dir)
        .include(dvp_include)
        .include(cuda_include)
        .flag("-std=c++11")
        .flag("-Wall")
        .flag("-Wno-unused-parameter")
        .compile("decklink_shim");
    
    // Link libraries
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native=/opt/NVIDIA_GPUDirect/gpudirect/sdk/linux/x86_64");
    println!("cargo:rustc-link-lib=dvp");
}
