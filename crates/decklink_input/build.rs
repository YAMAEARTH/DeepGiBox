// /Users/yamaearth/Documents/3_1/Capstone/Blackmagic DeckLink SDK 14.4/rust/build.rs
use std::{env, path::PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let crate_root = manifest_dir
        .parent()
        .expect("crate directory must have a parent (workspace crates folder)");
    let workspace_root = crate_root
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| crate_root.to_path_buf());
    let sdk_root = env::var("DECKLINK_SDK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace_root.clone());
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| String::from(""));

    let shim_dir = workspace_root.join("shim");
    let shim_cpp = shim_dir.join("shim.cpp");
    if !shim_cpp.exists() {
        panic!(
            "DeckLink shim source not found at {}. Move shim.cpp into the crate or update build.rs",
            shim_cpp.display()
        );
    }

    let mut include_candidates = vec![
        manifest_dir.join("include"),
        manifest_dir.join("shim").join("include"),
        workspace_root.join("include"),
        shim_dir.join("include"),
    ];
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .flag_if_supported("-std=c++17")
        .file(shim_cpp.as_path())
        .warnings(true);

    let mut extra_link_search: Vec<PathBuf> = Vec::new();
    let mut extra_link_libs: Vec<&'static str> = Vec::new();

    match target_os.as_str() {
        "linux" => {
            println!("cargo:rustc-link-lib=dylib=DeckLinkAPI");

            let lib_candidates = [
                sdk_root.join("Linux").join("x86_64").join("lib"),
                sdk_root.join("Linux").join("bin"),
                sdk_root.join("Linux").join("bin").join("x86_64"),
                sdk_root.join("Linux").join("x86_64"),
                sdk_root.join("Linux").join("lib"),
                sdk_root.join("Linux").join("bin").join("intel"),
                sdk_root.join("Linux"),
                sdk_root.join("lib"),
                PathBuf::from("/usr/local/lib"),
                PathBuf::from("/usr/lib"),
            ];
            for dir in lib_candidates.iter().filter(|p| p.exists()) {
                println!("cargo:rustc-link-search=native={}", dir.display());
            }

            include_candidates.push(sdk_root.join("Linux").join("include"));
            include_candidates.push(sdk_root.join("include"));
            include_candidates.push(PathBuf::from("/usr/include/DeckLink"));

            // NVIDIA GPUDirect for Video paths
            let dvp_include_env = env::var("DVP_INCLUDE_PATH").map(PathBuf::from).ok();
            let dvp_lib_env = env::var("DVP_LIB_PATH").map(PathBuf::from).ok();
            let gpudirect_root = env::var("GPUDIRECT_ROOT")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("/opt/DVP/gpudirect"));
            let gpudirect_include_default =
                gpudirect_root.join("sdk").join("linux").join("include");
            let gpudirect_lib_default = gpudirect_root.join("sdk").join("linux").join("lib64");

            if let Some(path) = dvp_include_env.filter(|p| p.exists()) {
                include_candidates.push(path);
            } else if gpudirect_include_default.exists() {
                include_candidates.push(gpudirect_include_default);
            } else {
                println!(
                    "cargo:warning=GPUDirect include path not found; set GPUDIRECT_ROOT or DVP_INCLUDE_PATH"
                );
            }

            if let Some(path) = dvp_lib_env.filter(|p| p.exists()) {
                extra_link_search.push(path);
            } else if gpudirect_lib_default.exists() {
                extra_link_search.push(gpudirect_lib_default);
            } else {
                println!(
                    "cargo:warning=GPUDirect lib path not found; set GPUDIRECT_ROOT or DVP_LIB_PATH"
                );
            }

            // CUDA toolkit paths
            let cuda_root = env::var("CUDA_HOME")
                .or_else(|_| env::var("CUDA_PATH"))
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("/usr/local/cuda"));
            let cuda_include = cuda_root.join("include");
            let cuda_lib64 = cuda_root.join("lib64");
            if cuda_include.exists() {
                include_candidates.push(cuda_include);
            } else {
                println!(
                    "cargo:warning=CUDA include path not found at {}; set CUDA_HOME or CUDA_PATH",
                    cuda_root.join("include").display()
                );
            }
            if cuda_lib64.exists() {
                extra_link_search.push(cuda_lib64);
            } else {
                println!(
                    "cargo:warning=CUDA lib path not found at {}; set CUDA_HOME or CUDA_PATH",
                    cuda_root.join("lib64").display()
                );
            }

            extra_link_libs.extend_from_slice(&["dvp", "cuda", "cudart"]);
        }
        other => {
            panic!("Unsupported target OS for DeckLink capture shim: {}", other);
        }
    }

    // คอมไพล์ DeckLinkAPIDispatch.cpp เพื่อเรียกใช้สัญลักษณ์ที่มี version suffix ในไลบรารี
    let dispatch_candidates = [
        manifest_dir.join("include").join("DeckLinkAPIDispatch.cpp"),
        manifest_dir
            .join("shim")
            .join("include")
            .join("DeckLinkAPIDispatch.cpp"),
        shim_dir.join("include").join("DeckLinkAPIDispatch.cpp"),
        sdk_root
            .join("rust")
            .join("include")
            .join("DeckLinkAPIDispatch.cpp"),
        sdk_root
            .join("Linux")
            .join("include")
            .join("DeckLinkAPIDispatch.cpp"),
        sdk_root.join("include").join("DeckLinkAPIDispatch.cpp"),
    ];
    if let Some(dispatch) = dispatch_candidates.iter().find(|p| p.exists()) {
        println!("cargo:warning=cc adding: {}", dispatch.display());
        build.file(dispatch);
    } else {
        println!("cargo:warning=DeckLinkAPIDispatch.cpp not found in typical locations; relying on library symbols only");
    }

    for dir in include_candidates.iter().filter(|p| p.exists()) {
        println!("cargo:warning=cc include dir: {}", dir.display());
        build.include(dir);
    }

    for dir in extra_link_search.iter().filter(|p| p.exists()) {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    for lib in extra_link_libs {
        println!("cargo:rustc-link-lib=dylib={}", lib);
    }

    // ให้ build.rs รันใหม่เมื่อไฟล์ shim หรือค่า env เปลี่ยน
    println!("cargo:rerun-if-changed={}", shim_cpp.display());
    println!(
        "cargo:rerun-if-changed={}",
        shim_dir.join("include").display()
    );
    println!("cargo:rerun-if-env-changed=DECKLINK_SDK_DIR");
    println!("cargo:rerun-if-env-changed=GPUDIRECT_ROOT");
    println!("cargo:rerun-if-env-changed=DVP_INCLUDE_PATH");
    println!("cargo:rerun-if-env-changed=DVP_LIB_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    build.compile("shim");
}
