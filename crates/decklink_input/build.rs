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
        shim_dir
            .join("include")
            .join("DeckLinkAPIDispatch.cpp"),
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

    // ให้ build.rs รันใหม่เมื่อไฟล์ shim หรือค่า env เปลี่ยน
    println!("cargo:rerun-if-changed={}", shim_cpp.display());
    println!(
        "cargo:rerun-if-changed={}",
        shim_dir.join("include").display()
    );
    println!("cargo:rerun-if-env-changed=DECKLINK_SDK_DIR");

    build.compile("shim");
}
