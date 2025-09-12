// /Users/yamaearth/Documents/3_1/Capstone/Blackmagic DeckLink SDK 14.4/rust/build.rs
use std::{env, path::PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let sdk_root = env::var("DECKLINK_SDK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.parent().unwrap().to_path_buf());

    // Ensure the linker searches the standard macOS Frameworks directory
    println!("cargo:rustc-link-search=framework=/Library/Frameworks");
    println!("cargo:rustc-link-lib=framework=DeckLinkAPI");
    println!("cargo:rustc-link-lib=framework=CoreFoundation");
    println!("cargo:rustc-link-lib=framework=CoreVideo");

    let include_candidates = [
        manifest_dir.join("shim").join("include"),
        sdk_root.join("Mac").join("include"),
        sdk_root.join("include"),
        PathBuf::from("/Library/Frameworks/DeckLinkAPI.framework/Headers"),
    ];

    let mut build = cc::Build::new();
    build.cpp(true)
        .flag_if_supported("-std=c++17")
        .file("shim/shim.cpp")
        .warnings(true);

    // On macOS, link-time symbols like CreateDeckLinkIteratorInstance are
    // versioned in the framework. We must compile DeckLinkAPIDispatch.cpp
    // from the SDK to resolve them via CFBundle at runtime.
    let dispatch_candidates = [
        manifest_dir.join("include").join("DeckLinkAPIDispatch.cpp"),
        manifest_dir.join("shim").join("include").join("DeckLinkAPIDispatch.cpp"),
        sdk_root.join("rust").join("include").join("DeckLinkAPIDispatch.cpp"),
        sdk_root.join("include").join("DeckLinkAPIDispatch.cpp"),
        PathBuf::from("/Library/Frameworks/DeckLinkAPI.framework/Headers/DeckLinkAPIDispatch.cpp"),
    ];
    if let Some(dispatch) = dispatch_candidates.iter().find(|p| p.exists()) {
        println!("cargo:warning=cc adding: {}", dispatch.display());
        build.file(dispatch);
    } else {
        println!("cargo:warning=DeckLinkAPIDispatch.cpp not found in typical locations; relying on framework symbols only");
    }

    for dir in include_candidates.iter().filter(|p| p.exists()) {
        println!("cargo:warning=cc include dir: {}", dir.display());
        build.include(dir);
    }

    // Re-run triggers
    println!("cargo:rerun-if-changed=shim/shim.cpp");
    println!("cargo:rerun-if-changed=shim/include");
    println!("cargo:rerun-if-env-changed=DECKLINK_SDK_DIR");

    build.compile("shim");
}
