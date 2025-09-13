// /Users/yamaearth/Documents/3_1/Capstone/Blackmagic DeckLink SDK 14.4/rust/build.rs
use std::{env, path::PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let sdk_root = env::var("DECKLINK_SDK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // Try common DeckLink SDK installation paths
            let common_paths = [
                PathBuf::from("/opt/Blackmagic DeckLink SDK 15.0"),
                PathBuf::from("/opt/BlackmagicDeckLinkSDK"),
                manifest_dir.parent().unwrap().to_path_buf(),
            ];
            common_paths.into_iter().find(|p| p.exists()).unwrap_or_else(|| manifest_dir.parent().unwrap().to_path_buf())
        });

    // Platform-specific linking
    match target_os.as_str() {
        "macos" => {
            // Standard macOS Frameworks directory
            println!("cargo:rustc-link-search=framework=/Library/Frameworks");
            println!("cargo:rustc-link-lib=framework=DeckLinkAPI");
            println!("cargo:rustc-link-lib=framework=CoreFoundation");
            println!("cargo:rustc-link-lib=framework=CoreVideo");
        }
        "linux" => {
            // On Linux, link against the shared library provided by Desktop Video drivers
            // Typically installed as /usr/lib/libDeckLinkAPI.so
            println!("cargo:rustc-link-lib=DeckLinkAPI");

            // If user points DECKLINK_SDK_DIR to a custom unpacked SDK with libs, add plausible lib dirs
            for lib_dir in [
                sdk_root.join("Linux").join("Libraries"),
                sdk_root.join("lib"),
                sdk_root.join("Linux"),
                PathBuf::from("/usr/lib"),  // System installation
                PathBuf::from("/usr/lib/x86_64-linux-gnu"),  // Ubuntu multiarch
            ] {
                if lib_dir.exists() {
                    println!("cargo:rustc-link-search=native={}", lib_dir.display());
                }
            }
        }
        _ => {
            // Default: try linking the common name
            println!("cargo:rustc-link-lib=DeckLinkAPI");
        }
    }

let include_candidates = [
    manifest_dir.join("shim").join("include"),
    manifest_dir.join("include"),
    sdk_root.join("Linux").join("include"),  // Linux SDK path
    sdk_root.join("Mac").join("include"),
    sdk_root.join("include"),
    PathBuf::from("/opt/Blackmagic DeckLink SDK 15.0/Linux/include"),  // Common Linux installation
    PathBuf::from("/usr/include/decklink"),  // System installation
    PathBuf::from("/Library/Frameworks/DeckLinkAPI.framework/Headers"),
];    let mut build = cc::Build::new();
    build.cpp(true)
        .flag_if_supported("-std=c++17")
        .file("shim/shim.cpp")
        .warnings(true);

    // On macOS, link-time symbols like CreateDeckLinkIteratorInstance are
    // versioned in the framework. We must compile DeckLinkAPIDispatch.cpp
    // from the SDK to resolve them via CFBundle at runtime.
    if target_os == "macos" {
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
    }
    
    // On Linux, also include DeckLinkAPIDispatch.cpp if available
    if target_os == "linux" {
        let dispatch_candidates = [
            manifest_dir.join("include").join("DeckLinkAPIDispatch.cpp"),
            sdk_root.join("Linux").join("include").join("DeckLinkAPIDispatch.cpp"),
            sdk_root.join("include").join("DeckLinkAPIDispatch.cpp"),
            PathBuf::from("/opt/Blackmagic DeckLink SDK 15.0/Linux/include/DeckLinkAPIDispatch.cpp"),
        ];
        if let Some(dispatch) = dispatch_candidates.iter().find(|p| p.exists()) {
            println!("cargo:warning=cc adding Linux dispatch: {}", dispatch.display());
            build.file(dispatch);
        } else {
            println!("cargo:warning=Linux DeckLinkAPIDispatch.cpp not found; using direct library linking");
        }
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
