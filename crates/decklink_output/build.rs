use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let workspace_root = PathBuf::from(&manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    // DeckLink SDK include directory
    let decklink_include = workspace_root.join("include");
    
    println!("cargo:rerun-if-changed=output_shim.cpp");
    println!("cargo:rerun-if-changed=build.rs");

    // Compile the output shim
    cc::Build::new()
        .cpp(true)
        .file("output_shim.cpp")
        .include(&decklink_include)
        .flag("-std=c++14")
        .flag("-fPIC")
        .warnings(false) // Suppress warnings from DeckLink SDK
        .compile("decklink_output_shim");

    // Link with DeckLink dispatch (Linux)
    let decklink_dispatch = decklink_include.join("DeckLinkAPIDispatch.cpp");
    cc::Build::new()
        .cpp(true)
        .file(&decklink_dispatch)
        .include(&decklink_include)
        .flag("-std=c++14")
        .flag("-fPIC")
        .warnings(false)
        .compile("decklink_dispatch");

    // Link with system libraries required by DeckLink
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=stdc++");
}
