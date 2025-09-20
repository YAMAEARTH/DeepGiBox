fn main() {
    println!("cargo::rustc-check-cfg=cfg(ocvrs_has_cuda)");
    println!("cargo::rustc-check-cfg=cfg(ocvrs_has_module_cudawarping)");
    println!("cargo::rustc-check-cfg=cfg(ocvrs_has_module_cudaimgproc)");
}
