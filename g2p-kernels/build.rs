// AVX-512 and GFNI - their target features and intrinsics - are stable from rustc 1.89: the kernels
// using them are built with cfg g2p_avx512 where rustc has them.

fn main() {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let version = std::process::Command::new(rustc).arg("--version").output();
    let version = version.ok().and_then(|v| String::from_utf8(v.stdout).ok());
    // "rustc 1.MINOR.PATCH ..."
    let minor = version.and_then(|v| v.split(|c| c == ' ' || c == '.').nth(2)?.parse().ok());

    println!("cargo:rustc-check-cfg=cfg(g2p_avx512)");
    if minor.map_or(false, |m: u32| m >= 89) {
        println!("cargo:rustc-cfg=g2p_avx512");
    }
}
