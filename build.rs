//! Build script — stamps the Rust target triple into the binary so the
//! self-updater (`src/node/self_update.rs`, DESIGN-remote §9 P3) can compare a
//! signed update manifest's `target` against the running binary's own triple.
//!
//! `TARGET` is set by Cargo for every build script; re-exporting it as
//! `HIGGS_BUILD_TARGET` makes it reachable at compile time via
//! `env!("HIGGS_BUILD_TARGET")` in the crate (Cargo does NOT expose `TARGET` to
//! the crate itself, only to build scripts). Nothing else here — the heavy
//! llama.cpp FFI is built by the `llama-cpp-2`/`-sys` crates' own build scripts.

fn main() {
    // Emit as a rustc env so `env!("HIGGS_BUILD_TARGET")` resolves in-crate.
    // `TARGET` is always present in a build script's environment.
    let target = std::env::var("TARGET").expect("cargo sets TARGET for build scripts");
    println!("cargo:rustc-env=HIGGS_BUILD_TARGET={target}");
    // This script depends on nothing on disk; without an explicit rerun-if it
    // would re-run whenever any file changes. Pin it to itself only.
    println!("cargo:rerun-if-changed=build.rs");
}
