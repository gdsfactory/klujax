//! Builds and statically links the SuiteSparse KLU stack.
//!
//! Sources are compiled with the `cc` crate into static archives that are linked
//! into the final `klujax_ffi` cdylib, so there is no runtime dependency on a
//! system `libklu` / `libsuitesparse` (mirrors `eigenlight`'s UMFPACK crate).
//!
//! The SuiteSparse checkout is located in this order:
//!   1. `$KLUJAX_SUITESPARSE_DIR`
//!   2. the `vendor/SuiteSparse` submodule
//!   3. the repo-root `suitesparse/` checkout created by `just deps`

use cc::Build;
use std::env;
use std::path::PathBuf;

/// SuiteSparse components needed by KLU (`KLU/Source` pulls in the rest).
const INCLUDE_DIRS: &[&str] = &[
    "SuiteSparse_config",
    "AMD/Include",
    "COLAMD/Include",
    "BTF/Include",
    "KLU/Include",
];

/// Directories whose `*.c` files are compiled into one static archive.
///
/// The 64-bit-index variants (`amd_l*`, `btf_l_*`, `colamd_l`, `klu_l_*`,
/// `klu_zl_*`) are **skipped**: the FFI layer only calls the `int32` (`klu_*` /
/// `klu_z_*`) entry points, so building them would only slow the build and
/// bloat the archive. See `is_int64_variant`.
const SOURCE_DIRS: &[&str] = &["AMD/Source", "COLAMD/Source", "BTF/Source", "KLU/Source"];

/// SuiteSparse names its 64-bit-index variants with an `_l` / `_zl` infix
/// (e.g. `amd_l1.c`, `btf_l_order.c`, `colamd_l.c`, `klu_l_factor.c`,
/// `klu_zl_factor.c`). The `int32` variants (`amd_2.c`, `klu_z_factor.c`, …)
/// must not match.
fn is_int64_variant(stem: &str) -> bool {
    stem.contains("_l") || stem.contains("_zl")
}

fn locate_suitesparse(manifest: PathBuf) -> PathBuf {
    if let Ok(dir) = env::var("KLUJAX_SUITESPARSE_DIR") {
        return PathBuf::from(dir);
    }
    let submodule = manifest.join("../../vendor/SuiteSparse");
    if submodule.join("KLU/Include/klu.h").exists() {
        return submodule;
    }
    manifest.join("../../suitesparse")
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = locate_suitesparse(manifest);

    if !root.join("KLU/Include/klu.h").exists() {
        panic!(
            "SuiteSparse sources not found at `{}`.\n\
             Initialize the submodule:\n  \
             git submodule update --init vendor/SuiteSparse\n\
             or set KLUJAX_SUITESPARSE_DIR to a SuiteSparse checkout.",
            root.display()
        );
    }

    let mut build = Build::new();
    build
        .warnings(false)
        .flag_if_supported("-static")
        // The vendored C sources emit many benign warnings; silence them the
        // same way `eigenlight` does.
        .flag_if_supported("-Wno-sign-compare")
        .flag_if_supported("-Wno-unknown-pragmas")
        .flag_if_supported("-Wno-unused-variable")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-unused-function")
        .flag_if_supported("-Wno-parentheses")
        .flag_if_supported("-Wno-maybe-uninitialized")
        .flag_if_supported("-Wno-clobbered")
        .flag_if_supported("-Wno-empty-body")
        .flag_if_supported("-Wno-unused-but-set-variable")
        .flag_if_supported("-Wno-array-parameter")
        .flag_if_supported("-Wno-implicit-fallthrough");

    for inc in INCLUDE_DIRS {
        build.include(root.join(inc));
    }

    build.file(root.join("SuiteSparse_config/SuiteSparse_config.c"));

    for dir in SOURCE_DIRS {
        let dir = root.join(dir);
        for entry in std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "c") {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                if is_int64_variant(stem) {
                    continue;
                }
                build.file(path);
            }
        }
    }

    // Compiles to `libsuitesparse_klu.a` and emits the static link directives.
    build.compile("suitesparse_klu");

    println!("cargo:include={}", root.display());
    println!("cargo:rerun-if-changed={}", root.display());
    println!("cargo:rerun-if-env-changed=KLUJAX_SUITESPARSE_DIR");
}
