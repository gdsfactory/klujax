use std::env;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = env::var("KLUJAX_SUITESPARSE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest.join("../../suitesparse"));

    if !root.join("KLU/Include/klu.h").exists() {
        panic!(
            "SuiteSparse sources not found at `{}`. Run `just deps` or set \
             KLUJAX_SUITESPARSE_DIR.",
            root.display()
        );
    }

    let mut build = cc::Build::new();
    build.warnings(false);
    for inc in [
        "SuiteSparse_config",
        "AMD/Include",
        "COLAMD/Include",
        "BTF/Include",
        "KLU/Include",
    ] {
        build.include(root.join(inc));
    }

    build.file(root.join("SuiteSparse_config/SuiteSparse_config.c"));
    for dir in ["AMD/Source", "COLAMD/Source", "BTF/Source", "KLU/Source"] {
        let dir = root.join(dir);
        for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| {
            panic!("cannot read {}: {e}", dir.display());
        }) {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "c") {
                build.file(path);
            }
        }
    }

    build.compile("suitesparse_klu");
    println!("cargo:rerun-if-changed={}", root.display());
    println!("cargo:rerun-if-env-changed=KLUJAX_SUITESPARSE_DIR");
}
