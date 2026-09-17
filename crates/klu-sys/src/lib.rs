//! Raw FFI bindings to the SuiteSparse C KLU library.
//!
//! `klujax-ffi` calls the C KLU through this crate; the sources are statically
//! linked from the `vendor/SuiteSparse` submodule (override with
//! `KLUJAX_SUITESPARSE_DIR`).

// TODO(hardening/stage-5): generated/asserted layouts + documented unsafe.
#![allow(non_camel_case_types, non_snake_case)]

use core::ffi::{c_int, c_void};

/// `KLU_OK`.
pub const KLU_OK: c_int = 0;

/// `klu_symbolic` (full layout, mirrored from `klu.h`).
///
/// Layout is asserted against the vendored header by
/// `layouts_match_vendored_header`.
#[repr(C)]
pub struct klu_symbolic {
    pub symmetry: f64,
    pub est_flops: f64,
    pub lnz: f64,
    pub unz: f64,
    pub Lnz: *mut f64,
    pub n: i32,
    pub nz: i32,
    pub P: *mut i32,
    pub Q: *mut i32,
    pub R: *mut i32,
    pub nzoff: i32,
    pub nblocks: i32,
    pub maxblock: i32,
    pub ordering: i32,
    pub do_btf: i32,
    pub structural_rank: i32,
}

/// Opaque `klu_numeric`.
#[repr(C)]
pub struct klu_numeric {
    _private: [u8; 0],
}

/// `klu_common` control/status struct (layout mirrored from `klu.h`).
#[repr(C)]
pub struct klu_common {
    pub tol: f64,
    pub memgrow: f64,
    pub initmem_amd: f64,
    pub initmem: f64,
    pub maxwork: f64,
    pub btf: c_int,
    pub ordering: c_int,
    pub scale: c_int,
    pub user_order: Option<
        unsafe extern "C" fn(c_int, *mut c_int, *mut c_int, *mut c_int, *mut klu_common) -> c_int,
    >,
    pub user_data: *mut c_void,
    pub halt_if_singular: c_int,
    pub status: c_int,
    pub nrealloc: c_int,
    pub structural_rank: i32,
    pub numerical_rank: i32,
    pub singular_col: i32,
    pub noffdiag: i32,
    pub flops: f64,
    pub rcond: f64,
    pub condest: f64,
    pub rgrowth: f64,
    pub work: f64,
    pub memusage: usize,
    pub mempeak: usize,
}

extern "C" {
    pub fn klu_defaults(common: *mut klu_common);

    pub fn klu_analyze(
        n: c_int,
        ap: *mut c_int,
        ai: *mut c_int,
        common: *mut klu_common,
    ) -> *mut klu_symbolic;

    pub fn klu_factor(
        ap: *mut c_int,
        ai: *mut c_int,
        ax: *mut f64,
        symbolic: *mut klu_symbolic,
        common: *mut klu_common,
    ) -> *mut klu_numeric;

    pub fn klu_refactor(
        ap: *mut c_int,
        ai: *mut c_int,
        ax: *mut f64,
        symbolic: *mut klu_symbolic,
        numeric: *mut klu_numeric,
        common: *mut klu_common,
    ) -> c_int;

    pub fn klu_solve(
        symbolic: *mut klu_symbolic,
        numeric: *mut klu_numeric,
        ldim: c_int,
        nrhs: c_int,
        b: *mut f64,
        common: *mut klu_common,
    ) -> c_int;

    pub fn klu_tsolve(
        symbolic: *mut klu_symbolic,
        numeric: *mut klu_numeric,
        ldim: c_int,
        nrhs: c_int,
        b: *mut f64,
        common: *mut klu_common,
    ) -> c_int;

    pub fn klu_free_symbolic(symbolic: *mut *mut klu_symbolic, common: *mut klu_common) -> c_int;

    pub fn klu_free_numeric(numeric: *mut *mut klu_numeric, common: *mut klu_common) -> c_int;

    // Complex variants operate on interleaved `double` arrays.
    pub fn klu_z_factor(
        ap: *mut c_int,
        ai: *mut c_int,
        ax: *mut f64,
        symbolic: *mut klu_symbolic,
        common: *mut klu_common,
    ) -> *mut klu_numeric;

    pub fn klu_z_refactor(
        ap: *mut c_int,
        ai: *mut c_int,
        ax: *mut f64,
        symbolic: *mut klu_symbolic,
        numeric: *mut klu_numeric,
        common: *mut klu_common,
    ) -> c_int;

    pub fn klu_z_solve(
        symbolic: *mut klu_symbolic,
        numeric: *mut klu_numeric,
        ldim: c_int,
        nrhs: c_int,
        b: *mut f64,
        common: *mut klu_common,
    ) -> c_int;

    pub fn klu_z_tsolve(
        symbolic: *mut klu_symbolic,
        numeric: *mut klu_numeric,
        ldim: c_int,
        nrhs: c_int,
        b: *mut f64,
        conj_solve: c_int,
        common: *mut klu_common,
    ) -> c_int;
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::MaybeUninit;

    fn new_common() -> klu_common {
        // SAFETY: `klu_defaults` fully initialises the struct in place.
        unsafe {
            let mut common = MaybeUninit::<klu_common>::uninit();
            klu_defaults(common.as_mut_ptr());
            common.assume_init()
        }
    }

    #[test]
    fn layout_matches_vendored_header() {
        use std::path::PathBuf;
        use std::process::Command;

        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let root = std::env::var("KLUJAX_SUITESPARSE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let sub = manifest.join("../../vendor/SuiteSparse");
                if sub.join("KLU/Include/klu.h").exists() {
                    sub
                } else {
                    manifest.join("../../suitesparse")
                }
            });
        if !root.join("KLU/Include/klu.h").exists() {
            eprintln!("skipping: no SuiteSparse header at {}", root.display());
            return;
        }
        let inc_flags: Vec<String> = [
            "SuiteSparse_config",
            "AMD/Include",
            "COLAMD/Include",
            "BTF/Include",
            "KLU/Include",
        ]
        .iter()
        .map(|d| format!("-I{}", root.join(d).display()))
        .collect();

        let dir = std::env::temp_dir().join(format!("klujax_layout_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let c_path = dir.join("layout.c");
        std::fs::write(
            &c_path,
            r#"
#include <stdio.h>
#include <stddef.h>
#include "klu.h"
int main(void) {
    printf("common_size %zu\n", sizeof(klu_common));
    printf("common_status_off %zu\n", offsetof(klu_common, status));
    printf("symbolic_size %zu\n", sizeof(klu_symbolic));
    printf("symbolic_n_off %zu\n", offsetof(klu_symbolic, n));
    printf("symbolic_do_btf_off %zu\n", offsetof(klu_symbolic, do_btf));
    return 0;
}
"#,
        )
        .unwrap();
        let bin = dir.join("layout");
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_owned());
        let ok = Command::new(&cc)
            .args(&inc_flags)
            .arg(&c_path)
            .arg("-o")
            .arg(&bin)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            eprintln!("skipping: no usable C compiler ({cc})");
            return;
        }
        let text =
            String::from_utf8_lossy(&Command::new(&bin).output().unwrap().stdout).to_string();
        let get = |key: &str| -> usize {
            text.lines()
                .find_map(|l| {
                    let mut it = l.split_whitespace();
                    (it.next() == Some(key)).then(|| it.next().unwrap().parse().unwrap())
                })
                .unwrap_or_else(|| panic!("missing {key} in:\n{text}"))
        };

        assert_eq!(core::mem::size_of::<klu_common>(), get("common_size"));
        assert_eq!(
            core::mem::offset_of!(klu_common, status),
            get("common_status_off")
        );
        assert_eq!(core::mem::size_of::<klu_symbolic>(), get("symbolic_size"));
        assert_eq!(
            core::mem::offset_of!(klu_symbolic, n),
            get("symbolic_n_off")
        );
        assert_eq!(
            core::mem::offset_of!(klu_symbolic, do_btf),
            get("symbolic_do_btf_off")
        );
    }

    #[test]
    fn solve_2x2_diagonal_f64() {
        // SAFETY: all pointers below are valid stack arrays/C KLU handles.
        unsafe {
            let n = 2;
            let mut ap = [0i32, 1, 2];
            let mut ai = [0i32, 1];
            let mut ax = [2.0f64, 4.0];
            let mut common = new_common();

            let sym = klu_analyze(n, ap.as_mut_ptr(), ai.as_mut_ptr(), &mut common);
            assert!(!sym.is_null(), "klu_analyze failed");
            let num = klu_factor(
                ap.as_mut_ptr(),
                ai.as_mut_ptr(),
                ax.as_mut_ptr(),
                sym,
                &mut common,
            );
            assert!(!num.is_null(), "klu_factor failed");

            let mut b = [10.0f64, 20.0];
            let status = klu_solve(sym, num, n, 1, b.as_mut_ptr(), &mut common);
            // KLU returns a boolean (nonzero == success); `common.status` is the
            // actual status code (KLU_OK == 0).
            assert_ne!(status, 0, "klu_solve failed");
            assert!(common.status >= KLU_OK);
            assert!((b[0] - 5.0).abs() < 1e-12);
            assert!((b[1] - 5.0).abs() < 1e-12);

            let mut num = num;
            let mut sym = sym;
            klu_free_numeric(&mut num, &mut common);
            klu_free_symbolic(&mut sym, &mut common);
        }
    }
}
