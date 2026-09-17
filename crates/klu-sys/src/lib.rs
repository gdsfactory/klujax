//! Raw FFI bindings to the SuiteSparse C KLU library.
//!
//! **Transitional (M1).** This crate exists only so that the `klu` crate can be
//! backed by the proven C implementation while the XLA FFI seam is validated.
//! It is deleted in Stage 6 once the pure-Rust port (Stage 5) lands.
//!
//! The C sources are compiled from `../../suitesparse` (override with
//! `KLUJAX_SUITESPARSE_DIR`), which is fetched by `just deps`.

#![allow(non_camel_case_types, non_snake_case)]

use core::ffi::{c_int, c_void};

/// `KLU_OK`.
pub const KLU_OK: c_int = 0;

/// Opaque `klu_symbolic`.
#[repr(C)]
pub struct klu_symbolic {
    _private: [u8; 0],
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
        unsafe {
            let mut common = MaybeUninit::<klu_common>::uninit();
            klu_defaults(common.as_mut_ptr());
            common.assume_init()
        }
    }

    #[test]
    fn solve_2x2_diagonal_f64() {
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
