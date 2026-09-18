//! Scalar types KLU operates on.

use core::ffi::c_int;
use klu_sys::{klu_common, klu_numeric, klu_symbolic};

/// Interleaved `complex<double>`, ABI-compatible with `double[2]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C64 {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

/// Scalar element types KLU factorizes (`f64` and [`C64`]).
///
/// # Safety
/// Implementations must use the KLU routines matching [`Scalar::IS_COMPLEX`],
/// honor their documented buffer lengths, reinterpret `Self` as `f64` only when
/// `Self` has a compatible `#[repr(C)]` layout, and return only KLU-allocated
/// numeric objects for the supplied symbolic handle.
pub unsafe trait Scalar: Copy + 'static {
    /// Whether this is a complex scalar type.
    const IS_COMPLEX: bool;
    /// Additive identity.
    fn zero() -> Self;
    /// `a * b`.
    fn mul(a: Self, b: Self) -> Self;
    /// `a + b`.
    fn add(a: Self, b: Self) -> Self;
    /// `-a`.
    fn neg(a: Self) -> Self;
    /// `|a|` (for residuals).
    fn abs(a: Self) -> f64;
    /// Raw pointer to the slice (reinterpreted as `f64`).
    fn f64_ptr(s: &mut [Self]) -> *mut f64;
    /// Const raw pointer to the slice (reinterpreted as `f64`).
    fn f64_ptr_const(s: &[Self]) -> *const f64;

    /// `klu_factor` / `klu_z_factor`.
    ///
    /// # Safety
    /// CSC arrays and handles must be valid (see the callers in `raw`).
    unsafe fn lu_factor(
        bp: &mut [i32],
        bi: &mut [i32],
        bx: &mut [Self],
        sym: *mut klu_symbolic,
        common: *mut klu_common,
    ) -> *mut klu_numeric;

    /// `klu_refactor` / `klu_z_refactor`.
    ///
    /// # Safety
    /// As [`Scalar::lu_factor`], with a valid `num`.
    unsafe fn lu_refactor(
        bp: &mut [i32],
        bi: &mut [i32],
        bx: &mut [Self],
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        common: *mut klu_common,
    ) -> c_int;

    /// `klu_solve` / `klu_z_solve`.
    ///
    /// # Safety
    /// Matching valid handles and a valid RHS buffer.
    unsafe fn lu_solve(
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        n_col: usize,
        n_rhs: usize,
        b: &mut [Self],
        common: *mut klu_common,
    ) -> c_int;

    /// `klu_tsolve` / `klu_z_tsolve` (plain transpose).
    ///
    /// # Safety
    /// As [`Scalar::lu_solve`].
    unsafe fn lu_tsolve(
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        n_col: usize,
        n_rhs: usize,
        b: &mut [Self],
        common: *mut klu_common,
    ) -> c_int;
}

// SAFETY: f64 has the real KLU scalar layout and dispatches to real routines.
unsafe impl Scalar for f64 {
    const IS_COMPLEX: bool = false;
    fn zero() -> Self {
        0.0
    }
    fn mul(a: Self, b: Self) -> Self {
        a * b
    }
    fn add(a: Self, b: Self) -> Self {
        a + b
    }
    fn neg(a: Self) -> Self {
        -a
    }
    fn abs(a: Self) -> f64 {
        a.abs()
    }
    fn f64_ptr(s: &mut [Self]) -> *mut f64 {
        s.as_mut_ptr()
    }
    fn f64_ptr_const(s: &[Self]) -> *const f64 {
        s.as_ptr()
    }
    unsafe fn lu_factor(
        bp: &mut [i32],
        bi: &mut [i32],
        bx: &mut [Self],
        sym: *mut klu_symbolic,
        common: *mut klu_common,
    ) -> *mut klu_numeric {
        // SAFETY: the caller supplies matching live handles and valid CSC/RHS slices.
        unsafe {
            klu_sys::klu_factor(
                bp.as_mut_ptr(),
                bi.as_mut_ptr(),
                Self::f64_ptr(bx),
                sym,
                common,
            )
        }
    }
    unsafe fn lu_refactor(
        bp: &mut [i32],
        bi: &mut [i32],
        bx: &mut [Self],
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        common: *mut klu_common,
    ) -> c_int {
        // SAFETY: the caller supplies matching live handles and valid CSC/RHS slices.
        unsafe {
            klu_sys::klu_refactor(
                bp.as_mut_ptr(),
                bi.as_mut_ptr(),
                Self::f64_ptr(bx),
                sym,
                num,
                common,
            )
        }
    }
    unsafe fn lu_solve(
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        n_col: usize,
        n_rhs: usize,
        b: &mut [Self],
        common: *mut klu_common,
    ) -> c_int {
        // SAFETY: the caller supplies matching live handles and valid CSC/RHS slices.
        unsafe {
            klu_sys::klu_solve(
                sym,
                num,
                n_col as c_int,
                n_rhs as c_int,
                Self::f64_ptr(b),
                common,
            )
        }
    }
    unsafe fn lu_tsolve(
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        n_col: usize,
        n_rhs: usize,
        b: &mut [Self],
        common: *mut klu_common,
    ) -> c_int {
        // SAFETY: the caller supplies matching live handles and valid CSC/RHS slices.
        unsafe {
            klu_sys::klu_tsolve(
                sym,
                num,
                n_col as c_int,
                n_rhs as c_int,
                Self::f64_ptr(b),
                common,
            )
        }
    }
}

// SAFETY: C64 is two repr(C) f64 fields and dispatches to complex routines.
unsafe impl Scalar for C64 {
    const IS_COMPLEX: bool = true;
    fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }
    fn mul(a: Self, b: Self) -> Self {
        Self {
            re: a.re * b.re - a.im * b.im,
            im: a.re * b.im + a.im * b.re,
        }
    }
    fn add(a: Self, b: Self) -> Self {
        Self {
            re: a.re + b.re,
            im: a.im + b.im,
        }
    }
    fn neg(a: Self) -> Self {
        Self {
            re: -a.re,
            im: -a.im,
        }
    }
    fn abs(a: Self) -> f64 {
        a.re.hypot(a.im)
    }
    fn f64_ptr(s: &mut [Self]) -> *mut f64 {
        s.as_mut_ptr() as *mut f64
    }
    fn f64_ptr_const(s: &[Self]) -> *const f64 {
        s.as_ptr() as *const f64
    }
    unsafe fn lu_factor(
        bp: &mut [i32],
        bi: &mut [i32],
        bx: &mut [Self],
        sym: *mut klu_symbolic,
        common: *mut klu_common,
    ) -> *mut klu_numeric {
        // SAFETY: the caller supplies matching live handles and valid CSC/RHS slices.
        unsafe {
            klu_sys::klu_z_factor(
                bp.as_mut_ptr(),
                bi.as_mut_ptr(),
                Self::f64_ptr(bx),
                sym,
                common,
            )
        }
    }
    unsafe fn lu_refactor(
        bp: &mut [i32],
        bi: &mut [i32],
        bx: &mut [Self],
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        common: *mut klu_common,
    ) -> c_int {
        // SAFETY: the caller supplies matching live handles and valid CSC/RHS slices.
        unsafe {
            klu_sys::klu_z_refactor(
                bp.as_mut_ptr(),
                bi.as_mut_ptr(),
                Self::f64_ptr(bx),
                sym,
                num,
                common,
            )
        }
    }
    unsafe fn lu_solve(
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        n_col: usize,
        n_rhs: usize,
        b: &mut [Self],
        common: *mut klu_common,
    ) -> c_int {
        // SAFETY: the caller supplies matching live handles and valid CSC/RHS slices.
        unsafe {
            klu_sys::klu_z_solve(
                sym,
                num,
                n_col as c_int,
                n_rhs as c_int,
                Self::f64_ptr(b),
                common,
            )
        }
    }
    unsafe fn lu_tsolve(
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        n_col: usize,
        n_rhs: usize,
        b: &mut [Self],
        common: *mut klu_common,
    ) -> c_int {
        // conj_solve = 0 -> plain transpose A^T (matches the JAX wrapper).
        // SAFETY: the caller supplies matching live handles and valid CSC/RHS slices.
        unsafe {
            klu_sys::klu_z_tsolve(
                sym,
                num,
                n_col as c_int,
                n_rhs as c_int,
                Self::f64_ptr(b),
                0,
                common,
            )
        }
    }
}
