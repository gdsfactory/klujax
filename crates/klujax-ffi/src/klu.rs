//! Safe wrapper around the raw `klu-sys` FFI.
//!
//! This is the single unsafe boundary for KLU calls in `klujax-ffi`: every
//! `unsafe` operation against `klu-sys` lives here, and callers get ordinary
//! `Result`-returning functions. `engine.rs` is therefore `forbid(unsafe_code)`.

#![allow(
    unsafe_code,
    clippy::too_many_arguments,
    clippy::undocumented_unsafe_blocks,
    clippy::missing_safety_doc
)]
// This module *is* the KLU unsafe boundary: every `unsafe` block below is
// inside a safe `fn` whose documented preconditions are met by the callers in
// `engine.rs`. `engine.rs` is `forbid(unsafe_code)`.

use crate::error::ErrorInfo;
use core::ffi::c_int;
use core::mem::MaybeUninit;
use klu_sys::{
    klu_analyze, klu_common, klu_defaults, klu_free_numeric, klu_free_symbolic, klu_numeric,
    klu_symbolic, KLU_OK,
};

/// Interleaved `complex<double>`, ABI-compatible with `double[2]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

/// Scalar types KLU operates on.
///
/// # Safety
/// `f64_ptr`/`f64_ptr_const` must reinterpret the slice as `f64`; sound for
/// `f64` and `C64` (`#[repr(C)]` of two `f64`).
pub unsafe trait Scalar: Copy + 'static {
    /// Whether this is a complex scalar type.
    const IS_COMPLEX: bool;
    /// The matching `XLA_FFI_DataType`.
    const DTYPE: c_int;
    /// Additive identity.
    fn zero() -> Self;
    /// `a * b`.
    fn mul(a: Self, b: Self) -> Self;
    /// `a + b`.
    fn add(a: Self, b: Self) -> Self;
    /// Raw pointer to the slice (reinterpreted as `f64`).
    fn f64_ptr(s: &mut [Self]) -> *mut f64;
    /// Const raw pointer to the slice (reinterpreted as `f64`).
    fn f64_ptr_const(s: &[Self]) -> *const f64;

    /// `klu_factor` / `klu_z_factor`.
    ///
    /// # Safety
    /// CSC arrays and handles must be valid (see the callers here).
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

unsafe impl Scalar for f64 {
    const IS_COMPLEX: bool = false;
    const DTYPE: c_int = crate::xla_ffi::dtype::F64;
    fn zero() -> Self {
        0.0
    }
    fn mul(a: Self, b: Self) -> Self {
        a * b
    }
    fn add(a: Self, b: Self) -> Self {
        a + b
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

unsafe impl Scalar for C64 {
    const IS_COMPLEX: bool = true;
    const DTYPE: c_int = crate::xla_ffi::dtype::C128;
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
        // conj_solve = 0 -> plain transpose A^T (matches the C++ wrapper).
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

/// Safe owner of a `klu_common` control/status block.
pub struct Common(klu_common);

impl Default for Common {
    fn default() -> Self {
        Self::new()
    }
}

impl Common {
    /// A `klu_common` initialised by `klu_defaults`.
    pub fn new() -> Self {
        // SAFETY: `klu_defaults` initialises the whole struct in place.
        let mut common = MaybeUninit::<klu_common>::uninit();
        unsafe { klu_defaults(common.as_mut_ptr()) };
        Self(unsafe { common.assume_init() })
    }

    /// Whether the last operation succeeded (`common.status >= KLU_OK`).
    pub fn ok(&self) -> bool {
        self.0.status >= KLU_OK
    }

    /// The raw KLU status code from the last operation.
    pub fn status(&self) -> c_int {
        self.0.status
    }

    fn ptr(&mut self) -> *mut klu_common {
        &mut self.0
    }
}

/// Validate and wrap a raw `klu_symbolic*` address.
fn sym_ptr(sym: u64) -> Result<*mut klu_symbolic, ErrorInfo> {
    let ptr = sym as *mut klu_symbolic;
    if ptr.is_null() {
        Err(ErrorInfo::invalid("symbolic pointer is null"))
    } else {
        Ok(ptr)
    }
}

fn num_ptr(num: u64) -> Result<*mut klu_numeric, ErrorInfo> {
    let ptr = num as *mut klu_numeric;
    if ptr.is_null() {
        Err(ErrorInfo::invalid("numeric pointer is null"))
    } else {
        Ok(ptr)
    }
}

/// Number of columns of the analyzed matrix.
pub fn symbolic_n(sym: u64) -> Result<usize, ErrorInfo> {
    let ptr = sym_ptr(sym)?;
    // SAFETY: `ptr` is a valid `klu_symbolic` (checked non-null; valid per the
    // handle contract), and `n` is an `i32` field of the mirrored struct.
    Ok(unsafe { (*ptr).n } as usize)
}

/// `klu_analyze` on a CSC pattern; returns the raw handle as `u64`.
pub fn analyze(n_col: usize, bp: &mut [i32], bi: &mut [i32]) -> Result<u64, ErrorInfo> {
    let mut common = Common::new();
    // SAFETY: `bp`/`bi` are valid CSC arrays; `common` is a valid owner.
    let sym = unsafe {
        klu_analyze(
            n_col as c_int,
            bp.as_mut_ptr(),
            bi.as_mut_ptr(),
            common.ptr(),
        )
    };
    if sym.is_null() || !common.ok() {
        return Err(ErrorInfo::internal("klu_analyze failed."));
    }
    Ok(sym as u64)
}

/// Numeric factorization of one matrix; returns the raw handle as `u64`.
pub fn factor<T: Scalar>(
    common: &mut Common,
    bp: &mut [i32],
    bi: &mut [i32],
    bx: &mut [T],
    sym: u64,
) -> Result<u64, ErrorInfo> {
    let ptr = sym_ptr(sym)?;
    // SAFETY: CSC arrays and handles are valid per the caller's contract.
    let num = unsafe { T::lu_factor(bp, bi, bx, ptr, common.ptr()) };
    if num.is_null() || !common.ok() {
        return Err(ErrorInfo::invalid(
            "klu_factor/z_factor failed (singular matrix?)",
        ));
    }
    Ok(num as u64)
}

/// Recompute a factorization in place.
pub fn refactor<T: Scalar>(
    common: &mut Common,
    bp: &mut [i32],
    bi: &mut [i32],
    bx: &mut [T],
    sym: u64,
    num: u64,
) -> Result<(), ErrorInfo> {
    let sym = sym_ptr(sym)?;
    let num = num_ptr(num)?;
    // SAFETY: handles/CSC arrays are valid per the caller's contract.
    let status = unsafe { T::lu_refactor(bp, bi, bx, sym, num, common.ptr()) };
    if status == 0 || !common.ok() {
        return Err(ErrorInfo::invalid(
            "klu_refactor/z_refactor failed (singular matrix?)",
        ));
    }
    Ok(())
}

/// Solve `A x = b` (or `A^T x = b` when `transpose`) in place.
pub fn solve<T: Scalar>(
    common: &mut Common,
    sym: u64,
    num: u64,
    n_col: usize,
    n_rhs: usize,
    b: &mut [T],
    transpose: bool,
) -> Result<(), ErrorInfo> {
    let sym = sym_ptr(sym)?;
    let num = num_ptr(num)?;
    // SAFETY: handles and `b` are valid per the caller's contract.
    let status = unsafe {
        if transpose {
            T::lu_tsolve(sym, num, n_col, n_rhs, b, common.ptr())
        } else {
            T::lu_solve(sym, num, n_col, n_rhs, b, common.ptr())
        }
    };
    if status == 0 || !common.ok() {
        return Err(ErrorInfo::invalid(if transpose {
            "klu_tsolve/z_tsolve failed"
        } else {
            "klu_solve/z_solve failed"
        }));
    }
    Ok(())
}

/// Free a numeric handle.
pub fn free_numeric(common: &mut Common, num: u64) {
    if num == 0 {
        return;
    }
    let mut ptr = num as *mut klu_numeric;
    // SAFETY: `ptr` is a valid handle (or zero, handled above).
    unsafe { klu_free_numeric(&mut ptr, common.ptr()) };
}

/// Free a symbolic handle.
///
/// The caller must ensure it is not used afterwards (handles are freed by the
/// XLA `free_*` targets / Python `close()`).
pub fn free_symbolic(common: &mut Common, sym: u64) {
    if sym == 0 {
        return;
    }
    let mut ptr = sym as *mut klu_symbolic;
    // SAFETY: `ptr` is a valid handle (or zero, handled above).
    unsafe { klu_free_symbolic(&mut ptr, common.ptr()) };
}
