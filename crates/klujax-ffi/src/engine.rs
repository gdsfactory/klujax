//! C-backed KLU engine.
//!
//! Mirrors the numeric logic of the original `klujax.cpp`, calling the C KLU
//! through `klu-sys`.
//!
//! All functions here take plain Rust slices so they can be unit-tested without
//! constructing XLA call frames.

// TODO(hardening/stage-3, stage-6): dedupe and route KLU calls through a safe
// wrapper so this module can be `#![forbid(unsafe_code)]`; remove this allow.
#![allow(clippy::undocumented_unsafe_blocks)]

use crate::error::ErrorInfo;
use crate::xla_ffi::dtype;
use core::ffi::c_int;
use klu_sys::{
    klu_analyze, klu_common, klu_defaults, klu_factor, klu_free_numeric, klu_free_symbolic,
    klu_numeric, klu_refactor, klu_solve, klu_symbolic, klu_tsolve, klu_z_factor, klu_z_refactor,
    klu_z_solve, klu_z_tsolve, KLU_OK,
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
/// `f64_ptr`/`f64_ptr_const` must reinterpret the slice's memory as `f64`,
/// which is sound for `f64` and `C64` (`#[repr(C)]` of two `f64`).
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

    /// KLU numeric factorization (`klu_factor` / `klu_z_factor`).
    ///
    /// # Safety
    /// `bp`/`bi`/`bx` must describe a valid CSC matrix, `sym` a valid symbolic
    /// handle and `common` a valid `klu_common`.
    unsafe fn lu_factor(
        bp: &mut [i32],
        bi: &mut [i32],
        bx: &mut [Self],
        sym: *mut klu_symbolic,
        common: *mut klu_common,
    ) -> *mut klu_numeric;

    /// KLU refactorization (`klu_refactor` / `klu_z_refactor`).
    ///
    /// # Safety
    /// As [`Scalar::lu_factor`]; `num` must be a valid numeric handle.
    unsafe fn lu_refactor(
        bp: &mut [i32],
        bi: &mut [i32],
        bx: &mut [Self],
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        common: *mut klu_common,
    ) -> c_int;

    /// KLU solve (`klu_solve` / `klu_z_solve`).
    ///
    /// # Safety
    /// `sym`/`num` must be matching valid handles and `b` a valid RHS buffer.
    unsafe fn lu_solve(
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        n_col: usize,
        n_rhs: usize,
        b: &mut [Self],
        common: *mut klu_common,
    ) -> c_int;

    /// KLU plain-transpose solve (`klu_tsolve` / `klu_z_tsolve`).
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
    const DTYPE: c_int = dtype::F64;
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
            klu_factor(
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
            klu_refactor(
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
            klu_solve(
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
            klu_tsolve(
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
    const DTYPE: c_int = dtype::C128;
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
            klu_z_factor(
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
            klu_z_refactor(
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
            klu_z_solve(
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
            klu_z_tsolve(
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

fn new_common() -> klu_common {
    unsafe {
        let mut common = core::mem::MaybeUninit::<klu_common>::uninit();
        klu_defaults(common.as_mut_ptr());
        common.assume_init()
    }
}

/// Row-major `(n_lhs, n_col, n_rhs)` -> column-major `(n_lhs, n_rhs, n_col)`.
///
/// KLU expects column-major right-hand sides.
fn to_col_major<T: Scalar>(b: &[T], n_lhs: usize, n_col: usize, n_rhs: usize) -> Vec<T> {
    let mut out = vec![T::zero(); n_lhs * n_col * n_rhs];
    for m in 0..n_lhs {
        for n in 0..n_col {
            for p in 0..n_rhs {
                out[m * n_rhs * n_col + p * n_col + n] = b[m * n_col * n_rhs + n * n_rhs + p];
            }
        }
    }
    out
}

/// Column-major `(n_lhs, n_rhs, n_col)` -> row-major `(n_lhs, n_col, n_rhs)`.
fn to_row_major<T: Scalar>(x: &[T], n_lhs: usize, n_col: usize, n_rhs: usize) -> Vec<T> {
    let mut out = vec![T::zero(); n_lhs * n_col * n_rhs];
    for m in 0..n_lhs {
        for n in 0..n_col {
            for p in 0..n_rhs {
                out[m * n_col * n_rhs + n * n_rhs + p] = x[m * n_rhs * n_col + p * n_col + n];
            }
        }
    }
    out
}

/// COO -> CSC conversion info (rows sorted within each column).
///
/// Returns `(bi, bp, bk)` where `bk[k]` is the original COO index of CSC entry
/// `k`.
pub fn coo_to_csc(
    n_col: usize,
    n_nz: usize,
    ai: &[i32],
    aj: &[i32],
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut bp = vec![0i32; n_col + 1];
    for &j in aj.iter().take(n_nz) {
        bp[j as usize] += 1;
    }
    let mut cumsum = 0i32;
    for val in bp.iter_mut() {
        let temp = *val;
        *val = cumsum;
        cumsum += temp;
    }
    let mut bi = vec![0i32; n_nz];
    let mut bk = vec![0i32; n_nz];
    for n in 0..n_nz {
        let col = aj[n] as usize;
        let dest = bp[col] as usize;
        bi[dest] = ai[n];
        bk[dest] = n as i32;
        bp[col] += 1;
    }
    let mut last = 0i32;
    for val in bp.iter_mut() {
        std::mem::swap(val, &mut last);
    }
    (bi, bp, bk)
}

/// Mirror of the C++ `validate_args` for base-case `(n_lhs,n_nz)` / `(3D x)`.
pub fn validate(
    ai: &[i32],
    aj: &[i32],
    n_lhs: usize,
    n_col: usize,
    n_rhs: usize,
    n_nz: usize,
) -> Result<(), ErrorInfo> {
    if ai.len() != n_nz {
        return Err(ErrorInfo::invalid(
            "n_nz mismatch: Ai.shape[0] != Ax.shape[1]",
        ));
    }
    if aj.len() != n_nz {
        return Err(ErrorInfo::invalid(
            "n_nz mismatch: Aj.shape[0] != Ax.shape[1]",
        ));
    }
    let _ = (n_lhs, n_rhs);
    for n in 0..n_nz {
        if ai[n] < 0 {
            return Err(ErrorInfo::invalid("Ai contains negative index"));
        }
        if ai[n] as usize >= n_col {
            return Err(ErrorInfo::invalid("Ai.max() >= n_col"));
        }
        if aj[n] < 0 {
            return Err(ErrorInfo::invalid("Aj contains negative index"));
        }
        if aj[n] as usize >= n_col {
            return Err(ErrorInfo::invalid("Aj.max() >= n_col"));
        }
    }
    Ok(())
}

/// Gather the CSC-ordered values of `ax` for left-hand side `i`.
fn gather_ax<T: Scalar>(ax: &[T], bk: &[i32], i: usize, n_nz: usize) -> Vec<T> {
    let m = i * n_nz;
    (0..n_nz).map(|k| ax[m + bk[k] as usize]).collect()
}

/// Symbolic analysis; returns the raw `klu_symbolic*` as `u64`.
pub fn analyze_raw(n_col: usize, ai: &[i32], aj: &[i32]) -> Result<u64, ErrorInfo> {
    let n_nz = ai.len();
    validate(ai, aj, 1, n_col, 1, n_nz)?;
    let (mut bi, mut bp, _) = coo_to_csc(n_col, n_nz, ai, aj);
    let mut common = new_common();
    let sym = unsafe {
        klu_analyze(
            n_col as c_int,
            bp.as_mut_ptr(),
            bi.as_mut_ptr(),
            &mut common,
        )
    };
    if sym.is_null() || common.status < KLU_OK {
        return Err(ErrorInfo::internal("klu_analyze failed."));
    }
    Ok(sym as u64)
}

/// Numeric factorization of one matrix; returns the raw `klu_numeric*` as `u64`.
pub fn factor_raw<T: Scalar>(ai: &[i32], aj: &[i32], ax: &[T], sym: u64) -> Result<u64, ErrorInfo> {
    let root = sym as *mut klu_symbolic;
    if root.is_null() {
        return Err(ErrorInfo::invalid("symbolic pointer is null"));
    }
    let n_col = unsafe { (*root).n as usize };
    let n_nz = ai.len();
    validate(ai, aj, 1, n_col, 1, n_nz)?;
    let (mut bi, mut bp, bk) = coo_to_csc(n_col, n_nz, ai, aj);
    let mut bx: Vec<T> = bk.iter().map(|&k| ax[k as usize]).collect();
    let mut common = new_common();
    let num = unsafe { T::lu_factor(&mut bp, &mut bi, &mut bx, root, &mut common) };
    if num.is_null() || common.status < KLU_OK {
        return Err(ErrorInfo::invalid(
            "klu_factor/z_factor failed (singular matrix?)",
        ));
    }
    Ok(num as u64)
}

/// Numeric factorization of a batch sharing one sparsity pattern.
pub fn factor_batch_raw<T: Scalar>(
    ai: &[i32],
    aj: &[i32],
    ax: &[T],
    n_lhs: usize,
    sym: u64,
) -> Result<Vec<u64>, ErrorInfo> {
    let root = sym as *mut klu_symbolic;
    if root.is_null() {
        return Err(ErrorInfo::invalid("symbolic pointer is null"));
    }
    let n_col = unsafe { (*root).n as usize };
    let n_nz = ai.len();
    validate(ai, aj, n_lhs, n_col, 1, n_nz)?;
    let (mut bi, mut bp, bk) = coo_to_csc(n_col, n_nz, ai, aj);
    let mut common = new_common();
    let mut out = Vec::with_capacity(n_lhs);
    for i in 0..n_lhs {
        let mut bx = gather_ax(ax, &bk, i, n_nz);
        let num = unsafe { T::lu_factor(&mut bp, &mut bi, &mut bx, root, &mut common) };
        if num.is_null() || common.status < KLU_OK {
            for addr in out.drain(..) {
                let mut p = addr as *mut klu_numeric;
                unsafe { klu_free_numeric(&mut p, &mut common) };
            }
            return Err(ErrorInfo::invalid(
                "klu_factor/z_factor failed (singular matrix?)",
            ));
        }
        out.push(num as u64);
    }
    Ok(out)
}

/// Recompute the numeric factorization of a batch in place; returns the same
/// handles (so XLA can see the refactor -> solve dependency edge).
pub fn refactor_batch_raw<T: Scalar>(
    ai: &[i32],
    aj: &[i32],
    ax: &[T],
    n_lhs: usize,
    sym: u64,
    numeric: &[u64],
) -> Result<Vec<u64>, ErrorInfo> {
    let root = sym as *mut klu_symbolic;
    if root.is_null() {
        return Err(ErrorInfo::invalid("symbolic pointer is null"));
    }
    let n_col = unsafe { (*root).n as usize };
    let n_nz = ai.len();
    validate(ai, aj, n_lhs, n_col, 1, n_nz)?;
    if numeric.len() != n_lhs {
        return Err(ErrorInfo::invalid("numeric array size must match n_lhs"));
    }
    let (mut bi, mut bp, bk) = coo_to_csc(n_col, n_nz, ai, aj);
    let mut common = new_common();
    let mut out = vec![0u64; n_lhs];
    for i in 0..n_lhs {
        let addr = numeric[i];
        if addr == 0 {
            return Err(ErrorInfo::invalid("numeric pointer is null"));
        }
        let mut bx = gather_ax(ax, &bk, i, n_nz);
        let status = unsafe {
            T::lu_refactor(
                &mut bp,
                &mut bi,
                &mut bx,
                root,
                addr as *mut klu_numeric,
                &mut common,
            )
        };
        if status == 0 || common.status < KLU_OK {
            return Err(ErrorInfo::invalid(
                "klu_refactor/z_refactor failed (singular matrix?)",
            ));
        }
        out[i] = addr;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn solve_with_symbol_impl<T: Scalar>(
    ai: &[i32],
    aj: &[i32],
    ax: &[T],
    b: &[T],
    n_lhs: usize,
    n_col: usize,
    n_rhs: usize,
    sym: u64,
    transpose: bool,
) -> Result<Vec<T>, ErrorInfo> {
    let n_nz = ax.len() / n_lhs;
    validate(ai, aj, n_lhs, n_col, n_rhs, n_nz)?;
    let (mut bi, mut bp, bk) = coo_to_csc(n_col, n_nz, ai, aj);

    // b (n_lhs, n_col, n_rhs) row-major -> x_temp col-major.
    let mut x_temp = to_col_major(b, n_lhs, n_col, n_rhs);

    let root = sym as *mut klu_symbolic;
    if root.is_null() {
        return Err(ErrorInfo::invalid("symbolic pointer is null"));
    }
    let mut common = new_common();

    for i in 0..n_lhs {
        let mut bx = gather_ax(ax, &bk, i, n_nz);
        let num = unsafe { T::lu_factor(&mut bp, &mut bi, &mut bx, root, &mut common) };
        if num.is_null() || common.status < KLU_OK {
            return Err(ErrorInfo::invalid(
                "klu_factor/z_factor failed (singular matrix?)",
            ));
        }
        let status = if transpose {
            let n = i * n_rhs * n_col;
            unsafe {
                T::lu_tsolve(
                    root,
                    num,
                    n_col,
                    n_rhs,
                    &mut x_temp[n..n + n_rhs * n_col],
                    &mut common,
                )
            }
        } else {
            let n = i * n_rhs * n_col;
            unsafe {
                T::lu_solve(
                    root,
                    num,
                    n_col,
                    n_rhs,
                    &mut x_temp[n..n + n_rhs * n_col],
                    &mut common,
                )
            }
        };
        let mut num = num;
        unsafe { klu_free_numeric(&mut num, &mut common) };
        if status == 0 || common.status < KLU_OK {
            return Err(ErrorInfo::invalid(if transpose {
                "klu_tsolve/z_tsolve failed"
            } else {
                "klu_solve/z_solve failed"
            }));
        }
    }

    // x_temp col-major -> x row-major.
    let x = to_row_major(&x_temp, n_lhs, n_col, n_rhs);
    Ok(x)
}

/// `solve` (analyze + factor + solve) for a batched system.
#[allow(clippy::too_many_arguments)]
pub fn solve_raw<T: Scalar>(
    ai: &[i32],
    aj: &[i32],
    ax: &[T],
    b: &[T],
    n_lhs: usize,
    n_col: usize,
    n_rhs: usize,
) -> Result<Vec<T>, ErrorInfo> {
    let n_nz = ax.len() / n_lhs;
    validate(ai, aj, n_lhs, n_col, n_rhs, n_nz)?;
    let (mut bi, mut bp, bk) = coo_to_csc(n_col, n_nz, ai, aj);
    let mut x_temp = to_col_major(b, n_lhs, n_col, n_rhs);
    let mut common = new_common();
    let root = unsafe {
        klu_analyze(
            n_col as c_int,
            bp.as_mut_ptr(),
            bi.as_mut_ptr(),
            &mut common,
        )
    };
    if root.is_null() || common.status < KLU_OK {
        return Err(ErrorInfo::invalid("klu_analyze failed."));
    }
    let mut result = Ok(());
    for i in 0..n_lhs {
        let mut bx = gather_ax(ax, &bk, i, n_nz);
        let num = unsafe { T::lu_factor(&mut bp, &mut bi, &mut bx, root, &mut common) };
        if num.is_null() || common.status < KLU_OK {
            result = Err(ErrorInfo::invalid(
                "klu_factor/z_factor failed (singular matrix?)",
            ));
            break;
        }
        let n = i * n_rhs * n_col;
        let status = unsafe {
            T::lu_solve(
                root,
                num,
                n_col,
                n_rhs,
                &mut x_temp[n..n + n_rhs * n_col],
                &mut common,
            )
        };
        let mut num = num;
        unsafe { klu_free_numeric(&mut num, &mut common) };
        if status == 0 || common.status < KLU_OK {
            result = Err(ErrorInfo::invalid("klu_solve/z_solve failed"));
            break;
        }
    }
    let mut root = root;
    unsafe { klu_free_symbolic(&mut root, &mut common) };
    result?;

    let x = to_row_major(&x_temp, n_lhs, n_col, n_rhs);
    Ok(x)
}

/// `solve_with_symbol`.
#[allow(clippy::too_many_arguments)]
pub fn solve_with_symbol_raw<T: Scalar>(
    ai: &[i32],
    aj: &[i32],
    ax: &[T],
    b: &[T],
    n_lhs: usize,
    n_col: usize,
    n_rhs: usize,
    sym: u64,
) -> Result<Vec<T>, ErrorInfo> {
    solve_with_symbol_impl(ai, aj, ax, b, n_lhs, n_col, n_rhs, sym, false)
}

/// `tsolve_with_symbol`.
#[allow(clippy::too_many_arguments)]
pub fn tsolve_with_symbol_raw<T: Scalar>(
    ai: &[i32],
    aj: &[i32],
    ax: &[T],
    b: &[T],
    n_lhs: usize,
    n_col: usize,
    n_rhs: usize,
    sym: u64,
) -> Result<Vec<T>, ErrorInfo> {
    solve_with_symbol_impl(ai, aj, ax, b, n_lhs, n_col, n_rhs, sym, true)
}

/// `solve_with_numeric` / `tsolve_with_numeric`.
///
/// `numeric` is a batch of raw handles (length 1 broadcasts). `b` is
/// `(n_lhs, n_col, n_rhs)`; output is the same shape.
pub fn solve_with_numeric_raw<T: Scalar>(
    sym: u64,
    numeric: &[u64],
    b: &[T],
    n_lhs: usize,
    n_col: usize,
    n_rhs: usize,
    transpose: bool,
) -> Result<Vec<T>, ErrorInfo> {
    let root = sym as *mut klu_symbolic;
    if root.is_null() {
        return Err(ErrorInfo::invalid("symbolic pointer is null"));
    }
    let broadcast_numeric = numeric.len() == 1;
    if !broadcast_numeric && numeric.len() != n_lhs {
        return Err(ErrorInfo::invalid("numeric and b batch size mismatch"));
    }
    let mut x_temp = to_col_major(b, n_lhs, n_col, n_rhs);
    let mut common = new_common();
    for i in 0..n_lhs {
        let addr = if broadcast_numeric {
            numeric[0]
        } else {
            numeric[i]
        };
        if addr == 0 {
            return Err(ErrorInfo::invalid("numeric pointer is null"));
        }
        let num = addr as *mut klu_numeric;
        let n = i * n_rhs * n_col;
        let status = if transpose {
            unsafe {
                T::lu_tsolve(
                    root,
                    num,
                    n_col,
                    n_rhs,
                    &mut x_temp[n..n + n_rhs * n_col],
                    &mut common,
                )
            }
        } else {
            unsafe {
                T::lu_solve(
                    root,
                    num,
                    n_col,
                    n_rhs,
                    &mut x_temp[n..n + n_rhs * n_col],
                    &mut common,
                )
            }
        };
        if status == 0 || common.status < KLU_OK {
            return Err(ErrorInfo::invalid(if transpose {
                "klu_tsolve/z_tsolve failed"
            } else {
                "klu_solve/z_solve failed"
            }));
        }
    }
    let x = to_row_major(&x_temp, n_lhs, n_col, n_rhs);
    Ok(x)
}

/// `b = A @ x`, batched over `n_lhs`.
#[allow(clippy::too_many_arguments)]
pub fn dot_raw<T: Scalar>(
    ai: &[i32],
    aj: &[i32],
    ax: &[T],
    x: &[T],
    n_lhs: usize,
    n_col: usize,
    n_rhs: usize,
) -> Result<Vec<T>, ErrorInfo> {
    let n_nz = ax.len() / n_lhs;
    validate(ai, aj, n_lhs, n_col, n_rhs, n_nz)?;
    let mut b = vec![T::zero(); n_lhs * n_col * n_rhs];
    for m in 0..n_lhs {
        for k in 0..n_nz {
            let i = ai[k] as usize;
            let j = aj[k] as usize;
            for p in 0..n_rhs {
                let idx = m * n_col * n_rhs + i * n_rhs + p;
                let xv = x[m * n_col * n_rhs + j * n_rhs + p];
                b[idx] = T::add(b[idx], T::mul(ax[m * n_nz + k], xv));
            }
        }
    }
    Ok(b)
}

/// Free a symbolic handle.
pub fn free_symbolic_raw(sym: u64) -> i32 {
    if sym == 0 {
        return 0;
    }
    let mut common = new_common();
    let mut ptr = sym as *mut klu_symbolic;
    unsafe { klu_free_symbolic(&mut ptr, &mut common) };
    common.status
}

/// Free a batch of numeric handles.
pub fn free_numeric_raw(numeric: &[u64]) -> i32 {
    let mut common = new_common();
    for &addr in numeric {
        if addr == 0 {
            continue;
        }
        let mut ptr = addr as *mut klu_numeric;
        unsafe { klu_free_numeric(&mut ptr, &mut common) };
    }
    common.status
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_round_trip() {
        let (n_lhs, n_col, n_rhs) = (2usize, 3usize, 4usize);
        let b: Vec<f64> = (0..(n_lhs * n_col * n_rhs)).map(|i| i as f64).collect();
        let cm = to_col_major(&b, n_lhs, n_col, n_rhs);
        assert_eq!(b, to_row_major(&cm, n_lhs, n_col, n_rhs));

        let bz: Vec<C64> = b.iter().map(|&x| C64 { re: x, im: -x }).collect();
        let cmz = to_col_major(&bz, n_lhs, n_col, n_rhs);
        assert_eq!(bz, to_row_major(&cmz, n_lhs, n_col, n_rhs));
    }

    fn diagonal(n: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let ai: Vec<i32> = (0..n as i32).collect();
        let aj = ai.clone();
        let ax: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        (ai, aj, ax)
    }

    #[cfg(not(miri))]
    #[test]
    fn solve_f64() {
        let (ai, aj, ax) = diagonal(4);
        let b = vec![1.0, 4.0, 9.0, 16.0];
        let x = solve_raw(&ai, &aj, &ax, &b, 1, 4, 1).unwrap();
        for (i, &xi) in x.iter().enumerate() {
            assert!((xi - (i + 1) as f64).abs() < 1e-12);
        }
    }

    #[cfg(not(miri))]
    #[test]
    fn analyze_factor_solve_tsolve_f64() {
        let (ai, aj, ax) = diagonal(3);
        let sym = analyze_raw(3, &ai, &aj).unwrap();
        let b = vec![2.0, 4.0, 9.0];
        let x = solve_with_symbol_raw(&ai, &aj, &ax, &b, 1, 3, 1, sym).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-12);
        let xt = tsolve_with_symbol_raw(&ai, &aj, &ax, &b, 1, 3, 1, sym).unwrap();
        // A is diagonal, so tsolve == solve: x = [2, 2, 3].
        assert!((xt[1] - 2.0).abs() < 1e-12);

        let num = factor_raw(&ai, &aj, &ax, sym).unwrap();
        let xn = solve_with_numeric_raw(sym, &[num], &b, 1, 3, 1, false).unwrap();
        assert!((xn[2] - 3.0).abs() < 1e-12);
        assert_eq!(free_numeric_raw(&[num]), 0);
        assert_eq!(free_symbolic_raw(sym), 0);
    }

    #[cfg(not(miri))]
    #[test]
    fn solve_c128() {
        let (ai, aj, _) = diagonal(2);
        let ax = vec![C64 { re: 2.0, im: 0.0 }, C64 { re: 0.0, im: 4.0 }];
        let b = vec![C64 { re: 4.0, im: 0.0 }, C64 { re: 0.0, im: 8.0 }];
        let x = solve_raw(&ai, &aj, &ax, &b, 1, 2, 1).unwrap();
        // A = diag(2, 4i); x = [4/2, 8i/(4i)] = [2, 2].
        assert!((x[0].re - 2.0).abs() < 1e-12);
        assert!((x[1].re - 2.0).abs() < 1e-12);
    }

    #[test]
    fn dot_f64() {
        let ai = vec![0i32, 1];
        let aj = vec![0i32, 1];
        let ax = vec![2.0f64, 4.0];
        let x = vec![3.0f64, 5.0];
        let b = dot_raw(&ai, &aj, &ax, &x, 1, 2, 1).unwrap();
        assert_eq!(b, vec![6.0, 20.0]);
    }
}

/// Property tests for the pure-Rust logic (run under `cargo miri`).
#[cfg(test)]
mod props {
    use super::*;
    use proptest::prelude::*;

    /// Tiny deterministic PRNG so the tests need no extra dependency.
    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state >> 33
    }

    proptest! {
        #[test]
        fn coo_to_csc_is_valid_csc(n_col in 1usize..8, n_nz in 0usize..24, seed in any::<u64>()) {
            let mut st = seed | 1;
            let ai: Vec<i32> = (0..n_nz).map(|_| (lcg(&mut st) as usize % n_col) as i32).collect();
            let aj: Vec<i32> = (0..n_nz).map(|_| (lcg(&mut st) as usize % n_col) as i32).collect();
            let (bi, bp, bk) = coo_to_csc(n_col, n_nz, &ai, &aj);
            prop_assert_eq!(bp.len(), n_col + 1);
            prop_assert_eq!(bp[0], 0);
            prop_assert_eq!(*bp.last().unwrap(), n_nz as i32);
            prop_assert!(bp.windows(2).all(|w| w[0] <= w[1]));
            for j in 0..n_col {
                let (lo, hi) = (bp[j] as usize, bp[j + 1] as usize);
                for &row in &bi[lo..hi] {
                    prop_assert!(row >= 0 && (row as usize) < n_col);
                }
            }
            let mut perm = bk.clone();
            perm.sort_unstable();
            prop_assert_eq!(perm, (0..n_nz as i32).collect::<Vec<_>>());
        }

        #[test]
        fn transpose_round_trip_prop(
            n_lhs in 1usize..4, n_col in 1usize..4, n_rhs in 1usize..4, seed in any::<u64>(),
        ) {
            let mut st = seed | 1;
            let n = n_lhs * n_col * n_rhs;
            let b: Vec<f64> = (0..n).map(|_| lcg(&mut st) as f64).collect();
            let cm = to_col_major(&b, n_lhs, n_col, n_rhs);
            prop_assert_eq!(&b, &to_row_major(&cm, n_lhs, n_col, n_rhs));
        }
    }
}
