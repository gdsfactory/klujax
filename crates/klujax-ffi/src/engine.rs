//! Numeric KLU engine: COO→CSC, batching, and RHS layout.
//!
//! Built on the safe [`crate::klu`] wrapper, so this module contains **no
//! `unsafe`** (enforced by `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

use crate::error::ErrorInfo;
use crate::klu::{self, Common};

pub use crate::klu::{Scalar, C64};

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

/// Gather the CSC-ordered values of `ax` for left-hand side `i`.
fn gather_ax<T: Scalar>(ax: &[T], bk: &[i32], i: usize, n_nz: usize) -> Vec<T> {
    let m = i * n_nz;
    (0..n_nz).map(|k| ax[m + bk[k] as usize]).collect()
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
    // Canonicalize rows so analysis and factorization agree even when their
    // COO inputs arrive in different orders.
    for col in 0..n_col {
        let lo = bp[col] as usize;
        let hi = bp[col + 1] as usize;
        let mut entries: Vec<_> = bi[lo..hi]
            .iter()
            .copied()
            .zip(bk[lo..hi].iter().copied())
            .collect();
        entries.sort_unstable_by_key(|&(row, _)| row);
        for (k, (row, original)) in (lo..hi).zip(entries) {
            bi[k] = row;
            bk[k] = original;
        }
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
    if n_col > i32::MAX as usize || n_nz > i32::MAX as usize {
        return Err(ErrorInfo::invalid("matrix exceeds KLU int32 dimensions"));
    }
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

/// Symbolic analysis; returns an opaque symbolic ID as `u64`.
pub fn analyze_raw(n_col: usize, ai: &[i32], aj: &[i32]) -> Result<u64, ErrorInfo> {
    let n_nz = ai.len();
    validate(ai, aj, 1, n_col, 1, n_nz)?;
    let (mut bi, mut bp, _) = coo_to_csc(n_col, n_nz, ai, aj);
    klu::analyze(n_col, &mut bp, &mut bi)
}

/// Numeric factorization of one matrix; returns an opaque numeric ID as `u64`.
pub fn factor_raw<T: Scalar>(ai: &[i32], aj: &[i32], ax: &[T], sym: u64) -> Result<u64, ErrorInfo> {
    let n_col = klu::symbolic_n(sym)?;
    let n_nz = ai.len();
    validate(ai, aj, 1, n_col, 1, n_nz)?;
    let (mut bi, mut bp, bk) = coo_to_csc(n_col, n_nz, ai, aj);
    if ax.len() != n_nz {
        return Err(ErrorInfo::invalid(
            "matrix value count does not match pattern",
        ));
    }
    let mut bx: Vec<T> = bk.iter().map(|&k| ax[k as usize]).collect();
    let mut common = Common::new();
    klu::factor(&mut common, &mut bp, &mut bi, &mut bx, sym)
}

/// Numeric factorization of a batch sharing one sparsity pattern.
pub fn factor_batch_raw<T: Scalar>(
    ai: &[i32],
    aj: &[i32],
    ax: &[T],
    n_lhs: usize,
    sym: u64,
) -> Result<Vec<u64>, ErrorInfo> {
    let n_col = klu::symbolic_n(sym)?;
    let n_nz = ai.len();
    if n_nz.checked_mul(n_lhs) != Some(ax.len()) {
        return Err(ErrorInfo::invalid(
            "matrix value count does not match batch/pattern",
        ));
    }
    validate(ai, aj, n_lhs, n_col, 1, n_nz)?;
    let (mut bi, mut bp, bk) = coo_to_csc(n_col, n_nz, ai, aj);
    let mut common = Common::new();
    let mut out = Vec::with_capacity(n_lhs);
    for i in 0..n_lhs {
        let mut bx = gather_ax(ax, &bk, i, n_nz);
        match klu::factor(&mut common, &mut bp, &mut bi, &mut bx, sym) {
            Ok(num) => out.push(num),
            Err(e) => {
                for addr in out.drain(..) {
                    klu::free_numeric(&mut common, addr);
                }
                return Err(e);
            }
        }
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
    let n_col = klu::symbolic_n(sym)?;
    let n_nz = ai.len();
    if n_nz.checked_mul(n_lhs) != Some(ax.len()) {
        return Err(ErrorInfo::invalid(
            "matrix value count does not match batch/pattern",
        ));
    }
    validate(ai, aj, n_lhs, n_col, 1, n_nz)?;
    if numeric.len() != n_lhs {
        return Err(ErrorInfo::invalid("numeric array size must match n_lhs"));
    }
    let (mut bi, mut bp, bk) = coo_to_csc(n_col, n_nz, ai, aj);
    let mut common = Common::new();
    let mut out = vec![0u64; n_lhs];
    for i in 0..n_lhs {
        let addr = numeric[i];
        if addr == 0 {
            return Err(ErrorInfo::invalid("numeric pointer is null"));
        }
        let mut bx = gather_ax(ax, &bk, i, n_nz);
        klu::refactor(&mut common, &mut bp, &mut bi, &mut bx, sym, addr)?;
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
    let mut x_temp = to_col_major(b, n_lhs, n_col, n_rhs);
    let mut common = Common::new();

    for i in 0..n_lhs {
        let mut bx = gather_ax(ax, &bk, i, n_nz);
        let num = klu::factor(&mut common, &mut bp, &mut bi, &mut bx, sym)?;
        let n = i * n_rhs * n_col;
        let result = klu::solve(
            &mut common,
            sym,
            num,
            n_col,
            n_rhs,
            &mut x_temp[n..n + n_rhs * n_col],
            transpose,
        );
        klu::free_numeric(&mut common, num);
        result?;
    }
    Ok(to_row_major(&x_temp, n_lhs, n_col, n_rhs))
}

/// `solve` (analyze + factor + solve) for a batched system.
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
    let mut common = Common::new();
    let root = klu::analyze(n_col, &mut bp, &mut bi)?;

    let mut result: Result<(), ErrorInfo> = Ok(());
    for i in 0..n_lhs {
        let mut bx = gather_ax(ax, &bk, i, n_nz);
        let num = match klu::factor(&mut common, &mut bp, &mut bi, &mut bx, root) {
            Ok(num) => num,
            Err(e) => {
                result = Err(e);
                break;
            }
        };
        let n = i * n_rhs * n_col;
        let res = klu::solve(
            &mut common,
            root,
            num,
            n_col,
            n_rhs,
            &mut x_temp[n..n + n_rhs * n_col],
            false,
        );
        klu::free_numeric(&mut common, num);
        if let Err(e) = res {
            result = Err(e);
            break;
        }
    }
    klu::free_symbolic(&mut common, root);
    result?;
    Ok(to_row_major(&x_temp, n_lhs, n_col, n_rhs))
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
pub fn solve_with_numeric_raw<T: Scalar>(
    sym: u64,
    numeric: &[u64],
    b: &[T],
    n_lhs: usize,
    n_col: usize,
    n_rhs: usize,
    transpose: bool,
) -> Result<Vec<T>, ErrorInfo> {
    let _ = klu::symbolic_n(sym)?; // null-check the symbolic handle
    let broadcast_numeric = numeric.len() == 1;
    if !broadcast_numeric && numeric.len() != n_lhs {
        return Err(ErrorInfo::invalid("numeric and b batch size mismatch"));
    }
    let mut x_temp = to_col_major(b, n_lhs, n_col, n_rhs);
    let mut common = Common::new();
    for i in 0..n_lhs {
        let addr = if broadcast_numeric {
            numeric[0]
        } else {
            numeric[i]
        };
        if addr == 0 {
            return Err(ErrorInfo::invalid("numeric pointer is null"));
        }
        let n = i * n_rhs * n_col;
        klu::solve(
            &mut common,
            sym,
            addr,
            n_col,
            n_rhs,
            &mut x_temp[n..n + n_rhs * n_col],
            transpose,
        )?;
    }
    Ok(to_row_major(&x_temp, n_lhs, n_col, n_rhs))
}

/// `b = A @ x`, batched over `n_lhs`.
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
    let mut common = Common::new();
    klu::free_symbolic(&mut common, sym);
    common.status()
}

/// Free a batch of numeric handles.
pub fn free_numeric_raw(numeric: &[u64]) -> i32 {
    let mut common = Common::new();
    for &addr in numeric {
        klu::free_numeric(&mut common, addr);
    }
    common.status()
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;

    fn diagonal(n: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let ai: Vec<i32> = (0..n as i32).collect();
        let aj = ai.clone();
        let ax: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        (ai, aj, ax)
    }

    #[test]
    fn solve_f64() {
        let (ai, aj, ax) = diagonal(4);
        let b = vec![1.0, 4.0, 9.0, 16.0];
        let x = solve_raw(&ai, &aj, &ax, &b, 1, 4, 1).unwrap();
        for (i, &xi) in x.iter().enumerate() {
            assert!((xi - (i + 1) as f64).abs() < 1e-12);
        }
    }

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

        #[test]
        fn dot_matches_manual(n_col in 2usize..5, n_rhs in 1usize..3, seed in any::<u64>()) {
            let mut st = seed | 1;
            let ai = vec![0i32, 1];
            let aj = vec![0i32, 1];
            let ax = vec![2.0f64, 3.0];
            let x: Vec<f64> = (0..n_col * n_rhs).map(|_| lcg(&mut st) as f64).collect();
            let b = dot_raw(&ai, &aj, &ax, &x, 1, n_col, n_rhs).unwrap();
            for p in 0..n_rhs {
                // A = diag(2, 3): row 0 -> 2*x[0], row 1 -> 3*x[1].
                prop_assert_eq!(b[p], 2.0 * x[p]);
                prop_assert_eq!(b[n_rhs + p], 3.0 * x[n_rhs + p]);
            }
        }
    }
}
