//! The numeric (factorized) handle.

use core::marker::PhantomData;

use crate::error::{Error, Result};
use crate::raw;
use crate::scalar::Scalar;
use crate::symbolic::Symbolic;

/// A numeric factorization of a specific matrix.
///
/// Owns its KLU allocation (freed on drop) and keeps the [`Symbolic`] analysis
/// alive, so [`Numeric::solve`] needs no symbolic argument. It is not `Clone`:
/// a factorization owns a single native object. Use [`Numeric::symbolic`] to
/// reach the analysis, or re-factor in place with [`Numeric::refactor`].
pub struct Numeric<T: Scalar> {
    sym: Symbolic,
    id: u64,
    _t: PhantomData<T>,
}

impl<T: Scalar> Numeric<T> {
    pub(crate) fn new(sym: Symbolic, id: u64) -> Self {
        Self {
            sym,
            id,
            _t: PhantomData,
        }
    }

    /// The symbolic analysis this factorization was built from.
    pub fn symbolic(&self) -> &Symbolic {
        &self.sym
    }

    /// Solve `A x = b`. `b` is a flat row-major `(n_col, n_rhs)` buffer; a plain
    /// vector (`len == n_col`) is the `n_rhs == 1` case.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>> {
        self.solve_impl(b, false)
    }

    /// Solve `Aᵀ x = b` (plain transpose).
    pub fn solve_transpose(&self, b: &[T]) -> Result<Vec<T>> {
        self.solve_impl(b, true)
    }

    fn solve_impl(&self, b: &[T], transpose: bool) -> Result<Vec<T>> {
        let n = raw::symbolic_n(self.sym.id())?;
        if n == 0 || !b.len().is_multiple_of(n) {
            return Err(Error::invalid("RHS length must be a multiple of n_col"));
        }
        let n_rhs = b.len() / n;

        // KLU wants column-major B; transpose row-major (n_col, n_rhs) in/out.
        let mut cm = vec![T::zero(); n * n_rhs];
        for i in 0..n {
            for p in 0..n_rhs {
                cm[p * n + i] = b[i * n_rhs + p];
            }
        }
        let mut common = raw::Common::new();
        raw::solve(
            &mut common,
            self.sym.id(),
            self.id,
            n,
            n_rhs,
            &mut cm,
            transpose,
        )?;

        let mut out = vec![T::zero(); n * n_rhs];
        for i in 0..n {
            for p in 0..n_rhs {
                out[i * n_rhs + p] = cm[p * n + i];
            }
        }
        Ok(out)
    }

    /// Re-factor this numeric object with new values (same pattern).
    pub fn refactor(&mut self, ax: &[T]) -> Result<()> {
        let mut common = raw::Common::new();
        raw::refactor_values(&mut common, ax, self.sym.id(), self.id)
    }
}

impl<T: Scalar> Drop for Numeric<T> {
    fn drop(&mut self) {
        raw::release_numeric(self.id);
    }
}
