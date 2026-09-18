//! The symbolic (structure-only) analysis handle.

use std::sync::Arc;

use crate::error::Result;
use crate::matrix::Csc;
use crate::numeric::Numeric;
use crate::raw;
use crate::scalar::Scalar;

struct Inner(u64);

impl Drop for Inner {
    fn drop(&mut self) {
        raw::release_symbolic(self.0);
    }
}

/// Reusable symbolic analysis of a sparsity pattern.
///
/// Cheap to [`Clone`] (reference-counted) and freed automatically when the last
/// owner drops. It is enough to drive [`Symbolic::solve`] on its own; to reuse a
/// numeric factorization, keep a [`Numeric`] instead.
#[derive(Clone)]
pub struct Symbolic(Arc<Inner>);

impl Symbolic {
    pub(crate) fn from_id(id: u64) -> Self {
        Self(Arc::new(Inner(id)))
    }

    pub(crate) fn id(&self) -> u64 {
        self.0 .0
    }

    /// Number of columns of the analyzed matrix.
    pub fn n_col(&self) -> Result<usize> {
        raw::symbolic_n(self.id())
    }

    /// Factor new values (same pattern) and return a [`Numeric`].
    ///
    /// `ax` must be in the canonical CSC order, i.e. [`Csc::values`].
    pub fn factor<T: Scalar>(&self, ax: &[T]) -> Result<Numeric<T>> {
        let mut common = raw::Common::new();
        let id = raw::factor_values(&mut common, ax, self.id())?;
        Ok(Numeric::new(self.clone(), id))
    }

    /// Factor `ax` and solve `A x = b` (one shot, reusing this analysis).
    pub fn solve<T: Scalar>(&self, ax: &[T], b: &[T]) -> Result<Vec<T>> {
        self.factor(ax)?.solve(b)
    }

    /// Factor `ax` and solve `Aᵀ x = b` (plain transpose).
    pub fn solve_transpose<T: Scalar>(&self, ax: &[T], b: &[T]) -> Result<Vec<T>> {
        self.factor(ax)?.solve_transpose(b)
    }
}

impl<T: Scalar> Csc<T> {
    /// Analyze the sparsity pattern (structure only); no values are used.
    pub fn analyze(&self) -> Result<Symbolic> {
        let mut bp = self.bp.clone();
        let mut bi = self.bi.clone();
        let id = raw::analyze(self.n, &mut bp, &mut bi)?;
        Ok(Symbolic::from_id(id))
    }

    /// Analyze + factor once, returning a reusable [`Numeric`].
    pub fn factorize(&self) -> Result<Numeric<T>> {
        self.analyze()?.factor(self.values())
    }

    /// One-shot `analyze + factor + solve` (equivalent to [`crate::solve`]).
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>> {
        self.factorize()?.solve(b)
    }

    /// One-shot transposed solve `Aᵀ x = b`.
    pub fn solve_transpose(&self, b: &[T]) -> Result<Vec<T>> {
        self.factorize()?.solve_transpose(b)
    }
}
