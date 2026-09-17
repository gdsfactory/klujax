//! Pure-Rust implementation of the KLU sparse linear solver.
//!
//! This crate is intentionally free of any FFI/JAX concerns: it must be usable
//! as a standalone Rust library. The JAX/XLA integration lives in the
//! `klujax-ffi` crate.
//!
//! Status: skeleton (Stage 1). The algorithm is ported in Stage 5; until then
//! the entry points return [`KluError::NotImplemented`].

#![forbid(unsafe_code)]

pub mod common;
pub mod error;
pub mod handle;

pub use common::{KluCommon, Ordering, Scaling};
pub use error::{KluError, KluStatus};
pub use handle::{Numeric, Symbolic};

/// Symbolic analysis of the sparsity pattern `(ai, aj)` of an `n_col x n_col`
/// matrix in COO format.
pub fn analyze(_n_col: usize, _ai: &[i32], _aj: &[i32]) -> Result<Symbolic, KluError> {
    Err(KluError::NotImplemented)
}

/// Numeric factorization of a matrix given its symbolic analysis.
pub fn factor(_ai: &[i32], _aj: &[i32], _ax: &[f64], _sym: &Symbolic) -> Result<Numeric, KluError> {
    Err(KluError::NotImplemented)
}

/// Solve `A x = b` using a numeric factorization.
pub fn solve(_num: &Numeric, _b: &mut [f64], _sym: &Symbolic) -> Result<(), KluError> {
    Err(KluError::NotImplemented)
}

/// Free a symbolic analysis (no-op for the pure-Rust owning type; provided for
/// API symmetry with the C-backed wrapper).
pub fn free_symbolic(_sym: Symbolic) {}

/// Free a numeric factorization (no-op for the pure-Rust owning type).
pub fn free_numeric(_num: Numeric) {}
