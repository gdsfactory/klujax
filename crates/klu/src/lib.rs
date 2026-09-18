//! Ergonomic Rust bindings to the SuiteSparse KLU sparse linear solver.
//!
//! Build a matrix with [`Coo`], then pick how much state to reuse:
//!
//! 1. [`solve`] — analyze + factor + solve in one call, no handles.
//! 2. [`Symbolic::solve`] — analyze the pattern once, re-factor + solve per call.
//! 3. [`Numeric::solve`] — factor once, solve many right-hand sides.
//!
//! Handles are reference-counted and freed on drop (no manual `free_*`), and
//! [`Error`] implements [`std::error::Error`].
//!
//! # One-shot
//! ```
//! # fn main() -> klu::Result<()> {
//! use klu::{Coo, solve};
//! let a = Coo::new(3)?
//!     .push(0, 0, 2.0)?
//!     .push(1, 1, 4.0)?
//!     .push(2, 2, 5.0)?
//!     .build()?;
//! let b = [2.0, 8.0, 10.0];
//! let x = solve(&a, &b)?;
//! assert!(a.residual(&x, &b)? < 1e-12);
//! # Ok(())
//! # }
//! ```
//!
//! # Reuse the symbolic analysis
//! ```
//! # fn main() -> klu::Result<()> {
//! use klu::Coo;
//! let mut a = Coo::new(2)?
//!     .push(0, 0, 2.0)?
//!     .push(1, 1, 4.0)?
//!     .build()?;
//! let symbolic = a.analyze()?;         // structure only, once
//! for scale in [1.0, 2.0, 3.0] {
//!     a.set_values(&[2.0 * scale, 4.0 * scale])?;
//!     let x = symbolic.solve(a.values(), &[10.0 * scale, 20.0 * scale])?;
//!     assert!((x[0] - 5.0).abs() < 1e-12);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Reuse the numeric factorization
//! ```
//! # fn main() -> klu::Result<()> {
//! use klu::Coo;
//! let a = Coo::new(2)?
//!     .push(0, 0, 2.0)?
//!     .push(1, 1, 4.0)?
//!     .build()?;
//! let numeric = a.factorize()?;        // factor once
//! for b in [[2.0, 8.0], [4.0, 8.0]] {
//!     let x = numeric.solve(&b)?;       // solve many
//!     assert!(a.residual(&x, &b)? < 1e-12);
//! }
//! let _ = numeric.symbolic();          // optional access to the analysis
//! # Ok(())
//! # }
//! ```

pub mod error;
pub mod matrix;
pub mod numeric;
pub mod raw;
pub mod scalar;
pub mod symbolic;

pub use error::{Error, ErrorKind, Result};
pub use matrix::{Coo, Csc};
pub use numeric::Numeric;
pub use raw::Common;
pub use scalar::{Scalar, C64};
pub use symbolic::Symbolic;

/// Solve `A x = b` in one shot: analyze + factor + solve. No handles to free.
pub fn solve<T: Scalar>(a: &Csc<T>, b: &[T]) -> Result<Vec<T>> {
    a.solve(b)
}

/// Solve `Aᵀ x = b` in one shot.
pub fn solve_transpose<T: Scalar>(a: &Csc<T>, b: &[T]) -> Result<Vec<T>> {
    a.solve_transpose(b)
}
