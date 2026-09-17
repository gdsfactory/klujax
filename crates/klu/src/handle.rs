//! Opaque owning handles for symbolic and numeric factorizations.

/// Symbolic analysis handle.
///
/// Pure-Rust owning type; later stages replace the placeholder fields with the
/// real symbolic data structures (permutations, block structure, etc.).
#[derive(Debug)]
pub struct Symbolic {
    /// Number of columns of the analyzed matrix.
    pub n_col: usize,
    /// Number of nonzeros of the analyzed matrix.
    pub n_nz: usize,
}

/// Numeric factorization handle.
#[derive(Debug)]
pub struct Numeric {
    /// Number of right-hand sides / left-hand-side batch dimension.
    pub n_lhs: usize,
}
