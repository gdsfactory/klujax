//! KLU control parameters, mirroring SuiteSparse's `klu_common`.

/// Ordering method used for the symbolic analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ordering {
    /// Approximate minimum degree (AMD). KLU default.
    Amd,
    /// Column approximate minimum degree (COLAMD).
    Colamd,
    /// Natural (no reordering).
    Natural,
}

/// Scaling method applied before factorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scaling {
    /// No scaling.
    None,
    /// Sum of absolute values.
    Sum,
    /// Maximum absolute value.
    Max,
}

/// Control parameters and status for a KLU solve.
#[derive(Debug, Clone)]
pub struct KluCommon {
    /// Partial pivoting tolerance in `[0, 1]`.
    pub tol: f64,
    /// Memory growth factor.
    pub memgrow: f64,
    /// Initial memory for AMD (fraction of `nz`).
    pub initmem_amd: f64,
    /// Initial memory (fraction of `nz`).
    pub initmem: f64,
    /// Maximum work factor.
    pub maxwork: f64,
    /// Whether to use block triangular form.
    pub btf: bool,
    /// Ordering used for the analysis.
    pub ordering: Ordering,
    /// Scaling used before factorization.
    pub scale: Scaling,
}

impl Default for KluCommon {
    fn default() -> Self {
        Self {
            tol: 0.001,
            memgrow: 1.2,
            initmem_amd: 1.2,
            initmem: 10.0,
            maxwork: 0.0,
            btf: true,
            ordering: Ordering::Amd,
            scale: Scaling::None,
        }
    }
}
