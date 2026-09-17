//! KLU control parameters and status, mirroring SuiteSparse's `klu_common`.
//!
//! In the pure-Rust implementation, memory management is handled by the Rust
//! allocator, so the `SuiteSparse_config` allocator hooks are intentionally not
//! ported (see `work.md` Stage 5.1).

use crate::error::KluStatus;

/// Ordering method used for the symbolic analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Ordering {
    /// Approximate minimum degree (AMD). KLU default.
    Amd = 0,
    /// Column approximate minimum degree (COLAMD).
    Colamd = 1,
    /// User-provided `P` and `Q` (not supported by the pure-Rust port yet).
    Given = 2,
    /// User-provided ordering function (not supported by the pure-Rust port).
    User = 3,
}

/// Scaling method applied before factorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Scaling {
    /// No scaling, and no error check (SuiteSparse `scale = -1`).
    NoneNoCheck = -1,
    /// No scaling.
    None = 0,
    /// Sum of absolute values.
    Sum = 1,
    /// Maximum absolute value.
    Max = 2,
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
    /// Maximum work factor for BTF (`<= 0` means no limit).
    pub maxwork: f64,
    /// Whether to use block triangular form.
    pub btf: bool,
    /// Ordering used for the analysis.
    pub ordering: Ordering,
    /// Scaling used before factorization.
    pub scale: Scaling,
    /// Stop quickly on a singular matrix.
    pub halt_if_singular: bool,
    /// Status code from the last operation.
    pub status: KluStatus,
    /// Number of reallocations of `L` and `U`.
    pub nrealloc: i32,
    /// Structural rank (or `-1` if not computed).
    pub structural_rank: i32,
    /// First `k` with a zero `U(k,k)` (or `-1` if not computed).
    pub numerical_rank: i32,
    /// Column of the first zero pivot (or `-1` if not computed).
    pub singular_col: i32,
    /// Number of off-diagonal pivots (`-1` if not computed).
    pub noffdiag: i32,
}

impl Default for KluCommon {
    fn default() -> Self {
        Self::defaults()
    }
}

impl KluCommon {
    /// Equivalent to SuiteSparse's `klu_defaults`.
    pub fn defaults() -> Self {
        Self {
            tol: 0.001,
            memgrow: 1.2,
            initmem_amd: 1.2,
            initmem: 10.0,
            maxwork: 0.0,
            btf: true,
            ordering: Ordering::Amd,
            scale: Scaling::None,
            halt_if_singular: true,
            status: KluStatus::Ok,
            nrealloc: 0,
            structural_rank: -1,
            numerical_rank: -1,
            singular_col: -1,
            noffdiag: -1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_suitesparse() {
        let c = KluCommon::defaults();
        assert_eq!(c.tol, 0.001);
        assert_eq!(c.memgrow, 1.2);
        assert_eq!(c.initmem_amd, 1.2);
        assert_eq!(c.initmem, 10.0);
        assert_eq!(c.maxwork, 0.0);
        assert!(c.btf);
        assert_eq!(c.ordering, Ordering::Amd);
        assert_eq!(c.scale, Scaling::None);
        assert!(c.halt_if_singular);
        assert_eq!(c.status, KluStatus::Ok);
        assert_eq!(c.structural_rank, -1);
    }

    #[test]
    fn ordering_repr_matches_c() {
        assert_eq!(Ordering::Amd as i32, 0);
        assert_eq!(Ordering::Colamd as i32, 1);
        assert_eq!(Scaling::NoneNoCheck as i32, -1);
        assert_eq!(Scaling::Max as i32, 2);
    }
}
