//! Sparse matrix types: COO (triplet) input and canonical CSC storage.

use crate::error::{Error, Result};
use crate::scalar::Scalar;

/// A sparse matrix in coordinate (triplet) format.
///
/// Duplicates are summed and rows are canonicalized when [`Coo::build`] is
/// called, so the resulting [`Csc`] is a valid KLU input.
///
/// # Examples
/// ```
/// # fn main() -> klu::Result<()> {
/// use klu::Coo;
/// let a = Coo::new(2)?
///     .push(0, 0, 2.0)?
///     .push(0, 1, 1.0)?
///     .push(1, 0, 1.0)?
///     .push(1, 1, 3.0)?
///     .build()?;
/// assert_eq!(a.n(), 2);
/// assert_eq!(a.nnz(), 4);
/// # Ok(())
/// # }
/// ```
pub struct Coo<T: Scalar> {
    n: usize,
    rows: Vec<i32>,
    cols: Vec<i32>,
    vals: Vec<T>,
}

impl<T: Scalar> Coo<T> {
    /// Start building an `n x n` matrix (KLU requires a square matrix).
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 || n > i32::MAX as usize {
            return Err(Error::invalid("matrix dimension must be in 1..=i32::MAX"));
        }
        Ok(Self {
            n,
            rows: Vec::new(),
            cols: Vec::new(),
            vals: Vec::new(),
        })
    }

    /// Append one `(row, col, value)` triplet (consuming builder).
    pub fn push(mut self, row: usize, col: usize, value: T) -> Result<Self> {
        if row >= self.n || col >= self.n {
            return Err(Error::invalid("index out of bounds"));
        }
        self.rows.push(row as i32);
        self.cols.push(col as i32);
        self.vals.push(value);
        Ok(self)
    }

    /// Coalesce duplicates and return a canonical (column-major, sorted) CSC.
    pub fn build(self) -> Result<Csc<T>> {
        let n = self.n;
        let n_nz = self.vals.len();
        if n_nz > i32::MAX as usize {
            return Err(Error::invalid("too many nonzeros"));
        }

        // Count entries per column, then turn counts into column starts.
        let mut bp = vec![0i32; n + 1];
        for &j in &self.cols {
            bp[j as usize] += 1;
        }
        let mut cumsum = 0i32;
        for v in bp.iter_mut() {
            let t = *v;
            *v = cumsum;
            cumsum += t;
        }

        // Scatter into CSC order (rows not yet sorted within a column).
        let mut tmp_rows = vec![0i32; n_nz];
        let mut tmp_vals = vec![T::zero(); n_nz];
        let mut cursor = bp.clone();
        for k in 0..n_nz {
            let col = self.cols[k] as usize;
            let dest = cursor[col] as usize;
            tmp_rows[dest] = self.rows[k];
            tmp_vals[dest] = self.vals[k];
            cursor[col] += 1;
        }

        // Sort rows within each column and sum duplicates.
        let mut out_bp = Vec::with_capacity(n + 1);
        let mut out_bi: Vec<i32> = Vec::with_capacity(n_nz);
        let mut out_ax: Vec<T> = Vec::with_capacity(n_nz);
        out_bp.push(0);
        for col in 0..n {
            let lo = bp[col] as usize;
            let hi = bp[col + 1] as usize;
            let start = out_bi.len();
            let mut entries: Vec<(i32, T)> = (lo..hi).map(|k| (tmp_rows[k], tmp_vals[k])).collect();
            entries.sort_unstable_by_key(|&(row, _)| row);
            for (row, value) in entries {
                if out_bi.len() > start && *out_bi.last().unwrap() == row {
                    let last = out_ax.last_mut().unwrap();
                    *last = T::add(*last, value);
                } else {
                    out_bi.push(row);
                    out_ax.push(value);
                }
            }
            out_bp.push(out_bi.len() as i32);
        }

        Ok(Csc {
            n,
            bp: out_bp,
            bi: out_bi,
            ax: out_ax,
        })
    }
}

/// Canonical sparse matrix in CSC (compressed sparse column) format.
pub struct Csc<T: Scalar> {
    pub(crate) n: usize,
    pub(crate) bp: Vec<i32>,
    pub(crate) bi: Vec<i32>,
    pub(crate) ax: Vec<T>,
}

impl<T: Scalar> Csc<T> {
    /// Matrix dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of stored nonzeros.
    pub fn nnz(&self) -> usize {
        self.ax.len()
    }

    /// The values, in canonical CSC order (matches [`Symbolic::factor`] input).
    pub fn values(&self) -> &[T] {
        &self.ax
    }

    /// Mutable access to the values (same pattern).
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.ax
    }

    /// Overwrite all values (pattern unchanged). Length must match [`Csc::nnz`].
    pub fn set_values(&mut self, values: &[T]) -> Result<()> {
        if values.len() != self.ax.len() {
            return Err(Error::invalid("value count does not match pattern"));
        }
        self.ax.copy_from_slice(values);
        Ok(())
    }

    /// Max-norm residual `‖A x − b‖∞` (a small independent sanity check).
    /// Returns `NaN` if any component of the residual has a `NaN` magnitude.
    pub fn residual(&self, x: &[T], b: &[T]) -> Result<f64> {
        if x.len() != self.n || b.len() != self.n {
            return Err(Error::invalid("x and b must have length n"));
        }
        let mut r = vec![T::zero(); self.n];
        for (col, &xj) in x.iter().enumerate() {
            for k in self.bp[col] as usize..self.bp[col + 1] as usize {
                let i = self.bi[k] as usize;
                r[i] = T::add(r[i], T::mul(self.ax[k], xj));
            }
        }
        let mut norm = 0.0f64;
        for i in 0..self.n {
            let magnitude = T::abs(T::add(r[i], T::neg(b[i])));
            if magnitude.is_nan() {
                return Ok(f64::NAN);
            }
            norm = norm.max(magnitude);
        }
        Ok(norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::C64;

    #[test]
    fn residual_preserves_finite_and_infinite_norms() -> Result<()> {
        let a = Coo::new(2)?.push(0, 0, 2.0)?.push(1, 1, 3.0)?.build()?;
        assert_eq!(a.residual(&[1.0, 2.0], &[2.0, 6.0])?, 0.0);
        assert_eq!(a.residual(&[1.0, 2.0], &[5.0, 2.0])?, 4.0);
        assert_eq!(
            a.residual(&[f64::INFINITY, 2.0], &[2.0, 6.0])?,
            f64::INFINITY
        );
        Ok(())
    }

    #[test]
    fn real_residual_propagates_nan() -> Result<()> {
        for row in 0..2 {
            let mut a = Coo::new(2)?.push(0, 0, 1.0)?.push(1, 1, 1.0)?.build()?;
            let mut invalid = [1.0; 2];
            invalid[row] = f64::NAN;
            assert!(a.residual(&invalid, &[1.0; 2])?.is_nan());
            assert!(a.residual(&[1.0; 2], &invalid)?.is_nan());
            a.set_values(&invalid)?;
            assert!(a.residual(&[1.0; 2], &[1.0; 2])?.is_nan());
        }
        Ok(())
    }

    #[test]
    fn complex_residual_propagates_nan() -> Result<()> {
        let one = C64 { re: 1.0, im: 0.0 };
        for nan in [
            C64 {
                re: f64::NAN,
                im: 0.0,
            },
            C64 {
                re: 1.0,
                im: f64::NAN,
            },
        ] {
            for row in 0..2 {
                let mut a = Coo::new(2)?.push(0, 0, one)?.push(1, 1, one)?.build()?;
                let mut invalid = [one; 2];
                invalid[row] = nan;
                assert!(a.residual(&invalid, &[one; 2])?.is_nan());
                assert!(a.residual(&[one; 2], &invalid)?.is_nan());
                a.set_values(&invalid)?;
                assert!(a.residual(&[one; 2], &[one; 2])?.is_nan());
            }
        }
        Ok(())
    }
}
