//! Low-level, handle-ID based KLU operations.
//!
//! This module is the single unsafe boundary around `klu-sys`. Every `unsafe`
//! operation against the C library lives here. Handles are opaque, never-reused
//! `u64` IDs into a process-global registry; the ergonomic types in
//! [`crate::Symbolic`]/[`crate::Numeric`] and the XLA layer in `klujax-ffi` are
//! both built on top of these functions.

use crate::error::{Error, Result};
use crate::scalar::Scalar;
use core::ffi::c_int;
use core::mem::MaybeUninit;
use klu_sys::{
    klu_analyze, klu_common, klu_defaults, klu_free_numeric, klu_free_symbolic, klu_numeric,
    klu_symbolic, KLU_OK,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Safe owner of a `klu_common` control/status block.
pub struct Common(klu_common);

impl Default for Common {
    fn default() -> Self {
        Self::new()
    }
}

impl Common {
    /// A `klu_common` initialised by `klu_defaults`.
    pub fn new() -> Self {
        // KLU leaves singular_col untouched. Zero every field before defaults;
        // integers, floats, raw pointers and Option<extern fn> all admit zero.
        let mut common = MaybeUninit::<klu_common>::zeroed();
        // SAFETY: common is writable and fully zero-initialized, including the
        // fields klu_defaults does not touch. Defaults preserves valid fields.
        unsafe {
            klu_defaults(common.as_mut_ptr());
            Self(common.assume_init())
        }
    }

    /// Whether the last operation succeeded (`common.status >= KLU_OK`).
    pub fn ok(&self) -> bool {
        self.0.status >= KLU_OK
    }

    /// The raw KLU status code from the last operation.
    pub fn status(&self) -> c_int {
        self.0.status
    }

    fn ptr(&mut self) -> *mut klu_common {
        &mut self.0
    }
}

/// Registry entries own native allocations. Only opaque, never-reused IDs cross
/// the Python/XLA boundary. Per-entry locks serialize access to each KLU object;
/// unrelated symbolic analyses can execute concurrently.
enum Entry {
    Symbolic {
        ptr: *mut klu_symbolic,
        n: usize,
        bp: Vec<i32>,
        bi: Vec<i32>,
    },
    Numeric {
        ptr: *mut klu_numeric,
        // Identity only: numeric cleanup is independent of the symbolic's
        // lifetime. Solve/refactor must look up and retain both live entries.
        symbolic: u64,
        complex: bool,
        usable: bool,
    },
}

// SAFETY: allocations have no thread affinity. Entries are private, and all
// access to their pointers (including free) is protected by their owning Mutex.
unsafe impl Send for Entry {}

impl Drop for Entry {
    fn drop(&mut self) {
        let mut common = Common::new();
        // SAFETY: this entry uniquely owns its allocation; the last Arc has
        // gone away, so no operation can still access it. Match scalar layout.
        unsafe {
            match self {
                Self::Symbolic { ptr, .. } => {
                    klu_free_symbolic(ptr, common.ptr());
                }
                Self::Numeric { ptr, complex, .. } => {
                    if *complex {
                        klu_sys::klu_z_free_numeric(ptr, common.ptr());
                    } else {
                        klu_free_numeric(ptr, common.ptr());
                    }
                }
            }
        }
    }
}

#[derive(Default)]
struct Registry {
    next: u64,
    entries: HashMap<u64, (bool, Arc<Mutex<Entry>>)>,
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Registry::default()))
}

fn insert(entry: Entry) -> Result<u64> {
    let mut registry = registry()
        .lock()
        .map_err(|_| Error::internal("handle registry poisoned"))?;
    registry.next = registry
        .next
        .checked_add(1)
        .ok_or_else(|| Error::internal("handle IDs exhausted"))?;
    let id = registry.next;
    let numeric = matches!(entry, Entry::Numeric { .. });
    registry
        .entries
        .insert(id, (numeric, Arc::new(Mutex::new(entry))));
    Ok(id)
}

fn lookup(id: u64) -> Result<Arc<Mutex<Entry>>> {
    registry()
        .lock()
        .map_err(|_| Error::internal("handle registry poisoned"))?
        .entries
        .get(&id)
        .map(|(_, entry)| Arc::clone(entry))
        .ok_or_else(|| Error::invalid("invalid or closed KLU handle"))
}

fn lock(entry: &Mutex<Entry>) -> Result<std::sync::MutexGuard<'_, Entry>> {
    entry
        .lock()
        .map_err(|_| Error::internal("KLU handle poisoned"))
}

fn pattern(n: usize, bp: &[i32], bi: &[i32]) -> Result<()> {
    if n == 0
        || n > i32::MAX as usize
        || bi.len() > i32::MAX as usize
        || bp.len() != n + 1
        || bp.first() != Some(&0)
        || bp.last().copied() != Some(bi.len() as i32)
        || bp.windows(2).any(|w| w[0] < 0 || w[0] > w[1])
        || bi.iter().any(|&i| i < 0 || i as usize >= n)
    {
        return Err(Error::invalid("invalid CSC pattern"));
    }
    // KLU requires unique row indices in each column (sorting is optional).
    let mut seen = vec![usize::MAX; n];
    for col in 0..n {
        for &row in &bi[bp[col] as usize..bp[col + 1] as usize] {
            if seen[row as usize] == col {
                return Err(Error::invalid("duplicate CSC entry"));
            }
            seen[row as usize] = col;
        }
    }
    Ok(())
}

type SymbolicParts<'a> = (*mut klu_symbolic, usize, &'a [i32], &'a [i32]);

fn symbolic(entry: &Entry) -> Result<SymbolicParts<'_>> {
    match entry {
        Entry::Symbolic { ptr, n, bp, bi } => Ok((*ptr, *n, bp, bi)),
        _ => Err(Error::invalid("expected symbolic handle")),
    }
}

fn numeric<T: Scalar>(entry: &Entry, sym: u64) -> Result<*mut klu_numeric> {
    match entry {
        Entry::Numeric {
            ptr,
            symbolic,
            complex,
            ..
        } if *symbolic == sym && *complex == T::IS_COMPLEX => Ok(*ptr),
        _ => Err(Error::invalid(
            "numeric handle does not match symbolic or scalar type",
        )),
    }
}

/// Number of columns of a live symbolic handle.
pub fn symbolic_n(sym: u64) -> Result<usize> {
    let entry = lookup(sym)?;
    let entry = lock(&entry)?;
    Ok(symbolic(&entry)?.1)
}

/// The canonical CSC pattern `(bp, bi)` stored with a symbolic handle.
pub(crate) fn symbolic_pattern(sym: u64) -> Result<(usize, Vec<i32>, Vec<i32>)> {
    let entry = lookup(sym)?;
    let entry = lock(&entry)?;
    let (_, n, bp, bi) = symbolic(&entry)?;
    Ok((n, bp.to_vec(), bi.to_vec()))
}

/// Analyze validated CSC arrays and return an owning opaque handle ID.
pub fn analyze(n_col: usize, bp: &mut [i32], bi: &mut [i32]) -> Result<u64> {
    pattern(n_col, bp, bi)?;
    let mut common = Common::new();
    // SAFETY: pattern checked dimensions, lengths, indices and uniqueness.
    let ptr = unsafe {
        klu_analyze(
            n_col as c_int,
            bp.as_mut_ptr(),
            bi.as_mut_ptr(),
            common.ptr(),
        )
    };
    if ptr.is_null() {
        return Err(Error::internal("klu_analyze failed."));
    }
    let entry = Entry::Symbolic {
        ptr,
        n: n_col,
        bp: bp.to_vec(),
        bi: bi.to_vec(),
    };
    if !common.ok() {
        return Err(Error::internal("klu_analyze failed."));
    }
    insert(entry)
}

/// Factor a matrix with exactly the analyzed pattern and matching value count.
pub fn factor<T: Scalar>(
    common: &mut Common,
    bp: &mut [i32],
    bi: &mut [i32],
    bx: &mut [T],
    sym: u64,
) -> Result<u64> {
    let owner = lookup(sym)?;
    let entry = lock(&owner)?;
    let (ptr, _, expected_bp, expected_bi) = symbolic(&entry)?;
    if bp != expected_bp || bi != expected_bi || bx.len() != bi.len() {
        return Err(Error::invalid(
            "factor arrays do not match symbolic pattern",
        ));
    }
    // SAFETY: the locked owner keeps ptr live; CSC and values match its validated pattern.
    let ptr = unsafe { T::lu_factor(bp, bi, bx, ptr, common.ptr()) };
    if ptr.is_null() {
        return Err(Error::singular(
            "klu_factor/z_factor failed (singular matrix?)",
        ));
    }
    let numeric = Entry::Numeric {
        ptr,
        symbolic: sym,
        complex: T::IS_COMPLEX,
        usable: true,
    };
    if !common.ok() {
        return Err(Error::singular(
            "klu_factor/z_factor failed (singular matrix?)",
        ));
    }
    insert(numeric)
}

/// Factor values for a symbolic handle using its stored (canonical) pattern.
///
/// `ax` must be in the canonical CSC order returned by [`symbolic_pattern`].
pub fn factor_values<T: Scalar>(common: &mut Common, ax: &[T], sym: u64) -> Result<u64> {
    let (n, mut bp, mut bi) = symbolic_pattern(sym)?;
    if ax.len() != bi.len() {
        return Err(Error::invalid("matrix value count does not match pattern"));
    }
    let mut bx = ax.to_vec();
    let root = lookup(sym)?;
    let entry = lock(&root)?;
    let (ptr, _, _, _) = symbolic(&entry)?;
    // SAFETY: the locked symbolic owner keeps ptr live; pattern and bx match it.
    let num = unsafe { T::lu_factor(&mut bp, &mut bi, &mut bx, ptr, common.ptr()) };
    if num.is_null() {
        return Err(Error::singular(
            "klu_factor/z_factor failed (singular matrix?)",
        ));
    }
    let numeric = Entry::Numeric {
        ptr: num,
        symbolic: sym,
        complex: T::IS_COMPLEX,
        usable: true,
    };
    if !common.ok() {
        return Err(Error::singular(
            "klu_factor/z_factor failed (singular matrix?)",
        ));
    }
    let _ = n;
    insert(numeric)
}

/// Recompute a matching numeric factorization in place.
pub fn refactor<T: Scalar>(
    common: &mut Common,
    bp: &mut [i32],
    bi: &mut [i32],
    bx: &mut [T],
    sym: u64,
    num: u64,
) -> Result<()> {
    if sym == num {
        return Err(Error::invalid(
            "expected distinct symbolic and numeric handles",
        ));
    }
    let sym_owner = lookup(sym)?;
    let num_owner = lookup(num)?;
    let sym_entry = lock(&sym_owner)?;
    let (sym_ptr, _, expected_bp, expected_bi) = symbolic(&sym_entry)?;
    let mut num_entry = lock(&num_owner)?;
    let num_ptr = numeric::<T>(&num_entry, sym)?;
    if bp != expected_bp || bi != expected_bi || bx.len() != bi.len() {
        return Err(Error::invalid(
            "refactor arrays do not match symbolic pattern",
        ));
    }
    let Entry::Numeric { usable, .. } = &mut *num_entry else {
        unreachable!()
    };
    *usable = false;
    // SAFETY: both owners are locked and alive, with matching type, pattern and values.
    let status = unsafe { T::lu_refactor(bp, bi, bx, sym_ptr, num_ptr, common.ptr()) };
    if status == 0 || !common.ok() {
        return Err(Error::singular(
            "klu_refactor/z_refactor failed (singular matrix?)",
        ));
    }
    *usable = true;
    Ok(())
}

/// Re-factor values for a symbolic/numeric pair using the stored pattern.
pub fn refactor_values<T: Scalar>(common: &mut Common, ax: &[T], sym: u64, num: u64) -> Result<()> {
    let (_, mut bp, mut bi) = symbolic_pattern(sym)?;
    if ax.len() != bi.len() {
        return Err(Error::invalid("matrix value count does not match pattern"));
    }
    let mut bx = ax.to_vec();
    refactor(common, &mut bp, &mut bi, &mut bx, sym, num)
}

/// Solve using live, matching handles and an exactly sized RHS buffer.
pub fn solve<T: Scalar>(
    common: &mut Common,
    sym: u64,
    num: u64,
    n_col: usize,
    n_rhs: usize,
    b: &mut [T],
    transpose: bool,
) -> Result<()> {
    if sym == num {
        return Err(Error::invalid(
            "expected distinct symbolic and numeric handles",
        ));
    }
    let sym_owner = lookup(sym)?;
    let num_owner = lookup(num)?;
    let sym_entry = lock(&sym_owner)?;
    let (sym_ptr, n, _, _) = symbolic(&sym_entry)?;
    let num_entry = lock(&num_owner)?;
    let num_ptr = numeric::<T>(&num_entry, sym)?;
    if matches!(*num_entry, Entry::Numeric { usable: false, .. }) {
        return Err(Error::invalid(
            "numeric factorization invalid after failed refactor",
        ));
    }
    if n_col != n || n_rhs > i32::MAX as usize || n.checked_mul(n_rhs) != Some(b.len()) {
        return Err(Error::invalid(
            "RHS dimensions do not match symbolic matrix",
        ));
    }
    // SAFETY: locked matching owners ensure validity and exclusive KLU access;
    // checked dimensions guarantee every RHS column has n initialized scalars.
    let status = unsafe {
        if transpose {
            T::lu_tsolve(sym_ptr, num_ptr, n, n_rhs, b, common.ptr())
        } else {
            T::lu_solve(sym_ptr, num_ptr, n, n_rhs, b, common.ptr())
        }
    };
    if status == 0 || !common.ok() {
        return Err(Error::invalid("klu_solve/tsolve failed"));
    }
    Ok(())
}

fn remove(id: u64, numeric: bool) {
    // Drop outside the registry lock. Any in-flight operation retains an Arc,
    // so removal prevents new lookups without freeing memory still in use.
    let removed = {
        let mut registry = registry().lock().unwrap_or_else(|e| e.into_inner());
        let matches = registry
            .entries
            .get(&id)
            .is_some_and(|(kind, _)| *kind == numeric);
        if matches {
            registry.entries.remove(&id)
        } else {
            None
        }
    };
    drop(removed);
}

/// Release a numeric ID. Repeated frees and invalid IDs are harmless.
pub fn free_numeric(_common: &mut Common, num: u64) {
    remove(num, true);
}

/// Release a symbolic ID. Repeated frees and invalid IDs are harmless.
pub fn free_symbolic(_common: &mut Common, sym: u64) {
    remove(sym, false);
}

/// Release a numeric ID without a `Common` (for RAII owners).
pub fn release_numeric(num: u64) {
    remove(num, true);
}

/// Release a symbolic ID without a `Common` (for RAII owners).
pub fn release_symbolic(sym: u64) {
    remove(sym, false);
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use crate::scalar::C64;

    fn diagonal() -> (u64, u64) {
        let (mut bp, mut bi, mut bx) = ([0, 1, 2], [0, 1], [2.0, 4.0]);
        let sym = analyze(2, &mut bp, &mut bi).unwrap();
        let num = factor(&mut Common::new(), &mut bp, &mut bi, &mut bx, sym).unwrap();
        (sym, num)
    }

    #[test]
    fn defaults_initialize_singular_column() {
        assert_eq!(Common::new().0.singular_col, 0);
    }

    #[test]
    fn rejects_fabricated_handles_and_invalid_csc() {
        let mut common = Common::new();
        assert!(symbolic_n(u64::MAX).is_err());
        assert!(factor(&mut common, &mut [0, 1], &mut [0], &mut [1.0], u64::MAX).is_err());
        assert!(analyze(2, &mut [], &mut []).is_err());
        assert!(analyze(2, &mut [0, 3, 1], &mut [0]).is_err());
        assert!(analyze(2, &mut [0, 1, 2], &mut [0, 2]).is_err());
        assert!(analyze(2, &mut [0, 2, 2], &mut [0, 0]).is_err());
        free_numeric(&mut common, u64::MAX);
        free_symbolic(&mut common, u64::MAX);
    }

    #[test]
    fn validates_all_arrays_handle_kinds_and_scalar_types() {
        let (sym, num) = diagonal();
        let (other_sym, other_num) = diagonal();
        let mut common = Common::new();
        assert!(symbolic_n(num).is_err());
        assert!(factor(&mut common, &mut [0, 1, 2], &mut [0, 1], &mut [1.0], sym).is_err());
        assert!(factor(
            &mut common,
            &mut [0, 1, 2],
            &mut [1, 0],
            &mut [1.0, 2.0],
            sym
        )
        .is_err());
        assert!(solve(&mut common, sym, num, 2, 1, &mut [1.0], false).is_err());
        assert!(solve(&mut common, sym, num, 1, 2, &mut [1.0, 2.0], false).is_err());
        assert!(solve(&mut common, sym, sym, 2, 1, &mut [1.0, 2.0], false).is_err());
        assert!(solve(&mut common, other_sym, num, 2, 1, &mut [1.0, 2.0], false).is_err());
        assert!(solve(&mut common, sym, num, 2, 1, &mut [C64::zero(); 2], false).is_err());
        assert!(refactor(
            &mut common,
            &mut [0, 1, 2],
            &mut [0, 1],
            &mut [C64::zero(); 2],
            sym,
            num
        )
        .is_err());
        for id in [num, other_num] {
            free_numeric(&mut common, id);
        }
        for id in [sym, other_sym] {
            free_symbolic(&mut common, id);
        }
    }

    #[test]
    fn failed_refactor_invalidates_partial_numeric() {
        let mut common = Common::new();
        let (mut bp, mut bi) = ([0, 2, 4], [0, 1, 0, 1]);
        let sym = analyze(2, &mut bp, &mut bi).unwrap();
        let num = factor(
            &mut common,
            &mut bp,
            &mut bi,
            &mut [2.0, 1.0, 1.0, 4.0],
            sym,
        )
        .unwrap();
        assert!(refactor(&mut common, &mut bp, &mut bi, &mut [0.0; 4], sym, num).is_err());
        assert!(solve(&mut common, sym, num, 2, 1, &mut [1.0, 2.0], false).is_err());
        free_numeric(&mut common, num);
        free_symbolic(&mut common, sym);
    }

    #[test]
    fn cleanup_releases_ids_once_and_inflight_owners_keep_allocations_alive() {
        let (sym, num) = diagonal();
        let owner = lookup(sym).unwrap();
        let weak = Arc::downgrade(&owner);
        release_symbolic(sym);
        assert!(symbolic_n(sym).is_err());
        assert!(weak.upgrade().is_some());
        assert_eq!(symbolic(&lock(&owner).unwrap()).unwrap().1, 2);
        drop(owner);
        assert!(weak.upgrade().is_none());
        release_numeric(num);
        release_numeric(num);
        assert!(lookup(num).is_err());
        release_symbolic(sym);
        let (fresh_sym, fresh_num) = diagonal();
        assert_ne!(fresh_sym, sym);
        assert_ne!(fresh_num, num);
        release_numeric(fresh_num);
        release_symbolic(fresh_sym);
    }

    #[test]
    fn concurrent_solve_and_close_never_access_freed_memory() {
        let (sym, num) = diagonal();
        let worker = std::thread::spawn(move || {
            for _ in 0..100 {
                let mut rhs = [2.0, 8.0];
                if solve(&mut Common::new(), sym, num, 2, 1, &mut rhs, false).is_ok() {
                    assert_eq!(rhs, [1.0, 2.0]);
                }
            }
        });
        release_numeric(num);
        release_symbolic(sym);
        worker.join().unwrap();
    }

    fn numeric_outlives_symbolic<T: Scalar + std::fmt::Debug + PartialEq>(mut values: [T; 2]) {
        let mut common = Common::new();
        let (mut bp, mut bi) = ([0, 1, 2], [0, 1]);
        let sym = analyze(2, &mut bp, &mut bi).unwrap();
        let num = factor(&mut common, &mut bp, &mut bi, &mut values, sym).unwrap();
        let weak_sym = Arc::downgrade(&lookup(sym).unwrap());
        let weak_num = Arc::downgrade(&lookup(num).unwrap());

        release_symbolic(sym);
        assert!(lookup(sym).is_err());
        assert!(weak_sym.upgrade().is_none());
        assert!(weak_num.upgrade().is_some());

        for transpose in [false, true] {
            let mut rhs = values;
            let error = solve(&mut common, sym, num, 2, 1, &mut rhs, transpose).unwrap_err();
            assert_eq!(error.message, "invalid or closed KLU handle");
            assert_eq!(rhs, values);
        }

        release_numeric(num);
        assert!(lookup(num).is_err());
        assert!(weak_num.upgrade().is_none());
        release_numeric(num);
    }

    #[test]
    fn real_numeric_can_be_freed_after_symbolic_and_rejects_solve() {
        numeric_outlives_symbolic([2.0, 4.0]);
    }

    #[test]
    fn complex_numeric_can_be_freed_after_symbolic_and_rejects_solve() {
        numeric_outlives_symbolic([C64 { re: 2.0, im: 1.0 }, C64 { re: 4.0, im: -1.0 }]);
    }
}
