//! Safe wrapper around the raw `klu-sys` FFI.
//!
//! This is the single unsafe boundary for KLU calls in `klujax-ffi`: every
//! `unsafe` operation against `klu-sys` lives here, and callers get ordinary
//! `Result`-returning functions. `engine.rs` is therefore `forbid(unsafe_code)`.

#![allow(clippy::too_many_arguments)]

use crate::error::ErrorInfo;
use core::ffi::c_int;
use core::mem::MaybeUninit;
use klu_sys::{
    klu_analyze, klu_common, klu_defaults, klu_free_numeric, klu_free_symbolic, klu_numeric,
    klu_symbolic, KLU_OK,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Interleaved `complex<double>`, ABI-compatible with `double[2]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

/// Scalar types KLU operates on.
///
/// # Safety
/// `f64_ptr`/`f64_ptr_const` must reinterpret the slice as `f64`; sound for
/// `f64` and `C64` (`#[repr(C)]` of two `f64`). Implementations must use the
/// KLU routines matching `IS_COMPLEX`, honor their documented buffer lengths,
/// and return only KLU-allocated numeric objects for the supplied symbolic.
pub unsafe trait Scalar: crate::call_frame::Element + 'static {
    /// Whether this is a complex scalar type.
    const IS_COMPLEX: bool;
    /// The matching `XLA_FFI_DataType`.
    const DTYPE: c_int;
    /// Additive identity.
    fn zero() -> Self;
    /// `a * b`.
    fn mul(a: Self, b: Self) -> Self;
    /// `a + b`.
    fn add(a: Self, b: Self) -> Self;
    /// Raw pointer to the slice (reinterpreted as `f64`).
    fn f64_ptr(s: &mut [Self]) -> *mut f64;
    /// Const raw pointer to the slice (reinterpreted as `f64`).
    fn f64_ptr_const(s: &[Self]) -> *const f64;

    /// `klu_factor` / `klu_z_factor`.
    ///
    /// # Safety
    /// CSC arrays and handles must be valid (see the callers here).
    unsafe fn lu_factor(
        bp: &mut [i32],
        bi: &mut [i32],
        bx: &mut [Self],
        sym: *mut klu_symbolic,
        common: *mut klu_common,
    ) -> *mut klu_numeric;

    /// `klu_refactor` / `klu_z_refactor`.
    ///
    /// # Safety
    /// As [`Scalar::lu_factor`], with a valid `num`.
    unsafe fn lu_refactor(
        bp: &mut [i32],
        bi: &mut [i32],
        bx: &mut [Self],
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        common: *mut klu_common,
    ) -> c_int;

    /// `klu_solve` / `klu_z_solve`.
    ///
    /// # Safety
    /// Matching valid handles and a valid RHS buffer.
    unsafe fn lu_solve(
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        n_col: usize,
        n_rhs: usize,
        b: &mut [Self],
        common: *mut klu_common,
    ) -> c_int;

    /// `klu_tsolve` / `klu_z_tsolve` (plain transpose).
    ///
    /// # Safety
    /// As [`Scalar::lu_solve`].
    unsafe fn lu_tsolve(
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        n_col: usize,
        n_rhs: usize,
        b: &mut [Self],
        common: *mut klu_common,
    ) -> c_int;
}

// SAFETY: f64 has the real KLU scalar layout and dispatches to real routines.
unsafe impl Scalar for f64 {
    const IS_COMPLEX: bool = false;
    const DTYPE: c_int = crate::xla_ffi::dtype::F64;
    fn zero() -> Self {
        0.0
    }
    fn mul(a: Self, b: Self) -> Self {
        a * b
    }
    fn add(a: Self, b: Self) -> Self {
        a + b
    }
    fn f64_ptr(s: &mut [Self]) -> *mut f64 {
        s.as_mut_ptr()
    }
    fn f64_ptr_const(s: &[Self]) -> *const f64 {
        s.as_ptr()
    }
    unsafe fn lu_factor(
        bp: &mut [i32],
        bi: &mut [i32],
        bx: &mut [Self],
        sym: *mut klu_symbolic,
        common: *mut klu_common,
    ) -> *mut klu_numeric {
        // SAFETY: the caller supplies matching live handles and valid CSC/RHS slices.
        unsafe {
            klu_sys::klu_factor(
                bp.as_mut_ptr(),
                bi.as_mut_ptr(),
                Self::f64_ptr(bx),
                sym,
                common,
            )
        }
    }
    unsafe fn lu_refactor(
        bp: &mut [i32],
        bi: &mut [i32],
        bx: &mut [Self],
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        common: *mut klu_common,
    ) -> c_int {
        // SAFETY: the caller supplies matching live handles and valid CSC/RHS slices.
        unsafe {
            klu_sys::klu_refactor(
                bp.as_mut_ptr(),
                bi.as_mut_ptr(),
                Self::f64_ptr(bx),
                sym,
                num,
                common,
            )
        }
    }
    unsafe fn lu_solve(
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        n_col: usize,
        n_rhs: usize,
        b: &mut [Self],
        common: *mut klu_common,
    ) -> c_int {
        // SAFETY: the caller supplies matching live handles and valid CSC/RHS slices.
        unsafe {
            klu_sys::klu_solve(
                sym,
                num,
                n_col as c_int,
                n_rhs as c_int,
                Self::f64_ptr(b),
                common,
            )
        }
    }
    unsafe fn lu_tsolve(
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        n_col: usize,
        n_rhs: usize,
        b: &mut [Self],
        common: *mut klu_common,
    ) -> c_int {
        // SAFETY: the caller supplies matching live handles and valid CSC/RHS slices.
        unsafe {
            klu_sys::klu_tsolve(
                sym,
                num,
                n_col as c_int,
                n_rhs as c_int,
                Self::f64_ptr(b),
                common,
            )
        }
    }
}

// SAFETY: C64 is two repr(C) f64 fields and dispatches to complex routines.
unsafe impl Scalar for C64 {
    const IS_COMPLEX: bool = true;
    const DTYPE: c_int = crate::xla_ffi::dtype::C128;
    fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }
    fn mul(a: Self, b: Self) -> Self {
        Self {
            re: a.re * b.re - a.im * b.im,
            im: a.re * b.im + a.im * b.re,
        }
    }
    fn add(a: Self, b: Self) -> Self {
        Self {
            re: a.re + b.re,
            im: a.im + b.im,
        }
    }
    fn f64_ptr(s: &mut [Self]) -> *mut f64 {
        s.as_mut_ptr() as *mut f64
    }
    fn f64_ptr_const(s: &[Self]) -> *const f64 {
        s.as_ptr() as *const f64
    }
    unsafe fn lu_factor(
        bp: &mut [i32],
        bi: &mut [i32],
        bx: &mut [Self],
        sym: *mut klu_symbolic,
        common: *mut klu_common,
    ) -> *mut klu_numeric {
        // SAFETY: the caller supplies matching live handles and valid CSC/RHS slices.
        unsafe {
            klu_sys::klu_z_factor(
                bp.as_mut_ptr(),
                bi.as_mut_ptr(),
                Self::f64_ptr(bx),
                sym,
                common,
            )
        }
    }
    unsafe fn lu_refactor(
        bp: &mut [i32],
        bi: &mut [i32],
        bx: &mut [Self],
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        common: *mut klu_common,
    ) -> c_int {
        // SAFETY: the caller supplies matching live handles and valid CSC/RHS slices.
        unsafe {
            klu_sys::klu_z_refactor(
                bp.as_mut_ptr(),
                bi.as_mut_ptr(),
                Self::f64_ptr(bx),
                sym,
                num,
                common,
            )
        }
    }
    unsafe fn lu_solve(
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        n_col: usize,
        n_rhs: usize,
        b: &mut [Self],
        common: *mut klu_common,
    ) -> c_int {
        // SAFETY: the caller supplies matching live handles and valid CSC/RHS slices.
        unsafe {
            klu_sys::klu_z_solve(
                sym,
                num,
                n_col as c_int,
                n_rhs as c_int,
                Self::f64_ptr(b),
                common,
            )
        }
    }
    unsafe fn lu_tsolve(
        sym: *mut klu_symbolic,
        num: *mut klu_numeric,
        n_col: usize,
        n_rhs: usize,
        b: &mut [Self],
        common: *mut klu_common,
    ) -> c_int {
        // conj_solve = 0 -> plain transpose A^T (matches the C++ wrapper).
        // SAFETY: the caller supplies matching live handles and valid CSC/RHS slices.
        unsafe {
            klu_sys::klu_z_tsolve(
                sym,
                num,
                n_col as c_int,
                n_rhs as c_int,
                Self::f64_ptr(b),
                0,
                common,
            )
        }
    }
}

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

fn insert(entry: Entry) -> Result<u64, ErrorInfo> {
    let mut registry = registry()
        .lock()
        .map_err(|_| ErrorInfo::internal("handle registry poisoned"))?;
    registry.next = registry
        .next
        .checked_add(1)
        .ok_or_else(|| ErrorInfo::internal("handle IDs exhausted"))?;
    let id = registry.next;
    let numeric = matches!(entry, Entry::Numeric { .. });
    registry
        .entries
        .insert(id, (numeric, Arc::new(Mutex::new(entry))));
    Ok(id)
}

fn lookup(id: u64) -> Result<Arc<Mutex<Entry>>, ErrorInfo> {
    registry()
        .lock()
        .map_err(|_| ErrorInfo::internal("handle registry poisoned"))?
        .entries
        .get(&id)
        .map(|(_, entry)| Arc::clone(entry))
        .ok_or_else(|| ErrorInfo::invalid("invalid or closed KLU handle"))
}

fn lock(entry: &Mutex<Entry>) -> Result<std::sync::MutexGuard<'_, Entry>, ErrorInfo> {
    entry
        .lock()
        .map_err(|_| ErrorInfo::internal("KLU handle poisoned"))
}

fn pattern(n: usize, bp: &[i32], bi: &[i32]) -> Result<(), ErrorInfo> {
    if n == 0
        || n > i32::MAX as usize
        || bi.len() > i32::MAX as usize
        || bp.len() != n + 1
        || bp.first() != Some(&0)
        || bp.last().copied() != Some(bi.len() as i32)
        || bp.windows(2).any(|w| w[0] < 0 || w[0] > w[1])
        || bi.iter().any(|&i| i < 0 || i as usize >= n)
    {
        return Err(ErrorInfo::invalid("invalid CSC pattern"));
    }
    // KLU requires unique row indices in each column (sorting is optional).
    let mut seen = vec![usize::MAX; n];
    for col in 0..n {
        for &row in &bi[bp[col] as usize..bp[col + 1] as usize] {
            if seen[row as usize] == col {
                return Err(ErrorInfo::invalid("duplicate CSC entry"));
            }
            seen[row as usize] = col;
        }
    }
    Ok(())
}

type SymbolicParts<'a> = (*mut klu_symbolic, usize, &'a [i32], &'a [i32]);

fn symbolic(entry: &Entry) -> Result<SymbolicParts<'_>, ErrorInfo> {
    match entry {
        Entry::Symbolic { ptr, n, bp, bi } => Ok((*ptr, *n, bp, bi)),
        _ => Err(ErrorInfo::invalid("expected symbolic handle")),
    }
}

fn numeric<T: Scalar>(entry: &Entry, sym: u64) -> Result<*mut klu_numeric, ErrorInfo> {
    match entry {
        Entry::Numeric {
            ptr,
            symbolic,
            complex,
            ..
        } if *symbolic == sym && *complex == T::IS_COMPLEX => Ok(*ptr),
        _ => Err(ErrorInfo::invalid(
            "numeric handle does not match symbolic or scalar type",
        )),
    }
}

/// Number of columns of a live symbolic handle.
pub fn symbolic_n(sym: u64) -> Result<usize, ErrorInfo> {
    let entry = lookup(sym)?;
    let entry = lock(&entry)?;
    Ok(symbolic(&entry)?.1)
}

/// Analyze validated CSC arrays and return an owning opaque handle ID.
pub fn analyze(n_col: usize, bp: &mut [i32], bi: &mut [i32]) -> Result<u64, ErrorInfo> {
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
        return Err(ErrorInfo::internal("klu_analyze failed."));
    }
    let entry = Entry::Symbolic {
        ptr,
        n: n_col,
        bp: bp.to_vec(),
        bi: bi.to_vec(),
    };
    if !common.ok() {
        return Err(ErrorInfo::internal("klu_analyze failed."));
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
) -> Result<u64, ErrorInfo> {
    let owner = lookup(sym)?;
    let entry = lock(&owner)?;
    let (ptr, _, expected_bp, expected_bi) = symbolic(&entry)?;
    if bp != expected_bp || bi != expected_bi || bx.len() != bi.len() {
        return Err(ErrorInfo::invalid(
            "factor arrays do not match symbolic pattern",
        ));
    }
    // SAFETY: the locked owner keeps ptr live; CSC and values match its validated pattern.
    let ptr = unsafe { T::lu_factor(bp, bi, bx, ptr, common.ptr()) };
    if ptr.is_null() {
        return Err(ErrorInfo::invalid(
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
        return Err(ErrorInfo::invalid(
            "klu_factor/z_factor failed (singular matrix?)",
        ));
    }
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
) -> Result<(), ErrorInfo> {
    if sym == num {
        return Err(ErrorInfo::invalid(
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
        return Err(ErrorInfo::invalid(
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
        return Err(ErrorInfo::invalid(
            "klu_refactor/z_refactor failed (singular matrix?)",
        ));
    }
    *usable = true;
    Ok(())
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
) -> Result<(), ErrorInfo> {
    if sym == num {
        return Err(ErrorInfo::invalid(
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
        return Err(ErrorInfo::invalid(
            "numeric factorization invalid after failed refactor",
        ));
    }
    if n_col != n || n_rhs > i32::MAX as usize || n.checked_mul(n_rhs) != Some(b.len()) {
        return Err(ErrorInfo::invalid(
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
        return Err(ErrorInfo::invalid("klu_solve/tsolve failed"));
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

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;

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

    fn numeric_outlives_symbolic<T: Scalar + std::fmt::Debug + PartialEq>(mut values: [T; 2]) {
        let mut common = Common::new();
        let (mut bp, mut bi) = ([0, 1, 2], [0, 1]);
        let sym = analyze(2, &mut bp, &mut bi).unwrap();
        let num = factor(&mut common, &mut bp, &mut bi, &mut values, sym).unwrap();
        let weak_sym = Arc::downgrade(&lookup(sym).unwrap());
        let weak_num = Arc::downgrade(&lookup(num).unwrap());

        free_symbolic(&mut common, sym);
        assert!(lookup(sym).is_err());
        assert!(weak_sym.upgrade().is_none());
        assert!(weak_num.upgrade().is_some());

        for transpose in [false, true] {
            let mut rhs = values;
            let error = solve(&mut common, sym, num, 2, 1, &mut rhs, transpose).unwrap_err();
            assert_eq!(
                error.message.to_str().unwrap(),
                "invalid or closed KLU handle"
            );
            assert_eq!(rhs, values);
        }

        free_numeric(&mut common, num);
        assert!(lookup(num).is_err());
        assert!(weak_num.upgrade().is_none());
        free_numeric(&mut common, num);
    }

    #[test]
    fn real_numeric_can_be_freed_after_symbolic_and_rejects_solve() {
        numeric_outlives_symbolic([2.0, 4.0]);
    }

    #[test]
    fn complex_numeric_can_be_freed_after_symbolic_and_rejects_solve() {
        numeric_outlives_symbolic([C64 { re: 2.0, im: 1.0 }, C64 { re: 4.0, im: -1.0 }]);
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
        crate::capi::klujax_free_symbolic(sym);
        assert!(symbolic_n(sym).is_err());
        assert!(weak.upgrade().is_some());
        assert_eq!(symbolic(&lock(&owner).unwrap()).unwrap().1, 2);
        drop(owner);
        assert!(weak.upgrade().is_none());
        let handles = [num, num, u64::MAX];
        // SAFETY: handles is a live array of initialized u64 values.
        unsafe {
            crate::capi::klujax_free_numeric(handles.as_ptr(), handles.len());
        }
        assert!(lookup(num).is_err());
        crate::capi::klujax_free_symbolic(sym);
        let (fresh_sym, fresh_num) = diagonal();
        assert_ne!(fresh_sym, sym);
        assert_ne!(fresh_num, num);
        free_numeric(&mut Common::new(), fresh_num);
        free_symbolic(&mut Common::new(), fresh_sym);
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
        free_numeric(&mut Common::new(), num);
        free_symbolic(&mut Common::new(), sym);
        worker.join().unwrap();
    }
}
