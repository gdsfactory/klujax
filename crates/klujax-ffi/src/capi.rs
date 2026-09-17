//! Plain C-ABI symbols called directly from Python via `ctypes` (not XLA).
//!
//! These back the pure-Python `KLUSymbolic` / `KLUNumeric` handle classes once
//! the algorithm is wired up. In Stage 1 they are no-ops.

use core::ffi::c_char;

/// Return a static version string. The pointer is valid for the lifetime of the
/// process.
#[no_mangle]
pub extern "C" fn klujax_version() -> *const c_char {
    c"klujax-ffi 0.5.2-rs".as_ptr()
}

/// Free a single symbolic handle (raw pointer stored as `u64`).
///
/// Returns `0` on success. Stage 1: no-op.
#[no_mangle]
pub extern "C" fn klujax_free_symbolic(_raw: u64) -> i32 {
    0
}

/// Free `len` numeric handles stored in `ptrs`.
///
/// Returns `0` on success. Stage 1: no-op.
///
/// # Safety
/// If non-null, `ptrs` must point to `len` valid `u64` values.
#[no_mangle]
pub unsafe extern "C" fn klujax_free_numeric(_ptrs: *const u64, _len: usize) -> i32 {
    0
}
