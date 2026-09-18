//! Plain C-ABI cleanup symbols used by the Python handle owners.

use core::ffi::c_char;

/// Return a version string valid for the lifetime of the process.
#[no_mangle]
pub extern "C" fn klujax_version() -> *const c_char {
    c"klujax-ffi 0.5.2-rs".as_ptr()
}

/// Release a symbolic ID. Invalid or already released IDs are harmless.
#[no_mangle]
pub extern "C" fn klujax_free_symbolic(raw: u64) -> i32 {
    crate::engine::free_symbolic_raw(raw)
}

/// Release numeric IDs. Invalid or already released IDs are harmless.
///
/// # Safety
/// For nonzero `len`, `ptrs` must point to `len` initialized, aligned `u64`
/// values, readable for this call. The slice must fit in `isize::MAX` bytes.
#[no_mangle]
pub unsafe extern "C" fn klujax_free_numeric(ptrs: *const u64, len: usize) -> i32 {
    if len == 0 {
        return 0;
    }
    if ptrs.is_null() {
        return -1;
    }
    // SAFETY: the caller guarantees a valid slice for the duration of this call.
    let handles = unsafe { core::slice::from_raw_parts(ptrs, len) };
    crate::engine::free_numeric_raw(handles)
}
