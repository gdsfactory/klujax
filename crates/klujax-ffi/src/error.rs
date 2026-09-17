//! Error construction and panic containment for XLA FFI handlers.

use crate::xla_ffi::{error_code, XLA_FFI_CallFrame, XLA_FFI_Error, XLA_FFI_Error_Create_Args};
use core::ffi::{c_int, CStr};
use core::mem::size_of;
use core::ptr::null_mut;
use std::ffi::CString;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// A status code plus an owned message, converted to an `XLA_FFI_Error` at the
/// FFI boundary.
#[derive(Debug, Clone)]
pub struct ErrorInfo {
    /// `XLA_FFI_Error_Code_*`.
    pub code: c_int,
    /// Owned, NUL-terminated message.
    pub message: CString,
}

impl ErrorInfo {
    /// Construct from a code and message (interior NULs are stripped).
    pub fn new(code: c_int, message: impl Into<Vec<u8>>) -> Self {
        let bytes = message.into();
        // `error_message` must not contain interior NULs.
        let bytes: Vec<u8> = bytes.into_iter().filter(|&b| b != 0).collect();
        let message = CString::new(bytes).unwrap_or_default();
        Self { code, message }
    }

    /// `INVALID_ARGUMENT` error.
    pub fn invalid(message: impl Into<Vec<u8>>) -> Self {
        Self::new(error_code::INVALID_ARGUMENT, message)
    }

    /// `INTERNAL` error.
    pub fn internal(message: impl Into<Vec<u8>>) -> Self {
        Self::new(error_code::INTERNAL, message)
    }

    /// `UNIMPLEMENTED` error (used by the Stage 1 handler stubs).
    pub fn unimplemented(message: impl Into<Vec<u8>>) -> Self {
        Self::new(error_code::UNIMPLEMENTED, message)
    }
}

/// Allocate an `XLA_FFI_Error` through the XLA runtime.
///
/// # Safety
///
/// `frame` must be the non-null call frame passed by XLA, and its `api`
/// pointer must be valid.
pub unsafe fn make_error(
    frame: *const XLA_FFI_CallFrame,
    code: c_int,
    message: &CStr,
) -> *mut XLA_FFI_Error {
    if frame.is_null() {
        return null_mut();
    }
    let api = unsafe { (*frame).api };
    if api.is_null() {
        return null_mut();
    }
    let create = match unsafe { (*api).XLA_FFI_Error_Create } {
        Some(create) => create,
        None => return null_mut(),
    };
    let mut args = XLA_FFI_Error_Create_Args {
        struct_size: size_of::<XLA_FFI_Error_Create_Args>(),
        extension_start: null_mut(),
        message: message.as_ptr(),
        errc: code,
    };
    unsafe { create(&mut args) }
}

/// Run `f`, converting `Ok(())` to success, `Err(ErrorInfo)` to an XLA error,
/// and any panic to an `INTERNAL` XLA error. Never unwinds across the FFI
/// boundary.
///
/// # Safety
///
/// `frame` must be a valid call frame passed by XLA (or null, in which case on
/// error a null pointer is returned).
pub unsafe fn guard<F>(frame: *mut XLA_FFI_CallFrame, f: F) -> *mut XLA_FFI_Error
where
    F: FnOnce() -> Result<(), ErrorInfo>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => null_mut(),
        Ok(Err(info)) => unsafe { make_error(frame, info.code, &info.message) },
        Err(_) => unsafe { make_error(frame, error_code::INTERNAL, c"panic in XLA FFI handler") },
    }
}
