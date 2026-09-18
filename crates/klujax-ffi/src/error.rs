//! Error construction and panic containment for XLA FFI handlers.

use crate::xla_ffi::{
    error_code, XLA_FFI_Api_Version, XLA_FFI_CallFrame, XLA_FFI_Error, XLA_FFI_Error_Create_Args,
    XLA_FFI_Metadata_Extension, XLA_FFI_TypeId, XLA_FFI_EXTENSION_METADATA,
};
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
    // SAFETY: `frame` is non-null and valid per this fn's `# Safety`.
    let api = unsafe { (*frame).api };
    if api.is_null() {
        return null_mut();
    }
    // SAFETY: `api` is non-null; `XLA_FFI_Api`'s prefix layout is pinned in
    // `xla_ffi.rs` (and offset-tested there).
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
    // SAFETY: `create` is a valid XLA callback and `args` is fully initialised.
    unsafe { create(&mut args) }
}

/// Run `f`, converting `Ok(())` to success, `Err(ErrorInfo)` to an XLA error,
/// and any panic to an `INTERNAL` XLA error. Never unwinds across the FFI
/// boundary.
///
/// If the call frame carries the metadata extension (a registration-time probe
/// asking for the handler's supported API version), the metadata is populated
/// and success is returned without invoking `f`.
///
/// # Safety
///
/// `frame` must be a valid call frame passed by XLA (or null, in which case on
/// error a null pointer is returned).
pub unsafe fn guard<F>(frame: *mut XLA_FFI_CallFrame, f: F) -> *mut XLA_FFI_Error
where
    F: FnOnce() -> Result<(), ErrorInfo>,
{
    // SAFETY: `frame` may be null; `metadata_extension` handles that.
    if let Some(ext) = unsafe { metadata_extension(frame) } {
        // SAFETY: `ext` is a valid metadata extension (checked above).
        unsafe { populate_metadata(ext) };
        return null_mut();
    }
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => null_mut(),
        // SAFETY: `frame` is valid per this fn's `# Safety`.
        Ok(Err(info)) => unsafe { make_error(frame, info.code, &info.message) },
        // SAFETY: as above.
        Err(_) => unsafe { make_error(frame, error_code::INTERNAL, c"panic in XLA FFI handler") },
    }
}

/// Return the metadata extension if this call is a metadata probe.
unsafe fn metadata_extension(
    frame: *mut XLA_FFI_CallFrame,
) -> Option<*mut XLA_FFI_Metadata_Extension> {
    if frame.is_null() {
        return None;
    }
    // SAFETY: `frame` is non-null (checked) and valid per the caller.
    let ext = unsafe { (*frame).extension_start };
    // SAFETY: `ext` is non-null (checked); `XLA_FFI_Extension_Base`'s layout is
    // pinned in `xla_ffi.rs`.
    if ext.is_null() || unsafe { (*ext).type_ } != XLA_FFI_EXTENSION_METADATA {
        return None;
    }
    Some(ext as *mut XLA_FFI_Metadata_Extension)
}

/// Populate the metadata extension with the FFI API version we implement.
unsafe fn populate_metadata(ext: *mut XLA_FFI_Metadata_Extension) {
    // SAFETY: `ext` is a valid metadata extension per the caller.
    let meta = unsafe { (*ext).metadata };
    if meta.is_null() {
        return;
    }
    // SAFETY: `meta` is non-null and points to the out-param metadata struct
    // the XLA runtime owns for this registration probe.
    unsafe {
        (*meta).api_version = XLA_FFI_Api_Version {
            struct_size: size_of::<XLA_FFI_Api_Version>(),
            extension_start: null_mut(),
            major_version: 0,
            minor_version: 3,
        };
        (*meta).traits = 0;
        (*meta).state_type_id = XLA_FFI_TypeId { type_id: 0 };
    }
}

impl From<klu::Error> for ErrorInfo {
    /// Map a `klu` crate error onto an XLA error code.
    fn from(error: klu::Error) -> Self {
        let code = match error.kind {
            klu::ErrorKind::Internal => error_code::INTERNAL,
            klu::ErrorKind::InvalidArgument | klu::ErrorKind::Singular => {
                error_code::INVALID_ARGUMENT
            }
        };
        Self::new(code, error.message.into_bytes())
    }
}
