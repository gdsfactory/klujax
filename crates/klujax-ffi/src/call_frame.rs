//! Helpers to decode an [`XLA_FFI_CallFrame`] into Rust slices.
//!
//! The C++ binding DSL does this implicitly; hand-rolling it requires matching
//! the argument/result ordering used by `jax.ffi.ffi_call`.

use crate::error::ErrorInfo;
use crate::xla_ffi::{
    XLA_FFI_Buffer, XLA_FFI_CallFrame, XLA_FFI_ARG_TYPE_BUFFER, XLA_FFI_RET_TYPE_BUFFER,
};
use core::ffi::c_int;
use core::slice;

/// Return the argument buffer at index `i`.
///
/// # Safety
///
/// `frame` must be a valid call frame from XLA.
pub unsafe fn arg_buffer(
    frame: *mut XLA_FFI_CallFrame,
    i: usize,
) -> Result<*const XLA_FFI_Buffer, ErrorInfo> {
    let args = unsafe { &(*frame).args };
    if args.args.is_null() || (i as i64) >= args.size {
        return Err(ErrorInfo::invalid(format!("missing argument {i}")));
    }
    if !args.types.is_null() && unsafe { *args.types.add(i) } != XLA_FFI_ARG_TYPE_BUFFER {
        return Err(ErrorInfo::invalid(format!("argument {i} is not a buffer")));
    }
    let ptr = unsafe { *args.args.add(i) } as *const XLA_FFI_Buffer;
    if ptr.is_null() {
        return Err(ErrorInfo::invalid(format!("argument {i} is null")));
    }
    Ok(ptr)
}

/// Return the result buffer at index `i`.
///
/// # Safety
///
/// `frame` must be a valid call frame from XLA.
pub unsafe fn ret_buffer(
    frame: *mut XLA_FFI_CallFrame,
    i: usize,
) -> Result<*mut XLA_FFI_Buffer, ErrorInfo> {
    let rets = unsafe { &(*frame).rets };
    if rets.rets.is_null() || (i as i64) >= rets.size {
        return Err(ErrorInfo::invalid(format!("missing result {i}")));
    }
    if !rets.types.is_null() && unsafe { *rets.types.add(i) } != XLA_FFI_RET_TYPE_BUFFER {
        return Err(ErrorInfo::invalid(format!("result {i} is not a buffer")));
    }
    let ptr = unsafe { *rets.rets.add(i) } as *mut XLA_FFI_Buffer;
    if ptr.is_null() {
        return Err(ErrorInfo::invalid(format!("result {i} is null")));
    }
    Ok(ptr)
}

/// The dimensions of a buffer.
///
/// # Safety
///
/// `buf` must point to a valid [`XLA_FFI_Buffer`], and its `dims` array
/// must have length `rank`.
pub unsafe fn dims<'a>(buf: *const XLA_FFI_Buffer) -> &'a [i64] {
    let buf = unsafe { &*buf };
    if buf.dims.is_null() || buf.rank <= 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(buf.dims, buf.rank as usize) }
    }
}

/// Number of elements in a buffer.
///
/// # Safety
///
/// See [`dims`].
pub unsafe fn element_count(buf: *const XLA_FFI_Buffer) -> usize {
    unsafe { dims(buf) }.iter().map(|&d| d as usize).product()
}

/// Check that `buf` has dtype `expected`.
///
/// # Safety
///
/// See [`dims`].
pub unsafe fn expect_dtype(
    buf: *const XLA_FFI_Buffer,
    expected: c_int,
    what: &str,
) -> Result<(), ErrorInfo> {
    let got = unsafe { (*buf).dtype };
    if got != expected {
        return Err(ErrorInfo::invalid(format!(
            "{what}: unexpected dtype {got}, expected {expected}"
        )));
    }
    Ok(())
}

/// Read-only view over a buffer's `f64` data.
///
/// # Safety
///
/// `buf` must have dtype `F64` and at least [`element_count`] elements.
pub unsafe fn as_f64<'a>(buf: *const XLA_FFI_Buffer) -> &'a [f64] {
    let n = unsafe { element_count(buf) };
    let data = unsafe { (*buf).data } as *const f64;
    if data.is_null() || n == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(data, n) }
    }
}

/// Mutable view over a buffer's `f64` data.
///
/// # Safety
///
/// `buf` must have dtype `F64` and be uniquely owned by the caller.
pub unsafe fn as_f64_mut<'a>(buf: *mut XLA_FFI_Buffer) -> &'a mut [f64] {
    let n = unsafe { element_count(buf) };
    let data = unsafe { (*buf).data } as *mut f64;
    if data.is_null() || n == 0 {
        &mut []
    } else {
        unsafe { slice::from_raw_parts_mut(data, n) }
    }
}

/// Read-only view over a buffer's `i32` data.
///
/// # Safety
///
/// `buf` must have dtype `S32`.
pub unsafe fn as_i32<'a>(buf: *const XLA_FFI_Buffer) -> &'a [i32] {
    let n = unsafe { element_count(buf) };
    let data = unsafe { (*buf).data } as *const i32;
    if data.is_null() || n == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(data, n) }
    }
}

/// Read-only view over a buffer's `u64` data.
///
/// # Safety
///
/// `buf` must have dtype `U64`.
pub unsafe fn as_u64<'a>(buf: *const XLA_FFI_Buffer) -> &'a [u64] {
    let n = unsafe { element_count(buf) };
    let data = unsafe { (*buf).data } as *const u64;
    if data.is_null() || n == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(data, n) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xla_ffi::{dtype, XLA_FFI_Args, XLA_FFI_Attrs, XLA_FFI_Rets, XLA_FFI_STAGE_EXECUTE};
    use core::ffi::c_void;
    use core::mem::size_of;
    use core::ptr::null_mut;

    fn empty_args() -> XLA_FFI_Args {
        XLA_FFI_Args {
            struct_size: size_of::<XLA_FFI_Args>(),
            extension_start: null_mut(),
            size: 0,
            types: null_mut(),
            args: null_mut(),
        }
    }

    fn empty_rets() -> XLA_FFI_Rets {
        XLA_FFI_Rets {
            struct_size: size_of::<XLA_FFI_Rets>(),
            extension_start: null_mut(),
            size: 0,
            types: null_mut(),
            rets: null_mut(),
        }
    }

    fn empty_attrs() -> XLA_FFI_Attrs {
        XLA_FFI_Attrs {
            struct_size: size_of::<XLA_FFI_Attrs>(),
            extension_start: null_mut(),
            size: 0,
            types: null_mut(),
            names: null_mut(),
            attrs: null_mut(),
        }
    }

    #[test]
    fn decodes_single_i32_arg() {
        let mut data = [1i32, 2, 3];
        let mut dims_arr = [3i64];
        let mut buf = XLA_FFI_Buffer {
            struct_size: size_of::<XLA_FFI_Buffer>(),
            extension_start: null_mut(),
            dtype: dtype::S32,
            data: data.as_mut_ptr() as *mut c_void,
            rank: 1,
            dims: dims_arr.as_mut_ptr(),
        };
        let mut types = [XLA_FFI_ARG_TYPE_BUFFER];
        let mut arg_ptrs = [&mut buf as *mut XLA_FFI_Buffer as *mut c_void];
        let mut args = empty_args();
        args.size = 1;
        args.types = types.as_mut_ptr();
        args.args = arg_ptrs.as_mut_ptr();

        let mut frame = XLA_FFI_CallFrame {
            struct_size: size_of::<XLA_FFI_CallFrame>(),
            extension_start: null_mut(),
            api: core::ptr::null(),
            ctx: null_mut(),
            stage: XLA_FFI_STAGE_EXECUTE,
            args,
            rets: empty_rets(),
            attrs: empty_attrs(),
            future: null_mut(),
        };

        unsafe {
            let b = arg_buffer(&mut frame, 0).unwrap();
            assert_eq!(dims(b), &[3]);
            assert_eq!(element_count(b), 3);
            assert_eq!(as_i32(b), &[1, 2, 3]);
            assert!(arg_buffer(&mut frame, 1).is_err());
            assert!(ret_buffer(&mut frame, 0).is_err());
        }
    }
}
