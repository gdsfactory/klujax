//! Borrowed, lifetime-safe decoding of an [`XLA_FFI_CallFrame`].
//!
//! The C++ binding DSL does this implicitly; hand-rolling it means building
//! slices from raw pointers. To keep that contained, all pointer work lives
//! here behind [`Frame`]/[`Buffer`]/[`BufferMut`], and every decoded slice is
//! tied to the frame's lifetime `'a` so it cannot outlive the handler call.

use crate::error::ErrorInfo;
use crate::xla_ffi::{
    XLA_FFI_Buffer, XLA_FFI_CallFrame, XLA_FFI_ARG_TYPE_BUFFER, XLA_FFI_RET_TYPE_BUFFER,
};
use core::ffi::{c_int, c_void};
use core::marker::PhantomData;
use core::slice;

/// A borrowed view of the XLA call frame.
///
/// `'a` is the handler-invocation lifetime; buffers decoded from it borrow `'a`
/// and therefore cannot escape the handler.
pub struct Frame<'a> {
    raw: *mut XLA_FFI_CallFrame,
    _marker: PhantomData<&'a XLA_FFI_CallFrame>,
}

impl<'a> Frame<'a> {
    /// Wrap the raw call frame.
    ///
    /// # Safety
    ///
    /// `raw` must be the valid, non-null call frame XLA passed to this handler,
    /// and it must stay alive for the whole (short) `'a`.
    pub unsafe fn from_raw(raw: *mut XLA_FFI_CallFrame) -> Self {
        Self {
            raw,
            _marker: PhantomData,
        }
    }

    /// The raw pointer, for the error/panic machinery in [`crate::error`].
    pub fn as_ptr(&self) -> *mut XLA_FFI_CallFrame {
        self.raw
    }

    /// Decode argument buffer `i`.
    ///
    /// # Safety
    ///
    /// The [`Frame::from_raw`] contract must hold (it does inside handlers).
    pub unsafe fn arg_buffer(&self, i: usize) -> Result<Buffer<'a>, ErrorInfo> {
        // SAFETY: `self.raw` is valid per `Frame::from_raw`'s contract.
        let args = unsafe { &(*self.raw).args };
        if args.args.is_null() || (i as i64) >= args.size {
            return Err(ErrorInfo::invalid(format!("missing argument {i}")));
        }
        // SAFETY: `args.types` has `args.size` entries and `i < size` (checked).
        if !args.types.is_null() && unsafe { *args.types.add(i) } != XLA_FFI_ARG_TYPE_BUFFER {
            return Err(ErrorInfo::invalid(format!("argument {i} is not a buffer")));
        }
        // SAFETY: `args.args` has `args.size` entries and `i < size` (checked).
        let ptr = unsafe { *args.args.add(i) } as *const XLA_FFI_Buffer;
        // SAFETY: `ptr` is the buffer XLA provided for argument `i`, valid for
        // the call-frame lifetime.
        unsafe { Buffer::from_ptr(ptr, "argument") }
    }

    /// Decode result buffer `i`.
    ///
    /// # Safety
    ///
    /// The [`Frame::from_raw`] contract must hold, and result buffers must be
    /// uniquely owned for this invocation.
    pub unsafe fn ret_buffer(&self, i: usize) -> Result<BufferMut<'a>, ErrorInfo> {
        // SAFETY: `self.raw` is valid per `Frame::from_raw`'s contract.
        let rets = unsafe { &(*self.raw).rets };
        if rets.rets.is_null() || (i as i64) >= rets.size {
            return Err(ErrorInfo::invalid(format!("missing result {i}")));
        }
        // SAFETY: `rets.types` has `rets.size` entries and `i < size` (checked).
        if !rets.types.is_null() && unsafe { *rets.types.add(i) } != XLA_FFI_RET_TYPE_BUFFER {
            return Err(ErrorInfo::invalid(format!("result {i} is not a buffer")));
        }
        // SAFETY: `rets.rets` has `rets.size` entries and `i < size` (checked).
        let ptr = unsafe { *rets.rets.add(i) } as *mut XLA_FFI_Buffer;
        // SAFETY: `ptr` is the result buffer XLA provided for slot `i`, valid
        // for the call and uniquely owned by this invocation.
        unsafe { BufferMut::from_ptr(ptr, "result") }
    }
}

/// Decode a raw XLA buffer pointer into its (validated) shape.
///
/// # Safety
///
/// `ptr` must point to a valid, non-null [`XLA_FFI_Buffer`] whose `dims`/`data`
/// remain valid for `'a` (i.e. for the call-frame lifetime).
unsafe fn buffer_parts<'a>(
    ptr: *const XLA_FFI_Buffer,
    what: &str,
) -> Result<(c_int, &'a [i64], *mut c_void), ErrorInfo> {
    if ptr.is_null() {
        return Err(ErrorInfo::invalid(format!("{what} is null")));
    }
    // SAFETY: `ptr` is non-null (checked) and valid per this fn's contract.
    let b = unsafe { &*ptr };
    let dims: &'a [i64] = if b.dims.is_null() || b.rank <= 0 {
        &[]
    } else {
        // SAFETY: for a valid buffer, `b.dims` has `b.rank` entries.
        unsafe { slice::from_raw_parts(b.dims, b.rank as usize) }
    };
    if dims.iter().any(|&d| d < 0) {
        return Err(ErrorInfo::invalid(format!(
            "{what} has a negative dimension"
        )));
    }
    Ok((b.dtype, dims, b.data))
}

/// A read-only view of an XLA argument buffer.
pub struct Buffer<'a> {
    dtype: c_int,
    dims: &'a [i64],
    data: *const c_void,
    _marker: PhantomData<&'a [u8]>,
}

impl<'a> Buffer<'a> {
    /// # Safety
    /// See [`buffer_parts`].
    unsafe fn from_ptr(ptr: *const XLA_FFI_Buffer, what: &str) -> Result<Self, ErrorInfo> {
        // SAFETY: the caller upholds `buffer_parts`' contract (see `# Safety`).
        let (dtype, dims, data) = unsafe { buffer_parts(ptr, what)? };
        Ok(Self {
            dtype,
            dims,
            data: data as *const c_void,
            _marker: PhantomData,
        })
    }

    /// The XLA dtype.
    pub fn dtype(&self) -> c_int {
        self.dtype
    }

    /// The buffer shape (borrowed from the call frame).
    pub fn dims(&self) -> &'a [i64] {
        self.dims
    }

    /// Number of elements.
    pub fn element_count(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    /// Check the dtype, erroring with `what` for a clear message.
    pub fn expect_dtype(&self, expected: c_int, what: &str) -> Result<(), ErrorInfo> {
        if self.dtype != expected {
            return Err(ErrorInfo::invalid(format!(
                "{what}: unexpected dtype {}, expected {expected}",
                self.dtype
            )));
        }
        Ok(())
    }

    /// Reinterpret the data as `&[T]`.
    ///
    /// The caller is responsible for checking the dtype first
    /// (see [`Buffer::expect_dtype`]).
    pub fn as_slice<T: Copy>(&self) -> &'a [T] {
        let n = self.element_count();
        if self.data.is_null() || n == 0 {
            return &[];
        }
        // SAFETY: `data` points to `n` contiguous elements of `dtype` (the XLA
        // buffer invariant), and `'a` is bounded by the call-frame lifetime.
        unsafe { slice::from_raw_parts(self.data as *const T, n) }
    }
}

/// A write-only view of an XLA result buffer.
pub struct BufferMut<'a> {
    dtype: c_int,
    dims: &'a [i64],
    data: *mut c_void,
    _marker: PhantomData<&'a mut [u8]>,
}

impl<'a> BufferMut<'a> {
    /// # Safety
    /// See [`buffer_parts`]; additionally the buffer must be uniquely owned.
    unsafe fn from_ptr(ptr: *mut XLA_FFI_Buffer, what: &str) -> Result<Self, ErrorInfo> {
        // SAFETY: the caller upholds `buffer_parts`' contract (see `# Safety`).
        let (dtype, dims, data) = unsafe { buffer_parts(ptr as *const XLA_FFI_Buffer, what)? };
        Ok(Self {
            dtype,
            dims,
            data,
            _marker: PhantomData,
        })
    }

    /// The XLA dtype.
    pub fn dtype(&self) -> c_int {
        self.dtype
    }

    /// The buffer shape.
    pub fn dims(&self) -> &'a [i64] {
        self.dims
    }

    /// Number of elements.
    pub fn element_count(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    /// Check the dtype.
    pub fn expect_dtype(&self, expected: c_int, what: &str) -> Result<(), ErrorInfo> {
        if self.dtype != expected {
            return Err(ErrorInfo::invalid(format!(
                "{what}: unexpected dtype {}, expected {expected}",
                self.dtype
            )));
        }
        Ok(())
    }

    /// Reinterpret the data as `&mut [T]`.
    ///
    /// The caller is responsible for checking the dtype first.
    pub fn as_slice_mut<T: Copy>(&mut self) -> &'a mut [T] {
        let n = self.element_count();
        if self.data.is_null() || n == 0 {
            return &mut [];
        }
        // SAFETY: as `Buffer::as_slice`, plus unique ownership of the result
        // buffer for this invocation.
        unsafe { slice::from_raw_parts_mut(self.data as *mut T, n) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xla_ffi::{dtype, XLA_FFI_Args, XLA_FFI_Attrs, XLA_FFI_Rets, XLA_FFI_STAGE_EXECUTE};
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

        let mut raw_frame = XLA_FFI_CallFrame {
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

        // SAFETY: `raw_frame` is a fully-initialised call frame that outlives
        // `frame` (a local).
        let frame = unsafe { Frame::from_raw(&mut raw_frame) };
        // SAFETY: the frame references a valid single-argument buffer.
        unsafe {
            let b = frame.arg_buffer(0).unwrap();
            assert_eq!(b.dims(), &[3]);
            assert_eq!(b.element_count(), 3);
            assert_eq!(b.dtype(), dtype::S32);
            assert_eq!(b.as_slice::<i32>(), &[1, 2, 3]);
            assert!(b.expect_dtype(dtype::S32, "arg").is_ok());
            assert!(b.expect_dtype(dtype::F64, "arg").is_err());
            assert!(frame.arg_buffer(1).is_err());
            assert!(frame.ret_buffer(0).is_err());
        }
    }
}
