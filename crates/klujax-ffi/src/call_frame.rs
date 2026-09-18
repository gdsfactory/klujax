//! Typed, borrowed views of XLA buffers. Output slots are consumed once, and
//! mutable slices borrow their owner rather than the entire call-frame lifetime.

use crate::error::ErrorInfo;
use crate::xla_ffi::{
    XLA_FFI_Buffer, XLA_FFI_CallFrame, XLA_FFI_ARG_TYPE_BUFFER, XLA_FFI_RET_TYPE_BUFFER,
};
use core::ffi::{c_int, c_void};
use core::marker::PhantomData;
use core::slice;
use std::cell::RefCell;
use std::collections::HashSet;
use std::ops::Range;

mod sealed {
    pub trait Sealed {}
    impl Sealed for i32 {}
    impl Sealed for u64 {}
    impl Sealed for f64 {}
    impl Sealed for crate::klu::C64 {}
}

/// The supported XLA scalar layouts. Sealed so safe callers cannot claim an
/// arbitrary Rust type matches a foreign dtype. All supported types admit zero.
pub trait Element: sealed::Sealed + Copy {
    const DTYPE: c_int;
}
impl Element for i32 {
    const DTYPE: c_int = crate::xla_ffi::dtype::S32;
}
impl Element for u64 {
    const DTYPE: c_int = crate::xla_ffi::dtype::U64;
}
impl Element for f64 {
    const DTYPE: c_int = crate::xla_ffi::dtype::F64;
}
impl Element for crate::klu::C64 {
    const DTYPE: c_int = crate::xla_ffi::dtype::C128;
}

/// A borrowed call frame. Successful output decodes consume their slot for the
/// whole invocation, even if the returned view is dropped.
///
/// Slot and overlap checks are local to this instance. The unsafe constructor's
/// one-Frame-per-call obligation prevents aliasing across separate instances;
/// neither the borrow checker nor these runtime checks enforce that obligation.
pub struct Frame<'a> {
    raw: *mut XLA_FFI_CallFrame,
    outputs: RefCell<HashSet<usize>>,
    regions: RefCell<Vec<(Range<usize>, bool)>>,
    _marker: PhantomData<&'a XLA_FFI_CallFrame>,
}

impl<'a> Frame<'a> {
    /// Borrow a foreign call frame.
    ///
    /// # Safety
    /// `raw` and its metadata arrays must be valid for `'a`. Each buffer must
    /// describe a live contiguous allocation of its advertised dtype and size,
    /// aligned for that dtype. Argument data must be initialized and immutable
    /// for `'a`; output data must be exclusively available for `'a`. Construct
    /// exactly one Frame per invocation and share it for all argument/result
    /// decoding, as the generated FFI wrappers do. Do not construct another Frame
    /// over the same raw pointer (or overlapping buffer storage) during `'a`,
    /// even after dropping this Frame: decoded views can outlive it. Otherwise,
    /// separate Frames could each return a mutable view of the same allocation.
    /// Metadata must not overlap output storage. Buffer-to-buffer overlap is
    /// checked on decode only within this Frame.
    pub unsafe fn from_raw(raw: *mut XLA_FFI_CallFrame) -> Self {
        Self {
            raw,
            outputs: RefCell::new(HashSet::new()),
            regions: RefCell::new(Vec::new()),
            _marker: PhantomData,
        }
    }

    /// The raw pointer for the error machinery.
    pub fn as_ptr(&self) -> *mut XLA_FFI_CallFrame {
        self.raw
    }

    fn buffer<T: Element>(
        &self,
        i: usize,
        output: bool,
        what: &str,
    ) -> Result<Buffer<'a>, ErrorInfo> {
        if self.raw.is_null() {
            return Err(ErrorInfo::invalid("null call frame"));
        }
        // SAFETY: Frame::from_raw guarantees live frame metadata and arrays.
        let ptr = unsafe {
            if output {
                let rets = &(*self.raw).rets;
                if rets.rets.is_null() || i as u128 >= rets.size as u128 || rets.size < 0 {
                    return Err(ErrorInfo::invalid(format!("missing result {i}")));
                }
                if rets.types.is_null() || *rets.types.add(i) != XLA_FFI_RET_TYPE_BUFFER {
                    return Err(ErrorInfo::invalid(format!("result {i} is not a buffer")));
                }
                *rets.rets.add(i) as *const XLA_FFI_Buffer
            } else {
                let args = &(*self.raw).args;
                if args.args.is_null() || i as u128 >= args.size as u128 || args.size < 0 {
                    return Err(ErrorInfo::invalid(format!("missing argument {i}")));
                }
                if args.types.is_null() || *args.types.add(i) != XLA_FFI_ARG_TYPE_BUFFER {
                    return Err(ErrorInfo::invalid(format!("argument {i} is not a buffer")));
                }
                *args.args.add(i) as *const XLA_FFI_Buffer
            }
        };
        if ptr.is_null() {
            return Err(ErrorInfo::invalid(format!("{what} is null")));
        }
        // SAFETY: metadata is valid per Frame::from_raw, and ptr is non-null.
        let b = unsafe { &*ptr };
        if b.dtype != T::DTYPE {
            return Err(ErrorInfo::invalid(format!("{what}: unexpected dtype")));
        }
        if b.rank < 0
            || (b.rank > 0 && b.dims.is_null())
            || b.rank as u128 > (isize::MAX as usize / size_of::<i64>()) as u128
        {
            return Err(ErrorInfo::invalid(format!(
                "{what}: invalid rank/dimensions"
            )));
        }
        let dims = if b.rank == 0 {
            &[]
        } else {
            // SAFETY: live dimension array per the frame contract; rank checked above.
            unsafe { slice::from_raw_parts(b.dims, b.rank as usize) }
        };
        let len = dims
            .iter()
            .try_fold(1usize, |n, &d| {
                usize::try_from(d).ok().and_then(|d| n.checked_mul(d))
            })
            .ok_or_else(|| {
                ErrorInfo::invalid(format!("{what}: invalid or overflowing dimensions"))
            })?;
        let bytes = len
            .checked_mul(size_of::<T>())
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| ErrorInfo::invalid("buffer size overflow"))?;
        let start = b.data as usize;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| ErrorInfo::invalid("buffer address overflow"))?;
        if bytes > 0 && (b.data.is_null() || !start.is_multiple_of(align_of::<T>())) {
            return Err(ErrorInfo::invalid("null or misaligned buffer data"));
        }
        if output && self.outputs.borrow().contains(&i) {
            return Err(ErrorInfo::invalid("result buffer already decoded"));
        }
        let mut regions = self.regions.borrow_mut();
        if bytes > 0
            && regions
                .iter()
                .any(|(r, writable)| (output || *writable) && start < r.end && r.start < end)
        {
            return Err(ErrorInfo::invalid("overlapping argument/result buffers"));
        }
        if output {
            self.outputs.borrow_mut().insert(i);
        }
        if bytes > 0 {
            regions.push((start..end, output));
        }
        Ok(Buffer {
            dims,
            data: b.data,
            len,
            _marker: PhantomData,
        })
    }

    /// Decode an argument after checking its dtype and allocation bounds.
    pub fn arg_buffer<T: Element>(&self, i: usize) -> Result<Buf<'a, T>, ErrorInfo> {
        let buffer = self.buffer::<T>(i, false, "argument")?;
        Ok(Buf {
            buffer,
            _t: PhantomData,
        })
    }

    /// Consume an output slot and initialize its storage before exposing it as
    /// Rust values. A second decode of the same slot returns an error.
    pub fn ret_buffer<T: Element>(&self, i: usize) -> Result<BufMut<'a, T>, ErrorInfo> {
        let buffer = self.buffer::<T>(i, true, "result")?;
        if buffer.len > 0 {
            // SAFETY: buffer is exclusively claimed, aligned, and bounds checked;
            // all sealed Element types have a valid all-zero representation.
            unsafe { buffer.data.cast::<T>().write_bytes(0, buffer.len) };
        }
        Ok(BufMut {
            buffer,
            _t: PhantomData,
        })
    }
}

// Private metadata; callers cannot reinterpret it with an arbitrary Rust type.
struct Buffer<'a> {
    dims: &'a [i64],
    data: *mut c_void,
    len: usize,
    _marker: PhantomData<&'a [u8]>,
}

/// A typed shared argument buffer.
pub struct Buf<'a, T: Element> {
    buffer: Buffer<'a>,
    _t: PhantomData<&'a T>,
}
impl<T: Element> Buf<'_, T> {
    pub fn dims(&self) -> &[i64] {
        self.buffer.dims
    }
    pub fn element_count(&self) -> usize {
        self.buffer.len
    }
    pub fn as_slice(&self) -> &[T] {
        if self.buffer.len == 0 {
            return &[];
        }
        // SAFETY: the sealed dtype and allocation bounds were checked on decode;
        // Frame guarantees initialized immutable argument storage for this borrow.
        unsafe { slice::from_raw_parts(self.buffer.data.cast::<T>(), self.buffer.len) }
    }
}
impl<T: Element> core::ops::Deref for Buf<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

/// An exclusively claimed typed result buffer.
///
/// One owner cannot yield two simultaneously live mutable slices. This checks
/// borrowing within a single BufMut, not the unsafe Frame construction contract:
/// ```compile_fail,E0499
/// use klujax_ffi::call_frame::BufMut;
/// fn alias(b: &mut BufMut<'_, f64>) {
///     let first = b.as_mut_slice();
///     let second = b.as_mut_slice();
///     //~^ ERROR cannot borrow `*b` as mutable more than once at a time
///     first[0] = second[0];
/// }
/// ```
pub struct BufMut<'a, T: Element> {
    buffer: Buffer<'a>,
    _t: PhantomData<&'a mut T>,
}
impl<T: Element> BufMut<'_, T> {
    pub fn dims(&self) -> &[i64] {
        self.buffer.dims
    }
    pub fn element_count(&self) -> usize {
        self.buffer.len
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.buffer.len == 0 {
            return &mut [];
        }
        // SAFETY: decode claimed this output once and initialized its checked
        // allocation. The returned slice borrows self exclusively.
        unsafe { slice::from_raw_parts_mut(self.buffer.data.cast::<T>(), self.buffer.len) }
    }
}
impl<T: Element> core::ops::Deref for BufMut<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        if self.buffer.len == 0 {
            return &[];
        }
        // SAFETY: decode checked dtype/bounds and initialized this output. This
        // shared borrow of self prevents a mutable slice while it is live.
        unsafe { slice::from_raw_parts(self.buffer.data.cast::<T>(), self.buffer.len) }
    }
}
impl<T: Element> core::ops::DerefMut for BufMut<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

pub trait DecodeArg<'a>: Sized {
    fn decode_arg(frame: &Frame<'a>, i: usize, what: &str) -> Result<Self, ErrorInfo>;
}
pub trait DecodeRet<'a>: Sized {
    fn decode_ret(frame: &Frame<'a>, i: usize, what: &str) -> Result<Self, ErrorInfo>;
}
impl<'a, T: Element> DecodeArg<'a> for Buf<'a, T> {
    fn decode_arg(frame: &Frame<'a>, i: usize, _what: &str) -> Result<Self, ErrorInfo> {
        frame.arg_buffer(i)
    }
}
impl<'a, T: Element> DecodeRet<'a> for BufMut<'a, T> {
    fn decode_ret(frame: &Frame<'a>, i: usize, _what: &str) -> Result<Self, ErrorInfo> {
        frame.ret_buffer(i)
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
        let b = frame.arg_buffer::<i32>(0).unwrap();
        assert_eq!(b.dims(), &[3]);
        assert_eq!(b.element_count(), 3);
        assert_eq!(b.as_slice(), &[1, 2, 3]);
        assert!(frame.arg_buffer::<f64>(0).is_err());
        assert!(frame.arg_buffer::<i32>(1).is_err());
        assert!(frame.ret_buffer::<i32>(0).is_err());
    }
    #[test]
    fn output_is_typed_initialized_and_consumed_once() {
        let mut data = [core::mem::MaybeUninit::<f64>::uninit(); 2];
        let mut dims = [2i64];
        let mut buffer = XLA_FFI_Buffer {
            struct_size: size_of::<XLA_FFI_Buffer>(),
            extension_start: null_mut(),
            dtype: dtype::F64,
            data: data.as_mut_ptr().cast(),
            rank: 1,
            dims: dims.as_mut_ptr(),
        };
        // Two result slots intentionally describe the same storage: the second
        // must be rejected without creating a second Rust reference.
        let mut pointers = [&mut buffer as *mut XLA_FFI_Buffer as *mut c_void; 2];
        let mut types = [XLA_FFI_RET_TYPE_BUFFER; 2];
        let mut rets = empty_rets();
        rets.size = 2;
        rets.types = types.as_mut_ptr();
        rets.rets = pointers.as_mut_ptr();
        let mut raw = XLA_FFI_CallFrame {
            struct_size: size_of::<XLA_FFI_CallFrame>(),
            extension_start: null_mut(),
            api: core::ptr::null(),
            ctx: null_mut(),
            stage: XLA_FFI_STAGE_EXECUTE,
            args: empty_args(),
            rets,
            attrs: empty_attrs(),
            future: null_mut(),
        };
        // SAFETY: all metadata is live and storage is exclusively available;
        // result overlap is handled by the decoder before exposing references.
        let frame = unsafe { Frame::from_raw(&mut raw) };
        assert!(frame.ret_buffer::<i32>(0).is_err());
        {
            let mut out = frame.ret_buffer::<f64>(0).unwrap();
            assert_eq!(&*out, &[0.0, 0.0]);
            out.as_mut_slice().copy_from_slice(&[3.0, 4.0]);
            assert!(frame.ret_buffer::<f64>(0).is_err());
            assert!(frame.ret_buffer::<f64>(1).is_err());
            assert_eq!(&*out, &[3.0, 4.0]);
        }
        assert!(frame.ret_buffer::<f64>(0).is_err());
    }
}
