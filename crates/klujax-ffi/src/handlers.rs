//! The XLA typed-FFI handler symbols registered with `jax.ffi.register_ffi_target`.
//!
//! With the default `c-backend` feature these decode the call frame and call the
//! C-backed engine; without it they are compile-only stubs.

#![allow(clippy::too_many_arguments)]
// Every `pub unsafe extern "C"` here is an XLA FFI handler with the uniform
// safety contract "`frame` is a valid call frame passed by XLA".
#![allow(clippy::missing_safety_doc)]

use crate::error::{guard, ErrorInfo};
use crate::xla_ffi::{XLA_FFI_CallFrame, XLA_FFI_Error};

#[cfg(not(feature = "c-backend"))]
mod stubs {
    use super::*;

    macro_rules! stub_handler {
        ($($name:ident),* $(,)?) => {
            $(
                /// XLA FFI handler (stub build).
                ///
                /// # Safety
                /// `frame` must be a valid call frame passed by XLA.
                #[no_mangle]
                pub unsafe extern "C" fn $name(
                    frame: *mut XLA_FFI_CallFrame,
                ) -> *mut XLA_FFI_Error {
                    unsafe {
                        guard(frame, || {
                            Err(ErrorInfo::unimplemented(concat!(
                                stringify!($name),
                                " is not implemented in the no-c-backend build"
                            )))
                        })
                    }
                }
            )*
        };
    }

    stub_handler!(
        dot_f64,
        dot_c128,
        solve_f64,
        solve_c128,
        solve_with_symbol_f64,
        solve_with_symbol_c128,
        tsolve_with_symbol_f64,
        tsolve_with_symbol_c128,
        factor_f64,
        factor_c128,
        refactor_f64,
        refactor_c128,
        refactor_and_solve_f64,
        refactor_and_solve_c128,
        solve_with_numeric_f64,
        solve_with_numeric_c128,
        tsolve_with_numeric_f64,
        tsolve_with_numeric_c128,
        free_numeric,
        free_symbolic,
        analyze,
    );
}

#[cfg(feature = "c-backend")]
pub use real::*;

#[cfg(feature = "c-backend")]
mod real {
    use super::*;
    use crate::call_frame::{arg_buffer, dims, element_count, expect_dtype, ret_buffer};
    use crate::engine::{self, Scalar};
    use crate::xla_ffi::{dtype, XLA_FFI_Buffer};
    use core::slice;

    // ---- generic argument / result access ------------------------------

    /// Read argument `i` as `&[T]` (F64 or C128 bytes reinterpreted as `T`).
    unsafe fn arg<T: Scalar>(
        frame: *mut XLA_FFI_CallFrame,
        i: usize,
    ) -> Result<&'static [T], ErrorInfo> {
        let b = unsafe { arg_buffer(frame, i)? };
        unsafe { expect_dtype(b, T::DTYPE, "arg")? };
        let n = unsafe { element_count(b) };
        let data = unsafe { (*b).data } as *const T;
        Ok(if data.is_null() || n == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(data, n) }
        })
    }

    /// Result buffer `i` as `&mut [T]`.
    unsafe fn ret<T: Scalar>(
        frame: *mut XLA_FFI_CallFrame,
        i: usize,
    ) -> Result<&'static mut [T], ErrorInfo> {
        let b = unsafe { ret_buffer(frame, i)? };
        unsafe { expect_dtype(b, T::DTYPE, "ret")? };
        let n = unsafe { element_count(b) };
        let data = unsafe { (*b).data } as *mut T;
        Ok(if data.is_null() || n == 0 {
            &mut []
        } else {
            unsafe { slice::from_raw_parts_mut(data, n) }
        })
    }

    unsafe fn s32_arg<'a>(frame: *mut XLA_FFI_CallFrame, i: usize) -> Result<&'a [i32], ErrorInfo> {
        let b = unsafe { arg_buffer(frame, i)? };
        unsafe { expect_dtype(b, dtype::S32, "arg")? };
        let n = unsafe { element_count(b) };
        let data = unsafe { (*b).data } as *const i32;
        Ok(if data.is_null() || n == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(data, n) }
        })
    }

    unsafe fn u64_arg<'a>(frame: *mut XLA_FFI_CallFrame, i: usize) -> Result<&'a [u64], ErrorInfo> {
        let b = unsafe { arg_buffer(frame, i)? };
        unsafe { expect_dtype(b, dtype::U64, "arg")? };
        let n = unsafe { element_count(b) };
        let data = unsafe { (*b).data } as *const u64;
        Ok(if data.is_null() || n == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(data, n) }
        })
    }

    unsafe fn u64_ret<'a>(
        frame: *mut XLA_FFI_CallFrame,
        i: usize,
    ) -> Result<&'a mut [u64], ErrorInfo> {
        let b = unsafe { ret_buffer(frame, i)? };
        unsafe { expect_dtype(b, dtype::U64, "ret")? };
        let n = unsafe { element_count(b) };
        let data = unsafe { (*b).data } as *mut u64;
        Ok(if data.is_null() || n == 0 {
            &mut []
        } else {
            unsafe { slice::from_raw_parts_mut(data, n) }
        })
    }

    unsafe fn out_dims(frame: *mut XLA_FFI_CallFrame, i: usize) -> Result<Vec<usize>, ErrorInfo> {
        let b = unsafe { arg_buffer(frame, i)? };
        let d = unsafe { dims(b) };
        if d.iter().any(|&x| x < 0) {
            return Err(ErrorInfo::invalid("negative dimension"));
        }
        Ok(d.iter().map(|&x| x as usize).collect())
    }

    unsafe fn b_dims(
        frame: *mut XLA_FFI_CallFrame,
        i: usize,
    ) -> Result<(usize, usize, usize), ErrorInfo> {
        let d = unsafe { out_dims(frame, i)? };
        if d.len() != 3 {
            return Err(ErrorInfo::invalid(
                "x must be normalized to 3D (n_lhs, n_col, n_rhs)",
            ));
        }
        Ok((d[0], d[1], d[2]))
    }

    unsafe fn sym_scalar(frame: *mut XLA_FFI_CallFrame, i: usize) -> Result<u64, ErrorInfo> {
        let s = unsafe { u64_arg(frame, i)? };
        if s.len() != 1 {
            return Err(ErrorInfo::invalid("symbolic must be scalar"));
        }
        if s[0] == 0 {
            return Err(ErrorInfo::invalid("symbolic pointer is null"));
        }
        Ok(s[0])
    }

    fn write<T: Copy>(out: &mut [T], src: &[T]) -> Result<(), ErrorInfo> {
        if out.len() != src.len() {
            return Err(ErrorInfo::invalid("output size mismatch"));
        }
        out.copy_from_slice(src);
        Ok(())
    }

    // ---- handlers ------------------------------------------------------

    #[no_mangle]
    pub unsafe extern "C" fn analyze(frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe {
            let ai = s32_arg(frame, 0)?;
            let aj = s32_arg(frame, 1)?;
            let nc = s32_arg(frame, 2)?;
            if nc.len() != 1 {
                return Err(ErrorInfo::invalid("n_col must be a scalar"));
            }
            let sym = engine::analyze_raw(nc[0] as usize, ai, aj)?;
            u64_ret(frame, 0)?[0] = sym;
            Ok(())
        })
    }

    #[no_mangle]
    pub unsafe extern "C" fn free_symbolic(frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe {
            let s = u64_arg(frame, 0)?;
            let mut status = 0;
            for &addr in s {
                status |= engine::free_symbolic_raw(addr);
            }
            let out = ret_buffer(frame, 0)?;
            expect_dtype(out, dtype::S32, "ret")?;
            *((*out).data as *mut i32) = status;
            Ok(())
        })
    }

    #[no_mangle]
    pub unsafe extern "C" fn free_numeric(frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe {
            let s = u64_arg(frame, 0)?;
            let status = engine::free_numeric_raw(s);
            let out = ret_buffer(frame, 0)?;
            expect_dtype(out, dtype::S32, "ret")?;
            *((*out).data as *mut i32) = status;
            Ok(())
        })
    }

    #[no_mangle]
    pub unsafe extern "C" fn factor_f64(frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe { factor_impl::<f64>(frame) })
    }
    #[no_mangle]
    pub unsafe extern "C" fn factor_c128(frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe { factor_impl::<engine::C64>(frame) })
    }

    unsafe fn factor_impl<T: Scalar>(frame: *mut XLA_FFI_CallFrame) -> Result<(), ErrorInfo> {
        let ai = unsafe { s32_arg(frame, 0)? };
        let aj = unsafe { s32_arg(frame, 1)? };
        let sym = unsafe { sym_scalar(frame, 3)? };
        let (n_lhs, n_nz) = {
            let d = unsafe { out_dims(frame, 2)? };
            if d.len() != 2 {
                return Err(ErrorInfo::invalid("Ax is not 2D."));
            }
            (d[0], d[1])
        };
        let ax = unsafe { arg::<T>(frame, 2)? };
        if ax.len() != n_lhs * n_nz {
            return Err(ErrorInfo::invalid("Ax size mismatch"));
        }
        let handles = engine::factor_batch_raw(ai, aj, ax, n_lhs, sym)?;
        write(unsafe { u64_ret(frame, 4)? }, &handles)
    }

    #[no_mangle]
    pub unsafe extern "C" fn refactor_f64(frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe { refactor_impl::<f64>(frame) })
    }
    #[no_mangle]
    pub unsafe extern "C" fn refactor_c128(frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe { refactor_impl::<engine::C64>(frame) })
    }

    unsafe fn refactor_impl<T: Scalar>(frame: *mut XLA_FFI_CallFrame) -> Result<(), ErrorInfo> {
        let ai = unsafe { s32_arg(frame, 0)? };
        let aj = unsafe { s32_arg(frame, 1)? };
        let sym = unsafe { sym_scalar(frame, 3)? };
        let numeric = unsafe { u64_arg(frame, 4)? };
        let n_lhs = {
            let d = unsafe { out_dims(frame, 2)? };
            if d.len() != 2 {
                return Err(ErrorInfo::invalid("Ax is not 2D."));
            }
            d[0]
        };
        let ax = unsafe { arg::<T>(frame, 2)? };
        let handles = engine::refactor_batch_raw(ai, aj, ax, n_lhs, sym, numeric)?;
        write(unsafe { u64_ret(frame, 5)? }, &handles)
    }

    #[no_mangle]
    pub unsafe extern "C" fn solve_f64(frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe { solve_impl::<f64>(frame) })
    }
    #[no_mangle]
    pub unsafe extern "C" fn solve_c128(frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe { solve_impl::<engine::C64>(frame) })
    }

    unsafe fn solve_impl<T: Scalar>(frame: *mut XLA_FFI_CallFrame) -> Result<(), ErrorInfo> {
        let ai = unsafe { s32_arg(frame, 0)? };
        let aj = unsafe { s32_arg(frame, 1)? };
        let (n_lhs, n_col, n_rhs) = unsafe { b_dims(frame, 3)? };
        let ax = unsafe { arg::<T>(frame, 2)? };
        let b = unsafe { arg::<T>(frame, 3)? };
        let x = engine::solve_raw(ai, aj, ax, b, n_lhs, n_col, n_rhs)?;
        write(unsafe { ret::<T>(frame, 4)? }, &x)
    }

    #[no_mangle]
    pub unsafe extern "C" fn solve_with_symbol_f64(
        frame: *mut XLA_FFI_CallFrame,
    ) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe {
            solve_with_symbol_impl_h::<f64>(frame, false)
        })
    }
    #[no_mangle]
    pub unsafe extern "C" fn solve_with_symbol_c128(
        frame: *mut XLA_FFI_CallFrame,
    ) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe {
            solve_with_symbol_impl_h::<engine::C64>(frame, false)
        })
    }
    #[no_mangle]
    pub unsafe extern "C" fn tsolve_with_symbol_f64(
        frame: *mut XLA_FFI_CallFrame,
    ) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe {
            solve_with_symbol_impl_h::<f64>(frame, true)
        })
    }
    #[no_mangle]
    pub unsafe extern "C" fn tsolve_with_symbol_c128(
        frame: *mut XLA_FFI_CallFrame,
    ) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe {
            solve_with_symbol_impl_h::<engine::C64>(frame, true)
        })
    }

    unsafe fn solve_with_symbol_impl_h<T: Scalar>(
        frame: *mut XLA_FFI_CallFrame,
        transpose: bool,
    ) -> Result<(), ErrorInfo> {
        let ai = unsafe { s32_arg(frame, 0)? };
        let aj = unsafe { s32_arg(frame, 1)? };
        let sym = unsafe { sym_scalar(frame, 4)? };
        let (n_lhs, n_col, n_rhs) = unsafe { b_dims(frame, 3)? };
        let ax = unsafe { arg::<T>(frame, 2)? };
        let b = unsafe { arg::<T>(frame, 3)? };
        let x = if transpose {
            engine::tsolve_with_symbol_raw(ai, aj, ax, b, n_lhs, n_col, n_rhs, sym)?
        } else {
            engine::solve_with_symbol_raw(ai, aj, ax, b, n_lhs, n_col, n_rhs, sym)?
        };
        write(unsafe { ret::<T>(frame, 5)? }, &x)
    }

    #[no_mangle]
    pub unsafe extern "C" fn solve_with_numeric_f64(
        frame: *mut XLA_FFI_CallFrame,
    ) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe {
            solve_with_numeric_impl_h::<f64>(frame, false)
        })
    }
    #[no_mangle]
    pub unsafe extern "C" fn solve_with_numeric_c128(
        frame: *mut XLA_FFI_CallFrame,
    ) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe {
            solve_with_numeric_impl_h::<engine::C64>(frame, false)
        })
    }
    #[no_mangle]
    pub unsafe extern "C" fn tsolve_with_numeric_f64(
        frame: *mut XLA_FFI_CallFrame,
    ) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe {
            solve_with_numeric_impl_h::<f64>(frame, true)
        })
    }
    #[no_mangle]
    pub unsafe extern "C" fn tsolve_with_numeric_c128(
        frame: *mut XLA_FFI_CallFrame,
    ) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe {
            solve_with_numeric_impl_h::<engine::C64>(frame, true)
        })
    }

    unsafe fn solve_with_numeric_impl_h<T: Scalar>(
        frame: *mut XLA_FFI_CallFrame,
        transpose: bool,
    ) -> Result<(), ErrorInfo> {
        let sym = unsafe { sym_scalar(frame, 0)? };
        let numeric = unsafe { u64_arg(frame, 1)? };
        let bd = unsafe { out_dims(frame, 2)? };
        let n_numeric = numeric.len();
        let d_b = bd.len();
        let (n_lhs_b, n_col, n_rhs) = match d_b {
            1 => (1, bd[0], 1),
            2 => {
                if n_numeric > 1 && bd[0] == n_numeric {
                    (bd[0], bd[1], 1)
                } else {
                    (1, bd[0], bd[1])
                }
            }
            3 => (bd[0], bd[1], bd[2]),
            _ => return Err(ErrorInfo::invalid("b must be 1D, 2D, or 3D")),
        };
        let broadcast_numeric = n_numeric == 1;
        let broadcast_b = n_lhs_b == 1;
        let n_lhs = if !broadcast_numeric && !broadcast_b {
            if n_numeric != n_lhs_b {
                return Err(ErrorInfo::invalid("numeric and b batch size mismatch"));
            }
            n_numeric
        } else if !broadcast_numeric {
            n_numeric
        } else if !broadcast_b {
            n_lhs_b
        } else {
            1
        };
        let b = unsafe { arg::<T>(frame, 2)? };
        let mut b_full = vec![T::zero(); n_lhs * n_col * n_rhs];
        for m in 0..n_lhs {
            let mb = if broadcast_b { 0 } else { m };
            for n in 0..n_col {
                for p in 0..n_rhs {
                    b_full[m * n_col * n_rhs + n * n_rhs + p] =
                        b[mb * n_col * n_rhs + n * n_rhs + p];
                }
            }
        }
        let x =
            engine::solve_with_numeric_raw(sym, numeric, &b_full, n_lhs, n_col, n_rhs, transpose)?;
        let out = unsafe { ret::<T>(frame, 3)? };
        let n = out.len().min(x.len());
        out[..n].copy_from_slice(&x[..n]);
        Ok(())
    }

    #[no_mangle]
    pub unsafe extern "C" fn refactor_and_solve_f64(
        frame: *mut XLA_FFI_CallFrame,
    ) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe { refactor_and_solve_impl_h::<f64>(frame) })
    }
    #[no_mangle]
    pub unsafe extern "C" fn refactor_and_solve_c128(
        frame: *mut XLA_FFI_CallFrame,
    ) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe {
            refactor_and_solve_impl_h::<engine::C64>(frame)
        })
    }

    unsafe fn refactor_and_solve_impl_h<T: Scalar>(
        frame: *mut XLA_FFI_CallFrame,
    ) -> Result<(), ErrorInfo> {
        let ai = unsafe { s32_arg(frame, 0)? };
        let aj = unsafe { s32_arg(frame, 1)? };
        let sym = unsafe { sym_scalar(frame, 4)? };
        let numeric = unsafe { u64_arg(frame, 5)? };
        let (n_lhs, n_col, n_rhs) = unsafe { b_dims(frame, 3)? };
        let ax = unsafe { arg::<T>(frame, 2)? };
        let b = unsafe { arg::<T>(frame, 3)? };
        let handles = engine::refactor_batch_raw(ai, aj, ax, n_lhs, sym, numeric)?;
        let x = engine::solve_with_numeric_raw(sym, &handles, b, n_lhs, n_col, n_rhs, false)?;
        write(unsafe { ret::<T>(frame, 6)? }, &x)?;
        write(unsafe { u64_ret(frame, 7)? }, &handles)
    }

    #[no_mangle]
    pub unsafe extern "C" fn dot_f64(frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe { dot_impl::<f64>(frame) })
    }
    #[no_mangle]
    pub unsafe extern "C" fn dot_c128(frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
        guard(frame, || unsafe { dot_impl::<engine::C64>(frame) })
    }

    unsafe fn dot_impl<T: Scalar>(frame: *mut XLA_FFI_CallFrame) -> Result<(), ErrorInfo> {
        let ai = unsafe { s32_arg(frame, 0)? };
        let aj = unsafe { s32_arg(frame, 1)? };
        let (n_lhs, n_col, n_rhs) = unsafe { b_dims(frame, 3)? };
        let ax = unsafe { arg::<T>(frame, 2)? };
        let x = unsafe { arg::<T>(frame, 3)? };
        let out = engine::dot_raw(ai, aj, ax, x, n_lhs, n_col, n_rhs)?;
        write(unsafe { ret::<T>(frame, 4)? }, &out)
    }

    // Keep the import used in all feature configurations.
    #[allow(unused_imports)]
    use XLA_FFI_Buffer as _XLA_FFI_Buffer;
}
