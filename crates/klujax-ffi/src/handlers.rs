//! The XLA typed-FFI handler symbols registered with `jax.ffi.register_ffi_target`.
//!
//! Each decodes the [`XLA_FFI_CallFrame`] into lifetime-safe [`Frame`] buffers
//! and calls the C-backed engine.

use crate::call_frame::{Buffer, BufferMut, Frame};
use crate::engine::{self, Scalar};
use crate::error::{guard, ErrorInfo};
use crate::xla_ffi::{dtype, XLA_FFI_CallFrame, XLA_FFI_Error};

// ---- typed argument / result access ---------------------------------------

fn arg<'a, T: Scalar>(frame: &Frame<'a>, i: usize) -> Result<&'a [T], ErrorInfo> {
    // SAFETY: `frame` came from `Frame::from_raw` in the enclosing handler.
    let b: Buffer<'a> = unsafe { frame.arg_buffer(i)? };
    b.expect_dtype(T::DTYPE, "argument")?;
    Ok(b.as_slice::<T>())
}

fn ret<'a, T: Scalar>(frame: &Frame<'a>, i: usize) -> Result<&'a mut [T], ErrorInfo> {
    // SAFETY: as `arg`, and the result buffer is uniquely owned by this call.
    let mut b: BufferMut<'a> = unsafe { frame.ret_buffer(i)? };
    b.expect_dtype(T::DTYPE, "result")?;
    Ok(b.as_slice_mut::<T>())
}

fn s32_arg<'a>(frame: &Frame<'a>, i: usize) -> Result<&'a [i32], ErrorInfo> {
    // SAFETY: `frame` is valid per the calling handler's contract.
    let b = unsafe { frame.arg_buffer(i)? };
    b.expect_dtype(dtype::S32, "argument")?;
    Ok(b.as_slice::<i32>())
}

fn u64_arg<'a>(frame: &Frame<'a>, i: usize) -> Result<&'a [u64], ErrorInfo> {
    // SAFETY: `frame` is valid per the calling handler's contract.
    let b = unsafe { frame.arg_buffer(i)? };
    b.expect_dtype(dtype::U64, "argument")?;
    Ok(b.as_slice::<u64>())
}

fn u64_ret<'a>(frame: &Frame<'a>, i: usize) -> Result<&'a mut [u64], ErrorInfo> {
    // SAFETY: result buffers are uniquely owned by this call.
    let mut b = unsafe { frame.ret_buffer(i)? };
    b.expect_dtype(dtype::U64, "result")?;
    Ok(b.as_slice_mut::<u64>())
}

fn i32_ret<'a>(frame: &Frame<'a>, i: usize) -> Result<&'a mut [i32], ErrorInfo> {
    // SAFETY: result buffers are uniquely owned by this call.
    let mut b = unsafe { frame.ret_buffer(i)? };
    b.expect_dtype(dtype::S32, "result")?;
    Ok(b.as_slice_mut::<i32>())
}

fn out_dims(frame: &Frame<'_>, i: usize) -> Result<Vec<usize>, ErrorInfo> {
    // SAFETY: `frame` is valid per the calling handler's contract.
    let b = unsafe { frame.arg_buffer(i)? };
    Ok(b.dims().iter().map(|&d| d as usize).collect())
}

fn b_dims(frame: &Frame<'_>, i: usize) -> Result<(usize, usize, usize), ErrorInfo> {
    let d = out_dims(frame, i)?;
    if d.len() != 3 {
        return Err(ErrorInfo::invalid(
            "x must be normalized to 3D (n_lhs, n_col, n_rhs)",
        ));
    }
    Ok((d[0], d[1], d[2]))
}

fn sym_scalar(frame: &Frame<'_>, i: usize) -> Result<u64, ErrorInfo> {
    let s = u64_arg(frame, i)?;
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

// ---- shared handler bodies ------------------------------------------------

fn factor_impl<T: Scalar>(frame: &Frame<'_>) -> Result<(), ErrorInfo> {
    let ai = s32_arg(frame, 0)?;
    let aj = s32_arg(frame, 1)?;
    let sym = sym_scalar(frame, 3)?;
    let (n_lhs, n_nz) = {
        let d = out_dims(frame, 2)?;
        if d.len() != 2 {
            return Err(ErrorInfo::invalid("Ax is not 2D."));
        }
        (d[0], d[1])
    };
    let ax = arg::<T>(frame, 2)?;
    if ax.len() != n_lhs * n_nz {
        return Err(ErrorInfo::invalid("Ax size mismatch"));
    }
    let handles = engine::factor_batch_raw(ai, aj, ax, n_lhs, sym)?;
    write(u64_ret(frame, 0)?, &handles)
}

fn refactor_impl<T: Scalar>(frame: &Frame<'_>) -> Result<(), ErrorInfo> {
    let ai = s32_arg(frame, 0)?;
    let aj = s32_arg(frame, 1)?;
    let sym = sym_scalar(frame, 3)?;
    let numeric = u64_arg(frame, 4)?;
    let n_lhs = {
        let d = out_dims(frame, 2)?;
        if d.len() != 2 {
            return Err(ErrorInfo::invalid("Ax is not 2D."));
        }
        d[0]
    };
    let ax = arg::<T>(frame, 2)?;
    let handles = engine::refactor_batch_raw(ai, aj, ax, n_lhs, sym, numeric)?;
    write(u64_ret(frame, 0)?, &handles)
}

fn solve_impl<T: Scalar>(frame: &Frame<'_>) -> Result<(), ErrorInfo> {
    let ai = s32_arg(frame, 0)?;
    let aj = s32_arg(frame, 1)?;
    let (n_lhs, n_col, n_rhs) = b_dims(frame, 3)?;
    let ax = arg::<T>(frame, 2)?;
    let b = arg::<T>(frame, 3)?;
    let x = engine::solve_raw(ai, aj, ax, b, n_lhs, n_col, n_rhs)?;
    write(ret::<T>(frame, 0)?, &x)
}

fn solve_with_symbol_impl_h<T: Scalar>(
    frame: &Frame<'_>,
    transpose: bool,
) -> Result<(), ErrorInfo> {
    let ai = s32_arg(frame, 0)?;
    let aj = s32_arg(frame, 1)?;
    let sym = sym_scalar(frame, 4)?;
    let (n_lhs, n_col, n_rhs) = b_dims(frame, 3)?;
    let ax = arg::<T>(frame, 2)?;
    let b = arg::<T>(frame, 3)?;
    let x = if transpose {
        engine::tsolve_with_symbol_raw(ai, aj, ax, b, n_lhs, n_col, n_rhs, sym)?
    } else {
        engine::solve_with_symbol_raw(ai, aj, ax, b, n_lhs, n_col, n_rhs, sym)?
    };
    write(ret::<T>(frame, 0)?, &x)
}

fn solve_with_numeric_impl_h<T: Scalar>(
    frame: &Frame<'_>,
    transpose: bool,
) -> Result<(), ErrorInfo> {
    let sym = sym_scalar(frame, 0)?;
    let numeric = u64_arg(frame, 1)?;
    let bd = out_dims(frame, 2)?;
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
    let b = arg::<T>(frame, 2)?;
    let mut b_full = vec![T::zero(); n_lhs * n_col * n_rhs];
    for m in 0..n_lhs {
        let mb = if broadcast_b { 0 } else { m };
        for n in 0..n_col {
            for p in 0..n_rhs {
                b_full[m * n_col * n_rhs + n * n_rhs + p] = b[mb * n_col * n_rhs + n * n_rhs + p];
            }
        }
    }
    let x = engine::solve_with_numeric_raw(sym, numeric, &b_full, n_lhs, n_col, n_rhs, transpose)?;
    let out = ret::<T>(frame, 0)?;
    let n = out.len().min(x.len());
    out[..n].copy_from_slice(&x[..n]);
    Ok(())
}

fn refactor_and_solve_impl_h<T: Scalar>(frame: &Frame<'_>) -> Result<(), ErrorInfo> {
    let ai = s32_arg(frame, 0)?;
    let aj = s32_arg(frame, 1)?;
    let sym = sym_scalar(frame, 4)?;
    let numeric = u64_arg(frame, 5)?;
    let (n_lhs, n_col, n_rhs) = b_dims(frame, 3)?;
    let ax = arg::<T>(frame, 2)?;
    let b = arg::<T>(frame, 3)?;
    let handles = engine::refactor_batch_raw(ai, aj, ax, n_lhs, sym, numeric)?;
    let x = engine::solve_with_numeric_raw(sym, &handles, b, n_lhs, n_col, n_rhs, false)?;
    write(ret::<T>(frame, 0)?, &x)?;
    write(u64_ret(frame, 1)?, &handles)
}

fn dot_impl<T: Scalar>(frame: &Frame<'_>) -> Result<(), ErrorInfo> {
    let ai = s32_arg(frame, 0)?;
    let aj = s32_arg(frame, 1)?;
    let (n_lhs, n_col, n_rhs) = b_dims(frame, 3)?;
    let ax = arg::<T>(frame, 2)?;
    let x = arg::<T>(frame, 3)?;
    let out = engine::dot_raw(ai, aj, ax, x, n_lhs, n_col, n_rhs)?;
    write(ret::<T>(frame, 0)?, &out)
}

// ---- the 21 exported handlers ---------------------------------------------
//
// Each constructs a `Frame` from the raw pointer (tying decoded slices to the
// call), runs the body inside `guard` (panic containment + XLA error creation).
// TODO(hardening/stage-2): replace this block with a table-driven macro.

macro_rules! handler {
    ($name:ident, |$frame:ident| $body:expr) => {
        /// XLA typed-FFI handler symbol.
        ///
        /// # Safety
        ///
        /// `frame_ptr` must be the valid, non-null call frame XLA passes to
        /// this handler, and it must stay alive for the duration of the call.
        #[no_mangle]
        pub unsafe extern "C" fn $name(frame_ptr: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
            // SAFETY: `frame_ptr` is the valid call frame XLA passed to this
            // handler; it stays alive for the duration of the call.
            unsafe {
                guard(frame_ptr, || {
                    let $frame = &Frame::from_raw(frame_ptr);
                    $body
                })
            }
        }
    };
}

handler!(analyze, |frame| {
    let ai = s32_arg(frame, 0)?;
    let aj = s32_arg(frame, 1)?;
    let nc = s32_arg(frame, 2)?;
    if nc.len() != 1 {
        return Err(ErrorInfo::invalid("n_col must be a scalar"));
    }
    let sym = engine::analyze_raw(nc[0] as usize, ai, aj)?;
    u64_ret(frame, 0)?[0] = sym;
    Ok(())
});

handler!(free_symbolic, |frame| {
    let s = u64_arg(frame, 0)?;
    let mut status = 0;
    for &addr in s {
        status |= engine::free_symbolic_raw(addr);
    }
    i32_ret(frame, 0)?[0] = status;
    Ok(())
});

handler!(free_numeric, |frame| {
    let s = u64_arg(frame, 0)?;
    let status = engine::free_numeric_raw(s);
    i32_ret(frame, 0)?[0] = status;
    Ok(())
});

handler!(factor_f64, |frame| factor_impl::<f64>(frame));
handler!(factor_c128, |frame| factor_impl::<engine::C64>(frame));
handler!(refactor_f64, |frame| refactor_impl::<f64>(frame));
handler!(refactor_c128, |frame| refactor_impl::<engine::C64>(frame));
handler!(solve_f64, |frame| solve_impl::<f64>(frame));
handler!(solve_c128, |frame| solve_impl::<engine::C64>(frame));
handler!(solve_with_symbol_f64, |frame| {
    solve_with_symbol_impl_h::<f64>(frame, false)
});
handler!(solve_with_symbol_c128, |frame| {
    solve_with_symbol_impl_h::<engine::C64>(frame, false)
});
handler!(tsolve_with_symbol_f64, |frame| {
    solve_with_symbol_impl_h::<f64>(frame, true)
});
handler!(tsolve_with_symbol_c128, |frame| {
    solve_with_symbol_impl_h::<engine::C64>(frame, true)
});
handler!(solve_with_numeric_f64, |frame| {
    solve_with_numeric_impl_h::<f64>(frame, false)
});
handler!(solve_with_numeric_c128, |frame| {
    solve_with_numeric_impl_h::<engine::C64>(frame, false)
});
handler!(tsolve_with_numeric_f64, |frame| {
    solve_with_numeric_impl_h::<f64>(frame, true)
});
handler!(tsolve_with_numeric_c128, |frame| {
    solve_with_numeric_impl_h::<engine::C64>(frame, true)
});
handler!(refactor_and_solve_f64, |frame| {
    refactor_and_solve_impl_h::<f64>(frame)
});
handler!(refactor_and_solve_c128, |frame| {
    refactor_and_solve_impl_h::<engine::C64>(frame)
});
handler!(dot_f64, |frame| dot_impl::<f64>(frame));
handler!(dot_c128, |frame| dot_impl::<engine::C64>(frame));
