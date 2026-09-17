//! The XLA typed-FFI handler symbols registered with `jax.ffi.register_ffi_target`.
//!
//! Stage 1 provides the full symbol surface as stubs so the Python `ctypes`
//! loader and smoke test can be exercised. The real implementations land in
//! Stage 2.

use crate::error::{guard, ErrorInfo};
use crate::xla_ffi::{XLA_FFI_CallFrame, XLA_FFI_Error};

macro_rules! stub_handler {
    ($($name:ident),* $(,)?) => {
        $(
            /// XLA FFI handler (Stage 1 stub).
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
                            " is not yet implemented (Stage 1 stub)"
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
