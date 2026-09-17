//! `klujax-ffi`: expose the Rust KLU solver to JAX through XLA's typed FFI.
//!
//! The crate is built as a `cdylib`. Handlers are plain `extern "C"` symbols
//! registered from Python with `jax.ffi.pycapsule(ctypes_fnptr)`. No PyO3, no
//! C++.

pub mod call_frame;
pub mod capi;
pub mod error;
pub mod handlers;
pub mod xla_ffi;

pub use error::{guard, make_error, ErrorInfo};
