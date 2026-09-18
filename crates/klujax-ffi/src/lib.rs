//! `klujax-ffi`: expose the Rust KLU solver to JAX through XLA's typed FFI.
//!
//! The crate is built as a `cdylib`. Handlers are plain `extern "C"` symbols
//! registered from Python with `jax.ffi.pycapsule(ctypes_fnptr)`. No PyO3, no
//! C++.
//!
//! # Unsafe boundary
//!
//! `unsafe` is confined to the code that must touch foreign memory, and every
//! such site carries a `// SAFETY:` comment (enforced by
//! `clippy::undocumented_unsafe_blocks`):
//!
//! - [`klu`] — the KLU C FFI (`klu-sys`); the only module that calls into the
//!   SuiteSparse library. Callers get `Result`-returning safe functions.
//! - [`call_frame`] — decoding `XLA_FFI_CallFrame`. [`call_frame::Frame`] ties
//!   every decoded slice to the call's lifetime; slices are built only inside
//!   [`call_frame::Buf`]/[`call_frame::BufMut`], which validate rank/dims
//!   at construction.
//! - [`error`] — creating `XLA_FFI_Error` values and handling the metadata
//!   probe.
//! - [`handlers`] — the generated `extern "C"` entry points (one documented
//!   `unsafe { guard(...) }` scope each).
//!
//! [`engine`] (the numeric logic) is `#![forbid(unsafe_code)]`. The remaining
//! unsafe-operation budget is tracked by `tools/unsafe_budget.sh`.

pub mod call_frame;
pub mod capi;
pub mod engine;
pub mod error;
pub mod handlers;
pub mod klu;
pub mod xla_ffi;

pub use error::{guard, make_error, ErrorInfo};
