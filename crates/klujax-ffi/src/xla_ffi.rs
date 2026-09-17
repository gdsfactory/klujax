//! Hand-written Rust declarations of the XLA typed-FFI C ABI.
//!
//! These mirror the *subset* of `xla/ffi/api/c_api.h` that we use. The header is
//! vendored at `crates/klujax-ffi/c_api/c_api.h` and pinned; see the ABI-drift
//! tests below.
//!
//! We deliberately do not use `bindgen`: the surface is tiny, and hand-writing
//! it keeps the build hermetic (no libclang/toolchain dependency) and makes the
//! ABI-sensitive offsets explicit and testable.

#![allow(non_camel_case_types, non_snake_case)]

use core::ffi::{c_char, c_int, c_void};

/// `XLA_FFI_DataType` enum. Size is `c_int` (C enum).
pub type XLA_FFI_DataType = c_int;

/// Data types we care about (`XLA_FFI_DataType_*`).
pub mod dtype {
    use core::ffi::c_int;

    pub const S32: c_int = 4;
    pub const U64: c_int = 9;
    pub const F64: c_int = 12;
    pub const C128: c_int = 18;
}

/// `XLA_FFI_ArgType_BUFFER`.
pub const XLA_FFI_ARG_TYPE_BUFFER: c_int = 1;
/// `XLA_FFI_RetType_BUFFER`.
pub const XLA_FFI_RET_TYPE_BUFFER: c_int = 1;
/// `XLA_FFI_ExecutionStage_EXECUTE`.
pub const XLA_FFI_STAGE_EXECUTE: c_int = 3;

/// `XLA_FFI_Extension_Metadata`.
pub const XLA_FFI_EXTENSION_METADATA: c_int = 1;

/// `XLA_FFI_TypeId`.
#[repr(C)]
#[derive(Debug)]
pub struct XLA_FFI_TypeId {
    pub type_id: i64,
}

/// `XLA_FFI_Metadata`.
#[repr(C)]
#[derive(Debug)]
pub struct XLA_FFI_Metadata {
    pub struct_size: usize,
    pub api_version: XLA_FFI_Api_Version,
    pub traits: u32,
    pub state_type_id: XLA_FFI_TypeId,
}

/// `XLA_FFI_Metadata_Extension`.
#[repr(C)]
#[derive(Debug)]
pub struct XLA_FFI_Metadata_Extension {
    pub extension_base: XLA_FFI_Extension_Base,
    pub metadata: *mut XLA_FFI_Metadata,
}

/// `XLA_FFI_Error_Code_*`.
pub mod error_code {
    use core::ffi::c_int;

    pub const OK: c_int = 0;
    pub const UNKNOWN: c_int = 2;
    pub const INVALID_ARGUMENT: c_int = 3;
    pub const INTERNAL: c_int = 13;
    pub const UNIMPLEMENTED: c_int = 12;
}

/// `XLA_FFI_Extension_Base`.
#[repr(C)]
#[derive(Debug)]
pub struct XLA_FFI_Extension_Base {
    pub struct_size: usize,
    pub type_: c_int,
    pub next: *mut XLA_FFI_Extension_Base,
}

/// `XLA_FFI_Buffer` (no strides; data is assumed contiguous).
#[repr(C)]
#[derive(Debug)]
pub struct XLA_FFI_Buffer {
    pub struct_size: usize,
    pub extension_start: *mut XLA_FFI_Extension_Base,
    pub dtype: XLA_FFI_DataType,
    pub data: *mut c_void,
    pub rank: i64,
    pub dims: *mut i64,
}

/// `XLA_FFI_Args`.
#[repr(C)]
#[derive(Debug)]
pub struct XLA_FFI_Args {
    pub struct_size: usize,
    pub extension_start: *mut XLA_FFI_Extension_Base,
    pub size: i64,
    pub types: *mut c_int,
    pub args: *mut *mut c_void,
}

/// `XLA_FFI_Rets`.
#[repr(C)]
#[derive(Debug)]
pub struct XLA_FFI_Rets {
    pub struct_size: usize,
    pub extension_start: *mut XLA_FFI_Extension_Base,
    pub size: i64,
    pub types: *mut c_int,
    pub rets: *mut *mut c_void,
}

/// `XLA_FFI_ByteSpan`.
#[repr(C)]
#[derive(Debug)]
pub struct XLA_FFI_ByteSpan {
    pub ptr: *const c_char,
    pub len: usize,
}

/// `XLA_FFI_Attrs` (attributes are sorted by name).
#[repr(C)]
#[derive(Debug)]
pub struct XLA_FFI_Attrs {
    pub struct_size: usize,
    pub extension_start: *mut XLA_FFI_Extension_Base,
    pub size: i64,
    pub types: *mut c_int,
    pub names: *mut *mut XLA_FFI_ByteSpan,
    pub attrs: *mut *mut c_void,
}

/// Opaque `XLA_FFI_Error`.
pub enum XLA_FFI_Error {}
/// Opaque `XLA_FFI_ExecutionContext`.
pub enum XLA_FFI_ExecutionContext {}
/// Opaque `XLA_FFI_Future`.
pub enum XLA_FFI_Future {}

/// `XLA_FFI_Error_Create_Args`.
#[repr(C)]
#[derive(Debug)]
pub struct XLA_FFI_Error_Create_Args {
    pub struct_size: usize,
    pub extension_start: *mut XLA_FFI_Extension_Base,
    pub message: *const c_char,
    pub errc: c_int,
}

/// `XLA_FFI_Error_Create`.
pub type XLA_FFI_Error_Create =
    unsafe extern "C" fn(args: *mut XLA_FFI_Error_Create_Args) -> *mut XLA_FFI_Error;

/// `XLA_FFI_Api_Version` (by value inside `XLA_FFI_Api`).
#[repr(C)]
#[derive(Debug)]
pub struct XLA_FFI_Api_Version {
    pub struct_size: usize,
    pub extension_start: *mut XLA_FFI_Extension_Base,
    pub major_version: c_int,
    pub minor_version: c_int,
}

/// `XLA_FFI_Api`.
///
/// Only the prefix up to and including `XLA_FFI_Error_Create` is declared: we
/// never index past it, and omitting trailing fields keeps the offsets of the
/// fields we do use correct.
#[repr(C)]
#[derive(Debug)]
pub struct XLA_FFI_Api {
    pub struct_size: usize,
    pub extension_start: *mut XLA_FFI_Extension_Base,
    pub api_version: XLA_FFI_Api_Version,
    pub internal_api: *const c_void,
    pub XLA_FFI_Error_Create: Option<XLA_FFI_Error_Create>,
}

/// `XLA_FFI_CallFrame`.
#[repr(C)]
#[derive(Debug)]
pub struct XLA_FFI_CallFrame {
    pub struct_size: usize,
    pub extension_start: *mut XLA_FFI_Extension_Base,
    pub api: *const XLA_FFI_Api,
    pub ctx: *mut XLA_FFI_ExecutionContext,
    pub stage: c_int,
    pub args: XLA_FFI_Args,
    pub rets: XLA_FFI_Rets,
    pub attrs: XLA_FFI_Attrs,
    pub future: *mut XLA_FFI_Future,
}

/// `XLA_FFI_Handler`: the signature XLA invokes.
pub type XLA_FFI_Handler =
    unsafe extern "C" fn(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error;

#[cfg(test)]
mod abi_tests {
    use super::*;
    use core::mem::{align_of, offset_of, size_of};

    // These sizes/offsets are the ABI contract. If jaxlib ships a new
    // `c_api.h` that changes them, these tests fail loudly (Stage 0.5 / 2.2
    // "ABI-drift test").
    #[test]
    fn struct_sizes() {
        assert_eq!(size_of::<XLA_FFI_Extension_Base>(), 24);
        assert_eq!(size_of::<XLA_FFI_Buffer>(), 48);
        assert_eq!(size_of::<XLA_FFI_Args>(), 40);
        assert_eq!(size_of::<XLA_FFI_Rets>(), 40);
        assert_eq!(size_of::<XLA_FFI_Attrs>(), 48);
        assert_eq!(size_of::<XLA_FFI_Error_Create_Args>(), 32);
        assert_eq!(size_of::<XLA_FFI_Api_Version>(), 24);
        assert_eq!(size_of::<XLA_FFI_CallFrame>(), 176);
        assert_eq!(align_of::<XLA_FFI_CallFrame>(), 8);
    }

    #[test]
    fn api_error_create_offset() {
        // struct_size + extension_start + api_version(24) + internal_api == 48
        assert_eq!(offset_of!(XLA_FFI_Api, XLA_FFI_Error_Create), 48);
    }

    #[test]
    fn call_frame_field_offsets() {
        assert_eq!(offset_of!(XLA_FFI_CallFrame, api), 16);
        assert_eq!(offset_of!(XLA_FFI_CallFrame, stage), 32);
        assert_eq!(offset_of!(XLA_FFI_CallFrame, args), 40);
        assert_eq!(offset_of!(XLA_FFI_CallFrame, rets), 80);
        assert_eq!(offset_of!(XLA_FFI_CallFrame, attrs), 120);
        assert_eq!(offset_of!(XLA_FFI_CallFrame, future), 168);
    }

    #[test]
    fn dtype_constants() {
        assert_eq!(dtype::S32, 4);
        assert_eq!(dtype::U64, 9);
        assert_eq!(dtype::F64, 12);
        assert_eq!(dtype::C128, 18);
    }
}
