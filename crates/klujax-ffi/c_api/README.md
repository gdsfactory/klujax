# Vendored XLA typed-FFI header

`c_api.h` is an unmodified copy of the header bundled with **jaxlib 0.9.2**:

```
jaxlib/include/xla/ffi/api/c_api.h
```

- sha256: `85fc385c2d3a6b539a05b9cf4c3535aa24b4b41040f9e111c1f2c11b0e2fa539`
- `XLA_FFI_API_MAJOR = 0`, `XLA_FFI_API_MINOR = 3`

It is vendored (rather than discovered at build time via
`jax.ffi.include_dir()`) so that builds are hermetic and reproducible. The Rust
declarations derived from it live in `../src/xla_ffi.rs`, and their sizes/offsets
are asserted in that module's `abi_tests`. Update this file *and* those tests
together when bumping the supported jaxlib.
