# klu-sys

Raw FFI bindings to the SuiteSparse C KLU library. The sources are statically
compiled by `build.rs` and linked into the `klujax_ffi` cdylib (no runtime
`libklu`/`libsuitesparse`).

## Pinned version

SuiteSparse is vendored as the `vendor/SuiteSparse` git submodule, pinned to
**v7.5.0** (see the root `.gitmodules`). `klujax-ffi` only calls the int32
`klu_*` / `klu_z_*` entry points, so `build.rs` skips the int64 (`*_l*` /
`*_zl*`) variants.

## Layout safety

`klu_common`, `klu_symbolic` and `klu_numeric` are hand-mirrored from
`KLU/Include/klu.h`. The `layout_matches_vendored_header` test compiles a small
C program against the vendored header and asserts `sizeof`/`offsetof` against
the Rust structs, so a header change that alters the layout fails a test instead
of corrupting memory at runtime. (`klu_numeric` is opaque and only ever used
behind a pointer, so it needs no layout mirror.)

## Bumping SuiteSparse

1. `git -C vendor/SuiteSparse fetch --tags && git -C vendor/SuiteSparse checkout <tag>`
2. `git add vendor/SuiteSparse` (updates the submodule gitlink)
3. `cargo test -p klu-sys` (runs the layout test) and the full test suite.
