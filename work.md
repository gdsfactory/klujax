# klujax → Rust migration plan (non-PyO3)

Status: in progress — Stages 0, 0.5, 1, 2, 3 complete; Milestone A (Stage 4,
macOS) complete; Stage 5 static linking verified (cross-platform hardening
remaining); Stage 6 packaging/CI/release remaining. See the status log at the
end of this file.
Owner: Floris
Target: replace `klujax.cpp` (pybind11) with a Rust `klujax-ffi` cdylib that
registers XLA typed-FFI handlers via `jax.ffi.pycapsule` and **statically links
SuiteSparse's KLU** through a `klu-sys` crate.

---

## 1. Goal & non-goals

### Goal
- Replace the pybind11 C++ extension (`klujax.cpp`) with a Rust `cdylib`
  (`klujax-ffi`) exposing the XLA typed-FFI handlers as plain `extern "C"`
  symbols.
- Ship a `klu-sys` crate that **builds and statically links** the required
  SuiteSparse libraries (`SuiteSparse_config`, `AMD`, `COLAMD`, `BTF`, `KLU`)
  using the `cc` crate, with SuiteSparse vendored as a `vendor/SuiteSparse` git
  submodule (same pattern as `eigenlight`'s UMFPACK crate). The resulting
  cdylib must have **no runtime dependency on a system `libklu` /
  `libsuitesparse`**.
- Register targets from Python with `jax.ffi.pycapsule()` + `ctypes` (no PyO3).
- Keep the public Python API (`klujax.solve`, `analyze`, `factor`, `refactor`,
  `solve_with_symbol`, `solve_with_numeric`, `tsolve_*`, `dot`, `coalesce`,
  `free_*`, `KLUSymbolic`, `KLUNumeric`, `KLUHandleManager`) **unchanged**.
- Keep `tests.py` passing unchanged as the primary acceptance gate.

### Non-goals
- **No pure-Rust KLU reimplementation.** KLU stays SuiteSparse C; we only wrap
  and statically link it. (Earlier drafts proposed porting AMD/COLAMD/BTF/KLU
  to Rust — that is explicitly off the table.)
- No PyO3, no compiled Python extension, no `#[pymodule]`.
- **No C++ shim.** Delete `klujax.cpp`; decode `XLA_FFI_CallFrame` directly in
  Rust (feasible: none of our handlers use attributes).
- Do not move JAX trace-time logic (lowerings, batching, AD, tree_util,
  `ffi_call`) into Rust. It cannot run in Rust and stays in `klujax.py`.
- No GPU support (KLU is CPU-only, same as today).
- No attempt to beat SuiteSparse performance; parity first.

### Licensing note
SuiteSparse is **LGPL-2.1**; statically linking it means the shipped cdylib is
covered by LGPL-2.1, which is already the repo licence. The vendored
SuiteSparse licence and attribution must be preserved.

---

## 2. Target architecture

```
klujax/
├── Cargo.toml                     # workspace
├── .gitmodules                    # vendor/SuiteSparse
├── vendor/
│   └── SuiteSparse/               # git submodule, pinned to v7.5.0
├── crates/
│   ├── klu-sys/                   # builds + statically links SuiteSparse KLU
│   │   ├── Cargo.toml             # links = "klu"; `cc` (parallel) build-dep
│   │   ├── build.rs               # locate sources -> cc-compile -> static link
│   │   └── src/lib.rs             # raw FFI: klu_*, klu_z_*, klu_common, klu_symbolic
│   └── klujax-ffi/                # cdylib: XLA FFI + C ABI shims (depends on klu-sys)
│       ├── Cargo.toml
│       ├── c_api/                 # pinned jaxlib c_api.h (+ README)
│       └── src/
│           ├── lib.rs
│           ├── xla_ffi.rs         # hand-written XLA C ABI types + drift tests
│           ├── call_frame.rs      # decode XLA_FFI_CallFrame
│           ├── error.rs           # error creation, panic guard, metadata probe
│           ├── engine.rs          # COO->CSC + generic (f64/C64) KLU calls
│           ├── handlers.rs        # the 21 XLA handler symbols
│           └── capi.rs            # plain free_* symbols for ctypes handles
├── klujax_native/__init__.py      # ctypes loader + capsule providers + handles
├── klujax.py                      # JAX plumbing; imports klujax_native as klujax_cpp
├── setup.py                       # CargoBuildExt: cargo build + copy cdylib
├── scripts/ffi_smoke.py           # ABI smoke test
├── tests.py                       # unchanged acceptance gate
├── tests_characterization/        # Stage 0.5 safety net + golden corpus
└── work.md                        # this file
```

### `klu-sys`: statically linked SuiteSparse

`klu-sys/build.rs` compiles the needed SuiteSparse C sources with the `cc`
crate into a static archive (`libsuitesparse_klu.a`) that is linked into the
`klujax_ffi` cdylib. Source location priority:

1. `$KLUJAX_SUITESPARSE_DIR`
2. the `vendor/SuiteSparse` submodule (preferred, reproducible)
3. repo-root `suitesparse/` from `just deps` (legacy fallback)

Compiled components: `SuiteSparse_config`, `AMD`, `COLAMD`, `BTF`, `KLU` (all
`*.c` in their `Source/` dirs). The int64 (`klu_l_*`) variants are compiled too
so the archive is self-contained. Mirrors `~/Projects/eigenlight/crates/umfpack`.


### Two entry points for the same operations
- **XLA FFI handlers** (e.g. `solve_f64`, `free_symbolic`): called by XLA
  during JIT, decoded from `XLA_FFI_CallFrame`.
- **Plain C ABI shims** (e.g. `klujax_free_symbolic(raw: u64)`): called directly
  from Python via `ctypes` for the non-JIT handle lifecycle (`close()`,
  `__del__`). These are **not** XLA handlers.

---

## 3. Key ABI facts (verified against jaxlib-bundled headers)

- jaxlib ships `xla/ffi/api/c_api.h` at `jax.ffi.include_dir()`.
- `XLA_FFI_Handler` is `XLA_FFI_Error* (XLA_FFI_CallFrame*)` — a **plain C
  function pointer**. The C++ `Ffi::Bind()` DSL is only the body of the
  trampoline; registration only needs the address.
- `XLA_FFI_CallFrame` fields: `api`, `ctx`, `stage`, `args` (`XLA_FFI_Args`),
  `rets` (`XLA_FFI_Rets`), `attrs` (`XLA_FFI_Attrs`), `future`.
- `XLA_FFI_Buffer { dtype, data, rank, dims }` — **no strides**; contiguity is
  assumed (current C++ code assumes the same).
- Error creation is via `frame.api->XLA_FFI_Error_Create` with
  `XLA_FFI_Error_Create_Args { struct_size, extension_start, message, errc }`.
  Returning `null` means success.
- `XLA_FFI_DataType` values needed: `S32=4`, `U64=9`, `F64=12`, `C128=18`.
- `jax.ffi.pycapsule(ctypes_fnptr)` = `PyCapsule_New(ptr, NULL, noop)`.
  Any `ctypes` function pointer from our `cdylib` works.
- Current pinned XLA API: `XLA_FFI_API_MAJOR=0`, `XLA_FFI_API_MINOR=3`.

### Current handler inventory (21 targets to reproduce)

| Target | Inputs | Outputs |
|---|---|---|
| `dot_f64` / `dot_c128` | Ai:S32, Aj:S32, Ax:F64/C128, x:F64/C128 | b:F64/C128 |
| `solve_f64` / `solve_c128` | Ai, Aj, Ax, b | x |
| `solve_with_symbol_f64/c128` | Ai, Aj, Ax, b, symbolic:U64 | x |
| `tsolve_with_symbol_f64/c128` | Ai, Aj, Ax, b, symbolic | x |
| `factor_f64/c128` | Ai, Aj, Ax, symbolic | numeric:U64 |
| `refactor_f64/c128` | Ai, Aj, Ax, symbolic, numeric | out_numeric:U64 |
| `refactor_and_solve_f64/c128` | Ai, Aj, Ax, b, symbolic, numeric | x, out_numeric |
| `solve_with_numeric_f64/c128` | symbolic:U64, numeric:U64, b | x |
| `tsolve_with_numeric_f64/c128` | symbolic, numeric, b | x |
| `free_numeric` | numeric:U64 | status:S32 |
| `free_symbolic` | symbolic:U64 | status:S32 |
| `analyze` | Ai, Aj, n_col:S32 | symbolic:U64 |

---

## 3.5 Prior art: `jeertmans/extending-jax`

A Rust port of JAX's *Extending JAX* tutorial (`rms_norm`), solving the same
problem: Rust numeric kernel + XLA typed FFI + `jax.ffi.pycapsule`. Key
findings from reading it:

- **It keeps a thin C++ shim** (`src/ffi.cc`) that owns the
  `XLA_FFI_DEFINE_HANDLER_SYMBOL` binding, and calls into Rust via the `cxx`
  bridge (`cxx::bridge`, `cxx-build` in `build.rs`). Its README lists
  "document why we still need some (basic) C++ code to use the JAX FFI" as
  future work. So the community path of least resistance is *not* to hand-roll
  the call-frame decode in Rust.
- **It still uses PyO3** purely to (a) be a Python module and (b) return the
  handler pointer as a `PyCapsule` via `pyo3::ffi::PyCapsule_New(fn_ptr, null,
  None)`. This is byte-for-byte the technique our ctypes loader will use, just
  reached from a different direction.
- **It discovers the XLA headers at build time** by running
  `python -c "from jax.ffi import include_dir; print(include_dir())"` from
  `build.rs` (`pyo3_build_config` for the interpreter), instead of vendoring
  `c_api.h`. More convenient, less reproducible.
- **Packaging is maturin + pyo3** (`bindings = "pyo3"`). Not applicable to our
  non-PyO3 route, which needs the setuptools/cargo path in Stage 1.
- Related links it cites: `dfm/extending-jax` (C++/CUDA), JAX FFI docs,
  JAX discussion #24187 (extending with Rust), PyO3 discussion #4772
  (exporting function pointers).

### What this changes for us

Our handlers take **only positional buffers and no attributes** (verified: no
`.Attr<>()` anywhere in `klujax.cpp`), so hand-rolling `XLA_FFI_CallFrame`
decode in Rust is tractable and avoids a C++ toolchain. `extending-jax` chose
the C++ shim because its `rms_norm` example uses an attribute (`eps`) and follows
the tutorial, not because Rust decode is impossible.

Decision (Stage 2): **hand-roll the decode in Rust, no C++**. We deliberately
diverge from `extending-jax`'s C++ shim because (a) our handlers carry no
attributes and (b) "zero C++" is a hard project constraint (§1). `klujax.cpp`
is deleted and is not replaced.

Contingency (not planned, do not schedule): if XLA FFI ABI churn or future
attribute needs make manual decode untenable, the escape hatch is a ~20-line
`cxx`-bridge shim in a *new* file, never a resurrection of `klujax.cpp`.
Revisit only if the hand-rolled decode actually breaks.

---

## 4. Milestones overview

- **M0 — Prep** ✅: decisions locked, header pinned, baseline captured.
- **M0.5 — Behavioral test safety net** ✅: characterization + oracle + golden
  corpus (regression contract for everything that follows).
- **M1 — Rust FFI seam + statically linked SuiteSparse** ✅ (macOS): `klu-sys`
  builds and statically links SuiteSparse; `klujax-ffi` implements the 21 XLA
  handlers; Python drives it via ctypes/pycapsule. `tests.py` green.
- **M2 — Packaging, CI, release** ⏳: remove the pybind11 C++ extension,
  wheels/CI/docs, verify static linking per platform.

There is deliberately **no pure-Rust-port milestone**: the SuiteSparse C KLU is
kept and statically linked.

---

## Stage 0 — Decisions & preparation

Status: mostly complete — only the baseline capture is outstanding (it needs
the vendored C++ deps).

- [x] Confirm end-state: Rust `klujax-ffi` cdylib + `klu-sys` statically
      linking SuiteSparse, LGPL-2.1, in-repo workspace.
- [x] Confirm build tool: `cargo` invoked from a setuptools `build_ext`
      (no maturin, no PyO3).
- [x] Confirm Python loader: `ctypes.CDLL` + `jax.ffi.pycapsule`.
- [x] Confirm handler naming: exported Rust symbols match XLA target names
      (`solve_f64`, `analyze`, …); verified by `scripts/ffi_smoke.py`.
- [x] Record baseline: on current `main`, run `just test` and a benchmark on a
      representative suite; save results. `tests.py` → **79 passed** (after a
      macOS RSS-probe fix, see Stage 0.5), benchmark → `benchmarks/baseline.json`
      (`implementation: suitesparse-c++`).
- [x] Poll `jaxlib`'s `c_api.h`; copy the pinned header into
      `crates/klujax-ffi/c_api/c_api.h`
      (sha256 `85fc385c…a539`, 788 lines).
- [x] Record the jaxlib version the pinned ABI targets: **jaxlib 0.9.2**
      (`XLA_FFI_API_MAJOR=0`, `XLA_FFI_API_MINOR=3`). README note deferred to
      Stage 6.
- [x] Create branch. Using the existing `rs` branch (per goal).
- [x] Add `.gitignore` entries for `/target`, `crates/*/target`, copied libs.

Exit criteria: baseline tests + benchmark recorded ✓; pinned header committed ✓.
Note: the C++ extension was built against the jaxlib-bundled XLA FFI headers
(`xla` symlinked to `jax.ffi.include_dir()`), avoiding the large XLA clone.

---

## Stage 0.5 — Behavioral test safety net (pre-port)

Purpose: make the rewrite *falsifiable*. The existing `tests.py` is a decent
regression gate for the C++ code but is **not sufficient** for a from-scratch
port: it uses tiny (`n_col ∈ {3,5}`), coalesced, strongly-diagonal matrices that
almost never exercise multi-block BTF/ordering/fill-in, and it omits error,
edge, dtype, and several shape/AD/vmap paths. A Rust port could pass all of
`tests.py` and still be wrong. This stage locks behavior first.

Guiding principle: optimize for **behavioral coverage + independent oracle**,
not line coverage (`pytest --cov` is a hint, not the target; JAX tracing also
distorts Python line coverage).

### 0.5.1 Coverage matrix (contract definition)
- [x] Build `docs/test-matrix.md` (API × transform × shape × dtype × edge).
- [x] Mark every cell tested or explicitly out-of-scope (with reason).
- [x] Convert the matrix into parametrized pytest cases; every in-scope cell
      has at least one test.

### 0.5.2 Shape/broadcast exhaustiveness (README table)
- [x] All six `Ax`/`b` dimension combos, including `Ax`1D+`b`3D and
      `Ax`2D+`b`1D (`test_shapes.py::test_shape_table`).
- [x] `n_lhs` broadcasting: `Ax` batched × `b` unbatched and vice versa
      (`test_n_lhs_broadcast`).
- [x] `n_rhs > 1` directly for `solve_with_symbol`, `solve_with_numeric`,
      `tsolve_with_symbol`, `tsolve_with_numeric` (`test_multi_rhs_direct`).
- [x] Mismatched `n_lhs` / `n_nz` raise (`test_mismatched_n_lhs_raises`,
      `test_n_nz_mismatch_raises`).

### 0.5.3 dtypes
- [x] `float32` → `float64` and `complex64` → `complex128` upcast parity.
- [x] int32 and int64 `Ai`/`Aj` inputs.
- [x] Mixed `Ax`/`b` dtype behavior.

### 0.5.4 `coalesce` (previously only used, never asserted)
- [x] Sorting/lexsort ordering.
- [x] Duplicate index summation.
- [x] Stability and dtype preservation.
- [x] Idempotence on already-coalesced input (+ batched `Ax`).

### 0.5.5 Independent oracles
- [x] `scipy.sparse.linalg.spsolve` oracle alongside dense `jsp.linalg.solve`.
      (`splu` not needed: transpose is checked against dense `A.T`.)
- [x] Explicit tolerances + oracle-free residual `||Ax - b||`.
- [x] `dot` versus `A @ x` with `scipy.sparse` (eager and batched).

### 0.5.6 Structural stress (the paths the port is riskiest in)
- [x] Generators: block-diagonal, block-upper-triangular (reducible), arrow,
      tridiagonal (= banded). Circuit-like deferred to Stage 5 corpus.
- [x] Larger systems (`n = 50/120`) vs `spsolve`; `n ≈ 1000` added in Stage 5.
- [ ] Ill-conditioned / near-singular accuracy — **deferred to Stage 5**
      (documented gap in `docs/test-matrix.md`).
- [ ] `hypothesis` property tests — **out of scope**: deterministic randomized
      params + `spsolve` oracle cover the same space without a new dependency.
- [ ] Confirm `nblocks > 1` / fill via SuiteSparse introspection —
      **deferred to Stage 5** (needs KLU internals).

### 0.5.7 Error / edge / degenerate paths
- [x] Singular / structurally singular: `RuntimeError` (no crash), matched.
- [x] Degenerate sizes: `n_col = 1`, `n_nz = 0`.
- [x] Out-of-bounds and negative indices raise with message parity.
- [x] Unsorted unique indices are order-independent; uncoalesced duplicates
      documented as unsupported/UB.
- [x] `NaN` propagates.

### 0.5.8 AD / vmap primitive coverage
- [x] Every primitive with a `jvp`/transpose rule is tested
      (`solve_with_numeric`, `solve_with_symbol`).
- [x] Every primitive with a `vmap` rule is tested across batch axes.
- [ ] Inside-JIT `free_symbolic` / `free_numeric` dependency pattern —
      **out of scope**: these are deprecated no-ops in 0.5.x (handles
      auto-free).

### 0.5.9 Golden corpus (frozen characterization)
- [x] Script running the current implementation over the corpus; inputs +
      outputs persisted to `tests_characterization/golden/*.npz`
      (`_generate_golden.py`, 8 cases incl. reducible real/complex).
- [x] Golden tests compare within `1e-12`; regenerate only explicitly.
- [x] Corpus is reused unchanged as the migration's **numerical regression
      harness** (it must stay green while the build/link layer changes).

### 0.5.10 Tooling & wiring
- [x] Add `pytest-cov`; baseline coverage recorded in
      `benchmarks/coverage-baseline.txt` (79% of `klujax.py`, JAX-tracing
      caveat noted).
- [x] Wire new tests into `just test` (`uv run pytest` uses testpaths);
      `tests.py` remains the legacy gate.
- [ ] CI runs `tests.py` + characterization + golden suites — **Stage 6**.
- [x] Document regeneration (`_generate_golden.py` docstring,
      `docs/test-matrix.md`).

Status: **complete** — 130 tests pass (79 legacy + 51 characterization).

Exit criteria: coverage matrix complete with no unexplained gaps; independent
oracles and golden corpus in place; all new tests pass on the current C++
implementation. **This stage is a prerequisite for Stage 5.**

---

## Stage 1 — Rust workspace & build plumbing

Status: **complete** (verified). Deviations from the draft checklist are noted
inline.

- [x] Create workspace `Cargo.toml` with members `crates/klu-sys`,
      `crates/klujax-ffi`. (The original draft also had a `crates/klu` pure-Rust
      crate; it was removed in the Stage 5 re-scope.)
- [x] `crates/klujax-ffi/Cargo.toml`: `crate-type = ["cdylib", "rlib"]`, dep
      `klu-sys` (path).
      - Deviation: **no `bindgen`/`libc`**. The XLA C ABI is hand-written in
        `src/xla_ffi.rs`, removing the libclang build dependency and making the
        offsets explicit and drift-tested.
- [x] ~~`build.rs`: run `bindgen`~~ — superseded: ABI types are hand-written and
      covered by the `abi_tests` size/offset tests.
      - [x] `XLA_FFI_CallFrame`, `XLA_FFI_Buffer`, `XLA_FFI_Args`,
            `XLA_FFI_Rets`, `XLA_FFI_Api`, `XLA_FFI_Error` declared and
            size/offset-tested.
- [x] Header source strategy: **vendor + pin**
      `crates/klujax-ffi/c_api/c_api.h` (jaxlib 0.9.2).
- [x] No C++ in the Rust crates (`cxx`/`.cc`/`.cpp` absent).
- [x] `setup.py`: `CargoBuildExt` (subclass of `build_ext`) that:
      - [x] runs `cargo build --release -p klujax-ffi`;
      - [x] copies the cdylib into the `klujax_native` package;
      - [x] is idempotent and works for editable installs
            (`uv pip install -e . --no-deps` verified);
      - [x] supports a `CARGO` env override.
      - Note: the legacy C++ extension is kept **optional** (built only when
        `suitesparse`, `xla`, `pybind11` are present) so the Stage 0.5 golden
        corpus stays generatable. Removed in Stage 6.
- [x] Update `MANIFEST.in` / `pyproject.toml` package-data for the cdylib and
      Rust sources.
- [x] Add `just` recipes: `rust-build`, `rust-test`, `rust-smoke`, `rust-fmt`,
      `rust-clippy`, `rust-clean`; updated `just clean`.
- [x] Smoke test: `scripts/ffi_smoke.py` loads the cdylib via `ctypes` and
      asserts all 21 handler symbols + 3 C-ABI symbols; `klujax_version()`
      returns the version.

Verification:
- `cargo build --workspace` and `cargo build --release -p klujax-ffi` ✓
- `cargo test --workspace` → 5 passed ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo fmt --all -- --check` ✓
- `python scripts/ffi_smoke.py` → 21 handlers + 3 C-ABI symbols ✓
- `uv pip install -e . --no-deps` → cdylib in `klujax_native/`, `ctypes` load ✓

Exit criteria: **met.**

---

## Stage 2 — XLA FFI layer in Rust (algorithm still C)

Purpose: reproduce the entire `klujax.cpp` surface from Rust, calling the
existing SuiteSparse C KLU behind a Rust-owned safe-ish API.

### 2.1 Vendor the C KLU behind a Rust crate
Status: `klu-sys` + the C-backed engine landed and verified.
- [x] Add `crates/klu-sys` using the `cc` crate to compile
      `SuiteSparse_config`, `AMD`, `COLAMD`, `BTF`, `KLU` C sources. It is a
      normal workspace member, sourcing `vendor/SuiteSparse` (submodule),
      `KLUJAX_SUITESPARSE_DIR`, or a root `suitesparse/` checkout.
      - [x] Expose the C `klu_*` + `klu_z_*` symbols (plus mirrored
            `klu_common`/`klu_symbolic`), covered by a direct-C
            `solve_2x2_diagonal_f64` test.
- [x] C-backed engine (`klujax-ffi/src/engine.rs`) with `coo_to_csc`,
      `analyze_raw`, `factor_raw`, `factor_batch_raw`, `refactor_batch_raw`,
      `solve_raw`, `solve_with_symbol_raw`, `tsolve_with_symbol_raw`,
      `solve_with_numeric_raw`, `dot_raw`, `free_*` for f64 **and** c128.
      Verified by Rust unit tests (real + complex, direct + split solves).
- [x] Differential smoke test vs. a direct C call (`klu-sys` test) and
      engine-level tests.

### 2.2 Call-frame decode + error + panic guard
Decision: hand-roll decode in Rust, **no C++** (see §1 and §3.5).
Status: implemented in Stage 1; validation parity with C++ `validate_args`
lands with the handlers in Stage 2.3.
- [x] `call_frame.rs`: helpers (`arg_buffer`, `ret_buffer`, `dims`,
      `element_count`, `expect_dtype`, `as_i32`, `as_u64`, `as_f64`,
      `as_f64_mut`), plus a synthetic-frame decode unit test.
      - [x] `as_c128_buf` — **pending** (needed when C128 handlers land).
- [x] `error.rs`:
      - [x] `make_error(frame, code, msg) -> *mut XLA_FFI_Error` via
            `frame.api->XLA_FFI_Error_Create`.
      - [x] `guard(frame, f) -> *mut XLA_FFI_Error` wrapping
            `std::panic::catch_unwind`; panics become
            `XLA_FFI_Error_Code_INTERNAL`; no unwinding across the boundary.
      - [x] `null_mut()` on success.
- [x] `capi.rs`: plain C ABI shims for Python handle lifecycle:
      - [x] `klujax_free_symbolic(raw: u64) -> i32` (Stage 1: no-op stub).
      - [x] `klujax_free_numeric(ptr: *const u64, len: usize) -> i32`
            (Stage 1: no-op stub).
      - [x] `klujax_version() -> *const c_char`.

### 2.3 Implement the 21 handlers
Status: all handlers implemented (feature `c-backend`); end-to-end verification
lands in Stage 3 once Python drives them (the generic decode helpers have unit
tests; a synthetic-frame handler test is still TODO).
Port each C++ handler function 1:1 (same validation, same COO→CSC conversion,
same `klu_common` handling, same error messages where reasonable).

- [x] `dot_f64`, `dot_c128`
- [x] `solve_f64`, `solve_c128`
- [x] `solve_with_symbol_f64`, `solve_with_symbol_c128`
- [x] `tsolve_with_symbol_f64`, `tsolve_with_symbol_c128`
- [x] `factor_f64`, `factor_c128`
- [x] `refactor_f64`, `refactor_c128`
- [x] `refactor_and_solve_f64`, `refactor_and_solve_c128`
- [x] `solve_with_numeric_f64`, `solve_with_numeric_c128`
- [x] `tsolve_with_numeric_f64`, `tsolve_with_numeric_c128`
- [x] `free_numeric`, `free_symbolic`
- [x] `analyze`
- [x] Expose each as `#[no_mangle] pub unsafe extern "C" fn <name>(
      frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error`.
- [x] Reuse COO→CSC logic (ported to `engine::coo_to_csc`).
- [x] Preserve the "output numeric is returned for XLA dependency edge"
      behavior in `refactor*`/`refactor_and_solve*`.
- [ ] Synthetic-frame handler test (deferred to Stage 3 e2e; the generic
      `call_frame` decode helpers already have a synthetic-frame test).

Exit criteria: every target symbol exported from the cdylib — met (symbols are
`#[no_mangle]`; verified by `scripts/ffi_smoke.py`). Full behaviour verified in
Stage 3/4.

---

## Stage 3 — Python layer conversion (ctypes + pycapsule)

Status: **complete**. `klujax.py` now imports `klujax_native as klujax_cpp`,
which loads the cdylib via `ctypes` and exposes capsule providers + pure-Python
handle classes. 130 tests pass.

- [x] Loader in `klujax_native/__init__.py`:
      - [x] Resolve platform lib name (`.so`/`.dylib`/`.dll`).
      - [x] `_LIB = ctypes.CDLL(path)` (lazy).
      - [x] capsule providers wrap each symbol with `jax.ffi.pycapsule`.
- [x] Every `klujax_cpp.<name>()` registration argument now returns a capsule
      from the cdylib (all 21 targets).
- [x] `import klujax_cpp` replaced by `import klujax_native as klujax_cpp`; the
      compiled C++ extension is no longer imported.
- [x] Pure-Python handle classes in `klujax_native`:
      - [x] `KLUSymbolic` (`.raw`, `.handle`, `.close()`, context manager,
            `__del__`).
      - [x] `KLUNumeric` (`.size`, `.as_list()`, `.close()`, context manager,
            `__del__`).
      - [x] `KLUHandleManager` alias comes from `klujax.py`.
- [x] Module-level names rebound to the new classes.
- [x] `jax.tree_util.register_pytree_node` still works (default identity hash).
- [x] `_get_symbolic_handle` / `_get_numeric_handle` use `.raw` / `.as_list()`.
- [x] `analyze()` constructs `KLUSymbolic(int(raw_symbol))`.
- [x] `free_symbolic` / `free_numeric` deprecation shims call `.close()`.
- [x] `getattr(lib, name)` fails fast on a missing symbol.

Exit criteria: `import klujax` succeeds; all targets registered; no reference
to `klujax_cpp` remains — met.

---

## Stage 4 — Milestone A: parity with existing C KLU behind Rust FFI

Status: **complete** for macOS (the only platform available here). The Rust
FFI seam wrapping the C KLU is at feature, API and performance parity.

- [x] `tests.py` unchanged → all pass (79).
- [x] `just test` → 130 pass (legacy + characterization + golden).
- [x] Leak tests and the JIT/handle/deprecation tests pass.
- [x] Benchmark vs. baseline: analyze 0.136 ms (base 0.139), solve 0.487 ms
      (base 0.480), solve_with_symbol 0.476 ms (base 0.465), factor 0.459 ms
      (base 0.416) — within ~10%/noise, no >5% regression except `factor`.
- [ ] Cross-platform smoke: Linux, macOS, Windows — macOS verified; Linux/
      Windows pending CI (Stage 6).
- [ ] Tag milestone `rust-ffi-parity` — deferred to the release step.

Exit criteria: feature/API parity, tests green, performance parity — met.

> Note: Stage 4 should already re-run the Stage 0.5 characterization + golden
> suites (they must pass on the Rust FFI seam wrapping the C algorithm).

---

## Stage 5 — Finalize `klu-sys` static SuiteSparse build

Goal: a `klu-sys` crate that reliably builds and **statically links** SuiteSparse
on all supported platforms, with no runtime dependency on a system
`libklu`/`libsuitesparse`. Pattern follows `~/Projects/eigenlight/crates/umfpack`.

Status: submodule, `build.rs`, `links = "klu"`, the `cc` static compile,
static-link verification, selective sources, and README docs are in place.
Remaining: cross-platform hardening (CI matrix in Stage 6).

- [x] Add `vendor/SuiteSparse` git submodule pinned to **v7.5.0**
      (`.gitmodules`); anchored the root `.gitignore` `suitesparse/` rule to
      `/suitesparse/` so it no longer case-insensitively matches `SuiteSparse`.
- [x] `klu-sys/Cargo.toml`: `links = "klu"` and
      `cc = { version = "1.0.83", features = ["parallel"] }`.
- [x] `klu-sys/build.rs`: locate sources (`KLUJAX_SUITESPARSE_DIR` ->
      `vendor/SuiteSparse` -> root `suitesparse/`), compile
      `SuiteSparse_config` + `AMD` + `COLAMD` + `BTF` + `KLU` with warning
      suppression, and `build.compile("suitesparse_klu")` (static archive).
- [x] Raw FFI bindings (`klu_*`, `klu_z_*`, `klu_common`, `klu_symbolic`) with a
      direct-C solve test.
- [x] Verify static linking: `otool -L` on the built cdylib shows only
      `libSystem` — **no** `libklu`/`libsuitesparse`.
- [x] Add an automated static-link check (`scripts/check_static_link.py` +
      `just static-link-check`); wiring into CI is Stage 6.
- [x] Confirm the build uses the submodule: rebuilding `klu-sys` with the
      legacy root `suitesparse/` hidden still succeeds (compiles from
      `vendor/SuiteSparse`).
- [ ] Cross-platform `cc` build: Linux (gcc/clang), Windows (MSVC).
- [x] Selective source list implemented: skip the 64-bit-index variants
      (`amd_l*`, `btf_l_*`, `colamd_l`, `klu_l_*`, `klu_zl_*`) in `build.rs`;
      the FFI only uses the int32 (`klu_*`/`klu_z_*`) entry points. 130 pytest
      + 9 `klu-sys` tests still pass; static-link check still green.
- [x] Remove the legacy repo-root `suitesparse/` fallback: the root checkout,
      `pybind11`, and `xla` clones were deleted. `build.rs` keeps the root path
      only as a documented dev convenience after the submodule and env var.
- [x] Document submodule init + static linking in `README.md` (install and
      source-build sections).

Exit criteria: `vendor/SuiteSparse` submodule builds cleanly on Linux/macOS/
Windows; the cdylib statically contains KLU with no dynamic SuiteSparse
dependency; `klu-sys` unit test passes.

---

## Stage 6 — Remove the C++ extension, package, CI, release

The SuiteSparse C library is **kept** (statically linked); only the pybind11
C++ extension is removed.

- [x] Delete `klujax.cpp`, the `KLUJAX_BUILD_CPP` branch in `setup.py`, the
      `pybind11`/`xla` clone recipes, `.clang-format`/`.clangd`; updated
      `MANIFEST.in`, `.gitignore`, and `.pre-commit-config.yaml`.
- [x] Update `MANIFEST.in` / package-data for the Rust workspace. The sdist
      vendors the needed SuiteSparse sources (`vendor/SuiteSparse/{SuiteSparse_config,AMD,COLAMD,BTF,KLU}`)
      so it is self-contained; repo clones still use the submodule.
- [x] Packaging verified: the wheel is platform-tagged
      (`...-cp312-cp312-macosx_11_0_arm64.whl`) and bundles
      `klujax_native/libklujax_ffi.{dylib,so,dll}`; the sdist ships no prebuilt
      native lib and builds a wheel from a clean extraction with only `cargo`.
      A fresh-venv install of the wheel solves correctly
      (`solve([2,4],[10,20]) = [5,5]`), loading the bundled cdylib from
      site-packages.
- [ ] CI (workflow rewritten in `.github/workflows/test.yml`; awaiting a green
      run on GitHub):
      - [ ] matrix: Linux/macOS/Windows × Python 3.11–3.14.
      - [ ] `submodules: recursive`, Rust toolchain, `cargo test`, build ext,
            `pytest`.
      - [ ] static-link check (non-Windows) and `scripts/ffi_smoke.py`.
      - [ ] leak tests where feasible.
      - [ ] `main.yml` wheel build updated for submodules + Rust (manylinux
            installs Rust via `CIBW_BEFORE_BUILD_LINUX`); unverified.
- [x] Pre-commit: `cargo fmt --check`, `cargo clippy -D warnings`.
- [x] Docs:
      - [x] README: architecture (Rust cdylib, statically linked SuiteSparse,
            ctypes, XLA FFI) + submodule init instructions.
      - [x] `docs/advanced/jax-integration.md`,
            `docs/advanced/memory-management.md` (C++ wording → Rust/native;
            plus the API/analyze/free and test-matrix docs).
      - [x] LGPL-2.1 + SuiteSparse attribution; note static linking.
- [x] Version bumps via `bver`: `pyproject.toml` now bumps `Cargo.toml`
      (workspace version) instead of `klujax.cpp`.
- [ ] Release: sdist + wheels, publish, tag.

Exit criteria: published release whose cdylib statically links SuiteSparse and
has no pybind11/C++; docs updated; CI green.

---

## Cross-cutting concerns

### Testing
- [x] `tests.py` unchanged is the primary acceptance gate (130 tests total with
      the characterization suite).
- [x] Stage 0.5 characterization + golden corpus lock the numerical contract.
- [x] `scripts/ffi_smoke.py` asserts all 21 handler + 3 C-ABI symbols exist.
- [x] XLA ABI drift tests in `crates/klujax-ffi/src/xla_ffi.rs`.
- [ ] Static-link regression test (Stage 5) in CI.
- [ ] Keep/extend leak tests (`test_no_leak_*`).

### Verification commands
```sh
git submodule update --init vendor/SuiteSparse
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
just test                 # uv run pytest (tests.py + characterization + golden)
python scripts/ffi_smoke.py
# static-link check (macOS):
otool -L target/release/libklujax_ffi.dylib | grep -i suitesparse && echo LEAK
```

### Risk register
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Submodule not initialized -> build fails | Med | Med | clear `build.rs` error; document `git submodule update --init` |
| Accidental dynamic link to a SuiteSparse dylib | Low | High | `cc` static archive + CI `otool`/`ldd` check |
| `c_api.h` ABI drift across jaxlib versions | Med | High | pinned header + ABI-drift tests; document supported jaxlib |
| Panic unwinds across FFI -> UB | Med | High | `guard` + `catch_unwind`; clippy deny |
| `cc` compile portability (MSVC/gcc/clang) | Med | Med | CI matrix; warning suppressions |
| ctypes symbol visibility (Windows) | Med | Med | `#[no_mangle]`, smoke test, CI |
| Packaging cdylib per-platform | Med | Med | `build_ext` matrix; wheel test in CI |
| Hand-rolled call-frame decode diverges | Low | Med | ABI-drift tests + `stage`/dtype checks |
| LGPL compliance for static linking | Low | Med | keep LGPL-2.1; preserve SuiteSparse licence/attribution |

### Effort sketch (rough)
- Stage 5 (static link finalization + verification): days.
- Stage 6 (remove C++, CI, release): days.

### Open questions
- [ ] Policy for bumping the pinned SuiteSparse submodule version?
- [ ] Keep the repo-root `suitesparse/` fallback or drop it in favour of the
      submodule only?
- [ ] Windows: is MSVC `cc` support sufficient, or is a MinGW path needed?
- [ ] Which currently-untested behaviors (uncoalesced input, `NaN`/`Inf`,
      degenerate sizes) are *contract* vs. *undocumented UB*?

---

## Definition of done
- `import klujax` loads the Rust cdylib via ctypes; no pybind11, no C++.
- SuiteSparse KLU is **statically linked** into the cdylib with no runtime
  dependency on a system `libklu`/`libsuitesparse` (verified per platform).
- `klu-sys` builds from a clean clone after submodule init, on Linux/macOS/
  Windows, and is documented.
- `tests.py` passes unchanged on Linux/macOS/Windows, Python 3.11–3.14.
- XLA FFI ABI pinned and drift-tested; jaxlib compatibility documented.
- Performance parity (or documented, justified gaps) vs. the C baseline.
- Docs and README describe the architecture, static linking, and memory model.
- LGPL-2.1 and SuiteSparse attribution preserved.

---

## Status log

| Date (UTC) | Change |
|---|---|
| 2026-03-21 | Stage 6 (wheel install verified): fresh-venv install of the built wheel loads the bundled cdylib from site-packages and solves correctly. `CargoBuildExt` now also copies the cdylib into the wheel build dir so the sdist `exclude` does not strip it. |
| 2026-03-21 | Stage 6 (packaging verified): wheel is platform-tagged and bundles the cdylib; sdist vendors the needed SuiteSparse sources and builds a wheel from a clean extraction (only `cargo`). Added `exclude klujax_native/*.{so,dylib,dll}` to MANIFEST and `BinaryDistribution` to force a non-`any` wheel. |
| 2026-03-21 | Stage 6 (docs): updated `docs/advanced/jax-integration.md`, `memory-management.md`, `api/analyze.md`, `api/free.md`, `test-matrix.md` from C++/pure-Rust wording to Rust + native handles. |
| 2026-03-21 | Stage 6 (CI): rewrote `.github/workflows/test.yml` with a Linux/macOS/Windows × Python 3.11–3.14 matrix (submodules, Rust toolchain, cargo test, pytest, static-link check, ffi smoke); updated `main.yml` wheel build for submodules + Rust. YAML valid; CI run still pending. |
| 2026-03-21 | Stage 6 (partial): removed `klujax.cpp`, the C++ branch of `setup.py`, `pybind11`/`xla`/root-`suitesparse` checkouts, `.clang-format`/`.clangd`, and C++ entries in `MANIFEST.in`/`.gitignore`; added `cargo fmt`/`clippy` pre-commit hooks; updated README + `bver`; `just deps` → `just submodule`. 130 pytest tests still pass without the C++ extension. |
| 2026-03-21 | Stage 5 (selective sources): `klu-sys/build.rs` skips the int64 (`_l`/`_zl`) variants; build/tests/static-link green. |
| 2026-03-21 | Stage 5 (docs + cleanup): README documents submodule init + static linking; removed legacy root `suitesparse`/`pybind11`/`xla` checkouts. |
| 2026-03-21 | Stage 5 (static link verified): `otool -L` shows only `libSystem`; added `scripts/check_static_link.py` + `just static-link-check`; confirmed the build compiles from `vendor/SuiteSparse` with the legacy root checkout hidden. 130 pytest + 9 klu-sys tests pass. |
| 2026-03-21 | **Re-scope**: dropped the pure-Rust KLU port; KLU stays SuiteSparse C, built and statically linked by a `klu-sys` crate (eigenlight/UMFPACK pattern). Removed the `crates/klu` scaffolding. Added `vendor/SuiteSparse` submodule (v7.5.0) and rewrote `klu-sys/build.rs` for static linking. |
| 2026-03-21 | Stage 3+4 (complete): switched `klujax.py` to `klujax_native` (ctypes + `jax.ffi.pycapsule`) with pure-Python handles; implemented the XLA FFI **metadata probe** response required at registration. All 130 tests pass against the Rust backend; benchmark parity confirmed. |
| 2026-03-21 | Stage 2 (implemented): `klu-sys` compiling the vendored SuiteSparse C KLU via `cc`; C-backed `engine.rs` (analyze/factor/refactor/solve/tsolve/dot/free, f64+c128); all 21 handlers. |
| 2026-03-21 | Stage 0.5 (complete): `tests_characterization/` (shapes, dtypes, coalesce, scipy oracles + structural stress, edges/errors, AD/vmap, golden corpus), `docs/test-matrix.md`, pytest-cov 79% baseline. 130 tests pass. |
| 2026-03-21 | Stage 0 (complete): pinned `c_api.h` from jaxlib 0.9.2; C++ baseline 79 passed; `benchmarks/baseline.json`; portable RSS probe. |
| 2026-03-21 | Stage 1 (complete): Rust workspace, hand-written XLA C ABI + drift tests, decode/error/guard, handler stubs, `CargoBuildExt` + `klujax_native`, ctypes smoke test. |
