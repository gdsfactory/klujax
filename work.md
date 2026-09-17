# klujax → Rust migration plan (non-PyO3)

Status: in progress — Stage 0, 0.5, 1 complete; Stage 2 implemented (handlers
pending e2e verification in Stage 3). See the status log at the end of this
file.
Owner: Floris
Target: replace `klujax.cpp` (pybind11 + SuiteSparse) with a pure-Rust
implementation exposed to Python via a `cdylib` + `ctypes` + `jax.ffi.pycapsule`,
**without PyO3**.

---

## 1. Goal & non-goals

### Goal
- Ship `klujax` backed by a **pure-Rust KLU** implementation, reusable as a
  standalone Rust crate.
- Expose the XLA typed-FFI handlers as **plain `extern "C"` symbols** in a
  `cdylib`, registered with JAX via `jax.ffi.pycapsule()` and `ctypes`.
- Keep the public Python API (`klujax.solve`, `analyze`, `factor`, `refactor`,
  `solve_with_symbol`, `solve_with_numeric`, `tsolve_*`, `dot`, `coalesce`,
  `free_*`, `KLUSymbolic`, `KLUNumeric`, `KLUHandleManager`) **unchanged**.
- Keep `tests.py` passing unchanged as the primary acceptance gate.

### Non-goals
- No PyO3, no compiled Python extension, no `#[pymodule]`.
- **No C++ at all.** Delete `klujax.cpp` and do not introduce a replacement
  C++ shim. The XLA typed-FFI handler is implemented directly in Rust by
  decoding `XLA_FFI_CallFrame` (feasible because none of our handlers use
  attributes).
- Do not move JAX trace-time logic (lowerings, batching, AD, tree_util,
  `ffi_call`) into Rust. It cannot run in Rust and stays in `klujax.py`.
- No GPU support (KLU is CPU-only, same as today).
- No attempt to beat SuiteSparse performance in the first pass; parity first.

### Licensing note
A faithful port of SuiteSparse/C KLU is a derivative work and stays
**LGPL-2.1**. The repo is already LGPL-2.1, so this is consistent — but the
standalone `klu` crate must carry LGPL too unless a clean-room reimplementation
is done (much harder, not planned here).

---

## 2. Target architecture

```
klujax/
├── Cargo.toml                     # workspace
├── crates/
│   ├── klu/                       # pure Rust KLU stack (no FFI, no JAX)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── config.rs          # SuiteSparse_config equivalent
│   │       ├── amd.rs             # AMD ordering
│   │       ├── colamd.rs          # COLAMD ordering
│   │       ├── btf.rs             # block triangular form
│   │       ├── symbolic.rs        # klu_analyze
│   │       ├── numeric.rs         # klu_factor / klu_refactor
│   │       ├── solve.rs           # klu_solve / klu_tsolve
│   │       ├── complex.rs         # klu_z_* (C128)
│   │       └── common.rs          # klu_common, status codes, memory
│   └── klujax-ffi/                # cdylib: XLA FFI + C ABI shims
│       ├── Cargo.toml
│       ├── build.rs               # bindgen over vendored c_api.h
│       ├── c_api/                 # pinned copy of jaxlib's xla/ffi/api/c_api.h
│       └── src/
│           ├── lib.rs
│           ├── call_frame.rs      # decode XLA_FFI_CallFrame
│           ├── error.rs           # XLA_FFI_Error creation + panic guard
│           ├── handlers/          # one module per handler group
│           └── capi.rs            # plain free_* symbols for ctypes handles
├── klujax/
│   ├── __init__.py                # (or keep top-level klujax.py)
│   ├── _ffi.py                    # ctypes loader + capsule + target registration
│   └── _handles.py                # pure-Python KLUSymbolic / KLUNumeric
├── klujax.py                      # JAX plumbing, unchanged except imports
├── setup.py                       # custom build_ext: cargo build + copy cdylib
├── tests.py                       # unchanged acceptance gate
└── work.md                        # this file
```

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

- **M0 — Prep**: decisions locked, headers vendored, baseline captured.
- **M0.5 — Behavioral test safety net**: characterization + oracle + golden
  corpus. Must be complete before M2 (pure-Rust port).
- **M1 — Rust FFI seam (wrap existing C KLU)**: new Rust `cdylib` + ctypes
  Python, algorithm still SuiteSparse C. `tests.py` green. Proves plumbing.
- **M2 — Pure-Rust KLU port**: replace C module-by-module, differential-tested.
- **M3 — Remove C, package, release**: pure Rust, wheels, docs, CI.

Deliberately wrap-then-port: M1 de-risks the novel plumbing (XLA FFI decode,
ctypes, packaging) with zero numerical risk; M2 then works against a stable,
already-integrated API.

---

## Stage 0 — Decisions & preparation

Status: mostly complete — only the baseline capture is outstanding (it needs
the vendored C++ deps).

- [x] Confirm end-state: pure-Rust `klu` crate, LGPL-2.1, in-repo workspace.
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
- [x] Corpus is reused unchanged as the **C-vs-Rust differential harness** in
      Stage 5.

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

- [x] Create workspace `Cargo.toml` with members `crates/klu`,
      `crates/klujax-ffi`.
- [x] `crates/klujax-ffi/Cargo.toml`: `crate-type = ["cdylib", "rlib"]`, dep
      `klu` (path).
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
Status: `klu-sys` + the C-backed engine landed and verified; frame handlers next.
- [x] Keep the existing `suitesparse/` checkout (fetched by `just deps`).
- [x] Add `crates/klu-sys` (temporary, M1-only) using the `cc` crate to compile
      `SuiteSparse_config`, `AMD`, `COLAMD`, `BTF`, `KLU` C sources. Not a
      default workspace member (needs `suitesparse/`); build with `-p klu-sys`
      or `--workspace`.
      - [x] Expose the C `klu_*` + `klu_z_*` symbols (plus mirrored
            `klu_common`), covered by a direct-C `solve_2x2_diagonal_f64` test.
- [x] C-backed engine (`klujax-ffi/src/engine.rs`, feature `c-backend`) with
      `coo_to_csc`, `analyze_raw`, `factor_raw`, `refactor_raw`,
      `solve_raw`, `solve_with_symbol_raw`, `tsolve_with_symbol_raw`,
      `solve_with_numeric_raw`, `dot_raw`, `free_*` for f64 **and** c128.
      Deviation: lives in `klujax-ffi` (not `crates/klu`) so that `klu` stays
      pure-Rust for Stage 5; the handler layer is unaffected.
      Verified by 9 Rust unit tests (real + complex, direct + split solves).
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

- [ ] Add `_native` loader (e.g. `klujax/_ffi.py` or import block in
      `klujax.py`):
      - [ ] Resolve platform lib name (`.so`/`.dylib`/`.dll`).
      - [ ] `_LIB = ctypes.CDLL(path)`.
      - [ ] `def _capsule(name): return jax.ffi.pycapsule(getattr(_LIB, name))`.
- [ ] Replace every `klujax_cpp.<name>()` registration argument with
      `_capsule("<name>")` in the `jax.ffi.register_ffi_target` calls.
      - [ ] `dot_f64`, `dot_c128`, `solve_f64`, `solve_c128`, `analyze`,
            `solve_with_symbol_*`, `tsolve_with_symbol_*`, `factor_*`,
            `refactor_*`, `solve_with_numeric_*`, `tsolve_with_numeric_*`,
            `refactor_and_solve_*`, `free_*`.
- [ ] Remove `import klujax_cpp`.
- [ ] Implement `_handles.py` pure-Python classes matching current API:
      - [ ] `KLUSymbolic(raw: int)` with `.raw`, `.handle`, `.close()`,
            `__enter__`, `__exit__`, `__del__` (double-free guarded),
            `.close_dependency`-compatible behavior if referenced.
      - [ ] `KLUNumeric(values: Sequence[int])` with `.size`, `.as_list()`,
            `.close()`, `__enter__`, `__exit__`, `__del__`.
      - [ ] `KLUHandleManager = KLUSymbolic` alias.
- [ ] Rebind module-level names: `KLUSymbolic`, `KLUNumeric`,
      `KLUHandleManager` now come from `_handles`, not `klujax_cpp`.
- [ ] Keep `jax.tree_util.register_pytree_node(...)` calls working with the
      new classes (they must be hashable / usable as aux data).
- [ ] `_get_symbolic_handle` / `_get_numeric_handle`: use `.raw` /
      `.as_list()` as today.
- [ ] Ensure `analyze()` constructor call works:
      `KLUSymbolic(int(raw_symbol))`.
- [ ] Ensure `free_symbolic` / `free_numeric` Python deprecation shims still
      call `.close()`.
- [ ] Verify `CDLL` symbol access raises a clear error if a symbol is missing
      (fail fast at import).

Exit criteria: `import klujax` succeeds; all targets registered; no reference
to `klujax_cpp` remains.

---

## Stage 4 — Milestone A: parity with existing C KLU behind Rust FFI

- [ ] Run `tests.py` unchanged → all pass.
- [ ] Run `just test` and the leak tests (`test_no_leak_*`).
- [ ] Run `test_analyze_inside_jit`, `test_context_manager_*`,
      `test_double_close_*`, `test_deprecated_free_*` specifically.
- [ ] Benchmark vs. baseline; assert within noise (no >5% regression).
- [ ] Cross-platform smoke: Linux, macOS, Windows (cdylib load + one solve).
- [ ] Tag milestone `rust-ffi-parity`.

Exit criteria: feature/API parity, tests green, performance parity.

> Note: Stage 4 should already re-run the Stage 0.5 characterization + golden
> suites (they must pass on the Rust FFI seam wrapping the C algorithm).

---

## Stage 5 — Pure-Rust KLU port (M2)

Port bottom-up. Each sub-stage is differential-tested against the C oracle
(kept behind a cargo feature `c-oracle` until Stage 6).

Reference source (pinned v7.5.0): `suitesparse/{SuiteSparse_config,AMD,COLAMD,BTF,KLU}`.

### 5.1 Core types & config
- [ ] `common.rs`: `KluCommon` (tol, memgrow, initmem, status, ordering, scale,
      etc.), status codes, `klu_defaults`.
- [ ] `config.rs`: memory allocation wrappers, `SuiteSparse_config` defaults
      (only the pieces KLU uses; drop timer/printf unless needed).
- [ ] Error/status mapping to `XLA_FFI_Error_Code` and to `i32` for C API shims.

### 5.2 Ordering — AMD
- [ ] Port `AMD/Source/*` (amd_1, amd_2, amd_aat, amd_postorder, amd_valid,
      amd_defaults, …).
- [ ] Differential test: for random SPD/structure matrices, permutation matches
      C AMD exactly.
- [ ] Benchmark fill-in & time parity.

### 5.3 Ordering — COLAMD
- [ ] Port `COLAMD/Source/*`.
- [ ] Differential test: exact permutation match; `colamd_recommended` sizing.
- [ ] Wire ordering selection via `KluCommon::ordering`.

### 5.4 Block triangular form — BTF
- [ ] Port `BTF/Source/*` (`btf_maxtrans`, `btf_strongcomp`, `btf_order`).
- [ ] Differential test: match `R`, `P`, `Q` outputs exactly.

### 5.5 Symbolic analysis (`klu_analyze`)
- [ ] Port `KLU/Source/klu_analyze*` incl. `klu_analyze_given`, BTF/ordering
      usage, and the symbolic workspace structs.
- [ ] Differential test: match `pinv`, `q`, `R`, `Lnz`, `nzoff`, `nblocks`,
      `maxblock`, `Pblock`, etc. on a corpus.

### 5.6 Numeric factorization
- [ ] Port `klu_factor`, `klu_kernel`, `klu_scale`, `klu_mem`, `klu_sort`,
      `klu_dump`, `klu_refactor` for f64.
- [ ] Differential test: compare `Lval`, `Uval`, `Lip`, `Uip`, `P`, `Q`,
      `R`, `p`, `nzoff` within tight tolerance (aim bitwise given faithful
      fp order); flag any divergence.
- [ ] Singular-matrix behavior parity (error status, not crash).

### 5.7 Solve / tsolve / refactor
- [ ] Port `klu_solve`, `klu_tsolve`, `klu_refactor`.
- [ ] Differential test: solution vectors within tolerance for RHS batches.
- [ ] Verify `klu_tsolve` transpose semantics (conj vs plain) matches current
      wrapper (`conj_solve=0`).

### 5.8 Complex (`klu_z_*`)
- [ ] Port all complex variants (factor/refactor/solve/tsolve/analyze,
      scale, kernel) reusing the generic core.
- [ ] Differential test for C128.

### 5.9 Cut over
- [ ] Make `crates/klu` default to the pure-Rust implementation.
- [ ] Keep C oracle behind `--features c-oracle` only for tests.
- [ ] Re-run `tests.py` + full parity suite.
- [ ] Benchmark vs. SuiteSparse; document any gaps (fill-in, time, memory).

Exit criteria: no C KLU in the default build; all tests + parity tests green;
performance documented.

---

## Stage 6 — Remove C, packaging, CI, release

- [ ] Delete `klujax.cpp`, `setup.py` C-extension logic, `suitesparse/`,
      `xla/`, `pybind11/` clone recipes from `justfile`.
- [ ] Delete `crates/klu-sys` and the `c-oracle` feature after final parity
      confirmation (or keep in a `dev/` oracle crate).
- [ ] Update `MANIFEST.in` / package-data: Rust workspace, no C sources.
- [ ] Ensure wheels bundle the cdylib per platform; `pip install .` from sdist
      requires only `cargo`.
- [ ] CI:
      - [ ] matrix: Linux/macOS/Windows × Python 3.11–3.14.
      - [ ] install Rust toolchain, `cargo test`, build ext, `pytest`.
      - [ ] leak tests under `pytest -W error` where feasible.
- [ ] Pre-commit: add `cargo fmt --check`, `cargo clippy -D warnings`.
- [ ] Docs:
      - [ ] README: architecture section (Rust core, ctypes, XLA FFI).
      - [ ] `docs/advanced/jax-integration.md`: update ABI explanation.
      - [ ] `docs/advanced/memory-management.md`: pure-Python handles + Rust
            free shims; keep ghost-pointer guidance.
      - [ ] Note LGPL-2.1 and attribution to SuiteSparse.
- [ ] Version bumps via `bver`: update `Cargo.toml` workspace version too
      (extend bver file list).
- [ ] Release: build sdist + wheels, publish, tag.

Exit criteria: published release with pure-Rust core; docs updated; CI green.

---

## Cross-cutting concerns

### Testing
- [ ] `tests.py` unchanged is the contract for M1 and M2.
- [ ] Stage 0.5 characterization + golden corpus is the contract for M2
      (must be green before and after each ported module).
- [ ] New `tests_parity.py` (cargo test + pytest) for C-vs-Rust oracle.
- [ ] New `tests_ffi_abi.py`: assert every expected symbol exists in the
      cdylib and that `jax.ffi.pycapsule` accepts it.
- [ ] Add an ABI-drift test: compare generated `bindgen` enum values
      (`S32/U64/F64/C128`) to expected constants; fail loudly on drift.
- [ ] Keep/extend leak tests (`test_no_leak_*`).

### Verification commands
```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
just test            # pytest tests.py
uv run pytest tests_parity.py
```

### Risk register
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `c_api.h` ABI drift across jaxlib versions | Med | High | pin header; ABI-drift test; document supported jaxlib |
| Panic unwinds across FFI → UB | Med | High | `ffi_guard` + `catch_unwind` everywhere; clippy deny |
| Exact fp parity impossible for port | Med | Med | tolerance-based tests; document divergence |
| Porting AMD/COLAMD/BTF/KLU is very large | High | High | wrap-then-port staging; differential gates each module |
| ctypes symbol visibility (Windows) | Med | Med | `#[unsafe(no_mangle)]`, exported test, CI |
| Packaging cdylib per-platform | Med | Med | build_ext matrix; wheel test in CI |
| Hand-rolled call-frame decode diverges | Low | Med | ABI-drift test in `tests_ffi_abi.py`; assert handler `stage == EXECUTE` and dtype/dim checks |
| LGPL contamination of standalone crate | Low | Med | keep crate LGPL-2.1; clear attribution |

### Effort sketch (rough)
- Stage 0–4 (plumbing + wrap C): days–weeks.
- Stage 5 (pure port): weeks–months; AMD/COLAMD/BTF are the bulk.
- Stage 6 (release): days.

### Open questions
- [ ] Is exact SuiteSparse numerical parity required, or "functionally
      equivalent" acceptable? (Affects how faithfully each kernel is ported
      and the golden-corpus tolerances. **Resolve before Stage 5.**)
- [ ] Which currently-untested behaviors (uncoalesced input, `NaN`/`Inf`,
      degenerate sizes) are *contract* vs. *undocumented UB*? Pin in Stage 0.5.
- [ ] Do we need the `klu_l_*` int64 variants? (Current wrapper uses int32
      only — assume no.)
- [ ] Keep standalone `klu` crate in this repo or split to its own repo?
- [ ] Fallback plan if the port stalls: link SuiteSparse’s AMD/BTF C but keep
      KLU core in Rust? (Partial-purity escape hatch.)

---

## Definition of done
- `import klujax` loads a pure-Rust cdylib via ctypes; no pybind11, no C KLU.
- `tests.py` passes unchanged on Linux/macOS/Windows, Python 3.11–3.14.
- `klu` crate is reusable standalone and documented.
- XLA FFI ABI pinned and drift-tested; jaxlib compatibility documented.
- Performance parity (or documented, justified gaps) vs. the C baseline.
- Docs and README describe the new architecture and memory model.
- LGPL-2.1 and SuiteSparse attribution preserved.

---

## Status log

| Date (UTC) | Change |
|---|---|
| 2026-03-21 | Stage 2.3 (implemented): all 21 XLA handlers wired to the engine (analyze/factor/refactor/solve/tsolve/dot/free, f64+c128, numeric broadcast). Compiles under both feature configs; e2e verification in Stage 3. |
| 2026-03-21 | Stage 2.1 (complete modulo handlers): added C-backed `engine.rs` (analyze/factor/refactor/solve/tsolve/dot/free, f64+c128) feature-gated behind `c-backend`; 9 Rust tests pass. |
| 2026-03-21 | Stage 2.1 (partial): added `crates/klu-sys` compiling the vendored SuiteSparse C KLU via `cc`; direct-C solve test passes. |
| 2026-03-21 | Stage 0.5 (complete): added `tests_characterization/` (shapes, dtypes, coalesce, scipy oracles + structural stress, edges/errors, AD/vmap, frozen golden corpus), `docs/test-matrix.md`, pytest-cov + 79% baseline. 130 tests pass. |
| 2026-03-21 | Stage 0 (complete): pinned `c_api.h` from jaxlib 0.9.2; built the C++ extension against jaxlib headers; `tests.py` 79 passed; benchmark → `benchmarks/baseline.json`. Fixed Linux-only RSS probe in `tests.py` to be macOS/Windows portable. |
| 2026-03-21 | Stage 1 (complete): Rust workspace (`klu`, `klujax-ffi`), hand-written XLA C ABI + drift tests, decode/error/guard, 21 handler stubs, C-ABI shims, `CargoBuildExt` + `klujax_native`, just recipes, ctypes smoke test. 5 Rust tests pass; clippy/fmt clean; editable install loads cdylib. |
