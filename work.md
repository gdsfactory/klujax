# klujax Rust hardening plan

Status: **in progress** — Stage 0 (guardrails + unsafe budget) complete;
Stages 1–6 remain. Baseline: **222** code-only `unsafe` occurrences. Supersedes
the previous migration plan (the migration is done: Rust `cdylib`, XLA FFI,
`klu-sys` statically linking SuiteSparse, Python via ctypes/`pycapsule`; 130
pytest + sax downstream green on macOS/Linux).

This plan addresses the code-quality debt identified in the post-migration
review. It is deliberately **behaviour-preserving**: no public API, numerical,
or ABI changes. Every stage must keep the existing gates green.

---

## 1. Context and baseline

Repo layout:

```
crates/klu-sys/          raw FFI + build.rs (SuiteSparse submodule, cc, static)
crates/klujax-ffi/src/   xla_ffi.rs, call_frame.rs, error.rs, engine.rs,
                         handlers.rs, capi.rs, lib.rs
klujax_native/           ctypes loader + capsule providers + handle classes
```

Measured baseline (before this plan):

| Metric | Value |
|---|---|
| Rust core size | 2254 lines (`engine.rs` 825, `handlers.rs` 443, `call_frame.rs` 241, `xla_ffi.rs` 246, `klu-sys` 316, `error.rs` 137, `capi.rs` 32, `lib.rs` 14) |
| `unsafe` lexical occurrences | ~220 (`handlers.rs` 133, `engine.rs` 37, `call_frame.rs` 32, `error.rs` 15, `klu-sys` 3, `xla_ffi.rs` 2, `capi.rs` 1) |
| Unbounded/`'static` slice helpers | 8 (`as_f64`, `as_f64_mut`, `as_i32`, `as_u64` in `call_frame.rs`; `arg<T>`/`ret<T>` in `handlers.rs`; `dims`) |
| `#[no_mangle]` handlers | 21, hand-written |
| Module-level lint suppressions | `#![allow(clippy::missing_safety_doc)]`, `#![allow(clippy::too_many_arguments)]` |
| Rust unit tests | 10 |
| Python tests | 79 legacy + 51 characterization (+ optional sax integration) |

### Success metrics (targets)

1. **0** slice helpers with an unbounded/`'static` lifetime derived from a raw
   pointer.
2. **0** module-level `missing_safety_doc` suppressions; every `pub unsafe fn`
   has a `# Safety` section (enforced by `clippy::undocumented_unsafe_blocks`).
3. `unsafe` lexical count reduced by **≥ 50%** (baseline ~220 → target < 110),
   and confined to `klu-sys` + a single FFI/decode boundary.
4. **`engine.rs` has 0 `unsafe`** (all raw KLU calls behind a safe wrapper).
5. Handler generation is table/macro-driven; `handlers.rs` < 250 lines and the
   21 symbols are provably present.
6. `cargo miri` passes on the pure-Rust decode/COO→CSC tests.
7. `klu_common`/`klu_symbolic` layouts are either bindgen-generated or
   size/offset-asserted against the C header.
8. All existing gates stay green: `cargo test`, `cargo clippy -D warnings`,
   `cargo fmt --check`, 130 pytest, `ffi_smoke`, static-link checks, sax suite.

### Non-goals

- No algorithm or numerical changes; the golden corpus must not move.
- No public Python API changes.
- No new features (GPU, new dtypes, etc.).
- Reintroducing a `crates/klu` safe wrapper is in scope (Stage 6); reimplementing
  KLU in Rust is **not**.

---

## Stage 0 — Baseline, guardrails, and an `unsafe` budget

Goal: make the debt measurable and prevent regressions while refactoring.

Status: **complete**. Code-only baseline is **222**; the lints are `deny` with a
documented per-module allow-list (ratchet) that later stages remove.

- [x] Recorded the baseline in `tools/unsafe_baseline.txt`: per-file code-only
      `unsafe` counts (handlers 132, engine 37, call_frame 32, error 15, …) and
      the unbounded-lifetime list.
- [x] Added `[workspace.lints.rust] unsafe_op_in_unsafe_fn = "deny"` and
      `[lints] workspace = true` in both crates.
- [x] Enabled `clippy::undocumented_unsafe_blocks` and
      `clippy::missing_safety_doc` as `deny` in `[workspace.lints.clippy]`
      (remove the per-module allows as Stages 1/2/3/5/6 land).
- [x] Added `tools/unsafe_budget.sh` (+ `tools/unsafe_budget.txt`): counts
      code-only `unsafe` (comments stripped) and fails over budget; baseline 222.
- [x] Wired `just unsafe-budget` + `just miri`, a pre-commit hook, and a
      `test.yml` step.
- [x] Ratchet: backlog modules carry `#![allow(clippy::undocumented_unsafe_blocks)]`
      and (in `handlers.rs`) `unsafe_op_in_unsafe_fn` / `missing_safety_doc`,
      each with a `TODO(hardening/stage-N)`. Clippy is green with the lints
      denied globally.

Exit criteria: baseline committed ✓; `unsafe_op_in_unsafe_fn` and
`undocumented_unsafe_blocks` enforced (deny) with a tracked allow-list ✓; budget
script runs and passes ✓. **Met.**

---

## Stage 1 — Lifetime-safe buffer wrappers (kill `&'static`)

Goal: make it *impossible* to return slices that outlive the XLA call frame.

Current problem:

```rust
// hidden unbound lifetime
pub unsafe fn as_f64<'a>(buf: *const XLA_FFI_Buffer) -> &'a [f64];
unsafe fn arg<T: Scalar>(frame: *mut XLA_FFI_CallFrame, i: usize) -> Result<&'static [T], ErrorInfo>;
```

Plan:

- [ ] Add a `Frame<'a>` newtype in `call_frame.rs` that borrows the frame for the
      handler duration:
      `pub struct Frame<'a> { raw: *mut XLA_FFI_CallFrame, _marker: PhantomData<&'a ...> }`,
      constructed once per handler from the raw pointer.
- [ ] Move decode onto `Frame<'a>`:
      - `Frame::arg_buffer(i) -> Result<Buffer<'a>>`
      - `Frame::ret_buffer(i) -> Result<BufferMut<'a>>`
      - `Buffer<'a>` methods: `dims() -> &'a [i64]`, `element_count()`,
        `as_f64() -> &'a [f64]`, `as_i32()`, `as_u64()`, `as_c128()`, …
      - `BufferMut<'a>`: `as_f64_mut() -> &'a mut [f64]`, etc.
- [ ] Make `Buffer<'a>` own the dtype/rank validation (checked at construction):
      no `unsafe` slice construction outside `Buffer`.
- [ ] Replace `handlers.rs` `arg<T>(frame, i) -> &'static [T]` with
      `frame.arg::<T>(i) -> Result<&'a [T], ErrorInfo>` where `'a` is the frame
      borrow. Add `T::read(buf)`/`T::write(buf)` on `Scalar` for F64/C128.
- [ ] Keep the `dtype` check inside the accessor so misuse is an error, not UB.
- [ ] Update the synthetic-frame tests to go through `Frame`.

Exit criteria: no helper returns a pointer-derived slice with an unbound
lifetime; `grep -rn "&'static" crates/klujax-ffi/src` is empty; tests green.

---

## Stage 2 — Generate the 21 handlers + real safety docs

Goal: delete boilerplate and the blanket lint suppression.

Current problem: 21 near-identical `#[no_mangle] pub unsafe extern "C" fn`s,
plus `#![allow(clippy::missing_safety_doc)]` and
`#![allow(clippy::too_many_arguments)]`.

Plan:

- [ ] Define a declarative target table: `(name, dtype_family, args, rets,
      impl_fn)` for all 21 targets.
- [ ] Write one macro (or a small codegen) that, per target, emits:
      - `#[no_mangle] pub unsafe extern "C" fn <name>(frame) -> *mut Error`
        with a **`# Safety`** doc and the uniform contract;
      - a `unsafe { guard(frame, || impl) }` call where `impl` receives the
        decoded `Frame` + typed buffers.
- [ ] Move the per-target numeric logic into ordinary `fn`s that take safe
      inputs (`Config`, `&[T]`, dims) and return `Result<(), ErrorInfo>` — no
      raw pointers in the logic.
- [ ] Delete `#![allow(clippy::missing_safety_doc)]`; rely on the generated
      `# Safety` sections. Keep `too_many_arguments` suppression only where a
      table entry genuinely needs it (prefer a `Params` struct instead).
- [ ] Add a test that the generated symbol set matches `klujax_native.TARGETS`
      (already partially covered by `ffi_smoke.py`; add a Rust-side check).

Exit criteria: `handlers.rs` < 250 lines; no blanket safety-doc suppression;
`ffi_smoke` finds 21 + 3 symbols; handlers behave identically (130 pytest).

---

## Stage 3 — Deduplicate batching / row-major ⇄ col-major logic

Goal: one implementation of the batch/transpose machinery.

Current duplication in `engine.rs`:

- `solve_with_symbol_impl`, `solve_raw`, `solve_with_numeric_raw`,
  `factor_batch_raw`, `refactor_batch_raw`, `dot_raw` each repeat:
  - the row-major → col-major transpose of `b`/`x`;
  - the per-`lhs` loop over `Bk`/`Bx`;
  - the complex/real `T::IS_COMPLEX` branching.

Plan:

- [ ] Extract `fn to_col_major<T>(b: &[T], n_lhs, n_col, n_rhs) -> Vec<T>` and
      `fn to_row_major<T>(x: &[T], …) -> Vec<T>` with round-trip unit tests.
- [ ] Extract `fn per_lhs<T>(ax, bk, bp, bi, n_lhs, n_nz, f: impl Fn(&mut [T]) -> Result<()>)`.
- [ ] Extract a single `fn solve_lhs<T>(…)` used by both `solve*` and `tsolve*`
      (a `transpose: bool` parameter instead of duplicated bodies).
- [ ] Collapse the `T::IS_COMPLEX` branches by routing through `T`'s methods
      (`T::klu_factor`, `T::klu_solve`, `T::klu_tsolve`, `T::klu_refactor`) so
      the generic code is branch-free — mirrors the old `KluTraits<T>`.
- [ ] Keep `#[allow(clippy::too_many_arguments)]` only on the one generic core.
- [ ] Add tests: transpose round-trip; equivalence of refactored helpers on the
      existing golden corpus (no tolerance loosening).

Exit criteria: no repeated transpose loops (grep is clean); `engine.rs` shrinks
materially; golden + 130 pytest unchanged.

---

## Stage 4 — Property tests + miri for the decode and `unsafe` paths

Goal: stress the hand-rolled ABI decode, which is the riskiest untested code.

Plan:

- [ ] Split pure-Rust, miri-friendly logic from the FFI boundary, so miri can run
      without calling into C KLU:
      - COO→CSC (`coo_to_csc`), dims/dtype validation, transpose helpers.
- [ ] Add `proptest` (dev-dependency) strategies for:
      - random `(n, n_nz, ai, aj)` including duplicates, unsorted, out-of-range;
      - synthetic call frames with wrong dtype/rank/size;
      - `b` shapes across all six `Ax`/`b` combinations.
- [ ] Property assertions:
      - decode never reads out of bounds; mismatches return `Err`, never panic;
      - `coo_to_csc` output is a valid CSC (sorted rows per column, `Bp`
        monotone, `Bk` a permutation);
      - transpose helpers round-trip.
- [ ] Add `just miri` (`cargo +nightly miri test -p klujax-ffi --lib`, with the
      C-KLU tests gated by `#[cfg(not(miri))]`).
- [ ] Add `just fuzz` (optional, `cargo-fuzz` target for the frame decoder) — or
      document a `libFuzzer` harness and keep it out of required CI.
- [ ] Ensure `unsafe_op_in_unsafe_fn` stays clean under the new tests.

Exit criteria: `cargo miri test` green for the pure-Rust subset; proptest
suite green; no panics on malformed frames.

---

## Stage 5 — Make `klu-sys` layout-safe

Goal: stop hand-maintaining C struct layouts that can silently drift.

Current problem: `klu_common` is fully hand-mirrored; `klu_symbolic` is a
*partial* struct (only fields up to `n`), so it compiles even if the header
changes.

Plan (pick one; prefer A if `libclang` is acceptable at build time):

- [ ] **A. bindgen**: generate bindings from the vendored `vendor/SuiteSparse`
      headers in `klu-sys/build.rs` (`bindgen` build-dependency). Map the
      generated names to the existing `klu-sys` API to avoid churn in
      `engine.rs`.
- [ ] **B. Layout assertions**: if bindgen is undesirable, add `const` size/
      offset assertions (mirroring `xla_ffi::abi_tests`) for `klu_common`,
      `klu_symbolic`, `klu_numeric`, computed from the C header values.
- [ ] Fully define `klu_symbolic` (all fields) or restrict access to a single
      `fn n(sym) -> usize` accessor with a documented contract.
- [ ] Add a "header drift" test: parse the vendored `klu.h` for `sizeof`/
      `offsetof` expectations (or bindgen output) and assert against Rust.
- [ ] Document the supported SuiteSparse version and the update procedure in
      `crates/klu-sys/README.md`.

Exit criteria: struct layouts are generated or asserted; a header bump that
changes layout fails a test/compile, not at runtime.

---

## Stage 6 — Minimize `unsafe` (cross-cutting)

Goal: shrink and fence the unsafe surface; make most of the crate safe code.

Plan:

- [ ] Introduce a safe `klu` wrapper (re-add `crates/klu`, or a `klu` module in
      `klujax-ffi`) that owns the single `unsafe` boundary around `klu-sys`:
      - safe `struct Symbolic`, `struct Numeric` (RAII `Drop`);
      - safe `analyze/factor/refactor/solve/tsolve/free`;
      - no raw pointers leak out.
- [ ] Rewrite `engine.rs` against the safe wrapper → **`engine.rs` becomes
      `#![forbid(unsafe_code)]`** (target metric 4).
- [ ] Fence the FFI boundary:
      - `unsafe` only in `call_frame.rs` (pointer→slice, documented) and the
        generated handler macro;
      - `unsafe fn` bodies use explicit `unsafe {}` (Stage 0 lint).
- [ ] Add `# Safety` docs to the remaining `pub unsafe fn`s; delete the blanket
      suppressions.
- [ ] Prefer safe alternatives where possible:
      - use `NonNull`/`Option<NonNull<_>>` instead of raw null checks;
      - `slice::from_raw_parts` centralized in `Buffer::new`;
      - avoid `transmute` entirely (none currently; keep it that way and add a
        `clippy::transmute_ptr_to_ptr`/`forbid(transmute)` check);
      - keep integer casts checked (`try_into`) at the boundary.
- [ ] Lower the `tools/unsafe_budget.sh` budget after each stage; record the
      reduction in the status log.
- [ ] Add a crate-level doc (`lib.rs`) explaining the unsafe boundary and the
      `Buffer`/`Frame` invariants.

Exit criteria: `unsafe` count < 110 (≥50% reduction), `engine.rs` unsafe-free,
no blanket lint suppressions, budget enforced.

---

## Verification (must stay green throughout)

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
just miri                 # pure-Rust decode subset (Stage 4)
tools/unsafe_budget.sh    # no increase over budget
just test                 # 130 pytest
python scripts/ffi_smoke.py
python scripts/check_static_link.py
just verify-linux         # Linux static-link (Docker)
just verify-sax           # downstream sax integration
```

Golden corpus (`tests_characterization/golden/`) must be byte-for-byte
unchanged after every stage.

## Risks

| Risk | Mitigation |
|---|---|
| Refactor changes numerics | golden corpus + 130 pytest as hard gates; no tolerance changes |
| Lifetime refactor fights the borrow checker | keep `unsafe` confined to `Buffer::new`; do not over-engineer |
| miri cannot run C-KLU tests | isolate pure-Rust logic; `#[cfg(not(miri))]` on KLU-calling tests |
| bindgen adds a libclang build dep | fall back to layout assertions (Stage 5B) |
| Safe wrapper adds overhead | RAII handles are pointer-sized; benchmark before/after |
| Unsafe budget too strict | lower per-stage, never raise |

## Status log

| Date | Change |
|---|---|
| 2026-03-21 | Stage 0 (complete): `tools/unsafe_budget.{sh,txt}` (code-only count, baseline **222**); workspace lints `unsafe_op_in_unsafe_fn`, `undocumented_unsafe_blocks`, `missing_safety_doc` = `deny` with a documented per-module ratchet; `just unsafe-budget`/`miri` + pre-commit + CI wiring. All gates green (clippy, fmt, 4 rust test bins, 130 pytest, ffi smoke, static link). |
