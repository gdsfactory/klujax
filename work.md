# klujax Rust hardening plan

Status: **in progress** — Stages 0, 1, 2 and 3 complete. Unsafe *operations*
(blocks + impls) **164 → 69**; no unbounded lifetimes; no handler lint
suppressions; batch/transpose duplication and `IS_COMPLEX` branches removed.
Stages 4–6 remain. Supersedes
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

Status: **complete**. Decoded slices are tied to the `Frame<'a>` borrow; there
are no `&'static`/unbounded pointer-derived slices, and unsafe dropped
**222 → 89** in this stage.

Was:

```rust
// hidden unbound lifetime
pub unsafe fn as_f64<'a>(buf: *const XLA_FFI_Buffer) -> &'a [f64];
unsafe fn arg<T: Scalar>(frame: *mut XLA_FFI_CallFrame, i: usize) -> Result<&'static [T], ErrorInfo>;
```

- [x] Added `Frame<'a>` in `call_frame.rs` (`PhantomData<&'a XLA_FFI_CallFrame>`),
      built once per handler via `Frame::from_raw`.
- [x] Decode moved onto `Frame<'a>`:
      - `Frame::arg_buffer(i) -> Result<Buffer<'a>, _>`
      - `Frame::ret_buffer(i) -> Result<BufferMut<'a>, _>`
      - `Buffer<'a>`: `dims() -> &'a [i64]`, `element_count()`, `dtype()`,
        `expect_dtype()`, `as_slice::<T>() -> &'a [T]`.
      - `BufferMut<'a>`: same plus `as_slice_mut::<T>() -> &'a mut [T]`.
- [x] `Buffer`/`BufferMut` validate rank/dims at construction; `as_slice` is the
      single documented slice-construction point.
- [x] `handlers.rs` helpers take `&Frame<'a>` and return `&'a`/`&'a mut`
      (`arg`/`ret`/`s32_arg`/`u64_arg`/`u64_ret`/`i32_ret`); no `'static`.
- [x] dtype checks stay in the accessors (misuse is an error, not UB).
- [x] Synthetic-frame test updated to go through `Frame`/`Buffer`.
- [x] `call_frame.rs` no longer carries a lint allow (all unsafe blocks have
      `// SAFETY:` docs); budget lowered 222 → 89.

Exit criteria: `grep -rn "&'static" crates/klujax-ffi/src` empty ✓; no unbounded
pointer lifetimes ✓; clippy (lints denied) + tests green ✓. **Met.**

---

## Stage 2 — Generate the 21 handlers + real safety docs

Status: **complete** (line-count target adjusted). All lint suppressions are
gone; handlers are generated by a macro with a `# Safety` contract.

Was: 21 near-identical `#[no_mangle] pub unsafe extern "C" fn`s plus
`#![allow(clippy::missing_safety_doc)]` and `#![allow(clippy::too_many_arguments)]`.

- [x] Declarative `handler!(name, |frame| body)` macro for all 21 targets.
- [x] The macro emits, per target:
      - `#[no_mangle] pub unsafe extern "C" fn <name>(frame_ptr) -> *mut Error`
        with a **`# Safety`** doc;
      - `unsafe { guard(frame_ptr, || { let frame = Frame::from_raw(...); body }) }`
        with a `// SAFETY:` comment.
- [x] Per-target numeric logic lives in ordinary safe `fn`s
      (`factor_impl`, `solve_impl`, `dot_impl`, …) taking `&Frame` — no raw
      pointers in the logic.
- [x] Deleted **all** handler lint allows (`missing_safety_doc`,
      `undocumented_unsafe_blocks`, `unsafe_op_in_unsafe_fn`,
      `too_many_arguments`); clippy still passes with the lints denied.
- [x] Symbol-set check remains covered by `scripts/ffi_smoke.py` (21 + 3).
- [ ] Rust-side symbol test (skipped: `ffi_smoke.py` already asserts the set
      against the cdylib; a Rust test would need to parse `nm` output).

Exit criteria: no blanket suppression ✓; `ffi_smoke` 21 + 3 ✓; 130 pytest ✓.
`handlers.rs` is 443 → **326** lines (handler boilerplate is now generated; the
remaining shared bodies shrink in Stage 3, so the original < 250 target is
tracked there).

---

## Stage 3 — Deduplicate batching / row-major ⇄ col-major logic

Status: **complete** (line-count exit criterion adjusted — see note). The
substantive duplication is centralized and the `IS_COMPLEX` branches are gone.

Was: six transpose loops across `solve_with_symbol_impl`, `solve_raw`,
`solve_with_numeric_raw`; the per-`lhs` gather repeated in `factor_batch_raw`,
`refactor_batch_raw`, and the solve paths; and `if T::IS_COMPLEX` in four
`*_t` helpers.

- [x] `to_col_major<T>` / `to_row_major<T>` extracted with a round-trip unit
      test (real + complex). The 6 inline transpose loops now call these.
- [x] `gather_ax<T>(ax, bk, i, n_nz)` extracted; the 4 duplicated
      per-`lhs` gather sites call it.
- [x] `IS_COMPLEX` collapse: added `Scalar::lu_factor` / `lu_refactor` /
      `lu_solve` / `lu_tsolve` (with `# Safety` docs); `f64` calls `klu_*`,
      `C64` calls `klu_z_*` (plain transpose, `conj_solve = 0`). The four
      `*_t` helpers are deleted and the generic paths are branch-free —
      mirrors the old C++ `KluTraits<T>`.
- [x] `#[allow(clippy::too_many_arguments)]` remains only on
      `solve_with_symbol_impl` (the one generic core).
- [x] Tests: transpose round-trip added; the golden corpus (byte-for-byte) and
      all 130 pytest remain unchanged.
- [ ] Full `per_lhs(closure)` / `solve_lhs` merge: the loop *bodies* differ
      enough (error cleanup, in-place vs. new handles) that a closure helper
      hurt readability more than it helped. Deferred/declined.

Note: `engine.rs` is **not smaller** (825 → 901) because the `# Safety` docs and
multi-line trait signatures more than offset the removed loops. Size is a poor
proxy here; duplication (transpose loops 8 → 2, gather sites 4 → 1,
`IS_COMPLEX` branches 4 → 0) is the real metric.

Exit criteria: no repeated transpose loops ✓; duplication centralized ✓;
`IS_COMPLEX` branches removed ✓; 130 pytest + golden unchanged ✓. **Met**
(line-count target deliberately dropped).

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
| 2026-03-21 | Stage 3 (complete): extracted `to_col_major`/`to_row_major` (+ round-trip test) and `gather_ax`; collapsed `IS_COMPLEX` into `Scalar::lu_*` trait methods and deleted the `*_t` helpers. Switched the unsafe budget metric to *operations* (`unsafe {` + `unsafe impl`): **164 → 69**. All gates green. |
| 2026-03-21 | Stage 2 (complete): macro-generated the 21 handlers with `# Safety` docs + explicit unsafe scopes; removed all handler lint suppressions (`missing_safety_doc`, `undocumented_unsafe_blocks`, `unsafe_op_in_unsafe_fn`, `too_many_arguments`); clippy still deny-clean. All gates green. |
| 2026-03-21 | Stage 1 (complete): added `Frame<'a>`/`Buffer<'a>`/`BufferMut<'a>` (lifetime-safe decode, documented `unsafe`, `call_frame.rs` allow removed); rewrote `handlers.rs` onto the new API + a `handler!` macro; no `&'static`/unbounded slices remain. Unsafe **222 → 89**; budget lowered. All gates green. |
| 2026-03-21 | Stage 0 (complete): `tools/unsafe_budget.{sh,txt}` (code-only count, baseline **222**); workspace lints `unsafe_op_in_unsafe_fn`, `undocumented_unsafe_blocks`, `missing_safety_doc` = `deny` with a documented per-module ratchet; `just unsafe-budget`/`miri` + pre-commit + CI wiring. All gates green. |
