# Test coverage matrix (Rust migration)

This matrix defines the behavioral contract the implementation must
satisfy (see `work.md` Stage 0.5). It is deliberately about *behavioral*
coverage, not line coverage: JAX tracing distorts Python line coverage, and the
structural paths that matter (reducible / multi-block matrices, fill-in) need
structural inputs, not more branches of the same tiny matrix.

- ✅ covered
- ⚪ out of scope (with reason)
- ❌ gap (to be closed)

Test files:

| File | Scope |
|---|---|
| `tests.py` | legacy gate (unchanged behavior) |
| `tests_characterization/test_shapes.py` | shape table, broadcasting, multi-RHS |
| `tests_characterization/test_dtypes.py` | dtype upcast, index widths |
| `tests_characterization/test_coalesce.py` | coalesce semantics |
| `tests_characterization/test_oracles.py` | scipy oracle, structural stress |
| `tests_characterization/test_edges.py` | errors, degenerate sizes |
| `tests_characterization/test_transforms.py` | AD / vmap for split primitives |
| `tests_characterization/test_golden.py` | frozen golden corpus |

## Public API × transforms

| API | eager | `jit` | `vmap` | `jvp`/`jacfwd` | `vjp`/`jacrev` | `pmap` |
|---|---|---|---|---|---|---|
| `solve` | ✅ tests.py | ✅ tests.py | ✅ tests.py | ✅ tests.py | ✅ tests.py | ✅ tests.py |
| `dot` | ✅ tests.py | ✅ tests.py | ✅ tests.py | ✅ tests.py | ✅ tests.py | ⚪ (no pmap test) |
| `analyze` | ✅ | ✅ `test_analyze_inside_jit` | ✅ (feed to split solves) | n/a (no arrays) | n/a | n/a |
| `factor` | ✅ | ✅ | ✅ `test_transforms` | n/a | n/a | ✅ `test_refactor_pmap` |
| `refactor` | ✅ | ✅ | ✅ `test_refactor_vmap` | ⚪ (identity handle) | ⚪ | ✅ `test_refactor_pmap` |
| `solve_with_symbol` | ✅ | ✅ | ✅ `test_transforms` | ✅ `test_solve_with_symbol_jvp`, `test_refactor_grad` | ✅ `tests.py` | ✅ `test_refactor_pmap` |
| `solve_with_numeric` | ✅ | ✅ | ✅ `test_transforms` | ✅ `test_transforms` | ✅ `test_transforms` | ✅ `test_refactor_pmap` |
| `tsolve_with_symbol` | ✅ | ✅ | ✅ `test_transforms` | ⚪ (no JVP rule registered) | ⚪ | ⚪ |
| `tsolve_with_numeric` | ✅ | ✅ | ✅ `test_transforms` (batched direct) | ⚪ | ⚪ | ⚪ |
| `refactor_and_solve` | ✅ | ✅ | ⚪ | ⚪ | ⚪ | ⚪ |
| `coalesce` | ✅ `test_coalesce` | ⚪ (documented not JIT-able) | ⚪ | ⚪ | ⚪ | ⚪ |
| `free_symbolic` / `free_numeric` | ✅ tests.py | ⚪ (deprecated) | ⚪ | ⚪ | ⚪ | ⚪ |
| `KLUSymbolic` / `KLUNumeric` | ✅ tests.py | n/a | n/a | n/a | n/a | n/a |

## Shape combinations (`Ax` × `b`)

| `Ax` | `b` | assumed `b` | covered |
|---|---|---|---|
| 1D | 1D | `n_col` | ✅ `test_shape_table` |
| 1D | 2D | `n_col × n_rhs` | ✅ `test_shape_table` |
| 1D | 3D | `n_lhs × n_col × n_rhs` | ✅ `test_shape_table`, `test_n_lhs_broadcast` |
| 2D | 1D | `n_col` | ✅ `test_shape_table` |
| 2D | 2D | `n_lhs × n_col` | ✅ `test_shape_table`, `test_n_lhs_broadcast` |
| 2D | 3D | `n_lhs × n_col × n_rhs` | ✅ `test_shape_table`, `test_n_lhs_broadcast` |
| mismatch | — | raises `ValueError` | ✅ `test_mismatched_n_lhs_raises` |

Direct `n_rhs > 1` (not via `vmap`): ✅ `test_multi_rhs_direct`.

## dtypes

| Input | Expected | covered |
|---|---|---|
| `float64` | `float64` | ✅ tests.py |
| `complex128` | `complex128` | ✅ tests.py |
| `float32` | upcast to `float64` | ✅ `test_dtypes` |
| `complex64` | upcast to `complex128` | ✅ `test_dtypes` |
| int32 indices | works | ✅ `test_dtypes`, tests.py |
| int64 indices | works | ✅ `test_dtypes` |
| mixed `Ax`/`b` | `float64` output | ✅ `test_dtypes` |

## Structure / oracles

| Case | Oracle | covered |
|---|---|---|
| random `n=120`, `n_nz=720` | `scipy.sparse.linalg.spsolve` + residual | ✅ `test_oracles` |
| tridiagonal | spsolve + residual | ✅ `test_oracles` |
| block diagonal (3×5) | spsolve + residual | ✅ `test_oracles` |
| block-upper-triangular (reducible, multi-BTF) | spsolve + residual | ✅ `test_oracles` |
| arrow matrix | spsolve + residual | ✅ `test_oracles` |
| ill-conditioned | — | ❌ (expected accuracy undocumented) |
| fill-in / `nblocks > 1` introspection | SuiteSparse introspection | ❌ (not asserted via introspection) |

## Error / edge / degenerate

| Case | Expected | covered |
|---|---|---|
| singular matrix | `RuntimeError` ("singular") | ✅ `test_edges` |
| `n_col == 1` | solves | ✅ `test_edges` |
| `n_nz == 0` | `RuntimeError` (singular) | ✅ `test_edges` |
| out-of-bounds index | `RuntimeError` ("Ai.max() >= n_col") | ✅ `test_edges` |
| negative index | `RuntimeError` ("negative index") | ✅ `test_edges` |
| unsorted (unique) indices | order-independent | ✅ `test_edges` |
| uncoalesced duplicates | unsupported / UB | ⚪ documented; use `coalesce` |
| `NaN`/`Inf` in `Ax` | propagates | ✅ `test_edges` |
| mismatched `n_nz` | `RuntimeError` | ✅ `test_shapes` |

## Golden corpus

✅ `tests_characterization/golden/{inputs,outputs}.npz`, 8 cases (real/complex,
batched, block-triangular, `spsolve`-independent). Regenerated only via
`_generate_golden.py`; reused unchanged as the migration's numerical regression
harness.

## Known gaps / out of scope

- Ill-conditioned accuracy bounds are not documented (❌).
- Multi-block BTF / fill-in is exercised structurally but not asserted via KLU
  introspection (❌; structural coverage only).
- `refactor_and_solve` AD/vmap/pmap not covered (⚪; low priority).
- `tsolve_*` AD is out of scope because no JVP/transpose primitives are
  registered for it (⚪).
