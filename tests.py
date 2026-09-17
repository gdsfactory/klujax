import gc
import os
import sys
from functools import wraps
from pathlib import Path

import jax
import jax.core
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest
from jax import lax

import klujax
from klujax import COMPLEX_DTYPES, coalesce

try:
    import resource
except ImportError:  # e.g. Windows
    resource = None

OPS_DENSE = {  # sparse to dense
    klujax.dot: lax.dot,
    klujax.solve: jsp.linalg.solve,
}


def log_test_name(f):
    @wraps(f)
    def new(*args, **kwargs):
        print(f"\n{f.__name__}", file=sys.stderr)
        if args:
            print(f"args={args}", file=sys.stderr)
        if kwargs:
            print(f"kwargs={kwargs}", file=sys.stderr)
        return f(*args, **kwargs)

    return new


def parametrize_dtypes(func):
    return pytest.mark.parametrize("dtype", [np.float64, np.complex128])(func)


def parametrize_ops(func):
    return pytest.mark.parametrize("op_sparse", [klujax.solve, klujax.dot])(func)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_1d(dtype, op_sparse):
    op_dense = OPS_DENSE[op_sparse]
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    x_sp = op_sparse(Ai, Aj, Ax, b)
    A = jnp.zeros((n_col, n_col), dtype=Ax.dtype).at[Ai, Aj].add(Ax)
    x = op_dense(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_2d(dtype, op_sparse):
    op_dense = jax.vmap(OPS_DENSE[op_sparse], (0, 0), 0)
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    x_sp = op_sparse(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = op_dense(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_2d_vmap(dtype, op_sparse):
    op_dense = OPS_DENSE[op_sparse]
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    x_sp = jax.vmap(op_sparse, (None, None, 1, 1), 0)(Ai, Aj, Ax.T, b.T)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = jax.vmap(op_dense, (0, 0), 0)(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_3d(dtype, op_sparse):
    op_dense = jax.vmap(OPS_DENSE[op_sparse], (0, 0), 0)
    Ai, Aj, Ax, b = _get_rand_arrs_3d((n_lhs := 2), 8, (n_col := 3), 4, dtype=dtype)
    x_sp = op_sparse(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = op_dense(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_3d_jacfwd(dtype, op_sparse):
    op_dense = jax.vmap(OPS_DENSE[op_sparse], (0, 0), 0)
    Ai, Aj, Ax, b = _get_rand_arrs_3d((n_lhs := 3), 15, (n_col := 5), 2, dtype=dtype)
    holomorphic = dtype in COMPLEX_DTYPES

    # jacobian on b
    jac_sp = jax.jacfwd(op_sparse, 3, holomorphic=holomorphic)(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    jac = jax.jacfwd(op_dense, 1, holomorphic=holomorphic)(A, b)
    _log_and_test_equality(jac_sp, jac)

    # jacobian on A
    jac_sp = jax.jacfwd(op_sparse, 2, holomorphic=holomorphic)(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    jac = jax.jacfwd(op_dense, 0, holomorphic=holomorphic)(A, b)[..., Ai, Aj]
    _log_and_test_equality(jac_sp, jac)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_3d_jacrev(dtype, op_sparse):
    op_dense = jax.vmap(OPS_DENSE[op_sparse], (0, 0), 0)
    Ai, Aj, Ax, b = _get_rand_arrs_3d((n_lhs := 3), 15, (n_col := 5), 2, dtype=dtype)
    holomorphic = dtype in COMPLEX_DTYPES

    # jacobian on b
    jac_sp = jax.jacrev(op_sparse, 3, holomorphic=holomorphic)(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    jac = jax.jacrev(op_dense, 1, holomorphic=holomorphic)(A, b)
    _log_and_test_equality(jac_sp, jac)

    # jacobian on A
    jac_sp = jax.jacrev(op_sparse, 2, holomorphic=holomorphic)(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    jac = jax.jacrev(op_dense, 0, holomorphic=holomorphic)(A, b)[..., Ai, Aj]
    _log_and_test_equality(jac_sp, jac)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_3d_vmap(dtype, op_sparse):
    op_dense = OPS_DENSE[op_sparse]
    Ai, Aj, Ax, b = _get_rand_arrs_3d((n_lhs := 3), 15, (n_col := 5), 2, dtype=dtype)
    _log(Ai_shape=Ai.shape, Aj_shape=Aj.shape, Ax_shape=Ax.shape, b_shape=b.shape)
    x_sp = jax.vmap(op_sparse, (None, None, None, -1), -1)(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = jax.vmap(op_dense, (0, 0), 0)(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_4d(dtype, op_sparse):
    Ai, Aj, Ax, b = _get_rand_arrs_3d(3, 15, 5, 2, dtype=dtype)
    with pytest.raises(ValueError):  # noqa: PT011
        op_sparse(Ai, Aj, Ax, b[None])

    with pytest.raises(ValueError):  # noqa: PT011
        op_sparse(Ai, Aj, Ax[None], b)

    with pytest.raises(ValueError):  # noqa: PT011
        op_sparse(Ai, Aj, Ax[None], b[None])


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_4d_vmap(dtype, op_sparse):
    Ai, Aj, Ax, b = _get_rand_arrs_3d(3, 8, 5, 2, dtype=dtype)
    bb = np.stack([b, (b2 := np.random.RandomState(seed=42).rand(*b.shape))], axis=1)
    r = jax.vmap(op_sparse, (None, None, None, 1), 0)(Ai, Aj, Ax, bb)
    r1 = op_sparse(Ai, Aj, Ax, b)
    r2 = op_sparse(Ai, Aj, Ax, b2)
    _log_and_test_equality(r[0], r1)
    _log_and_test_equality(r[1], r2)


@log_test_name
def test_analyze():
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)

    symbolic = klujax.analyze(Ai, Aj, n_col)
    assert isinstance(symbolic, klujax.KLUSymbolic)

    x = klujax.solve_with_symbol(Ai, Aj, Ax, b, symbolic)
    assert x.shape == (n_col,)


@log_test_name
@parametrize_dtypes
def test_solve_with_symbol(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)

    x_sp = klujax.solve_with_symbol(Ai, Aj, Ax, b, symbolic)

    A = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax)
    x = jsp.linalg.solve(A, b)
    _log_and_test_equality(x, x_sp)

    # Test JIT
    x_sp_jit = jax.jit(klujax.solve_with_symbol)(Ai, Aj, Ax, b, symbolic)
    _log_and_test_equality(x, x_sp_jit)


@log_test_name
@parametrize_dtypes
def test_solve_with_symbol_batched(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)

    x_sp = klujax.solve_with_symbol(Ai, Aj, Ax, b, symbolic)

    # Dense verification
    op_dense = jax.vmap(jsp.linalg.solve, (0, 0), 0)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = op_dense(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
def test_solve_with_numeric(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)
    numeric = klujax.factor(Ai, Aj, Ax, symbolic)

    x_sp = klujax.solve_with_numeric(numeric, b, symbolic)

    A = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax)
    x = jsp.linalg.solve(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
def test_solve_with_numeric_batched(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)
    numeric = klujax.factor(Ai, Aj, Ax, symbolic)

    x_sp = klujax.solve_with_numeric(numeric, b, symbolic)

    # Dense verification
    op_dense = jax.vmap(jsp.linalg.solve, (0, 0), 0)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = op_dense(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
def test_solve_with_numeric_vmap_1d_b(dtype):
    """vmap over solve_with_numeric where b is 1D (single-system, single-RHS)."""
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)

    # Create a batch of different Ax values (different matrix values, same sparsity)
    key = jax.random.PRNGKey(42)
    batch = 4
    Ax_batch = jax.random.normal(key, (batch, *Ax.shape), dtype=dtype) + 10.0

    def solve_one(Ax_i):
        num = klujax.factor(Ai, Aj, Ax_i, symbolic)
        return klujax.solve_with_numeric(num, b, symbolic)

    x_sp = jax.vmap(solve_one)(Ax_batch)

    # Dense reference
    A_batch = jnp.zeros((batch, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax_batch)
    x = jax.vmap(lambda A: jsp.linalg.solve(A, b))(A_batch)
    _log_and_test_equality(x, x_sp)


def _get_rand_arrs_1d(n_nz, n_col, *, dtype, seed=33):
    Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(seed), 4)
    Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    Ax = jax.random.normal(Axkey, (n_nz,), dtype=dtype)
    # Add diagonal to ensure matrix is invertible
    diag_i = jnp.arange(n_col, dtype=jnp.int32)
    diag_x = jnp.ones(n_col, dtype=dtype) * 10.0
    Ai = jnp.concatenate([Ai, diag_i])
    Aj = jnp.concatenate([Aj, diag_i])
    Ax = jnp.concatenate([Ax, diag_x])
    Ai, Aj, Ax = coalesce(Ai, Aj, Ax)
    b = jax.random.normal(bkey, (n_col,), dtype=dtype)
    return Ai, Aj, Ax, b


def _get_rand_arrs_2d(n_lhs, n_nz, n_col, *, dtype, seed=33):
    Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(seed), 4)
    Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    Ax = jax.random.normal(
        Axkey,
        (
            n_lhs,
            n_nz,
        ),
        dtype=dtype,
    )
    # Add diagonal to ensure matrix is invertible
    diag_i = jnp.arange(n_col, dtype=jnp.int32)
    diag_x = jnp.ones((n_lhs, n_col), dtype=dtype) * 10.0
    Ai = jnp.concatenate([Ai, diag_i])
    Aj = jnp.concatenate([Aj, diag_i])
    Ax = jnp.concatenate([Ax, diag_x], axis=1)
    Ai, Aj, Ax = coalesce(Ai, Aj, Ax)
    b = jax.random.normal(bkey, (n_lhs, n_col), dtype=dtype)
    return Ai, Aj, Ax, b


def _get_rand_arrs_3d(n_lhs, n_nz, n_col, n_rhs, *, dtype, seed=33):
    Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(seed), 4)
    Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    Ax = jax.random.normal(
        Axkey,
        (
            n_lhs,
            n_nz,
        ),
        dtype=dtype,
    )
    # Add diagonal to ensure matrix is invertible
    diag_i = jnp.arange(n_col, dtype=jnp.int32)
    diag_x = jnp.ones((n_lhs, n_col), dtype=dtype) * 10.0
    Ai = jnp.concatenate([Ai, diag_i])
    Aj = jnp.concatenate([Aj, diag_i])
    Ax = jnp.concatenate([Ax, diag_x], axis=1)
    Ai, Aj, Ax = coalesce(Ai, Aj, Ax)
    b = jax.random.normal(bkey, (n_lhs, n_col, n_rhs), dtype=dtype)
    return Ai, Aj, Ax, b


def _log_and_test_equality(x, x_sp):
    print(f"\nx=\n{x}")
    print(f"\nx_sp=\n{x_sp}")
    print(f"\ndiff=\n{np.round(x_sp - x, 9)}")
    print(f"\nis_equal=\n{_is_almost_equal(x, x_sp)}")
    np.testing.assert_array_almost_equal(x, x_sp)


def _log(**kwargs):
    for k, v in kwargs.items():
        print(f"{k}={v}")


def _is_almost_equal(arr1, arr2):
    try:
        np.testing.assert_array_almost_equal(arr1, arr2)
    except AssertionError:
        return False
    else:
        return True


def _make_ax2(Ai, Aj, Ax, *, dtype, seed=42):
    """New Ax values with the same sparsity pattern as Ax, with a strong diagonal."""
    Ax_raw = jax.random.normal(jax.random.PRNGKey(seed), Ax.shape, dtype=dtype)
    # Ensure diagonal entries are large (invertible) — Ai[k]==Aj[k] marks diagonal positions
    if Ax_raw.ndim == 1:
        Ax_raw = Ax_raw + jnp.where(
            Ai == Aj, jnp.full_like(Ax_raw, 10.0), jnp.zeros_like(Ax_raw)
        )
    else:  # (n_lhs, n_nz) batch case
        diag_mask = (Ai == Aj)[None, :]
        Ax_raw = Ax_raw + jnp.where(
            diag_mask, jnp.full_like(Ax_raw, 10.0), jnp.zeros_like(Ax_raw)
        )
    return Ax_raw


@log_test_name
@parametrize_dtypes
def test_refactor(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    Ax2 = _make_ax2(Ai, Aj, Ax, dtype=dtype)
    b2 = jax.random.normal(jax.random.PRNGKey(43), b.shape, dtype=dtype)

    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)

    num2 = klujax.refactor(Ai, Aj, Ax2, num, sym)
    assert isinstance(num2, klujax.KLUNumeric)
    assert num2 is num

    x_sp = klujax.solve_with_numeric(num2, b2, sym)
    A2 = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax2)
    x = jsp.linalg.solve(A2, b2)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
def test_refactor_batched(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    Ax2 = _make_ax2(Ai, Aj, Ax, dtype=dtype)
    b2 = jax.random.normal(jax.random.PRNGKey(43), b.shape, dtype=dtype)

    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)

    num2 = klujax.refactor(Ai, Aj, Ax2, num, sym)
    assert isinstance(num2, klujax.KLUNumeric)
    assert num2 is num

    x_sp = klujax.solve_with_numeric(num2, b2, sym)
    op_dense = jax.vmap(jsp.linalg.solve, (0, 0), 0)
    A2 = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax2)
    x = op_dense(A2, b2)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
def test_refactor_vmap(dtype):
    """vmap solve_with_symbol over n_rhs with shared sym used by refactor.

    Tests that vmap composes correctly with the symbolic analysis after refactor,
    and that refactor does not corrupt the symbolic object.
    """
    Ai, Aj, Ax, b = _get_rand_arrs_3d((n_lhs := 3), 15, (n_col := 5), 4, dtype=dtype)
    Ax2 = _make_ax2(Ai, Aj, Ax, dtype=dtype)

    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)
    klujax.refactor(Ai, Aj, Ax2, num, sym)

    # vmap solve_with_symbol over n_rhs axis (same pattern as test_3d_vmap)
    x_sp = jax.vmap(klujax.solve_with_symbol, (None, None, None, -1, None), -1)(
        Ai, Aj, Ax2, b, sym
    )

    A2 = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax2)
    x = jax.vmap(jax.vmap(jsp.linalg.solve, (0, 0), 0), (None, -1), -1)(A2, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
def test_refactor_pmap(dtype):
    """pmap solve_with_numeric across devices after refactor."""
    n_dev = jax.device_count()
    Ai, Aj, Ax, _ = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    Ax2 = _make_ax2(Ai, Aj, Ax, dtype=dtype)
    B = jax.random.normal(jax.random.PRNGKey(77), (n_dev, n_col), dtype=dtype)

    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)
    num2 = klujax.refactor(Ai, Aj, Ax2, num, sym)

    x_sp = jax.pmap(lambda b: klujax.solve_with_numeric(num2, b, sym))(B)

    A2 = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax2)
    x = jax.vmap(lambda b: jsp.linalg.solve(A2, b))(B)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
def test_refactor_grad(dtype):
    """AD through solve_with_symbol is unaffected by refactor on the shared symbolic analysis."""
    Ai, Aj, Ax, _ = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    Ax2 = _make_ax2(Ai, Aj, Ax, dtype=dtype)
    b2 = jax.random.normal(jax.random.PRNGKey(43), (n_col,), dtype=dtype)
    holomorphic = dtype in COMPLEX_DTYPES

    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)
    num2 = klujax.refactor(Ai, Aj, Ax2, num, sym)

    # refactor value and solve_with_symbol value must agree
    x_num = klujax.solve_with_numeric(num2, b2, sym)
    x_sym = klujax.solve_with_symbol(Ai, Aj, Ax2, b2, sym)
    _log_and_test_equality(x_num, x_sym)

    # jacfwd through solve_with_symbol w.r.t. Ax: tests that sym is not corrupted by refactor
    jac_sp = jax.jacfwd(
        lambda ax: klujax.solve_with_symbol(Ai, Aj, ax, b2, sym),
        holomorphic=holomorphic,
    )(Ax2)
    jac_dense = jax.jacfwd(
        lambda ax: jnp.linalg.solve(
            jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(ax), b2
        ),
        holomorphic=holomorphic,
    )(Ax2)
    _log_and_test_equality(jac_sp, jac_dense)


def test_solve_with_symbol_jvp():
    Ai = jnp.array([0, 1], dtype=jnp.int32)
    Aj = jnp.array([0, 1], dtype=jnp.int32)
    Ax = jnp.array([2.0, 4.0], dtype=jnp.float64)  # Values
    b = jnp.array([10.0, 20.0], dtype=jnp.float64)

    # Pre-compute symbolic factorization
    n_col = 2
    symbolic = klujax.analyze(Ai, Aj, n_col)

    def solve_step(Ax_vals, b_vec):
        return klujax.solve_with_symbol(Ai, Aj, Ax_vals, b_vec, symbolic)

    # Tangents (perturbations)
    tangent_Ax = jnp.array([0.1, 0.1], dtype=jnp.float64)
    tangent_b = jnp.array([0.0, 0.0], dtype=jnp.float64)

    # This triggers the JVP rule lookup
    primals, tangents = jax.jvp(solve_step, (Ax, b), (tangent_Ax, tangent_b))
    assert primals.shape == (2,)
    assert tangents.shape == (2,)


@log_test_name
@parametrize_dtypes
def test_tsolve_with_symbol(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)

    x_sp = klujax.tsolve_with_symbol(Ai, Aj, Ax, b, symbolic)

    # For tsolve, we compare against the transposed dense matrix A.T
    A = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax)
    x = jsp.linalg.solve(A.T, b)
    _log_and_test_equality(x, x_sp)

    # Test JIT
    x_sp_jit = jax.jit(klujax.tsolve_with_symbol)(Ai, Aj, Ax, b, symbolic)
    _log_and_test_equality(x, x_sp_jit)


@log_test_name
@parametrize_dtypes
def test_tsolve_with_symbol_batched(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)

    x_sp = klujax.tsolve_with_symbol(Ai, Aj, Ax, b, symbolic)

    # Dense verification for batched transpose
    op_dense = jax.vmap(jsp.linalg.solve, (0, 0), 0)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    A_T = jnp.swapaxes(A, -1, -2)
    x = op_dense(A_T, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
def test_tsolve_with_numeric(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)
    numeric = klujax.factor(Ai, Aj, Ax, symbolic)

    x_sp = klujax.tsolve_with_numeric(numeric, b, symbolic)

    A = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax)
    x = jsp.linalg.solve(A.T, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
def test_tsolve_with_numeric_batched(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)
    numeric = klujax.factor(Ai, Aj, Ax, symbolic)

    x_sp = klujax.tsolve_with_numeric(numeric, b, symbolic)

    # Dense verification for batched transpose
    op_dense = jax.vmap(jsp.linalg.solve, (0, 0), 0)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    A_T = jnp.swapaxes(A, -1, -2)
    x = op_dense(A_T, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
def test_refactor_and_solve(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    Ax2 = _make_ax2(Ai, Aj, Ax, dtype=dtype)
    b2 = jax.random.normal(jax.random.PRNGKey(43), b.shape, dtype=dtype)

    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)

    # Verifying specific API order: Ai, Aj, Ax, b, numeric, symbolic
    x_sp, num2 = klujax.refactor_and_solve(Ai, Aj, Ax2, b2, num, sym)

    assert isinstance(num2, klujax.KLUNumeric)
    assert num2 is num

    A2 = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax2)
    x = jsp.linalg.solve(A2, b2)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
def test_refactor_and_solve_batched(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    Ax2 = _make_ax2(Ai, Aj, Ax, dtype=dtype)
    b2 = jax.random.normal(jax.random.PRNGKey(43), b.shape, dtype=dtype)

    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)

    # Verifying specific API order: Ai, Aj, Ax, b, numeric, symbolic
    x_sp, num2 = klujax.refactor_and_solve(Ai, Aj, Ax2, b2, num, sym)

    assert isinstance(num2, klujax.KLUNumeric)
    assert num2 is num

    op_dense = jax.vmap(jsp.linalg.solve, (0, 0), 0)
    A2 = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax2)
    x = op_dense(A2, b2)
    _log_and_test_equality(x, x_sp)


def test_analyze_inside_jit():
    """Regression test: analyze() must not call int() on traced values."""
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)

    @jax.jit
    def solve_jitted(Ai, Aj, Ax, b):
        sym = klujax.analyze(Ai, Aj, n_col)
        return klujax.solve_with_symbol(Ai, Aj, Ax, b, sym)

    x_sp = solve_jitted(Ai, Aj, Ax, b)
    A = jnp.zeros((n_col, n_col), dtype=np.float64).at[Ai, Aj].add(Ax)
    x = jsp.linalg.solve(A, b)
    _log_and_test_equality(x, x_sp)


# RAII handle tests


def test_context_manager_symbolic():
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    with klujax.analyze(Ai, Aj, n_col) as sym:
        assert isinstance(sym, klujax.KLUSymbolic)
        x = klujax.solve_with_symbol(Ai, Aj, Ax, b, sym)
        assert x.shape == (n_col,)


def test_context_manager_numeric():
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)
    with num:
        x = klujax.solve_with_numeric(num, b, sym)
        assert x.shape == (n_col,)


def test_double_close_symbolic():
    Ai, Aj, _, _ = _get_rand_arrs_1d(15, 5, dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, 5)
    sym.close()
    sym.close()


def test_double_close_numeric():
    Ai, Aj, Ax, _ = _get_rand_arrs_1d(15, 5, dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, 5)
    num = klujax.factor(Ai, Aj, Ax, sym)
    num.close()
    num.close()


def test_deprecated_free_symbolic():
    Ai, Aj, _, _ = _get_rand_arrs_1d(15, 5, dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, 5)
    with pytest.warns(DeprecationWarning, match="free_symbolic is deprecated"):
        klujax.free_symbolic(sym)


def test_deprecated_free_numeric():
    Ai, Aj, Ax, _ = _get_rand_arrs_1d(15, 5, dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, 5)
    num = klujax.factor(Ai, Aj, Ax, sym)
    with pytest.warns(DeprecationWarning, match="free_numeric is deprecated"):
        klujax.free_numeric(num)


def _get_rss_kb():
    """Return current RSS in kilobytes (Linux, macOS; 0 elsewhere)."""
    try:
        with Path(f"/proc/{os.getpid()}/status").open() as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        pass
    if resource is None:
        return 0
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    return usage // 1024 if sys.platform == "darwin" else usage


def test_no_leak_symbolic():
    Ai, Aj, _, _ = _get_rand_arrs_1d(15, 5, dtype=np.float64)
    for _ in range(50):
        sym = klujax.analyze(Ai, Aj, 5)
    del sym
    gc.collect()

    rss_before = _get_rss_kb()
    for _ in range(5000):
        sym = klujax.analyze(Ai, Aj, 5)
    del sym
    gc.collect()
    rss_after = _get_rss_kb()

    assert rss_after - rss_before < 10_000, (
        f"RSS grew by {rss_after - rss_before} KB after 5000 analyze() calls"
    )


def test_no_leak_numeric():
    Ai, Aj, Ax, _ = _get_rand_arrs_1d(15, 5, dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, 5)
    for _ in range(50):
        num = klujax.factor(Ai, Aj, Ax, sym)
    del num
    gc.collect()

    rss_before = _get_rss_kb()
    for _ in range(5000):
        num = klujax.factor(Ai, Aj, Ax, sym)
    del num
    gc.collect()
    rss_after = _get_rss_kb()

    assert rss_after - rss_before < 10_000, (
        f"RSS grew by {rss_after - rss_before} KB after 5000 factor() calls"
    )
