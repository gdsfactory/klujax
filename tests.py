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
        x = klujax.solve_with_numeric(num, b, symbolic)
        klujax.free_numeric(num, dependency=x)
        return x

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
    """Return current RSS in kilobytes (Linux-only)."""
    with Path(f"/proc/{os.getpid()}/status").open() as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


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


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("constant_values", [False, True])
@pytest.mark.parametrize("consume_result", [False, True])
@pytest.mark.parametrize("multiple_consumers", [False, True])
def test_compiled_numeric_lifetime(
    dtype, constant_values, consume_result, multiple_consumers
):
    """Allocation/free survive DCE and execute once per dynamic loop iteration."""
    Ai, Aj, Ax, b = _get_rand_arrs_1d(100, 32, dtype=dtype)
    sym = klujax.analyze(Ai, Aj, 32)
    A = jnp.zeros((32, 32), dtype=dtype).at[Ai, Aj].add(Ax)
    expected = jsp.linalg.solve(A, b)[0]
    expected_t = jsp.linalg.solve(A.T, 2 * b)[0] if multiple_consumers else 0

    @jax.jit
    def run(n_iter, values):
        def body(i, acc):
            scale = 1.0 if constant_values else 1 + 1e-6 * i
            num = klujax.factor(Ai, Aj, values * scale, sym)
            x = klujax.solve_with_numeric(num, b, sym)
            y = (
                klujax.tsolve_with_numeric(num, 2 * b, sym)
                if multiple_consumers
                else jnp.zeros_like(x)
            )
            # The status is deliberately unused, matching issue #36.
            dependency = {"normal": x, "transpose": y} if multiple_consumers else x
            klujax.free_numeric(num, dependency=dependency)
            return acc + x[0] + y[0] if consume_result else acc

        return lax.fori_loop(0, n_iter, body, jnp.zeros((), dtype=dtype))

    run(1, Ax).block_until_ready()
    gc.collect()
    live_before, total_before = klujax.klujax_cpp._numeric_allocation_counts()
    for n_iter in (0, 100, 1000, 3000):
        result = run(n_iter, Ax).block_until_ready()
        scales = 1.0 if constant_values else 1 + 1e-6 * np.arange(n_iter)
        reference = (expected + expected_t) * np.sum(np.ones(n_iter) / scales)
        np.testing.assert_allclose(
            result, reference if consume_result else 0, rtol=1e-10
        )
        live, total = klujax.klujax_cpp._numeric_allocation_counts()
        assert live == live_before
        assert total - total_before == n_iter
        total_before = total


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("mapped", [False, True])
def test_compiled_numeric_lifetime_batched(dtype, mapped):
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, 5, dtype=dtype)
    sym = klujax.analyze(Ai, Aj, 5)
    values = jnp.stack([Ax, Ax * 2, Ax * 3])

    def solve_one(values):
        num = klujax.factor(Ai, Aj, values, sym)
        rhs = b if mapped else jnp.broadcast_to(b, (3, 5))
        x = klujax.solve_with_numeric(num, rhs, sym)
        klujax.free_numeric(num, dependency=x)
        return x

    run = jax.jit(jax.vmap(solve_one) if mapped else solve_one)
    A = jnp.zeros((5, 5), dtype=dtype).at[Ai, Aj].add(Ax)
    expected = jsp.linalg.solve(A, b)[None] / jnp.arange(1, 4)[:, None]
    gc.collect()
    live_before, total_before = klujax.klujax_cpp._numeric_allocation_counts()
    for _ in range(3):
        result = run(values).block_until_ready()
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        live, total = klujax.klujax_cpp._numeric_allocation_counts()
        assert live == live_before
        assert total - total_before == 3
        total_before = total


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
# refactor_and_solve has no batching rule, so the fused path is not vmapped.
@pytest.mark.parametrize(
    ("fused", "mapped"), [(False, False), (False, True), (True, False)]
)
def test_compiled_refactor_updates_raw_numeric(dtype, fused, mapped):
    """Inside jit, solves through the returned handle see the refactored values.

    Regression: refactor() returned its input handle, so the unused refactor
    primitive was dead-code eliminated and solves used the original factors.
    """
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, 5, dtype=dtype)
    Ax2 = _make_ax2(Ai, Aj, Ax, dtype=dtype)
    sym = klujax.analyze(Ai, Aj, 5)

    def solve(values, new_values):
        num = klujax.factor(Ai, Aj, values, sym)

        def body(i, carry):
            num, _ = carry
            target = new_values * (1 + i)  # the matrix changes every iteration
            if fused:
                x, num = klujax.refactor_and_solve(Ai, Aj, target, b, num, sym)
            else:
                num = klujax.refactor(Ai, Aj, target, num, sym)
                x = klujax.solve_with_numeric(num, b, sym)
            return num, x.reshape(b.shape)

        num, x = lax.fori_loop(0, 3, body, (num, jnp.zeros_like(b)))
        y = klujax.solve_with_numeric(num, b, sym).reshape(b.shape)
        klujax.free_numeric(num, dependency=(x, y))
        return x, y

    scales = jnp.arange(1, 4, dtype=dtype)[:, None]
    run = jax.jit(jax.vmap(solve) if mapped else solve)
    values, new_values = (Ax * scales, Ax2 * scales) if mapped else (Ax, Ax2)
    A2 = jnp.zeros((5, 5), dtype=dtype).at[Ai, Aj].add(Ax2)
    # The last refactor uses 3 * new_values.
    expected = jsp.linalg.solve(3 * A2, b)
    expected = expected[None] / scales if mapped else expected
    gc.collect()
    live_before, total_before = klujax.klujax_cpp._numeric_allocation_counts()
    x, y = run(values, new_values)
    np.testing.assert_allclose(x, expected, rtol=1e-10)
    np.testing.assert_allclose(y, expected, rtol=1e-10)
    live, total = klujax.klujax_cpp._numeric_allocation_counts()
    assert live == live_before
    assert total - total_before == (3 if mapped else 1)


def test_raw_free_requires_dependency():
    @jax.jit
    def free(raw):
        return klujax.free_numeric(raw)

    with pytest.raises(ValueError, match="require dependency"):
        free(jnp.zeros(1, dtype=jnp.uint64))


@pytest.mark.parametrize("with_dependency", [False, True])
def test_eager_numeric_cannot_be_freed_during_tracing(with_dependency):
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, 5, dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, 5)
    num = klujax.factor(Ai, Aj, Ax, sym)

    @jax.jit
    def run(rhs):
        x = klujax.solve_with_numeric(num, rhs, sym)
        klujax.free_numeric(num, dependency=x if with_dependency else None)
        return x

    with pytest.raises(ValueError, match="Close eager numeric owners outside"):
        run(b)
    # Rejecting the trace must leave the eager owner usable.
    klujax.solve_with_numeric(num, b, sym).block_until_ready()


def test_vmap_free_rejects_shared_numeric():
    raw = jnp.zeros(1, dtype=jnp.uint64)
    with pytest.raises(ValueError, match="shared numeric handle"):
        jax.jit(jax.vmap(lambda x: klujax.free_numeric(raw, dependency=x)))(
            jnp.ones((3, 5))
        )


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("stopped_dependency", [False, True])
@pytest.mark.parametrize("in_loop", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("transform", ["grad", "jvp", "vjp"])
def test_raw_numeric_autodiff_rejected(
    dtype, stopped_dependency, in_loop, transpose, transform
):
    """Reject before execution, including the stopped-dependency UAF reproducer."""
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, 5, dtype=dtype)
    sym = klujax.analyze(Ai, Aj, 5)
    solve = klujax.tsolve_with_numeric if transpose else klujax.solve_with_numeric

    def objective(rhs):
        def body(i, total):
            num = klujax.factor(Ai, Aj, Ax * (1 + 0.01 * i), sym)
            x = solve(num, rhs, sym)
            dependency = lax.stop_gradient(x) if stopped_dependency else x
            klujax.free_numeric(num, dependency=dependency)
            return total + jnp.real(jnp.vdot(x, x))

        return lax.fori_loop(0, 3, body, 0.0) if in_loop else body(0, 0.0)

    if transform == "grad":
        differentiated = jax.grad(objective)
    elif transform == "jvp":

        def differentiated(rhs):
            return jax.jvp(objective, (rhs,), (jnp.ones_like(rhs),))

    else:

        def differentiated(rhs):
            return jax.vjp(objective, rhs)[1](1.0)

    gc.collect()
    counts_before = klujax.klujax_cpp._numeric_allocation_counts()
    # Lower rather than execute: the old stopped-dependency graph can use freed
    # memory. This must fail during tracing, before any native allocations.
    with pytest.raises(ValueError, match="Autodiff through raw numeric handles"):
        jax.jit(differentiated).lower(b)
    assert klujax.klujax_cpp._numeric_allocation_counts() == counts_before


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("eager_numeric", [False, True])
def test_supported_numeric_autodiff_lifetime(dtype, eager_numeric):
    """Supported alternatives preserve gradients and bounded allocations in loops."""
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, 5, dtype=dtype)
    sym = klujax.analyze(Ai, Aj, 5)
    numeric = klujax.factor(Ai, Aj, Ax, sym) if eager_numeric else None
    A = jnp.zeros((5, 5), dtype=dtype).at[Ai, Aj].add(Ax)

    def objective(rhs):
        def body(i, total):
            x = (
                klujax.solve_with_numeric(numeric, rhs + i, sym)
                if eager_numeric
                else klujax.solve_with_symbol(Ai, Aj, Ax, rhs + i, sym)
            )
            return total + jnp.real(jnp.vdot(x, x))

        return lax.fori_loop(0, 3, body, 0.0)

    def dense_objective(rhs):
        solutions = [jsp.linalg.solve(A, rhs + i) for i in range(3)]
        return sum(jnp.real(jnp.vdot(x, x)) for x in solutions)

    expected = jax.value_and_grad(dense_objective)(b)
    run = jax.jit(jax.value_and_grad(objective))
    gc.collect()
    counts_before = klujax.klujax_cpp._numeric_allocation_counts()
    for _ in range(3):
        value, gradient = jax.block_until_ready(run(b))
        np.testing.assert_allclose(value, expected[0], rtol=1e-10)
        np.testing.assert_allclose(gradient, expected[1], rtol=1e-10)
        assert klujax.klujax_cpp._numeric_allocation_counts() == counts_before
    if numeric is not None:
        numeric.close()
