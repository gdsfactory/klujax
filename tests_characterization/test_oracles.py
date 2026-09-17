"""Stage 0.5.5 / 0.5.6 — independent oracles and structural stress.

Compares against `scipy.sparse.linalg.spsolve` (an independent implementation)
and exercises the structure that KLU's AMD/BTF/fill-in machinery exists for:
banded, block-diagonal, *reducible* (block-upper-triangular), and arrow
matrices. Also checks the oracle-free residual ||Ax - b||.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import klujax
from tests_characterization.helpers import DTYPES, rand_coo, rand_rhs


def _crand(shape, dtype, rng):
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
            dtype
        )
    return rng.standard_normal(shape).astype(dtype)


def _from_dense(A):
    Ai, Aj = np.nonzero(np.abs(A) > 0)
    Ax = A[Ai, Aj]
    return klujax.coalesce(
        jnp.asarray(Ai, jnp.int32), jnp.asarray(Aj, jnp.int32), jnp.asarray(Ax)
    )


def _scipy_solve(Ai, Aj, Ax, b, n_col):
    A = sp.coo_matrix(
        (np.asarray(Ax), (np.asarray(Ai), np.asarray(Aj))), shape=(n_col, n_col)
    ).tocsr()
    return spsolve(A, np.asarray(b))


def _check(Ai, Aj, Ax, b, n_col, rtol=1e-9):
    x = klujax.solve(Ai, Aj, Ax, b)
    ref = _scipy_solve(Ai, Aj, Ax, b, n_col)
    np.testing.assert_allclose(np.asarray(x), ref, rtol=rtol, atol=rtol)
    # Oracle-free residual.
    A = sp.coo_matrix(
        (np.asarray(Ax), (np.asarray(Ai), np.asarray(Aj))), shape=(n_col, n_col)
    ).tocsr()
    res = np.asarray(A @ np.asarray(x)) - np.asarray(b)
    assert np.linalg.norm(res) <= 1e-7 * (1 + np.linalg.norm(np.asarray(b)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_random_large(dtype):
    n, n_nz = 120, 720
    Ai, Aj, Ax = rand_coo(n, n_nz, dtype=dtype, seed=1)
    b = rand_rhs((n,), dtype, 2)
    _check(Ai, Aj, Ax, b, n)


@pytest.mark.parametrize("dtype", DTYPES)
def test_tridiagonal(dtype):
    n = 50
    A = np.diag(np.full(n, 4.0)).astype(dtype)
    A += np.diag(np.full(n - 1, -1.0).astype(dtype), 1)
    A += np.diag(np.full(n - 1, -1.0).astype(dtype), -1)
    Ai, Aj, Ax = _from_dense(A)
    b = rand_rhs((n,), dtype, 3)
    _check(Ai, Aj, Ax, b, n)


@pytest.mark.parametrize("dtype", DTYPES)
def test_block_diagonal(dtype):
    rng = np.random.default_rng(4)
    blocks, size = 3, 5
    n = blocks * size
    A = np.zeros((n, n), dtype=dtype)
    for k in range(blocks):
        s = k * size
        A[s : s + size, s : s + size] = _crand((size, size), dtype, rng) + 10 * np.eye(
            size, dtype=dtype
        )
    Ai, Aj, Ax = _from_dense(A)
    b = rand_rhs((n,), dtype, 5)
    _check(Ai, Aj, Ax, b, n)


@pytest.mark.parametrize("dtype", DTYPES)
def test_block_upper_triangular_reducible(dtype):
    """A reducible matrix (lower-left block is zero) forces multi-block BTF."""
    rng = np.random.default_rng(6)
    blocks, size = 3, 5
    n = blocks * size
    A = np.zeros((n, n), dtype=dtype)
    for i in range(blocks):
        si = i * size
        A[si : si + size, si : si + size] = _crand(
            (size, size), dtype, rng
        ) + 10 * np.eye(size, dtype=dtype)
        for j in range(i + 1, blocks):
            sj = j * size
            A[si : si + size, sj : sj + size] = 0.3 * _crand((size, size), dtype, rng)
    Ai, Aj, Ax = _from_dense(A)
    b = rand_rhs((n,), dtype, 7)
    _check(Ai, Aj, Ax, b, n)


@pytest.mark.parametrize("dtype", DTYPES)
def test_arrow_matrix(dtype):
    rng = np.random.default_rng(8)
    n = 40
    A = np.diag(np.full(n, 5.0)).astype(dtype)
    A[0, :] = _crand((n,), dtype, rng)
    A[:, 0] = _crand((n,), dtype, rng)
    A[0, 0] = 5.0
    Ai, Aj, Ax = _from_dense(A)
    b = rand_rhs((n,), dtype, 9)
    _check(Ai, Aj, Ax, b, n)


@pytest.mark.parametrize("dtype", DTYPES)
def test_dot_oracle(dtype):
    """`dot` (b = A x) against scipy sparse matmul, eager and batched."""
    n = 20
    Ai, Aj, Ax = rand_coo(n, 60, dtype=dtype, seed=20)
    A = sp.coo_matrix(
        (np.asarray(Ax), (np.asarray(Ai), np.asarray(Aj))), shape=(n, n)
    ).tocsr()

    x1 = rand_rhs((n,), dtype, 21)
    np.testing.assert_allclose(
        np.asarray(klujax.dot(Ai, Aj, Ax, x1)),
        A @ np.asarray(x1),
        rtol=1e-10,
        atol=1e-10,
    )

    x2 = rand_rhs((n, 3), dtype, 22)
    np.testing.assert_allclose(
        np.asarray(klujax.dot(Ai, Aj, Ax, x2)),
        A @ np.asarray(x2),
        rtol=1e-10,
        atol=1e-10,
    )

    Ai_b, Aj_b, Ax_b = rand_coo(n, 60, n_lhs=2, dtype=dtype, seed=23)
    A_b = [
        sp.coo_matrix(
            (np.asarray(Ax_b[k]), (np.asarray(Ai_b), np.asarray(Aj_b))), shape=(n, n)
        ).tocsr()
        for k in range(2)
    ]
    x3 = rand_rhs((2, n), dtype, 24)
    out = np.asarray(klujax.dot(Ai_b, Aj_b, Ax_b, x3))
    for k in range(2):
        np.testing.assert_allclose(
            out[k], A_b[k] @ np.asarray(x3)[k], rtol=1e-10, atol=1e-10
        )
