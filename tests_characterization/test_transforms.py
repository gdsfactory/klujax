"""Stage 0.5.8 — AD / vmap coverage for the split solve primitives.

`tests.py` covers AD/vmap for `solve`, `dot`, and `solve_with_symbol`, but not
for `solve_with_numeric`, `tsolve_with_*`, or `factor`/`refactor`. This file
fills those gaps.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
import klujax
from tests_characterization.helpers import DTYPES, dense, rand_coo, rand_rhs

N_COL = 6
N_NZ = 12


@pytest.mark.parametrize("dtype", DTYPES)
def test_vmap_solve_with_symbol_over_ax(dtype):
    Ai, Aj, Ax = rand_coo(N_COL, N_NZ, dtype=dtype, seed=1)
    b = rand_rhs((N_COL,), dtype, 2)
    sym = klujax.analyze(Ai, Aj, N_COL)

    batch = 4
    Ax_batch = jnp.stack([Ax + 0.1 * (k + 1) for k in range(batch)])
    out = jax.vmap(lambda ax: klujax.solve_with_symbol(Ai, Aj, ax, b, sym))(Ax_batch)

    for k in range(batch):
        A = dense(Ai, Aj, np.asarray(Ax_batch[k]), N_COL)
        ref = np.linalg.solve(A, np.asarray(b))
        np.testing.assert_allclose(np.asarray(out[k]), ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("dtype", DTYPES)
def test_vmap_tsolve_with_symbol_over_b(dtype):
    Ai, Aj, Ax = rand_coo(N_COL, N_NZ, dtype=dtype, seed=3)
    sym = klujax.analyze(Ai, Aj, N_COL)
    A = dense(Ai, Aj, Ax, N_COL)
    b_batch = rand_rhs((3, N_COL), dtype, 4)

    out = jax.vmap(lambda b: klujax.tsolve_with_symbol(Ai, Aj, Ax, b, sym))(b_batch)
    for k in range(3):
        ref = np.linalg.solve(A.T, np.asarray(b_batch[k]))
        np.testing.assert_allclose(np.asarray(out[k]), ref, rtol=1e-10, atol=1e-10)


def test_vmap_solve_with_numeric_over_b():
    Ai, Aj, Ax = rand_coo(N_COL, N_NZ, dtype=np.float64, seed=5)
    sym = klujax.analyze(Ai, Aj, N_COL)
    num = klujax.factor(Ai, Aj, Ax, sym)
    A = dense(Ai, Aj, Ax, N_COL)
    b_batch = rand_rhs((3, N_COL), np.float64, 6)

    out = jax.vmap(lambda b: klujax.solve_with_numeric(num, b, sym))(b_batch)
    for k in range(3):
        ref = np.linalg.solve(A, np.asarray(b_batch[k]))
        np.testing.assert_allclose(np.asarray(out[k]), ref, rtol=1e-10, atol=1e-10)


def test_solve_with_numeric_jvp_wrt_b():
    Ai, Aj, Ax = rand_coo(N_COL, N_NZ, dtype=np.float64, seed=7)
    sym = klujax.analyze(Ai, Aj, N_COL)
    num = klujax.factor(Ai, Aj, Ax, sym)
    A = dense(Ai, Aj, Ax, N_COL)
    b = rand_rhs((N_COL,), np.float64, 8)
    db = rand_rhs((N_COL,), np.float64, 9)

    _, tangent = jax.jvp(lambda b: klujax.solve_with_numeric(num, b, sym), (b,), (db,))
    ref = np.linalg.solve(A, np.asarray(db))
    np.testing.assert_allclose(np.asarray(tangent), ref, rtol=1e-10, atol=1e-10)


def test_solve_with_numeric_transpose_wrt_b():
    Ai, Aj, Ax = rand_coo(N_COL, N_NZ, dtype=np.float64, seed=10)
    sym = klujax.analyze(Ai, Aj, N_COL)
    num = klujax.factor(Ai, Aj, Ax, sym)
    A = dense(Ai, Aj, Ax, N_COL)
    b = rand_rhs((N_COL,), np.float64, 11)
    ct = rand_rhs((N_COL,), np.float64, 12)

    _, vjp = jax.vjp(lambda b: klujax.solve_with_numeric(num, b, sym), b)
    (b_bar,) = vjp(ct)
    ref = np.linalg.solve(A.T, np.asarray(ct))
    np.testing.assert_allclose(np.asarray(b_bar), ref, rtol=1e-10, atol=1e-10)


def test_refactor_preserves_factor_value():
    Ai, Aj, Ax = rand_coo(N_COL, N_NZ, dtype=np.float64, seed=13)
    sym = klujax.analyze(Ai, Aj, N_COL)
    num = klujax.factor(Ai, Aj, Ax, sym)
    Ax2 = Ax + 0.05
    num2 = klujax.refactor(Ai, Aj, Ax2, num, sym)
    assert num2 is num
    b = rand_rhs((N_COL,), np.float64, 14)
    x_num = klujax.solve_with_numeric(num2, b, sym)
    x_sym = klujax.solve_with_symbol(Ai, Aj, Ax2, b, sym)
    np.testing.assert_allclose(np.asarray(x_num), np.asarray(x_sym), rtol=1e-8)
