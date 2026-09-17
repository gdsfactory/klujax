"""Stage 0.5.2 — exhaustive Ax/b shape-table coverage.

The README documents six Ax/b dimension combinations. The legacy `tests.py`
only exercises a subset; this file covers all six, plus n_lhs broadcasting and
direct multi-RHS (`n_rhs > 1`) calls.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp
import klujax
from tests_characterization.helpers import DTYPES, dense, rand_coo, rand_rhs

N_COL = 6
N_NZ = 12


def _ref_solve(A, b):
    """Dense reference; A may be (n,n) or (n_lhs,n,n), b may be (n,), (n,rhs)..."""
    A = np.asarray(A)
    b = np.asarray(b)
    if A.ndim == 2:
        return np.linalg.solve(A, b)
    assert b.ndim == A.ndim  # (n_lhs, n, ...)
    return np.stack([np.linalg.solve(A[i], b[i]) for i in range(A.shape[0])])


@pytest.mark.parametrize("dtype", DTYPES)
def test_shape_table(dtype):
    n_lhs, n_rhs = 2, 3
    Ai1, Aj1, Ax1 = rand_coo(N_COL, N_NZ, dtype=dtype, seed=1)
    Ai2, Aj2, Ax2 = rand_coo(N_COL, N_NZ, n_lhs=n_lhs, dtype=dtype, seed=2)
    A1 = dense(Ai1, Aj1, Ax1, N_COL)
    A2 = dense(Ai2, Aj2, Ax2, N_COL)

    def solve_batched(A, b):
        b = np.asarray(b)
        return np.stack([np.linalg.solve(A[i], b[i]) for i in range(A.shape[0])])

    b1 = rand_rhs((N_COL,), dtype, 3)
    b2 = rand_rhs((N_COL, n_rhs), dtype, 4)
    b3 = rand_rhs((n_lhs, N_COL, n_rhs), dtype, 5)
    b4 = rand_rhs((N_COL,), dtype, 6)
    b5 = rand_rhs((n_lhs, N_COL), dtype, 7)
    b6 = rand_rhs((n_lhs, N_COL, n_rhs), dtype, 8)

    cases = [
        (Ai1, Aj1, Ax1, b1, (N_COL,), lambda: np.linalg.solve(A1, np.asarray(b1))),
        (
            Ai1,
            Aj1,
            Ax1,
            b2,
            (N_COL, n_rhs),
            lambda: np.linalg.solve(A1, np.asarray(b2)),
        ),
        (
            Ai1,
            Aj1,
            Ax1,
            b3,
            (n_lhs, N_COL, n_rhs),
            lambda: np.stack(
                [np.linalg.solve(A1, np.asarray(b3)[i]) for i in range(n_lhs)]
            ),
        ),
        (
            Ai2,
            Aj2,
            Ax2,
            b4,
            (n_lhs, N_COL),
            lambda: np.stack(
                [np.linalg.solve(A2[i], np.asarray(b4)) for i in range(n_lhs)]
            ),
        ),
        (
            Ai2,
            Aj2,
            Ax2,
            b5,
            (n_lhs, N_COL),
            lambda: solve_batched(A2, b5),
        ),
        (
            Ai2,
            Aj2,
            Ax2,
            b6,
            (n_lhs, N_COL, n_rhs),
            lambda: solve_batched(A2, b6),
        ),
    ]

    for Ai, Aj, Ax, b, out_shape, ref in cases:
        x = klujax.solve(Ai, Aj, Ax, b)
        assert x.shape == out_shape
        np.testing.assert_allclose(np.asarray(x), ref(), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("dtype", DTYPES)
def test_n_lhs_broadcast(dtype):
    n_lhs = 3
    # Ax 2D with n_lhs==1 broadcast against b 2D with n_lhs==3.
    Ai, Aj, Ax1 = rand_coo(N_COL, N_NZ, n_lhs=1, dtype=dtype, seed=12)
    b2 = rand_rhs((n_lhs, N_COL), dtype, 13)
    A1 = dense(Ai, Aj, Ax1, N_COL)[0]
    ref = np.stack([np.linalg.solve(A1, np.asarray(b2)[i]) for i in range(n_lhs)])
    x = klujax.solve(Ai, Aj, Ax1, b2)
    assert x.shape == (n_lhs, N_COL)
    np.testing.assert_allclose(np.asarray(x), ref, rtol=1e-10, atol=1e-10)

    # Ax 1D broadcast against b 3D (leading n_lhs).
    Ai1, Aj1, Ax1d = rand_coo(N_COL, N_NZ, dtype=dtype, seed=11)
    b3 = rand_rhs((n_lhs, N_COL, 2), dtype, 14)
    A1d = dense(Ai1, Aj1, Ax1d, N_COL)
    ref3 = np.stack([np.linalg.solve(A1d, np.asarray(b3)[i]) for i in range(n_lhs)])
    x = klujax.solve(Ai1, Aj1, Ax1d, b3)
    assert x.shape == (n_lhs, N_COL, 2)
    np.testing.assert_allclose(np.asarray(x), ref3, rtol=1e-10, atol=1e-10)

    # Ax 2D with n_lhs==3 broadcast against b 3D with n_lhs==1.
    Ai3, Aj3, Ax3 = rand_coo(N_COL, N_NZ, n_lhs=n_lhs, dtype=dtype, seed=15)
    b_one = rand_rhs((1, N_COL, 2), dtype, 16)
    A3 = dense(Ai3, Aj3, Ax3, N_COL)
    ref_b = np.stack(
        [np.linalg.solve(A3[i], np.asarray(b_one)[0]) for i in range(n_lhs)]
    )
    x = klujax.solve(Ai3, Aj3, Ax3, b_one)
    assert x.shape == (n_lhs, N_COL, 2)
    np.testing.assert_allclose(np.asarray(x), ref_b, rtol=1e-10, atol=1e-10)


def test_mismatched_n_lhs_raises():
    Ai, Aj, Ax = rand_coo(N_COL, N_NZ, n_lhs=2, dtype=np.float64, seed=21)
    b = rand_rhs((3, N_COL, 1), np.float64, 22)
    with pytest.raises(ValueError):
        klujax.solve(Ai, Aj, Ax, b)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_lhs,n_rhs", [(2, 3), (3, 1), (1, 4)])
def test_multi_rhs_direct(dtype, n_lhs, n_rhs):
    """Direct n_rhs>1 (not via vmap) for the split solve routines."""
    Ai, Aj, Ax = rand_coo(N_COL, N_NZ, n_lhs=n_lhs, dtype=dtype, seed=31)
    b = rand_rhs((n_lhs, N_COL, n_rhs), dtype, 32)
    A = dense(Ai, Aj, Ax, N_COL)
    ref = _ref_solve(A, np.asarray(b))

    symbolic = klujax.analyze(Ai, Aj, N_COL)

    x = klujax.solve_with_symbol(Ai, Aj, Ax, b, symbolic)
    np.testing.assert_allclose(np.asarray(x), ref, rtol=1e-10, atol=1e-10)

    num = klujax.factor(Ai, Aj, Ax, symbolic)
    x = klujax.solve_with_numeric(num, b, symbolic)
    np.testing.assert_allclose(np.asarray(x), ref, rtol=1e-10, atol=1e-10)

    x = klujax.tsolve_with_symbol(Ai, Aj, Ax, b, symbolic)
    ref_t = _ref_solve(np.swapaxes(A, -1, -2), np.asarray(b))
    np.testing.assert_allclose(np.asarray(x), ref_t, rtol=1e-10, atol=1e-10)

    x = klujax.tsolve_with_numeric(num, b, symbolic)
    np.testing.assert_allclose(np.asarray(x), ref_t, rtol=1e-10, atol=1e-10)


def test_n_nz_mismatch_raises():
    Ai, Aj, Ax = rand_coo(N_COL, N_NZ, dtype=np.float64, seed=41)
    b = rand_rhs((N_COL,), np.float64, 42)
    # Truncate Ai so it disagrees with Ax.
    with pytest.raises(RuntimeError):
        klujax.solve(Ai[:-1], Aj, Ax, b)
