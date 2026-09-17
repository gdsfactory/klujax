"""Stage 0.5.3 — dtype and index-width coverage.

klujax documents that float32/complex64 are upcast to float64/complex128.
`tests.py` only ever uses 64-bit dtypes, so this locks the documented behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp
import klujax
from klujax import COMPLEX_DTYPES
from tests_characterization.helpers import dense, rand_coo, rand_rhs

N_COL = 5
N_NZ = 8


def test_complex_dtypes_membership():
    assert np.complex64 in COMPLEX_DTYPES
    assert np.complex128 in COMPLEX_DTYPES
    assert np.float64 not in COMPLEX_DTYPES


def test_float32_upcasts_to_float64():
    Ai, Aj, Ax = rand_coo(N_COL, N_NZ, dtype=np.float32, seed=1)
    b = rand_rhs((N_COL,), np.float32, 2)
    assert Ax.dtype == jnp.float32 and b.dtype == jnp.float32
    x = klujax.solve(Ai, Aj, Ax, b)
    assert x.dtype == jnp.float64
    ref = np.linalg.solve(
        dense(Ai, Aj, Ax, N_COL).astype(np.float64),
        np.asarray(b).astype(np.float64),
    )
    np.testing.assert_allclose(np.asarray(x), ref, rtol=1e-5, atol=1e-6)


def test_complex64_upcasts_to_complex128():
    Ai, Aj, Ax = rand_coo(N_COL, N_NZ, dtype=np.complex64, seed=3)
    b = rand_rhs((N_COL,), np.complex64, 4)
    assert Ax.dtype == jnp.complex64 and b.dtype == jnp.complex64
    x = klujax.solve(Ai, Aj, Ax, b)
    assert x.dtype == jnp.complex128
    ref = np.linalg.solve(
        dense(Ai, Aj, Ax, N_COL).astype(np.complex128),
        np.asarray(b).astype(np.complex128),
    )
    np.testing.assert_allclose(np.asarray(x), ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("index_dtype", [jnp.int32, jnp.int64])
def test_index_widths(index_dtype):
    Ai, Aj, Ax = rand_coo(N_COL, N_NZ, dtype=np.float64, seed=5)
    b = rand_rhs((N_COL,), np.float64, 6)
    x32 = klujax.solve(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, b)
    x64 = klujax.solve(Ai.astype(jnp.int64), Aj.astype(jnp.int64), Ax, b)
    assert x64.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(x32), np.asarray(x64), rtol=1e-12, atol=1e-12)


def test_mixed_ax_b_dtype():
    Ai, Aj, Ax = rand_coo(N_COL, N_NZ, dtype=np.float64, seed=7)
    b = rand_rhs((N_COL,), np.float32, 8)
    x = klujax.solve(Ai, Aj, Ax, b)
    assert x.dtype == jnp.float64
    ref = np.linalg.solve(dense(Ai, Aj, Ax, N_COL), np.asarray(b).astype(np.float64))
    np.testing.assert_allclose(np.asarray(x), ref, rtol=1e-6, atol=1e-7)
