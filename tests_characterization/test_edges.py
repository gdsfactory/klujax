"""Stage 0.5.7 — error, edge, and degenerate paths.

These pin the *current* (C++) error behavior so the Rust port can match it.
Uncoalesced duplicate indices are treated as unsupported/UB (KLU requires a
coalesced matrix) and are covered via `coalesce` in `test_coalesce.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp
import klujax
from tests_characterization.helpers import dense, rand_coo, rand_rhs


def test_singular_raises_runtime_error():
    Ai = jnp.array([0, 0, 1, 1], jnp.int32)
    Aj = jnp.array([0, 1, 0, 1], jnp.int32)
    Ax = jnp.array([1.0, 1.0, 1.0, 1.0])
    b = jnp.array([1.0, 2.0])
    with pytest.raises(RuntimeError):
        klujax.solve(Ai, Aj, Ax, b)


def test_single_element_system():
    x = klujax.solve(
        jnp.array([0], jnp.int32),
        jnp.array([0], jnp.int32),
        jnp.array([5.0]),
        jnp.array([10.0]),
    )
    np.testing.assert_allclose(np.asarray(x), [2.0])


def test_empty_matrix_raises_singular():
    with pytest.raises(RuntimeError):
        klujax.solve(
            jnp.array([], jnp.int32),
            jnp.array([], jnp.int32),
            jnp.array([]),
            jnp.array([1.0, 2.0]),
        )


def test_out_of_bounds_index_raises():
    with pytest.raises(RuntimeError, match="Ai.max"):
        klujax.solve(
            jnp.array([0, 5], jnp.int32),
            jnp.array([0, 0], jnp.int32),
            jnp.array([1.0, 1.0]),
            jnp.array([1.0, 2.0]),
        )


def test_negative_index_raises():
    with pytest.raises(RuntimeError, match="negative index"):
        klujax.solve(
            jnp.array([0, -1], jnp.int32),
            jnp.array([0, 0], jnp.int32),
            jnp.array([1.0, 1.0]),
            jnp.array([1.0, 2.0]),
        )


def test_nan_propagates():
    Ai, Aj, _ = rand_coo(4, 6, dtype=np.float64, seed=1)
    Ax = jnp.array([jnp.nan, 4.0], dtype=jnp.float64)
    # Use a 2x2 system with an explicit NaN on the diagonal.
    Ai2 = jnp.array([0, 1], jnp.int32)
    Aj2 = jnp.array([0, 1], jnp.int32)
    x = klujax.solve(Ai2, Aj2, Ax, jnp.array([1.0, 2.0]))
    assert np.isnan(np.asarray(x)[0])


def test_unsorted_unique_indices():
    """Order within the COO triple must not matter for unique indices."""
    Ai = jnp.array([1, 0], jnp.int32)
    Aj = jnp.array([1, 0], jnp.int32)
    Ax = jnp.array([4.0, 2.0])
    b = jnp.array([10.0, 20.0])
    x = klujax.solve(Ai, Aj, Ax, b)
    ref = np.linalg.solve(dense(Ai, Aj, Ax, 2), np.asarray(b))
    np.testing.assert_allclose(np.asarray(x), ref, rtol=1e-12, atol=1e-12)
