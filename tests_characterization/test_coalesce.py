"""Stage 0.5.4 — `coalesce` semantics.

`coalesce` is used throughout `tests.py` but never asserted directly. Its
contract (lexsort by (Ai, Aj), sum duplicates, preserve dtype, idempotent)
matters for the Rust port because KLU requires coalesced input.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from klujax import coalesce


def test_sorts_by_row_then_column():
    Ai = jnp.array([1, 0, 1, 0], jnp.int32)
    Aj = jnp.array([1, 1, 0, 0], jnp.int32)
    Ax = jnp.array([4.0, 3.0, 2.0, 1.0])
    ci, cj, cx = coalesce(Ai, Aj, Ax)
    np.testing.assert_array_equal(np.asarray(ci), [0, 0, 1, 1])
    np.testing.assert_array_equal(np.asarray(cj), [0, 1, 0, 1])
    np.testing.assert_allclose(np.asarray(cx), [1.0, 3.0, 2.0, 4.0])


def test_sums_duplicates():
    Ai = jnp.array([0, 0, 0, 1], jnp.int32)
    Aj = jnp.array([0, 0, 1, 1], jnp.int32)
    Ax = jnp.array([1.0, 2.0, 5.0, 7.0])
    ci, cj, cx = coalesce(Ai, Aj, Ax)
    np.testing.assert_array_equal(np.asarray(ci), [0, 0, 1])
    np.testing.assert_array_equal(np.asarray(cj), [0, 1, 1])
    np.testing.assert_allclose(np.asarray(cx), [3.0, 5.0, 7.0])


def test_idempotent():
    Ai = jnp.array([0, 0, 1, 1], jnp.int32)
    Aj = jnp.array([0, 1, 0, 1], jnp.int32)
    Ax = jnp.array([1.0, 2.0, 3.0, 4.0])
    ci, cj, cx = coalesce(Ai, Aj, Ax)
    ci2, cj2, cx2 = coalesce(ci, cj, cx)
    np.testing.assert_array_equal(np.asarray(ci), np.asarray(ci2))
    np.testing.assert_array_equal(np.asarray(cj), np.asarray(cj2))
    np.testing.assert_allclose(np.asarray(cx), np.asarray(cx2))


def test_preserves_dtype():
    Ai = jnp.array([0, 0], jnp.int32)
    Aj = jnp.array([0, 1], jnp.int32)
    for dtype in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        Ax = jnp.array([1, 2], dtype=dtype)
        _, _, cx = coalesce(Ai, Aj, Ax)
        assert cx.dtype == dtype


def test_batched_ax():
    Ai = jnp.array([0, 0, 1], jnp.int32)
    Aj = jnp.array([0, 1, 1], jnp.int32)
    Ax = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ci, cj, cx = coalesce(Ai, Aj, Ax)
    assert cx.shape == (2, 3)
    np.testing.assert_array_equal(np.asarray(ci), [0, 0, 1])
    np.testing.assert_array_equal(np.asarray(cj), [0, 1, 1])
    np.testing.assert_allclose(np.asarray(cx), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
