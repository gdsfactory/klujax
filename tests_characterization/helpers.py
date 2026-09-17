"""Shared helpers for the Rust-migration characterization tests.

These tests lock down the *behavior* of the current (SuiteSparse-backed)
implementation so that the pure-Rust port can be validated against a frozen
contract. See `work.md` Stage 0.5.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

import klujax

DTYPES = [np.float64, np.complex128]


def _rand(shape, dtype, rng):
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
            dtype
        )
    return rng.standard_normal(shape).astype(dtype)


def rand_coo(
    n_col: int,
    n_nz: int,
    *,
    n_lhs: int | None = None,
    dtype=np.float64,
    seed: int = 0,
    diag: float = 10.0,
):
    """Random coalesced COO matrix with an injected strong diagonal.

    Returns ``(Ai, Aj, Ax)``; ``Ax`` is ``(n_nz + n_col,)`` when ``n_lhs`` is
    None and ``(n_lhs, n_nz + n_col)`` otherwise.
    """
    rng = np.random.default_rng(seed)
    ai = rng.integers(0, n_col, size=n_nz)
    aj = rng.integers(0, n_col, size=n_nz)
    diag_i = np.arange(n_col)
    ai = np.concatenate([ai, diag_i])
    aj = np.concatenate([aj, diag_i])
    if n_lhs is None:
        ax = _rand(n_nz + n_col, dtype, rng)
        ax[-n_col:] = diag
    else:
        ax = _rand((n_lhs, n_nz + n_col), dtype, rng)
        ax[:, -n_col:] = diag
    ai = jnp.asarray(ai, jnp.int32)
    aj = jnp.asarray(aj, jnp.int32)
    ax = jnp.asarray(ax)
    # KLU requires coalesced (sorted, unique) indices.
    return klujax.coalesce(ai, aj, ax)


def dense(Ai: Array, Aj: Array, Ax: Array, n_col: int) -> np.ndarray:
    """Build the dense matrix represented by a coalesced COO triple."""
    ai = np.asarray(Ai)
    aj = np.asarray(Aj)
    ax = np.asarray(Ax)
    if ax.ndim == 1:
        out = np.zeros((n_col, n_col), dtype=ax.dtype)
        np.add.at(out, (ai, aj), ax)
        return out
    out = np.zeros((ax.shape[0], n_col, n_col), dtype=ax.dtype)
    for m in range(ax.shape[0]):
        np.add.at(out[m], (ai, aj), ax[m])
    return out


def rand_rhs(shape, dtype, seed: int = 1) -> Array:
    return jnp.asarray(_rand(shape, dtype, np.random.default_rng(seed)))
