"""Regression coverage for native handle ownership and stale-ID rejection."""

import copy
import gc
import pickle
from concurrent.futures import ThreadPoolExecutor

import jax.numpy as jnp
import numpy as np
import pytest

import klujax


def system(dtype=jnp.float64):
    indices = jnp.array([0, 1], dtype=jnp.int32)
    values = jnp.array([2, 4], dtype=dtype)
    rhs = jnp.array([2, 8], dtype=dtype)
    return indices, indices, values, rhs


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_close_releases_ids_and_rejects_use(dtype):
    ai, aj, ax, rhs = system(dtype)
    symbolic = klujax.analyze(ai, aj, 2)
    numeric = klujax.factor(ai, aj, ax, symbolic)
    sym_id = jnp.uint64(symbolic.raw)
    num_ids = jnp.array(numeric.as_list(), dtype=jnp.uint64)
    np.testing.assert_allclose(
        klujax.solve_with_numeric(numeric, rhs, symbolic), [1, 2]
    )

    numeric.close()
    numeric.close()
    assert numeric.size == 0
    assert numeric._handles == []  # noqa: SLF001
    with pytest.raises(RuntimeError, match="closed"):
        numeric.as_list()
    with pytest.raises((RuntimeError, ValueError), match="invalid or closed"):
        klujax.solve_with_numeric(num_ids, rhs, sym_id).block_until_ready()

    symbolic.close()
    symbolic.close()
    assert symbolic._raw == 0  # noqa: SLF001
    with pytest.raises(RuntimeError, match="closed"):
        _ = symbolic.raw
    with pytest.raises((RuntimeError, ValueError), match="invalid or closed"):
        klujax.solve_with_symbol(ai, aj, ax, rhs, sym_id).block_until_ready()


@pytest.mark.parametrize("operation", [copy.copy, copy.deepcopy, pickle.dumps])
def test_owners_cannot_be_duplicated(operation):
    ai, aj, ax, rhs = system()
    with (
        klujax.analyze(ai, aj, 2) as symbolic,
        klujax.factor(ai, aj, ax, symbolic) as numeric,
    ):
        for owner in [symbolic, numeric]:
            with pytest.raises(TypeError, match="cannot be"):
                operation(owner)
        np.testing.assert_allclose(
            klujax.solve_with_numeric(numeric, rhs, symbolic), [1, 2]
        )


def test_finalizers_release_native_ids():
    ai, aj, ax, rhs = system()
    symbolic = klujax.analyze(ai, aj, 2)
    numeric = klujax.factor(ai, aj, ax, symbolic)
    sym_id = jnp.uint64(symbolic.raw)
    num_ids = jnp.array(numeric.as_list(), dtype=jnp.uint64)
    del numeric
    gc.collect()
    with pytest.raises((RuntimeError, ValueError), match="invalid or closed"):
        klujax.solve_with_numeric(num_ids, rhs, sym_id).block_until_ready()
    del symbolic
    gc.collect()
    with pytest.raises((RuntimeError, ValueError), match="invalid or closed"):
        klujax.solve_with_symbol(ai, aj, ax, rhs, sym_id).block_until_ready()


def test_concurrent_close_is_idempotent():
    ai, aj, ax, _ = system()
    symbolic = klujax.analyze(ai, aj, 2)
    numeric = klujax.factor(ai, aj, ax, symbolic)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: numeric.close(), range(20)))
        list(executor.map(lambda _: symbolic.close(), range(20)))
    assert numeric.size == 0
    assert symbolic._raw == 0  # noqa: SLF001


def test_unsorted_analysis_matches_sorted_factorization():
    ai = jnp.array([1, 0, 1, 0], dtype=jnp.int32)
    aj = jnp.array([0, 1, 1, 0], dtype=jnp.int32)
    ax = jnp.array([1.0, 1.0, 4.0, 2.0])
    with (
        klujax.analyze(ai, aj, 2) as symbolic,
        klujax.factor(ai, aj, ax, symbolic) as numeric,
    ):
        np.testing.assert_allclose(
            klujax.solve_with_numeric(numeric, jnp.array([3.0, 5.0]), symbolic),
            [1, 1],
        )
