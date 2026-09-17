"""Stage 0.5.9 — frozen golden corpus.

Recomputes every case in ``golden/inputs.npz`` and compares against the outputs
captured from the current implementation. The corpus is regenerated only with
an explicit, reviewed change (``_generate_golden.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import jax.numpy as jnp
import klujax

GOLDEN = Path(__file__).resolve().parent / "golden"


def _run(op, Ai, Aj, Ax, b, n_col):
    if op == "solve":
        return klujax.solve(Ai, Aj, Ax, b)
    if op == "dot":
        return klujax.dot(Ai, Aj, Ax, b)
    if op == "tsolve_with_symbol":
        sym = klujax.analyze(Ai, Aj, n_col)
        return klujax.tsolve_with_symbol(Ai, Aj, Ax, b, sym)
    if op == "solve_with_numeric":
        sym = klujax.analyze(Ai, Aj, n_col)
        num = klujax.factor(Ai, Aj, Ax, sym)
        return klujax.solve_with_numeric(num, b, sym)
    msg = f"unknown op {op!r}"
    raise ValueError(msg)


@pytest.mark.skipif(
    not (GOLDEN / "inputs.npz").exists(), reason="golden corpus not generated"
)
def test_golden_corpus():
    inputs = np.load(GOLDEN / "inputs.npz", allow_pickle=False)
    outputs = np.load(GOLDEN / "outputs.npz", allow_pickle=False)
    names = [str(n) for n in inputs["case_names"]]

    for name in names:
        Ai = jnp.asarray(inputs[f"{name}_Ai"])
        Aj = jnp.asarray(inputs[f"{name}_Aj"])
        Ax = jnp.asarray(inputs[f"{name}_Ax"])
        b = jnp.asarray(inputs[f"{name}_b"])
        n_col = int(inputs[f"{name}_n_col"])
        op = str(inputs[f"{name}_op"])
        expected = outputs[f"{name}_x"]

        got = np.asarray(_run(op, Ai, Aj, Ax, b, n_col))
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
