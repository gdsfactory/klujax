"""Generate the frozen golden corpus from the *current* (C++) implementation.

Run with:

    PYTHONPATH=. uv run python tests_characterization/_generate_golden.py

Writes ``golden/inputs.npz`` and ``golden/outputs.npz``. The corpus is reused
unchanged as the C-vs-Rust differential harness in Stage 5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import jax.numpy as jnp
import klujax
from tests_characterization.helpers import rand_coo, rand_rhs

GOLDEN = Path(__file__).resolve().parent / "golden"
N_COL = 8
N_NZ = 16

# (name, op, kind, n_lhs, n_rhs, dtype, seed)
CASES = [
    ("solve_rand_f64", "solve", "rand", None, None, np.float64, 1),
    ("solve_rand_c128", "solve", "rand", None, None, np.complex128, 2),
    ("solve_batched_f64", "solve", "rand", 3, 2, np.float64, 3),
    ("tsolve_symbol_f64", "tsolve_with_symbol", "rand", None, None, np.float64, 4),
    ("solve_numeric_c128", "solve_with_numeric", "rand", None, None, np.complex128, 5),
    ("dot_f64", "dot", "rand", None, 3, np.float64, 6),
    ("solve_blocktri_f64", "solve", "blocktri", None, None, np.float64, 7),
    ("solve_blocktri_c128", "solve", "blocktri", None, None, np.complex128, 8),
]


def _block_upper_triangular(dtype, blocks=3, size=5, seed=0):
    rng = np.random.default_rng(seed)
    n = blocks * size
    A = np.zeros((n, n), dtype=dtype)
    for i in range(blocks):
        si = i * size
        A[si : si + size, si : si + size] = rng.standard_normal(
            (size, size)
        ).astype(dtype) + 10 * np.eye(size, dtype=dtype)
        for j in range(i + 1, blocks):
            sj = j * size
            A[si : si + size, sj : sj + size] = 0.3 * rng.standard_normal(
                (size, size)
            ).astype(dtype)
    Ai, Aj = np.nonzero(np.abs(A) > 0)
    return A, Ai, Aj


def build_case(kind, n_lhs, n_rhs, dtype, seed):
    if kind == "rand":
        Ai, Aj, Ax = rand_coo(N_COL, N_NZ, n_lhs=n_lhs, dtype=dtype, seed=seed)
        n_col = N_COL
        b_shape = (N_COL,)
        if n_lhs is not None:
            b_shape = (n_lhs, N_COL) + ((n_rhs,) if n_rhs else ())
        b = rand_rhs(b_shape, dtype, seed + 1000)
        return Ai, Aj, Ax, b, n_col
    if kind == "blocktri":
        A, Ai, Aj = _block_upper_triangular(dtype, seed=seed)
        Ax = A[Ai, Aj]
        Ai, Aj, Ax = klujax.coalesce(
            jnp.asarray(Ai, jnp.int32), jnp.asarray(Aj, jnp.int32), jnp.asarray(Ax)
        )
        b = rand_rhs((A.shape[0],), dtype, seed + 1000)
        return Ai, Aj, Ax, b, A.shape[0]
    msg = f"unknown kind {kind!r}"
    raise ValueError(msg)


def run_op(op, Ai, Aj, Ax, b, n_col):
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


def main() -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, np.ndarray] = {}
    outputs: dict[str, np.ndarray] = {}

    for name, op, kind, n_lhs, n_rhs, dtype, seed in CASES:
        Ai, Aj, Ax, b, n_col = build_case(kind, n_lhs, n_rhs, dtype, seed)
        x = run_op(op, Ai, Aj, Ax, b, n_col)
        inputs[f"{name}_Ai"] = np.asarray(Ai)
        inputs[f"{name}_Aj"] = np.asarray(Aj)
        inputs[f"{name}_Ax"] = np.asarray(Ax)
        inputs[f"{name}_b"] = np.asarray(b)
        inputs[f"{name}_n_col"] = np.asarray(n_col)
        inputs[f"{name}_op"] = np.asarray(op)
        outputs[f"{name}_x"] = np.asarray(x)
        print(f"{name}: op={op} out_shape={np.asarray(x).shape} dtype={np.asarray(x).dtype}")

    inputs["case_names"] = np.asarray([c[0] for c in CASES])
    np.savez(GOLDEN / "inputs.npz", **inputs)
    np.savez(GOLDEN / "outputs.npz", **outputs)
    print(f"\nwrote {GOLDEN / 'inputs.npz'} and {GOLDEN / 'outputs.npz'}")


if __name__ == "__main__":
    main()
