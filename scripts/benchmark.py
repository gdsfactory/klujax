#!/usr/bin/env python
"""Micro-benchmark for klujax solve/factor paths.

Records a reproducible baseline used to check performance parity during the
Rust migration. Writes JSON to ``benchmarks/baseline.json`` by default.

Usage:
    python scripts/benchmark.py [output.json]
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import klujax

N_COL = 200
N_NZ = 2000
N_RHS = 8
REPEATS = 20
SEED = 0


def make_system(n_col: int, n_nz: int, dtype=jnp.float64):
    rng = np.random.default_rng(SEED)
    Ai = rng.integers(0, n_col, size=n_nz)
    Aj = rng.integers(0, n_col, size=n_nz)
    Ax = rng.standard_normal(n_nz)
    # Strong diagonal for a well-conditioned system.
    diag = np.arange(n_col)
    Ai = np.concatenate([Ai, diag])
    Aj = np.concatenate([Aj, diag])
    Ax = np.concatenate([Ax, np.full(n_col, 10.0)])
    Ai, Aj, Ax = klujax.coalesce(
        jnp.asarray(Ai, jnp.int32), jnp.asarray(Aj, jnp.int32), jnp.asarray(Ax, dtype)
    )
    b = jnp.asarray(rng.standard_normal((n_col, N_RHS)), dtype)
    return Ai, Aj, Ax, b


def _block(out) -> None:
    if isinstance(out, tuple):
        for item in out:
            jax.block_until_ready(item)
    else:
        jax.block_until_ready(out)


def timeit(fn, *args, repeats: int = REPEATS) -> float:
    _block(fn(*args))  # warmup / compile
    start = time.perf_counter()
    for _ in range(repeats):
        _block(fn(*args))
    return (time.perf_counter() - start) / repeats * 1e3  # ms


def main() -> int:
    out_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmarks/baseline.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    Ai, Aj, Ax, b = make_system(N_COL, N_NZ)
    symbolic = klujax.analyze(Ai, Aj, N_COL)

    solve_jit = jax.jit(klujax.solve)
    solve_with_symbol_jit = jax.jit(klujax.solve_with_symbol)

    results = {
        "implementation": "suitesparse-c++",
        "klujax_version": klujax.__version__,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "shape": {"n_col": N_COL, "n_nz": int(Ax.shape[0]), "n_rhs": N_RHS},
        "repeats": REPEATS,
        "ms": {
            "analyze": timeit(lambda: klujax.analyze(Ai, Aj, N_COL), repeats=REPEATS),
            "solve_jit": timeit(solve_jit, Ai, Aj, Ax, b),
            "solve_with_symbol_jit": timeit(
                solve_with_symbol_jit, Ai, Aj, Ax, b, symbolic
            ),
            "factor": timeit(lambda: klujax.factor(Ai, Aj, Ax, symbolic)),
        },
    }

    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
