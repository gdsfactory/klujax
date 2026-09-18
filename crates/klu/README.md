# klu

Ergonomic Rust bindings to the [SuiteSparse KLU](https://github.com/DrTimothyAldenDavis/SuiteSparse)
sparse linear solver. Part of the [klujax](https://github.com/gdsfactory/klujax)
project; also used as the backend for the `klujax-ffi` XLA/JAX layer.

KLU is a fast
[direct solver](https://en.wikipedia.org/wiki/Direct_solver) for sparse
`A x = b`, tuned for circuit-simulation-like matrices.

## Three ways to solve

```rust
use klu::{Coo, solve};

let a = Coo::new(2)?
    .push(0, 0, 2.0)?
    .push(0, 1, 1.0)?
    .push(1, 0, 1.0)?
    .push(1, 1, 3.0)?
    .build()?;

// 1. one-shot: analyze + factor + solve, no handles to free.
let x = solve(&a, &[3.0, 5.0])?;

// 2. reuse the symbolic analysis (pattern fixed, values change).
let symbolic = a.analyze()?;
let x = symbolic.solve(a.values(), &[3.0, 5.0])?;

// 3. reuse the numeric factorization (values fixed, many RHS).
let numeric = a.factorize()?;
let x = numeric.solve(&[3.0, 5.0])?;
```

Handles are reference-counted and freed on drop; there is no manual `free_*`.
`klu::Error` implements `std::error::Error`, and `Coo`/`Csc` are generic over
`f64` and `klu::C64` (complex).

## Notes

- The SuiteSparse sources are compiled by the `klu-sys` dependency. Initialize
  the `vendor/SuiteSparse` submodule (or set `KLUJAX_SUITESPARSE_DIR`) before
  building.
- KLU is CPU-only and uses `f64`/`complex128`.
