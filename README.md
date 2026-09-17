# KLUJAX

> version: 0.5.2

A sparse linear solver for JAX based on the efficient [KLU algorithm](https://ufdcimages.uflib.ufl.edu/UF/E0/01/17/21/00001/palamadai_e.pdf).

## CPU & float64

This library is a wrapper around the [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) KLU
algorithms. This means the algorithm is only implemented for
C-arrays and hence is **only available for CPU
arrays with double precision**, i.e. float64 or complex128.

Note that `float32`/`complex64` arrays will be cast to `float64`/`complex128`!

## Basic Usage

The `klujax` library provides a basic function `solve(Ai, Aj, Ax, b)`, which solves for `x` in
the sparse linear system `Ax=b`, where `A` is explicitly given in COO-format (`Ai`, `Aj`, `Ax`).

> NOTE: the sparse matrix represented by (`Ai`, `Aj`, `Ax`) needs to be [coalesced](https://pytorch.org/docs/stable/sparse.html#uncoalesced-sparse-coo-tensors)!
> KLUJAX provides a `coalesce` function (which unfortunately is not jax-jittable).

Supported shapes (`?` suffix means optional):

- `Ai`: `(n_nz,)`
- `Aj`: `(n_nz,)`
- `Ax`: `(n_lhs?, n_nz)`
- `b`: `(n_lhs?, n_col, n_rhs?)`
- `A` (represented by (`Ai`, `Aj`, `Ax`)): (`n_lhs?`, `n_col`, `n_col`)

KLUJAX will automatically select a sensible way to act on underdefined dimensions of Ax
and b:

| dim(Ax) | dim(b) | assumed shape(Ax) | assumed shape(b)      |
| ------- | ------ | ----------------- | --------------------- |
| 1D      | 1D     | n_nz              | n_col                 |
| 1D      | 2D     | n_nz              | n_col x n_rhs         |
| 1D      | 3D     | n_nz              | n_lhs x n_col x n_rhs |
| 2D      | 1D     | n_lhs x n_nz      | n_col                 |
| 2D      | 2D     | n_lhs x n_nz      | n_lhs x n_col         |
| 2D      | 3D     | n_lhs x n_nz      | n_lhs x n_col x n_rhs |

Where the `A` is always acting on the `n_col` dimension of `b`. The `n_lhs` dim is a
shared batch dimension between `A` and `b`.

Additional dimensions can be added with `jax.vmap` (alternatively any higher dimensional
problem can be reduced to the one above by properly transposing and reshaping `Ax` and `b`).

> NOTE: JAX now has an experimental sparse library (`jax.experimental.sparse`). Using
> this natively in KLUJAX is not yet supported (but converting from `BCOO` or `COO` to
> `Ai`, `Aj`, `Ax` is trivial).

## Basic Example

Script:

```python
import klujax
import jax.numpy as jnp

b = jnp.array([8, 45, -3, 3, 19])
A_dense = jnp.array(
    [
        [2, 3, 0, 0, 0],
        [3, 0, 4, 0, 6],
        [0, -1, -3, 2, 0],
        [0, 0, 1, 0, 0],
        [0, 4, 2, 0, 1],
    ]
)
Ai, Aj = jnp.where(jnp.abs(A_dense) > 0)
Ax = A_dense[Ai, Aj]

result_ref = jnp.linalg.inv(A_dense) @ b
result = klujax.solve(Ai, Aj, Ax, b)

print(jnp.abs(result - result_ref) < 1e-12)
print(result)
```

Output:

```
[ True True True True True]
[1. 2. 3. 4. 5.]
```

## Advanced Usage

For high-performance applications like transient simulations or iterative solvers, you should avoid using the high-level `klujax.solve` function. The `klujax.solve` is in fact a wrapper around three distinct parts of the KLU algorithm:

1. **Analyze (Symbolic)**: Inspects the sparsity pattern ($A_i, A_j$) to find optimal permutations and block triangular forms. This depends only on the structure of the matrix.
2. **Factorize (Numeric)**: Performs the actual LU decomposition. This depends on the values ($A_x$) and requires a symbolic handle.
3. **Solve (Numeric)**: Executes forward and backward substitution to find $x$. This depends on the right-hand side ($b$) and requires a numeric handle.

Significant performance gains are achieved by hoisting the "Analysis" or "Factorization" steps out of your inner loops.

### 1. High-Performance Transient Pattern (Reusing Symbolic)

In a simulation where the sparsity pattern is constant but the values ($A_x$) and right-hand side ($b$) change, you should perform the expensive `analyze` step exactly once outside your JIT loop.

```python
import jax
import klujax

# 1. Analyze once in Python (CPU)
# Returns a KLUHandleManager that automatically cleans up C++ memory
symbolic = klujax.analyze(Ai, Aj, n_col)

@jax.jit
def simulation_step(Ax_t, b_t, sym):
    # 2. Use the symbolic handle inside JIT
    # The solver will perform numeric factorization and solve
    return klujax.solve_with_symbol(Ai, Aj, Ax_t, b_t, sym)

for t in range(steps):
    x_t = simulation_step(Ax[t], b[t], symbolic)

```

### Fine-Grained Control (Numeric Factorization)

If you need to solve the same system with many different $b$ vectors while the matrix $A$ remains constant, you can further split the numeric factorization. This is often performed in a modified Newton-Raphson loop where the computationally expensive jacobian+factorization is only evaluated once and the **solve** stage is deemed "cheap" in comparison

```python
# Factorize the matrix once
numeric = klujax.factor(Ai, Aj, Ax, symbolic)

@jax.jit
def fast_solve(b_t, num, sym):
    # This call is extremely fast as it skips factorization entirely
    return klujax.solve_with_numeric(num, b_t, sym)

for i in range(100):
    x_i = fast_solve(b_batch[i], numeric, symbolic)
```

### Lifecycle & Pointer Pitfalls

Because `klujax.analyze` and `klujax.factor` generate `KLUHandleManager` objects which wrap low level C++ pointers, there are strict rules for avoiding memory leaks and segmentation faults.

#### The "Ghost Pointer" Problem inside JIT:

JAX's jit works by tracing your code. During tracing, Python objects like the `KLUHandleManager` are converted into symbolic Tracers.

- Outside JIT: The `KLUHandleManager` uses RAII (Resource Acquisition Is Initialization). When the Python variable is deleted or goes out of scope, the C++ memory is freed automatically.

- Inside JIT: If you create a handle (via `analyze` or `factor`) **inside** a JIT-compiled function, the Python manager is "lost" during the conversion to XLA. XLA will allocate the C++ memory at runtime, but it will **never** call the free function.

#### The Fix: Explicit Destruction with Dependencies

If you must create a handle inside JIT, you must manually call `free_symbolic` or `free_numeric` inside that same function. To prevent the compiler from freeing the pointer before the solve is finished, you must pass the solution as a dependency.

```python
@jax.jit
def dynamic_solve(Ai, Aj, Ax, b):
    # 1. Born inside JIT (No automatic cleanup!)
    sym = klujax.analyze(Ai, Aj, 5)

    # 2. Compute solution
    x = klujax.solve_with_symbol(Ai, Aj, Ax, b, sym)

    # 3. CRITICAL: Force XLA to free 'sym' ONLY AFTER 'x' is ready
    klujax.free_symbolic(sym, dependency=x)

    return x
```

#### Summary of Best Practices

1. **Hoist Creations**: Always try to call analyze or factor outside of JIT blocks.

2. **One Manager, One Free**: Do not manually call free_symbolic(manager) and then let the manager go out of scope; it will attempt a double-free (though the library has safeguards to prevent a crash).

3. **Check for Warnings**: If you see a UserWarning: Allocating KLU handle inside JIT, your code is currently leaking memory. Use the dependency pattern shown above to fix it.

## Installation

`klujax` is a Rust `cdylib` (built from the `klujax-ffi` crate) that **statically
links** SuiteSparse's C KLU (via the `klu-sys` crate). It is registered with JAX
through `jax.ffi.pycapsule` — there is no pybind11 and no compiled Python
extension.

```bash
pip install klujax
```

Pre-built wheels are provided for common platforms. If no compatible wheel is
found, pip builds from source (see [Installing from Source](#installing-from-source)).

## Installing from Source

> NOTE: Installing from source should only be necessary when developing the
> library. If you experience a source install as a user, please create an issue.

You need a Rust toolchain (`cargo`, see <https://rustup.rs>) and a C compiler.
The SuiteSparse sources are vendored as a git submodule, so initialize it first:

```sh
git clone https://github.com/gdsfactory/klujax
cd klujax
git submodule update --init --recursive   # vendor/SuiteSparse
pip install .
```

`klu-sys/build.rs` compiles the needed SuiteSparse C sources
(`SuiteSparse_config`, `AMD`, `COLAMD`, `BTF`, `KLU`) with the `cc` crate and
links them **statically** into the cdylib. Point `KLUJAX_SUITESPARSE_DIR` at a
different checkout to override the submodule.

### Linux / macOS

You need `cargo` and a C/C++ compiler (`gcc` or `clang`):

```sh
pip install .
```

### Windows

You need `cargo` and the MSVC build tools (Visual Studio with the "Desktop
development with C++" workload). Then, in an x64 Native Tools Command Prompt:

```cmd
pip install .
```

## License & Credits

© Floris Laporte 2022, LGPL-2.1

This library was partly based on:

- [torch_sparse_solve](https://github.com/flaport/torch_sparse_solve), LGPL-2.1
- [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse), LGPL-2.1
- [kagami-c/PyKLU](https://github.com/kagami-c/PyKLU), LGPL-2.1
- [scipy.sparse](https://github.com/scipy/scipy/tree/master/scipy/sparse), BSD-3

This library statically links an unmodified version of the
[SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) libraries
(vendored as the `vendor/SuiteSparse` git submodule and compiled by the
`klu-sys` crate).
This is in accordance with their
[LGPL licence](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/LICENSE.txt).
