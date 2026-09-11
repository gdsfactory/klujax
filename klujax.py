"""klujax: a KLU solver for JAX."""

# Metadata ============================================================================

__version__ = "0.5.1"
__author__ = "Floris Laporte"
__all__ = [
    "KLUHandleManager",
    "KLUNumeric",
    "KLUSymbolic",
    "analyze",
    "coalesce",
    "dot",
    "factor",
    "free_numeric",
    "free_symbolic",
    "refactor",
    "refactor_and_solve",
    "solve",
    "solve_with_numeric",
    "solve_with_symbol",
    "tsolve_with_numeric",
    "tsolve_with_symbol",
]

# Imports =============================================================================

import os
import sys
import warnings
from typing import Any

import jax
import jax.core
import jax.extend.core
import jax.numpy as jnp
import klujax_cpp  # ty: ignore[unresolved-import]
import numpy as np
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir
from jaxtyping import Array

KLUSymbolic = klujax_cpp.KLUSymbolic
KLUNumeric = klujax_cpp.KLUNumeric
KLUHandleManager = KLUSymbolic

jax.tree_util.register_pytree_node(
    klujax_cpp.KLUSymbolic,
    lambda s: ([], s),
    lambda s, _: s,
)
jax.tree_util.register_pytree_node(
    klujax_cpp.KLUNumeric,
    lambda s: ([], s),
    lambda s, _: s,
)

# Config ==============================================================================

DEBUG = bool(int(os.environ.get("KLUJAX_DEBUG", "0")))
jax.config.update(name="jax_enable_x64", val=True)
jax.config.update(name="jax_platform_name", val="cpu")
debug = lambda s: None if not DEBUG else print(s, file=sys.stderr)  # noqa: E731,T201
debug("KLUJAX DEBUG MODE.")

# Constants ===========================================================================

COMPLEX_DTYPES = (
    np.complex64,
    np.complex128,
    jnp.complex64,
    jnp.complex128,
)

# Main Functions ======================================================================


@jax.jit
def solve(Ai: Array, Aj: Array, Ax: Array, b: Array) -> Array:
    """Solve for x in the sparse linear system Ax=b.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector

    Returns:
        x: the result (x≈A^-1b)

    """
    debug("solve")
    Ai, Aj, Ax, b, shape = validate_args(Ai, Aj, Ax, b, x_name="b")
    if any(x.dtype in COMPLEX_DTYPES for x in (Ax, b)):
        debug("solve-complex128")
        x = solve_c128.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.complex128),
            b.astype(jnp.complex128),
        )
    else:
        debug("solve-float64")
        x = solve_f64.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.float64),
            b.astype(jnp.float64),
        )

    return x.reshape(*shape)


@jax.jit
def dot(Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    """Multiply a sparse matrix with a vector: Ax=b.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        x:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the vector multiplied by A

    Returns:
        b: the result (b=A@x)

    """
    debug("dot")
    Ai, Aj, Ax, x, shape = validate_args(Ai, Aj, Ax, x, x_name="x")
    if any(x.dtype in COMPLEX_DTYPES for x in (Ax, x)):
        debug("dot-complex128")
        b = dot_c128.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.complex128),
            x.astype(jnp.complex128),
        )
    else:
        debug("dot-float64")
        b = dot_f64.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.float64),
            x.astype(jnp.float64),
        )
    return b.reshape(*shape)


def coalesce(
    Ai: Array,
    Aj: Array,
    Ax: Array,
) -> tuple[Array, Array, Array]:
    """Coalesce a sparse matrix by summing duplicate indices.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [... x n_nz; float64|complex128]: the values of the sparse matrix A

    Returns:
        coalesced Ai, Aj, Ax

    """
    with jax.ensure_compile_time_eval():
        shape = Ax.shape

        order = jnp.lexsort((Aj, Ai))
        Ai = Ai[order]
        Aj = Aj[order]

        # Compute unique indices
        unique_mask = jnp.concatenate(
            [jnp.array([True]), (Ai[1:] != Ai[:-1]) | (Aj[1:] != Aj[:-1])],
        )
        unique_idxs = jnp.where(unique_mask)[0]

        # Assign each entry to a unique group
        groups = jnp.cumsum(unique_mask) - 1

        # Sum Ax values over groups
        Ai = Ai[unique_idxs]
        Aj = Aj[unique_idxs]

    Ax = Ax.reshape(-1, shape[-1])
    Ax = Ax[:, order]
    Ax = jax.vmap(jax.ops.segment_sum, [0, None], 0)(Ax, groups)

    return Ai, Aj, Ax.reshape(*shape[:-1], -1)


# Handle helpers =====================================================================


def _get_symbolic_handle(symbolic: Any) -> Array:  # noqa: ANN401
    if isinstance(symbolic, klujax_cpp.KLUSymbolic):
        return jnp.array(symbolic.raw, dtype=jnp.uint64)
    return symbolic


def _get_numeric_handle(numeric: Any) -> Array:  # noqa: ANN401
    if isinstance(numeric, klujax_cpp.KLUNumeric):
        return jnp.array(numeric.as_list(), dtype=jnp.uint64)
    return numeric


def free_symbolic(symbolic: Any, dependency: Any = None) -> Array:  # noqa: ANN401
    """Free a KLU handle (deprecated — handles are freed automatically)."""
    warnings.warn(
        "free_symbolic is deprecated; handles are freed automatically",
        DeprecationWarning,
        stacklevel=2,
    )
    if hasattr(symbolic, "close"):
        symbolic.close()
    return jnp.array(0, dtype=jnp.int32)


def free_numeric(numeric: Any, dependency: Any = None) -> Array:  # noqa: ANN401
    """Free a KLU handle (deprecated — handles are freed automatically)."""
    warnings.warn(
        "free_numeric is deprecated; handles are freed automatically",
        DeprecationWarning,
        stacklevel=2,
    )
    if hasattr(numeric, "close"):
        numeric.close()
    return jnp.array(0, dtype=jnp.int32)


# Split Solve routines =============================================================


def analyze(Ai: Array, Aj: Array, n_col: int) -> klujax_cpp.KLUSymbolic:
    """Analyze the sparsity pattern of a matrix A.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        n_col: [int]: the number of columns in the sparse matrix A

    Returns:
        symbolic: [KLUSymbolic]: the symbolic analysis object (freed automatically)

    """
    Ai = jnp.asarray(Ai, dtype=jnp.int32)
    Aj = jnp.asarray(Aj, dtype=jnp.int32)
    raw_symbol = analyze_p.bind(Ai, Aj, jnp.int32(n_col))
    if isinstance(raw_symbol, jax.core.Tracer):
        return raw_symbol
    return klujax_cpp.KLUSymbolic(int(raw_symbol))


def validate_numeric_solve(
    Ai: Array, Aj: Array, Ax: Array, b: Array
) -> tuple[Array, Array, Array, Array, tuple[int, ...]]:
    """Reduced set of validate_args for use with solve_with_symbol."""
    order = jnp.lexsort((Aj, Ai))
    Ai, Aj = Ai[order], Aj[order]
    Ax = Ax[..., order] if Ax.ndim == 2 else Ax[order]

    shape = b.shape

    # 2. Dimension expansion to base case: (n_lhs, n_nz) and (n_lhs, n_col, n_rhs)
    if Ax.ndim == 1 and b.ndim == 1:
        Ax, b = Ax[None, :], b[None, :, None]
    elif Ax.ndim == 1 and b.ndim == 2:
        Ax, b = Ax[None, :], b[None, :, :]
    elif Ax.ndim == 1 and b.ndim == 3:
        Ax = Ax[None, :]
    elif Ax.ndim == 2 and b.ndim == 1:
        b = b[None, :, None]
        shape = (Ax.shape[0], shape[0])
    elif Ax.ndim == 2 and b.ndim == 2:
        if Ax.shape[0] != b.shape[0] and Ax.shape[0] != 1 and b.shape[0] != 1:
            msg = f"Batch mismatch: {Ax.shape=} vs {b.shape=}"
            raise ValueError(msg)
        b = b[:, :, None]
        if b.shape[0] == 1 and Ax.shape[0] > 1:
            shape = (Ax.shape[0], *shape[1:])

    # 3. Final broadcasting for C++ FFI
    n_lhs = max(Ax.shape[0], b.shape[0])
    Ax = jnp.broadcast_to(Ax, (n_lhs, Ax.shape[1]))
    b = jnp.broadcast_to(b, (n_lhs, b.shape[1], b.shape[2]))

    if len(shape) == 3 and shape[0] != b.shape[0]:
        shape = (Ax.shape[0], shape[1], shape[2])

    return Ai, Aj, Ax, b, shape


@jax.jit
def _solve_with_symbol_jit(
    Ai: Array, Aj: Array, Ax: Array, b: Array, sym_h: Array
) -> Array:
    # Use the robust validator
    Ai, Aj, Ax, b, out_shape = validate_numeric_solve(Ai, Aj, Ax, b)

    is_complex = any(x.dtype in COMPLEX_DTYPES for x in (Ax, b))
    prim = solve_with_symbol_c128 if is_complex else solve_with_symbol_f64

    # Pass standardized arrays to the C++ extension
    x = prim.bind(
        Ai.astype(jnp.int32),
        Aj.astype(jnp.int32),
        Ax.astype(jnp.complex128 if is_complex else jnp.float64),
        b.astype(jnp.complex128 if is_complex else jnp.float64),
        sym_h.astype(jnp.uint64),
    )

    return x.reshape(*out_shape)


def solve_with_symbol(
    Ai: Array, Aj: Array, Ax: Array, b: Array, symbolic: klujax_cpp.KLUSymbolic | Array
) -> Array:
    """Solve Ax=b using a pre-computed symbolic analysis.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector
        symbolic: [KLUSymbolic|Array]: the symbolic analysis object or handle

    Returns:
        x: the result (x≈A^-1b)

    """
    handle = _get_symbolic_handle(symbolic)
    return _solve_with_symbol_jit(Ai, Aj, Ax, b, handle)


@jax.jit
def _tsolve_with_symbol_jit(
    Ai: Array, Aj: Array, Ax: Array, b: Array, sym_h: Array
) -> Array:
    Ai, Aj, Ax, b, out_shape = validate_numeric_solve(Ai, Aj, Ax, b)

    is_complex = any(x.dtype in COMPLEX_DTYPES for x in (Ax, b))
    prim = tsolve_with_symbol_c128 if is_complex else tsolve_with_symbol_f64

    x = prim.bind(
        Ai.astype(jnp.int32),
        Aj.astype(jnp.int32),
        Ax.astype(jnp.complex128 if is_complex else jnp.float64),
        b.astype(jnp.complex128 if is_complex else jnp.float64),
        sym_h.astype(jnp.uint64),
    )

    return x.reshape(*out_shape)


def tsolve_with_symbol(
    Ai: Array, Aj: Array, Ax: Array, b: Array, symbolic: klujax_cpp.KLUSymbolic | Array
) -> Array:
    """Solve A^T x=b (transpose solve) using a pre-computed symbolic analysis.

    Factors A numerically, then solves the transposed system using klu_tsolve.
    The symbolic handle describes the sparsity pattern of A (not A^T), and is
    reused as-is — KLU's triangular transpose solver handles the direction internally.

    For complex matrices, this solves A^T x = b (plain transpose, not conjugate).
    Use the conjugate transpose if you need A^H x = b.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector
        symbolic: [KLUSymbolic|Array]: the symbolic analysis object or handle

    Returns:
        x: the result (x≈(A^T)^-1 b)

    """
    handle = _get_symbolic_handle(symbolic)
    return _tsolve_with_symbol_jit(Ai, Aj, Ax, b, handle)


@jax.jit
def _factor_jit(Ai: Array, Aj: Array, Ax: Array, sym_h: Array) -> Array:
    dummy_b = jnp.zeros((1,), dtype=Ax.dtype)
    Ai, Aj, Ax, _, _ = validate_args(Ai, Aj, Ax, dummy_b)
    prim = factor_c128 if Ax.dtype in COMPLEX_DTYPES else factor_f64
    return prim.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, sym_h)


def factor(
    Ai: Array, Aj: Array, Ax: Array, symbolic: klujax_cpp.KLUSymbolic | Array
) -> klujax_cpp.KLUNumeric:
    """Compute the numeric factorization of a matrix A given its symbolic analysis.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        symbolic: [KLUSymbolic|Array]: the symbolic analysis object or handle

    Returns:
        numeric: [KLUNumeric]: the numeric factorization object (freed automatically).
            Inside vmap, returns a raw array — RAII cannot wrap traced values.

    """
    sym_h = _get_symbolic_handle(symbolic)
    raw_numeric = _factor_jit(Ai, Aj, Ax, sym_h)
    if isinstance(raw_numeric, jax.core.Tracer):
        return raw_numeric
    return klujax_cpp.KLUNumeric(list(np.asarray(raw_numeric).flat))


@jax.jit
def _refactor_jit(Ai: Array, Aj: Array, Ax: Array, sym_h: Array, num_h: Array) -> Array:
    dummy_b = jnp.zeros((1,), dtype=Ax.dtype)
    Ai, Aj, Ax, _, _ = validate_args(Ai, Aj, Ax, dummy_b)
    prim = refactor_c128 if Ax.dtype in COMPLEX_DTYPES else refactor_f64
    return prim.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, sym_h, num_h)


def refactor(
    Ai: Array,
    Aj: Array,
    Ax: Array,
    numeric: klujax_cpp.KLUNumeric | Array,
    symbolic: klujax_cpp.KLUSymbolic | Array,
) -> klujax_cpp.KLUNumeric | Array:
    """Re-factorize matrix A numerically, reusing the symbolic analysis.

    Use when the sparsity pattern is unchanged but values have changed.
    Modifies the numeric factorization in-place. Faster than calling factor().

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        numeric: [KLUNumeric|Array]: existing numeric factorization
            (modified in-place)
        symbolic: [KLUSymbolic|Array]: the symbolic analysis object or handle

    Returns:
        numeric: the same handle, updated in-place

    """
    num_h = _get_numeric_handle(numeric)
    sym_h = _get_symbolic_handle(symbolic)
    _refactor_jit(Ai, Aj, Ax, sym_h, num_h)
    return numeric


@jax.jit
def _solve_with_numeric_jit(num_h: Array, b: Array, sym_h: Array) -> Array:
    prim = (
        solve_with_numeric_c128 if b.dtype in COMPLEX_DTYPES else solve_with_numeric_f64
    )
    return prim.bind(sym_h.astype(jnp.uint64), num_h.astype(jnp.uint64), b)


def solve_with_numeric(
    numeric: klujax_cpp.KLUNumeric | Array,
    b: Array,
    symbolic: klujax_cpp.KLUSymbolic | Array,
) -> Array:
    """Solve Ax=b using a pre-computed numeric factorization.

    Args:
        numeric: [KLUNumeric|Array]: the numeric factorization object or handle
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector
        symbolic: [KLUSymbolic|Array]: the symbolic analysis object or handle

    Returns:
        x: the result (x≈A^-1b)

    """
    num_h = _get_numeric_handle(numeric)
    sym_h = _get_symbolic_handle(symbolic)
    return _solve_with_numeric_jit(num_h, b, sym_h)


@jax.jit
def _tsolve_with_numeric_jit(num_h: Array, b: Array, sym_h: Array) -> Array:
    prim = (
        tsolve_with_numeric_c128
        if b.dtype in COMPLEX_DTYPES
        else tsolve_with_numeric_f64
    )
    return prim.bind(sym_h.astype(jnp.uint64), num_h.astype(jnp.uint64), b)


def tsolve_with_numeric(
    numeric: klujax_cpp.KLUNumeric | Array,
    b: Array,
    symbolic: klujax_cpp.KLUSymbolic | Array,
) -> Array:
    """Solve A^T x=b (transpose solve) using a pre-computed numeric factorization.

    Uses klu_tsolve internally. The numeric factorization must have been computed
    for A (not A^T); KLU handles the transposition during the triangular solve.

    For complex matrices, this solves A^T x = b (plain transpose, not conjugate).

    Args:
        numeric: [KLUNumeric|Array]: the numeric factorization object or handle
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector
        symbolic: [KLUSymbolic|Array]: the symbolic analysis object or handle

    Returns:
        x: the result (x≈(A^T)^-1 b)

    """
    num_h = _get_numeric_handle(numeric)
    sym_h = _get_symbolic_handle(symbolic)
    return _tsolve_with_numeric_jit(num_h, b, sym_h)


@jax.jit
def _refactor_and_solve_jit(
    Ai: Array, Aj: Array, Ax: Array, b: Array, sym_h: Array, num_h: Array
) -> tuple[Array, Array]:
    # Use validate_numeric_solve to standardize shapes and get the expected out_shape
    Ai, Aj, Ax, b, out_shape = validate_numeric_solve(Ai, Aj, Ax, b)

    is_complex = any(x.dtype in COMPLEX_DTYPES for x in (Ax, b))
    prim = refactor_and_solve_c128 if is_complex else refactor_and_solve_f64

    x, out_num = prim.bind(
        Ai.astype(jnp.int32),
        Aj.astype(jnp.int32),
        Ax.astype(jnp.complex128 if is_complex else jnp.float64),
        b.astype(jnp.complex128 if is_complex else jnp.float64),
        sym_h.astype(jnp.uint64),
        num_h.astype(jnp.uint64),
    )

    # Reshape x back to the original dimensions of b
    return x.reshape(*out_shape), out_num


def refactor_and_solve(
    Ai: Array,
    Aj: Array,
    Ax: Array,
    b: Array,
    numeric: klujax_cpp.KLUNumeric | Array,
    symbolic: klujax_cpp.KLUSymbolic | Array,
) -> tuple[Array, klujax_cpp.KLUNumeric | Array]:
    """Fused in-place refactorization followed by triangular solve.

    Equivalent to calling refactor() then solve_with_numeric(), but executes as a
    single C++ kernel call. This avoids allocating the COO→CSC work buffer twice
    and saves a JAX dispatch round-trip, which matters in tight iteration loops.

    The numeric factorization is modified in-place (same behaviour as refactor()).

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the right-hand side
        numeric: [KLUNumeric|Array]: existing numeric factorization
            (modified in-place)
        symbolic: [KLUSymbolic|Array]: the symbolic analysis object or handle

    Returns:
        (x, numeric): solution array and the same numeric handle

    """
    num_h = _get_numeric_handle(numeric)
    sym_h = _get_symbolic_handle(symbolic)
    x, _ = _refactor_and_solve_jit(Ai, Aj, Ax, b, sym_h, num_h)
    return x, numeric


# Primitives ==========================================================================

dot_f64 = jax.extend.core.Primitive("dot_f64")
dot_c128 = jax.extend.core.Primitive("dot_c128")
solve_f64 = jax.extend.core.Primitive("solve_f64")
solve_c128 = jax.extend.core.Primitive("solve_c128")
analyze_p = jax.extend.core.Primitive("analyze")
solve_with_symbol_f64 = jax.extend.core.Primitive("solve_with_symbol_f64")
solve_with_symbol_c128 = jax.extend.core.Primitive("solve_with_symbol_c128")
tsolve_with_symbol_f64 = jax.extend.core.Primitive("tsolve_with_symbol_f64")
tsolve_with_symbol_c128 = jax.extend.core.Primitive("tsolve_with_symbol_c128")
factor_f64 = jax.extend.core.Primitive("factor_f64")
factor_c128 = jax.extend.core.Primitive("factor_c128")
solve_with_numeric_f64 = jax.extend.core.Primitive("solve_with_numeric_f64")
solve_with_numeric_c128 = jax.extend.core.Primitive("solve_with_numeric_c128")
tsolve_with_numeric_f64 = jax.extend.core.Primitive("tsolve_with_numeric_f64")
tsolve_with_numeric_c128 = jax.extend.core.Primitive("tsolve_with_numeric_c128")
refactor_f64 = jax.extend.core.Primitive("refactor_f64")
refactor_c128 = jax.extend.core.Primitive("refactor_c128")
refactor_and_solve_f64 = jax.extend.core.Primitive("refactor_and_solve_f64")
refactor_and_solve_c128 = jax.extend.core.Primitive("refactor_and_solve_c128")
refactor_and_solve_f64.multiple_results = True
refactor_and_solve_c128.multiple_results = True

# Implementations ========================================================


@dot_f64.def_impl
def dot_f64_impl(Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    return general_impl("dot_f64", Ai, Aj, Ax, x)


@dot_c128.def_impl
def dot_c128_impl(Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    return general_impl("dot_c128", Ai, Aj, Ax, x)


@solve_f64.def_impl
def solve_f64_impl(Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    return general_impl("solve_f64", Ai, Aj, Ax, x)


@solve_c128.def_impl
def solve_c128_impl(Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    return general_impl("solve_c128", Ai, Aj, Ax, x)


@solve_with_symbol_f64.def_impl
def solve_with_symbol_f64_impl(
    Ai: Array, Aj: Array, Ax: Array, b: Array, symbolic: Array
) -> Array:
    return general_impl("solve_with_symbol_f64", Ai, Aj, Ax, b, symbolic)


@solve_with_symbol_c128.def_impl
def solve_with_symbol_c128_impl(
    Ai: Array, Aj: Array, Ax: Array, b: Array, symbolic: Array
) -> Array:
    return general_impl("solve_with_symbol_c128", Ai, Aj, Ax, b, symbolic)


@tsolve_with_symbol_f64.def_impl
def tsolve_with_symbol_f64_impl(
    Ai: Array, Aj: Array, Ax: Array, b: Array, symbolic: Array
) -> Array:
    return general_impl("tsolve_with_symbol_f64", Ai, Aj, Ax, b, symbolic)


@tsolve_with_symbol_c128.def_impl
def tsolve_with_symbol_c128_impl(
    Ai: Array, Aj: Array, Ax: Array, b: Array, symbolic: Array
) -> Array:
    return general_impl("tsolve_with_symbol_c128", Ai, Aj, Ax, b, symbolic)


@analyze_p.def_impl
def analyze_impl(Ai: Array, Aj: Array, n_col: Array) -> Array:
    return jax.ffi.ffi_call("analyze", ShapedArray((), jnp.uint64))(Ai, Aj, n_col)


@factor_f64.def_impl
def factor_f64_impl(Ai, Aj, Ax, symbolic):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call("factor_f64", jax.ShapeDtypeStruct((n_lhs,), jnp.uint64))
    return call(Ai, Aj, Ax, symbolic)


@factor_c128.def_impl
def factor_c128_impl(Ai, Aj, Ax, symbolic):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call("factor_c128", jax.ShapeDtypeStruct((n_lhs,), jnp.uint64))
    return call(Ai, Aj, Ax, symbolic)


@refactor_f64.def_impl
def refactor_f64_impl(Ai, Aj, Ax, symbolic, numeric):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call("refactor_f64", jax.ShapeDtypeStruct((n_lhs,), jnp.uint64))
    return call(Ai, Aj, Ax, symbolic, numeric)


@refactor_c128.def_impl
def refactor_c128_impl(Ai, Aj, Ax, symbolic, numeric):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call("refactor_c128", jax.ShapeDtypeStruct((n_lhs,), jnp.uint64))
    return call(Ai, Aj, Ax, symbolic, numeric)


@solve_with_numeric_f64.def_impl
def solve_with_numeric_f64_impl(symbolic, numeric, b):
    call = jax.ffi.ffi_call(
        "solve_with_numeric_f64", jax.ShapeDtypeStruct(b.shape, b.dtype)
    )
    return call(symbolic, numeric, b)


@solve_with_numeric_c128.def_impl
def solve_with_numeric_c128_impl(symbolic, numeric, b):
    call = jax.ffi.ffi_call(
        "solve_with_numeric_c128", jax.ShapeDtypeStruct(b.shape, b.dtype)
    )
    return call(symbolic, numeric, b)


@tsolve_with_numeric_f64.def_impl
def tsolve_with_numeric_f64_impl(symbolic, numeric, b):
    call = jax.ffi.ffi_call(
        "tsolve_with_numeric_f64", jax.ShapeDtypeStruct(b.shape, b.dtype)
    )
    return call(symbolic, numeric, b)


@tsolve_with_numeric_c128.def_impl
def tsolve_with_numeric_c128_impl(symbolic, numeric, b):
    call = jax.ffi.ffi_call(
        "tsolve_with_numeric_c128", jax.ShapeDtypeStruct(b.shape, b.dtype)
    )
    return call(symbolic, numeric, b)


@refactor_and_solve_f64.def_impl
def refactor_and_solve_f64_impl(Ai, Aj, Ax, b, symbolic, numeric):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call(
        "refactor_and_solve_f64",
        (
            jax.ShapeDtypeStruct(b.shape, b.dtype),
            jax.ShapeDtypeStruct((n_lhs,), jnp.uint64),
        ),
    )
    return call(Ai, Aj, Ax, b, symbolic, numeric)


@refactor_and_solve_c128.def_impl
def refactor_and_solve_c128_impl(Ai, Aj, Ax, b, symbolic, numeric):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call(
        "refactor_and_solve_c128",
        (
            jax.ShapeDtypeStruct(b.shape, b.dtype),
            jax.ShapeDtypeStruct((n_lhs,), jnp.uint64),
        ),
    )
    return call(Ai, Aj, Ax, b, symbolic, numeric)


def general_impl(
    name: str, Ai: Array, Aj: Array, Ax: Array, x: Array, *args: Array
) -> Array:
    call = jax.ffi.ffi_call(
        name,
        jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    if not callable(call):
        msg = "jax.ffi.ffi_call did not return a callable."
        raise RuntimeError(msg)  # noqa: TRY004
    return call(Ai, Aj, Ax, x, *args)


# Lowerings ===========================================================================

jax.ffi.register_ffi_target(
    "dot_f64",
    klujax_cpp.dot_f64(),
    platform="cpu",
)

dot_f64_low = mlir.lower_fun(dot_f64_impl, multiple_results=False)
mlir.register_lowering(dot_f64, dot_f64_low)

jax.ffi.register_ffi_target(
    "dot_c128",
    klujax_cpp.dot_c128(),
    platform="cpu",
)

dot_c128_low = mlir.lower_fun(dot_c128_impl, multiple_results=False)
mlir.register_lowering(dot_c128, dot_c128_low)

jax.ffi.register_ffi_target(
    "solve_f64",
    klujax_cpp.solve_f64(),
    platform="cpu",
)

solve_f64_low = mlir.lower_fun(solve_f64_impl, multiple_results=False)
mlir.register_lowering(solve_f64, solve_f64_low)

jax.ffi.register_ffi_target(
    "solve_c128",
    klujax_cpp.solve_c128(),
    platform="cpu",
)

solve_c128_low = mlir.lower_fun(solve_c128_impl, multiple_results=False)
mlir.register_lowering(solve_c128, solve_c128_low)

jax.ffi.register_ffi_target(
    "analyze",
    klujax_cpp.analyze(),
    platform="cpu",
)

analyze_low = mlir.lower_fun(analyze_impl, multiple_results=False)
mlir.register_lowering(analyze_p, analyze_low)

jax.ffi.register_ffi_target(
    "solve_with_symbol_f64",
    klujax_cpp.solve_with_symbol_f64(),
    platform="cpu",
)

solve_with_symbol_f64_low = mlir.lower_fun(
    solve_with_symbol_f64_impl, multiple_results=False
)
mlir.register_lowering(solve_with_symbol_f64, solve_with_symbol_f64_low)

jax.ffi.register_ffi_target(
    "solve_with_symbol_c128",
    klujax_cpp.solve_with_symbol_c128(),
    platform="cpu",
)

solve_with_symbol_c128_low = mlir.lower_fun(
    solve_with_symbol_c128_impl, multiple_results=False
)
mlir.register_lowering(solve_with_symbol_c128, solve_with_symbol_c128_low)

jax.ffi.register_ffi_target(
    "tsolve_with_symbol_f64",
    klujax_cpp.tsolve_with_symbol_f64(),
    platform="cpu",
)

tsolve_with_symbol_f64_low = mlir.lower_fun(
    tsolve_with_symbol_f64_impl, multiple_results=False
)
mlir.register_lowering(tsolve_with_symbol_f64, tsolve_with_symbol_f64_low)

jax.ffi.register_ffi_target(
    "tsolve_with_symbol_c128",
    klujax_cpp.tsolve_with_symbol_c128(),
    platform="cpu",
)

tsolve_with_symbol_c128_low = mlir.lower_fun(
    tsolve_with_symbol_c128_impl, multiple_results=False
)
mlir.register_lowering(tsolve_with_symbol_c128, tsolve_with_symbol_c128_low)

jax.ffi.register_ffi_target("factor_f64", klujax_cpp.factor_f64(), platform="cpu")
factor_f64_low = mlir.lower_fun(factor_f64_impl, multiple_results=False)
mlir.register_lowering(factor_f64, factor_f64_low)

jax.ffi.register_ffi_target("factor_c128", klujax_cpp.factor_c128(), platform="cpu")
factor_c128_low = mlir.lower_fun(factor_c128_impl, multiple_results=False)
mlir.register_lowering(factor_c128, factor_c128_low)

jax.ffi.register_ffi_target("refactor_f64", klujax_cpp.refactor_f64(), platform="cpu")
refactor_f64_low = mlir.lower_fun(refactor_f64_impl, multiple_results=False)
mlir.register_lowering(refactor_f64, refactor_f64_low)

jax.ffi.register_ffi_target("refactor_c128", klujax_cpp.refactor_c128(), platform="cpu")
refactor_c128_low = mlir.lower_fun(refactor_c128_impl, multiple_results=False)
mlir.register_lowering(refactor_c128, refactor_c128_low)

jax.ffi.register_ffi_target(
    "solve_with_numeric_f64", klujax_cpp.solve_with_numeric_f64(), platform="cpu"
)
solve_with_numeric_f64_low = mlir.lower_fun(
    solve_with_numeric_f64_impl, multiple_results=False
)
mlir.register_lowering(solve_with_numeric_f64, solve_with_numeric_f64_low)

jax.ffi.register_ffi_target(
    "solve_with_numeric_c128", klujax_cpp.solve_with_numeric_c128(), platform="cpu"
)
solve_with_numeric_c128_low = mlir.lower_fun(
    solve_with_numeric_c128_impl, multiple_results=False
)
mlir.register_lowering(solve_with_numeric_c128, solve_with_numeric_c128_low)

jax.ffi.register_ffi_target(
    "tsolve_with_numeric_f64", klujax_cpp.tsolve_with_numeric_f64(), platform="cpu"
)
tsolve_with_numeric_f64_low = mlir.lower_fun(
    tsolve_with_numeric_f64_impl, multiple_results=False
)
mlir.register_lowering(tsolve_with_numeric_f64, tsolve_with_numeric_f64_low)

jax.ffi.register_ffi_target(
    "tsolve_with_numeric_c128", klujax_cpp.tsolve_with_numeric_c128(), platform="cpu"
)
tsolve_with_numeric_c128_low = mlir.lower_fun(
    tsolve_with_numeric_c128_impl, multiple_results=False
)
mlir.register_lowering(tsolve_with_numeric_c128, tsolve_with_numeric_c128_low)

jax.ffi.register_ffi_target(
    "refactor_and_solve_f64", klujax_cpp.refactor_and_solve_f64(), platform="cpu"
)
refactor_and_solve_f64_low = mlir.lower_fun(
    refactor_and_solve_f64_impl, multiple_results=True
)
mlir.register_lowering(refactor_and_solve_f64, refactor_and_solve_f64_low)

jax.ffi.register_ffi_target(
    "refactor_and_solve_c128", klujax_cpp.refactor_and_solve_c128(), platform="cpu"
)
refactor_and_solve_c128_low = mlir.lower_fun(
    refactor_and_solve_c128_impl, multiple_results=True
)
mlir.register_lowering(refactor_and_solve_c128, refactor_and_solve_c128_low)

# Abstract Evals ======================================================================


@dot_f64.def_abstract_eval
@dot_c128.def_abstract_eval
@solve_f64.def_abstract_eval
@solve_c128.def_abstract_eval
@solve_with_symbol_f64.def_abstract_eval
@solve_with_symbol_c128.def_abstract_eval
@tsolve_with_symbol_f64.def_abstract_eval
@tsolve_with_symbol_c128.def_abstract_eval
def general_abstract_eval(
    Ai: Array, Aj: Array, Ax: Array, b: Array, *args: Array
) -> ShapedArray:
    return ShapedArray(b.shape, b.dtype)


@analyze_p.def_abstract_eval
def analyze_abstract_eval(Ai: Array, Aj: Array, n_col: Array) -> ShapedArray:
    return ShapedArray((), jnp.uint64)


@factor_f64.def_abstract_eval
@factor_c128.def_abstract_eval
def factor_abstract_eval(Ai, Aj, Ax, symbolic):
    return ShapedArray((Ax.shape[0],), jnp.uint64)


@refactor_f64.def_abstract_eval
@refactor_c128.def_abstract_eval
def refactor_abstract_eval(Ai, Aj, Ax, symbolic, numeric):
    return ShapedArray((Ax.shape[0],), jnp.uint64)


@solve_with_numeric_f64.def_abstract_eval
@solve_with_numeric_c128.def_abstract_eval
@tsolve_with_numeric_f64.def_abstract_eval
@tsolve_with_numeric_c128.def_abstract_eval
@tsolve_with_numeric_f64.def_abstract_eval
@tsolve_with_numeric_c128.def_abstract_eval
def solve_with_numeric_abstract_eval(symbolic, numeric, b):
    # Output has same shape as input b
    return ShapedArray(b.shape, b.dtype)


@refactor_and_solve_f64.def_abstract_eval
@refactor_and_solve_c128.def_abstract_eval
def refactor_and_solve_abstract_eval(Ai, Aj, Ax, b, symbolic, numeric):
    # Returns (x with same shape as b, out_numeric with same shape as numeric)
    return ShapedArray(b.shape, b.dtype), ShapedArray(numeric.shape, jnp.uint64)


# Forward Differentiation =============================================================


def dot_f64_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    return dot_value_and_jvp(dot_f64, arg_values, arg_tangents)


ad.primitive_jvps[dot_f64] = dot_f64_value_and_jvp


def dot_c128_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    return dot_value_and_jvp(dot_c128, arg_values, arg_tangents)


ad.primitive_jvps[dot_c128] = dot_c128_value_and_jvp


def solve_f64_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    return solve_value_and_jvp(solve_f64, dot_f64, arg_values, arg_tangents)


ad.primitive_jvps[solve_f64] = solve_f64_value_and_jvp


def solve_c128_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    return solve_value_and_jvp(solve_c128, dot_c128, arg_values, arg_tangents)


ad.primitive_jvps[solve_c128] = solve_c128_value_and_jvp


def solve_value_and_jvp(
    prim_solve: jax.extend.core.Primitive,
    prim_dot: jax.extend.core.Primitive,
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    Ai, Aj, Ax, b = arg_values
    dAi, dAj, dAx, db = arg_tangents
    if not isinstance(dAi, ad.Zero) or not isinstance(dAj, ad.Zero):
        msg = "Sparse indices Ai and Aj should not require gradients."
        raise ValueError(msg)  # noqa: TRY004
    dAx = dAx if not isinstance(dAx, ad.Zero) else jnp.zeros_like(Ax)
    db = db if not isinstance(db, ad.Zero) else jnp.zeros_like(b)
    x = prim_solve.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, b)
    dA_x = prim_dot.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), dAx, x)
    invA_dA_x = prim_solve.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, dA_x)
    invA_db = prim_solve.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, db)
    return x, -invA_dA_x + invA_db


def dot_value_and_jvp(
    prim: jax.extend.core.Primitive,
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    Ai, Aj, Ax, b = arg_values
    dAi, dAj, dAx, db = arg_tangents
    if not isinstance(dAi, ad.Zero) or not isinstance(dAj, ad.Zero):
        msg = "Sparse indices Ai and Aj should not require gradients."
        raise ValueError(msg)  # noqa: TRY004
    dAx = dAx if not isinstance(dAx, ad.Zero) else jnp.zeros_like(Ax)
    db = db if not isinstance(db, ad.Zero) else jnp.zeros_like(b)
    x = prim.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, b)
    dA_b = prim.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), dAx, b)
    A_db = prim.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, db)
    return x, dA_b + A_db


# Batching (vmap) =====================================================================


def dot_f64_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap(dot_f64, vector_arg_values, batch_axes)


batching.primitive_batchers[dot_f64] = dot_f64_vmap


def dot_c128_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap(dot_c128, vector_arg_values, batch_axes)


batching.primitive_batchers[dot_c128] = dot_c128_vmap


def solve_f64_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap(solve_f64, vector_arg_values, batch_axes)


batching.primitive_batchers[solve_f64] = solve_f64_vmap


def solve_c128_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap(solve_c128, vector_arg_values, batch_axes)


batching.primitive_batchers[solve_c128] = solve_c128_vmap


def solve_with_symbol_f64_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_with_symbol(
        solve_with_symbol_f64, vector_arg_values, batch_axes
    )


batching.primitive_batchers[solve_with_symbol_f64] = solve_with_symbol_f64_vmap


def solve_with_symbol_c128_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_with_symbol(
        solve_with_symbol_c128, vector_arg_values, batch_axes
    )


batching.primitive_batchers[solve_with_symbol_c128] = solve_with_symbol_c128_vmap


def tsolve_with_symbol_f64_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_with_symbol(
        tsolve_with_symbol_f64, vector_arg_values, batch_axes
    )


batching.primitive_batchers[tsolve_with_symbol_f64] = tsolve_with_symbol_f64_vmap


def tsolve_with_symbol_c128_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_with_symbol(
        tsolve_with_symbol_c128, vector_arg_values, batch_axes
    )


batching.primitive_batchers[tsolve_with_symbol_c128] = tsolve_with_symbol_c128_vmap


def solve_with_numeric_f64_vmap(
    vector_arg_values: tuple[Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_with_numeric(
        solve_with_numeric_f64, vector_arg_values, batch_axes
    )


batching.primitive_batchers[solve_with_numeric_f64] = solve_with_numeric_f64_vmap


def solve_with_numeric_c128_vmap(
    vector_arg_values: tuple[Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_with_numeric(
        solve_with_numeric_c128, vector_arg_values, batch_axes
    )


batching.primitive_batchers[solve_with_numeric_c128] = solve_with_numeric_c128_vmap


def tsolve_with_numeric_f64_vmap(
    vector_arg_values: tuple[Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_with_numeric(
        tsolve_with_numeric_f64, vector_arg_values, batch_axes
    )


batching.primitive_batchers[tsolve_with_numeric_f64] = tsolve_with_numeric_f64_vmap


def tsolve_with_numeric_c128_vmap(
    vector_arg_values: tuple[Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_with_numeric(
        tsolve_with_numeric_c128, vector_arg_values, batch_axes
    )


batching.primitive_batchers[tsolve_with_numeric_c128] = tsolve_with_numeric_c128_vmap


def factor_f64_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_factor(factor_f64, vector_arg_values, batch_axes)


batching.primitive_batchers[factor_f64] = factor_f64_vmap


def factor_c128_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_factor(factor_c128, vector_arg_values, batch_axes)


batching.primitive_batchers[factor_c128] = factor_c128_vmap


def refactor_f64_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_refactor(refactor_f64, vector_arg_values, batch_axes)


batching.primitive_batchers[refactor_f64] = refactor_f64_vmap


def refactor_c128_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_refactor(refactor_c128, vector_arg_values, batch_axes)


batching.primitive_batchers[refactor_c128] = refactor_c128_vmap


def general_vmap(
    prim: jax.extend.core.Primitive,
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    Ai, Aj, Ax, x = vector_arg_values
    aAi, aAj, aAx, ax = batch_axes

    if aAi is not None:
        msg = "Ai cannot be vectorized."
        raise ValueError(msg)

    if aAj is not None:
        msg = "Aj cannot be vectorized."
        raise ValueError(msg)

    if aAx is not None and ax is not None:
        if Ax.ndim != 3 or x.ndim != 4:
            msg = (
                "Ax and x should be 3D and 4D respectively when vectorizing "
                f"over them simultaneously. Got: {Ax.shape=}; {x.shape=}."
            )
            raise ValueError(msg)
        # vectorize over n_lhs
        Ax = jnp.moveaxis(Ax, aAx, 0)
        x = jnp.moveaxis(x, ax, 0)
        shape = x.shape
        Ax = Ax.reshape(Ax.shape[0] * Ax.shape[1], Ax.shape[2])
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        return prim.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, x).reshape(
            *shape
        ), 0
    if aAx is not None:
        if Ax.ndim != 3 or x.ndim != 3:
            msg = (
                "Ax and x should both be 3D when vectorizing "
                f"over Ax. Got: {Ax.shape=}; {x.shape=}."
            )
            raise ValueError(msg)
        # vectorize over n_lhs
        ax = 0
        Ax = jnp.moveaxis(Ax, aAx, 0)
        x = jnp.broadcast_to(x[None], (Ax.shape[0], x.shape[0], x.shape[1], x.shape[2]))
        shape = x.shape
        Ax = Ax.reshape(Ax.shape[0] * Ax.shape[1], Ax.shape[2])
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        return prim.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, x).reshape(
            *shape
        ), 0
    if ax is not None:
        if Ax.ndim != 2 or x.ndim != 4:
            msg = (
                "Ax and x should both be 2D and 4D respectively when vectorizing "
                f"over x. Got: {Ax.shape=}; {x.shape=}."
            )
            raise ValueError(msg)
        # vectorize over n_rhs
        x = jnp.moveaxis(x, ax, 3)
        shape = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        return prim.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, x).reshape(
            *shape
        ), 3
    msg = "vmap failed. Please select an axis to vectorize over."
    raise ValueError(msg)


def general_vmap_with_symbol(
    prim: jax.extend.core.Primitive,
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    Ai, Aj, Ax, x, symbolic = vector_arg_values
    aAi, aAj, aAx, ax, asymbolic = batch_axes

    if aAi is not None:
        msg = "Ai cannot be vectorized."
        raise ValueError(msg)

    if aAj is not None:
        msg = "Aj cannot be vectorized."
        raise ValueError(msg)

    if asymbolic is not None:
        msg = "symbolic handle cannot be vectorized."
        raise ValueError(msg)

    if aAx is not None and ax is not None:
        if Ax.ndim != 3 or x.ndim != 4:
            msg = (
                "Ax and x should be 3D and 4D respectively when vectorizing "
                f"over them simultaneously. Got: {Ax.shape=}; {x.shape=}."
            )
            raise ValueError(msg)
        # vectorize over n_lhs
        Ax = jnp.moveaxis(Ax, aAx, 0)
        x = jnp.moveaxis(x, ax, 0)
        shape = x.shape
        Ax = Ax.reshape(Ax.shape[0] * Ax.shape[1], Ax.shape[2])
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        return prim.bind(
            Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, x, symbolic
        ).reshape(*shape), 0
    if aAx is not None:
        if Ax.ndim != 3 or x.ndim != 3:
            msg = (
                "Ax and x should both be 3D when vectorizing "
                f"over Ax. Got: {Ax.shape=}; {x.shape=}."
            )
            raise ValueError(msg)
        # vectorize over n_lhs
        ax = 0
        Ax = jnp.moveaxis(Ax, aAx, 0)
        x = jnp.broadcast_to(x[None], (Ax.shape[0], x.shape[0], x.shape[1], x.shape[2]))
        shape = x.shape
        Ax = Ax.reshape(Ax.shape[0] * Ax.shape[1], Ax.shape[2])
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        return prim.bind(
            Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, x, symbolic
        ).reshape(*shape), 0
    if ax is not None:
        if Ax.ndim != 2 or x.ndim != 4:
            msg = (
                "Ax and x should both be 2D and 4D respectively when vectorizing "
                f"over x. Got: {Ax.shape=}; {x.shape=}."
            )
            raise ValueError(msg)
        # vectorize over n_rhs
        x = jnp.moveaxis(x, ax, 3)
        shape = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        return prim.bind(
            Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, x, symbolic
        ).reshape(*shape), 3
    msg = "vmap failed. Please select an axis to vectorize over."
    raise ValueError(msg)


def general_vmap_with_numeric(
    prim: jax.extend.core.Primitive,
    vector_arg_values: tuple[Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None],
) -> tuple[Array, int]:
    symbolic, numeric, b = vector_arg_values
    asymbolic, anumeric, ab = batch_axes

    if asymbolic is not None:
        msg = "symbolic handle cannot be vectorized."
        raise ValueError(msg)

    # numeric and b should batch together
    if anumeric is not None and ab is not None:
        if numeric.ndim != 2 or b.ndim < 2:
            msg = (
                "numeric should be 2D and b should be at least 2D when vectorizing "
                f"over them simultaneously. Got: {numeric.shape=}; {b.shape=}."
            )
            raise ValueError(msg)
        numeric = jnp.moveaxis(numeric, anumeric, 0)
        b = jnp.moveaxis(b, ab, 0)
        shape = b.shape
        batch = shape[0]
        numeric = numeric.reshape(batch * numeric.shape[1])
        if b.ndim >= 3:
            b = b.reshape(batch * b.shape[1], *b.shape[2:])
        # b.ndim == 2: shape is (batch, n) — pass directly, matches multi-LHS convention
        result = prim.bind(symbolic, numeric, b)
        return result.reshape(*shape), 0

    if anumeric is not None:
        if numeric.ndim != 2:
            msg = (
                f"numeric should be 2D when vectorizing over it. Got: {numeric.shape=}."
            )
            raise ValueError(msg)
        numeric = jnp.moveaxis(numeric, anumeric, 0)
        batch = numeric.shape[0]
        b = jnp.broadcast_to(b[None], (batch, *b.shape))
        shape = b.shape
        numeric = numeric.reshape(batch * numeric.shape[1])
        if b.ndim >= 3:
            b = b.reshape(batch * b.shape[1], *b.shape[2:])
        # b.ndim == 2: shape is (batch, n) — pass directly
        return prim.bind(symbolic, numeric, b).reshape(*shape), 0

    if ab is not None:
        b = jnp.moveaxis(b, ab, -1)
        shape = b.shape
        if b.ndim >= 3:
            b = b.reshape(*b.shape[:-2], b.shape[-2] * b.shape[-1])
        # b.ndim == 2: shape is (n, batch) — pass directly as multi-RHS
        return prim.bind(symbolic, numeric, b).reshape(*shape), len(shape) - 1

    msg = "vmap failed. Please select an axis to vectorize over."
    raise ValueError(msg)


def general_vmap_factor(
    prim: jax.extend.core.Primitive,
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    Ai, Aj, Ax, symbolic = vector_arg_values
    aAi, aAj, aAx, asymbolic = batch_axes

    if aAi is not None:
        msg = "Ai cannot be vectorized."
        raise ValueError(msg)

    if aAj is not None:
        msg = "Aj cannot be vectorized."
        raise ValueError(msg)

    if asymbolic is not None:
        msg = "symbolic handle cannot be vectorized."
        raise ValueError(msg)

    if aAx is not None:
        if Ax.ndim != 3:
            msg = f"Ax should be 3D when vectorizing over it. Got: {Ax.shape=}."
            raise ValueError(msg)
        Ax = jnp.moveaxis(Ax, aAx, 0)
        batch, n_lhs, n_vals = Ax.shape
        Ax = Ax.reshape(batch * n_lhs, n_vals)
        return prim.bind(
            Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, symbolic
        ).reshape(batch, n_lhs), 0

    msg = "vmap failed. Please select an axis to vectorize over."
    raise ValueError(msg)


def general_vmap_refactor(
    prim: jax.extend.core.Primitive,
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    Ai, Aj, Ax, symbolic, numeric = vector_arg_values
    aAi, aAj, aAx, asymbolic, anumeric = batch_axes

    if aAi is not None:
        msg = "Ai cannot be vectorized."
        raise ValueError(msg)

    if aAj is not None:
        msg = "Aj cannot be vectorized."
        raise ValueError(msg)

    if asymbolic is not None:
        msg = "symbolic handle cannot be vectorized."
        raise ValueError(msg)

    if aAx is not None and anumeric is not None:
        if Ax.ndim != 3 or numeric.ndim != 2:
            msg = (
                "Ax and numeric should be 3D and 2D respectively when vectorizing "
                f"over them simultaneously. Got: {Ax.shape=}; {numeric.shape=}."
            )
            raise ValueError(msg)
        Ax = jnp.moveaxis(Ax, aAx, 0)
        numeric = jnp.moveaxis(numeric, anumeric, 0)
        batch, n_lhs, n_vals = Ax.shape
        Ax = Ax.reshape(batch * n_lhs, n_vals)
        numeric = numeric.reshape(batch * n_lhs)
        return prim.bind(
            Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, symbolic, numeric
        ).reshape(batch, n_lhs), 0

    if aAx is not None:
        if Ax.ndim != 3:
            msg = f"Ax should be 3D when vectorizing over it. Got: {Ax.shape=}."
            raise ValueError(msg)
        Ax = jnp.moveaxis(Ax, aAx, 0)
        batch, n_lhs, n_vals = Ax.shape
        numeric = jnp.broadcast_to(numeric[None], (batch, numeric.shape[0]))
        Ax = Ax.reshape(batch * n_lhs, n_vals)
        numeric = numeric.reshape(batch * n_lhs)
        return prim.bind(
            Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, symbolic, numeric
        ).reshape(batch, n_lhs), 0

    if anumeric is not None:
        if numeric.ndim != 2:
            msg = (
                f"numeric should be 2D when vectorizing over it. Got: {numeric.shape=}."
            )
            raise ValueError(msg)
        numeric = jnp.moveaxis(numeric, anumeric, 0)
        batch, n_lhs = numeric.shape
        Ax = jnp.broadcast_to(Ax[None], (batch, Ax.shape[0], Ax.shape[1]))
        Ax = Ax.reshape(batch * n_lhs, Ax.shape[2])
        numeric = numeric.reshape(batch * n_lhs)
        return prim.bind(
            Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, symbolic, numeric
        ).reshape(batch, n_lhs), 0

    msg = "vmap failed. Please select an axis to vectorize over."
    raise ValueError(msg)


# Transposition =======================================================================


def dot_f64_transpose(
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    x: Array,
) -> tuple[Array, Array, Array, Array]:
    return dot_transpose(dot_f64, ct, Ai, Aj, Ax, x)


ad.primitive_transposes[dot_f64] = dot_f64_transpose


def dot_c128_transpose(
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    x: Array,
) -> tuple[Array, Array, Array, Array]:
    return dot_transpose(dot_c128, ct, Ai, Aj, Ax, x)


ad.primitive_transposes[dot_c128] = dot_c128_transpose


def solve_f64_transpose(
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    b: Array,
) -> tuple[Array, Array, Array, Array]:
    return solve_transpose(solve_f64, ct, Ai, Aj, Ax, b)


ad.primitive_transposes[solve_f64] = solve_f64_transpose


def solve_c128_transpose(
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    b: Array,
) -> tuple[Array, Array, Array, Array]:
    return solve_transpose(solve_c128, ct, Ai, Aj, Ax, b)


ad.primitive_transposes[solve_c128] = solve_c128_transpose


def dot_transpose(
    prim: jax.extend.core.Primitive,
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    x: Array,
) -> tuple[Array, Array, Array, Array]:
    if ad.is_undefined_primal(Ai) or ad.is_undefined_primal(Aj):
        msg = "Sparse indices Ai and Aj should not require gradients."
        raise ValueError(msg)

    if ad.is_undefined_primal(x):
        # replace x by ct
        return Aj, Ai, Ax, prim.bind(Aj.astype(jnp.int32), Ai.astype(jnp.int32), Ax, ct)

    if ad.is_undefined_primal(Ax):
        # ∂L/∂Ax[m,n] = Σₖ ct[m, Ai[n], k] · x[m, Aj[n], k]
        return Ai, Aj, (ct[:, Ai] * x[:, Aj, :]).sum(-1), x

    msg = "No undefined primals in transpose."
    raise ValueError(msg)


def solve_transpose(
    prim: jax.extend.core.Primitive,
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    b: Array,
) -> tuple[Array, Array, Array, Array]:
    if ad.is_undefined_primal(Ai) or ad.is_undefined_primal(Aj):
        msg = "Sparse indices Ai and Aj should not require gradients."
        raise ValueError(msg)

    if ad.is_undefined_primal(b):
        b_bar = prim.bind(Aj.astype(jnp.int32), Ai.astype(jnp.int32), Ax, ct)
        return Ai, Aj, Ax, b_bar

    if ad.is_undefined_primal(Ax):
        Ax_bar = -(
            ct * prim.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, b)
        ).sum(-1)
        return Ai, Aj, Ax_bar, b

    msg = "No undefined primals in transpose."
    raise ValueError(msg)


# Validators ==========================================================================


def validate_args(  # noqa: C901,PLR0912
    Ai: Array, Aj: Array, Ax: Array, x: Array, x_name: str = "x"
) -> tuple[Array, Array, Array, Array, tuple[int, ...]]:
    # cases:
    # - (n_lhs, n_nz) x (n_lhs, n_col, n_rhs)
    # - (n_lhs, n_nz) x (n_lhs, n_col) --> (n_lhs, n_nz) x (n_lhs, n_col, 1)
    # - (n_nz,) x (n_lhs, n_col, n_rhs) --> (n_lhs, n_nz) x (n_lhs, n_col, n_rhs)
    # - (n_nz,) x (n_col, n_rhs) --> (1, n_nz) x (1, n_col, n_rhs)
    # - (n_nz,) x (n_col,) --> (1, n_nz) x (1, n_col, 1)
    if Ai.ndim != 1:
        msg = f"Ai should be 1D with shape (n_nz). Got: {Ai.shape=}."
        raise ValueError(msg)
    if Aj.ndim != 1:
        msg = f"Aj should be 1D with shape (n_nz,). Got: {Aj.shape=}."
        raise ValueError(msg)
    if Ax.ndim == 0 or Ax.ndim > 2:
        msg = (
            "Ax should be 1D with shape (n_nz,) "
            "or 2D with shape (n_lhs, n_nz). "
            f"Got: {Ax.shape=}."
        )
        raise ValueError(msg)
    if x.ndim == 0 or x.ndim > 3:
        msg = (
            f"{x_name} should be 1D with shape (n_col,) "
            "or 2D with shape (n_col, n_rhs) "
            "or 3D with shape (n_lhs, n_col, n_rhs). "
            f"Got: {x_name}.shape={x.shape}."
        )
        raise ValueError(msg)

    shape = x.shape

    if Ax.ndim == 1 and x.ndim == 1:  # expand Ax and b dims
        debug(f"assuming (n_nz:={Ax.shape[0]},) x (n_col:={x.shape[0]},)")
        Ax = Ax[None, :]
        x = x[None, :, None]
    elif Ax.ndim == 1 and x.ndim == 2:  # expand Ax and b dims
        debug(
            f"assuming (n_nz:={Ax.shape[0]},) x "
            f"(n_col:={x.shape[0]}, n_rhs:={x.shape[1]})"
        )
        Ax = Ax[None, :]
        x = x[None, :, :]
    elif Ax.ndim == 1 and x.ndim == 3:  # expand A dim (broadcast will happen in base)
        debug(
            f"assuming (n_nz:={Ax.shape[0]},) x "
            f"(n_lhs:={x.shape[0]}, n_col:={x.shape[1]}, n_rhs:={x.shape[2]})"
        )
        Ax = Ax[None, :]
    elif Ax.ndim == 2 and x.ndim == 1:  # expand dims to base case
        debug(
            f"assuming (n_lhs:={Ax.shape[0]}, n_nz:={Ax.shape[1]}) x "
            f"(n_col:={x.shape[0]},)"
        )
        x = x[None, :, None]
        shape = (Ax.shape[0], shape[0])  # we need to expand the shape here.
    elif Ax.ndim == 2 and x.ndim == 2:  # expand dims to base case
        debug(
            f"assuming (n_lhs:={Ax.shape[0]}, n_nz:={Ax.shape[1]}) x "
            f"(n_lhs:={x.shape[0]}, n_col:={x.shape[1]})"
        )
        if Ax.shape[0] != x.shape[0] and Ax.shape[0] != 1 and x.shape[0] != 1:
            msg = (
                f"Ax (2D) and {x_name} (2D) should have their first shape "
                f"index `n_lhs` match. Got: {Ax.shape=}; {x_name}.shape={x.shape}. "
                f"assuming (n_lhs:={Ax.shape[0]}, n_nz:={Ax.shape[1]}) x "
                f"(n_lhs:={x.shape[0]}, n_col:={x.shape[1]})"
            )
            raise ValueError(msg)
        x = x[:, :, None]
        if x.shape[0] == 1 and Ax.shape[0] > 0:
            shape = (Ax.shape[0], *shape[1:])

    if Ax.ndim != 2 or x.ndim != 3:
        msg = (
            f"Invalid shapes for Ax and {x_name}. "
            f"Got: {Ax.shape=}; {x_name}.shape={x.shape}. "
            f"Expected: Ax.shape=([n_lhs],n_nz); "
            f"{x_name}.shape=([n_lhs],n_col,[n_rhs])."
        )
        raise ValueError(msg)

    # base case
    debug(
        f"assuming (n_lhs:={Ax.shape[0]}, n_nz:={Ax.shape[1]}) x "
        f"(n_lhs:={x.shape[0]}, n_col:={x.shape[1]}, n_rhs:={x.shape[2]})"
    )
    if Ax.shape[0] != x.shape[0] and Ax.shape[0] != 1 and x.shape[0] != 1:
        msg = (
            f"Ax (2D) and {x_name} (3D) should have their first shape "
            f"index `n_lhs` match. Got: {Ax.shape=}; {x_name}.shape={x.shape}."
            f"assuming (n_lhs:={Ax.shape[0]}, n_nz:={Ax.shape[1]}) x "
            f"(n_lhs:={x.shape[0]}, n_col:={x.shape[1]}, n_rhs:={x.shape[2]})"
        )
        raise ValueError(msg)
    n_lhs = max(Ax.shape[0], x.shape[0])  # handle broadcastable 1-index
    Ax = jnp.broadcast_to(Ax, (n_lhs, Ax.shape[1]))
    x = jnp.broadcast_to(x, (n_lhs, x.shape[1], x.shape[2]))
    if len(shape) == 3 and shape[0] != x.shape[0]:
        shape = (Ax.shape[0], shape[1], shape[2])
    return Ai, Aj, Ax, x, shape


# Differentiation rules for solve_with_symbol =================================


def solve_with_symbol_value_and_jvp(
    prim_solve: jax.extend.core.Primitive,
    prim_dot: jax.extend.core.Primitive,
    arg_values: tuple[Array, Array, Array, Array, Array],
    arg_tangents: tuple[Any, Any, Any, Any, Any],
) -> tuple[Array, Array]:
    """Jacobian-vector product rule for `solve_with_symbol`.

    The rule for the JVP of a linear solve `x = A^-1 * b` is:
    `dx = A^-1 * (db - dA * x)`

    This function implements this rule efficiently. It first computes the primal
    solution `x`. Then it computes the right-hand side of the tangent system,
    `rhs = db - dA * x`. If `dA` is not zero, `dA * x` is computed using the
    `dot` primitive. If the `rhs` is not zero, `dx` is found by solving the
    system `A * dx = rhs`, reusing the symbolic factorization of A for
    efficiency.

    Args:
        prim_solve: The `solve_with_symbol` primitive.
        prim_dot: The corresponding `dot` primitive.
        arg_values: Primal values for `(Ai, Aj, Ax, b, symbolic)`.
        arg_tangents: Tangent values for `(Ai, Aj, Ax, b, symbolic)`.

    Returns:
        A tuple containing the primal output `x` and the tangent output `dx`.

    """
    Ai, Aj, Ax, b, symbolic = arg_values
    _t_Ai, _t_Aj, t_Ax, t_b, _t_symbolic = arg_tangents

    x = prim_solve.bind(Ai, Aj, Ax, b, symbolic)

    rhs = t_b

    if not isinstance(t_Ax, ad.Zero):
        dAx = prim_dot.bind(Ai, Aj, t_Ax, x)
        rhs = -dAx if isinstance(rhs, ad.Zero) else rhs - dAx

    if isinstance(rhs, ad.Zero):
        dx = jnp.zeros_like(x)
    else:
        dx = prim_solve.bind(Ai, Aj, Ax, rhs, symbolic)

    return x, dx


# Register JVP for Float64
def solve_with_symbol_f64_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array, Array],
    arg_tangents: tuple[Any, Any, Any, Any, Any],
) -> tuple[Array, Array]:
    """JVP rule for float64 solve_with_symbol."""
    return solve_with_symbol_value_and_jvp(
        solve_with_symbol_f64,
        dot_f64,
        arg_values,
        arg_tangents,
    )


ad.primitive_jvps[solve_with_symbol_f64] = solve_with_symbol_f64_value_and_jvp


# Register JVP for Complex128
def solve_with_symbol_c128_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array, Array],
    arg_tangents: tuple[Any, Any, Any, Any, Any],
) -> tuple[Array, Array]:
    """JVP rule for complex128 solve_with_symbol."""
    return solve_with_symbol_value_and_jvp(
        solve_with_symbol_c128,
        dot_c128,
        arg_values,
        arg_tangents,
    )


ad.primitive_jvps[solve_with_symbol_c128] = solve_with_symbol_c128_value_and_jvp


def solve_with_symbol_transpose(
    solve_prim: jax.extend.core.Primitive,
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    b: Array,
    symbolic: Array,
) -> tuple[Array, Array, Array, Array, None]:
    if ad.is_undefined_primal(Ai) or ad.is_undefined_primal(Aj):
        msg = "Sparse indices Ai and Aj should not require gradients."
        raise ValueError(msg)

    # Pick the right tsolve primitive
    if solve_prim is solve_with_symbol_f64:
        tsolve_prim = tsolve_with_symbol_f64
    else:
        tsolve_prim = tsolve_with_symbol_c128

    if ad.is_undefined_primal(b):
        # FAST PATH: Backpropagate through b using the reused handle!
        b_bar = tsolve_prim.bind(
            Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, ct, symbolic
        )
        return Ai, Aj, Ax, b_bar, None

    if ad.is_undefined_primal(Ax):
        Ax_bar = -(
            ct
            * tsolve_prim.bind(
                Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, b, symbolic
            )
        ).sum(-1)
        return Ai, Aj, Ax_bar, b, None

    msg = "No undefined primals in transpose."
    raise ValueError(msg)


def solve_with_symbol_f64_transpose(ct, Ai, Aj, Ax, b, symbolic):
    return solve_with_symbol_transpose(
        solve_with_symbol_f64, ct, Ai, Aj, Ax, b, symbolic
    )


ad.primitive_transposes[solve_with_symbol_f64] = solve_with_symbol_f64_transpose


def solve_with_symbol_c128_transpose(ct, Ai, Aj, Ax, b, symbolic):
    return solve_with_symbol_transpose(
        solve_with_symbol_c128, ct, Ai, Aj, Ax, b, symbolic
    )


ad.primitive_transposes[solve_with_symbol_c128] = solve_with_symbol_c128_transpose


# Differentiation rules for solve_with_numeric ================================


def solve_with_numeric_value_and_jvp(
    prim_solve: jax.extend.core.Primitive,
    arg_values: tuple[Array, Array, Array],
    arg_tangents: tuple[Any, Any, Any],
) -> tuple[Array, Array]:
    """Jacobian-vector product rule for `solve_with_numeric`.

    Since the matrix values are baked into the numeric handle, we only track
    gradients with respect to `b`. The rule is simply: dx = A^-1 * db.
    """
    symbolic, numeric, b = arg_values
    _, _, t_b = arg_tangents

    x = prim_solve.bind(symbolic, numeric, b)

    if isinstance(t_b, ad.Zero):
        dx = jnp.zeros_like(x)
    else:
        dx = prim_solve.bind(symbolic, numeric, t_b)

    return x, dx


def solve_with_numeric_f64_value_and_jvp(
    arg_values: tuple[Array, Array, Array],
    arg_tangents: tuple[Any, Any, Any],
) -> tuple[Array, Array]:
    return solve_with_numeric_value_and_jvp(
        solve_with_numeric_f64, arg_values, arg_tangents
    )


ad.primitive_jvps[solve_with_numeric_f64] = solve_with_numeric_f64_value_and_jvp


def solve_with_numeric_c128_value_and_jvp(
    arg_values: tuple[Array, Array, Array],
    arg_tangents: tuple[Any, Any, Any],
) -> tuple[Array, Array]:
    return solve_with_numeric_value_and_jvp(
        solve_with_numeric_c128, arg_values, arg_tangents
    )


ad.primitive_jvps[solve_with_numeric_c128] = solve_with_numeric_c128_value_and_jvp


def solve_with_numeric_transpose(
    solve_prim: jax.extend.core.Primitive,
    ct: Array,
    symbolic: Array,
    numeric: Array,
    b: Array,
) -> tuple[None, None, Array]:
    """Compute the transpose of solve_with_numeric.

    Uses the new tsolve primitives to efficiently solve A^T x_bar = b_bar
    using the existing numeric factorization handle.
    """
    tsolve_prim = (
        tsolve_with_numeric_f64
        if solve_prim is solve_with_numeric_f64
        else tsolve_with_numeric_c128
    )

    if ad.is_undefined_primal(b):
        b_bar = tsolve_prim.bind(symbolic, numeric, ct)
        # return None for symbolic and numeric handles, since they don't take gradients
        return None, None, b_bar

    msg = "No undefined primals in transpose."
    raise ValueError(msg)


def solve_with_numeric_f64_transpose(ct, symbolic, numeric, b):
    return solve_with_numeric_transpose(
        solve_with_numeric_f64, ct, symbolic, numeric, b
    )


ad.primitive_transposes[solve_with_numeric_f64] = solve_with_numeric_f64_transpose


def solve_with_numeric_c128_transpose(ct, symbolic, numeric, b):
    return solve_with_numeric_transpose(
        solve_with_numeric_c128, ct, symbolic, numeric, b
    )


ad.primitive_transposes[solve_with_numeric_c128] = solve_with_numeric_c128_transpose
