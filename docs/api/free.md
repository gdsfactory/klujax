---
title: free_symbolic / free_numeric
summary: Explicitly free KLU handles inside JIT
---

# free_symbolic / free_numeric

```python
klujax.free_symbolic(symbolic, dependency=None) -> None
klujax.free_numeric(numeric, dependency=None) -> None
```

Explicitly free the C++ memory behind a KLU handle. You only need these inside `jax.jit`-compiled functions — outside JIT, handles are freed automatically.

## Parameters

### free_symbolic

| Parameter    | Type             | Description                                                   |
| ------------ | ---------------- | ------------------------------------------------------------- |
| `symbolic`   | KLUHandleManager | Handle from [analyze](../api/analyze.md)                      |
| `dependency` | Array or None    | An array that must be computed **before** the handle is freed |

### free_numeric

| Parameter    | Type             | Description                                                              |
| ------------ | ---------------- | ------------------------------------------------------------------------ |
| `numeric`    | KLUHandleManager | Handle from [factor](../api/factor.md) or [refactor](../api/refactor.md) |
| `dependency` | Array or None    | An array that must be computed **before** the handle is freed            |

## Why dependency Matters

Inside JIT, the XLA compiler can reorder operations. Without a dependency, the compiler might free your handle **before** the solve that uses it has finished. The `dependency` parameter tells XLA: "don't free this until the dependency array is ready."

```mermaid
flowchart TD
    subgraph "Without dependency (WRONG)"
        A1["analyze"] --> S1["solve_with_symbol"]
        A1 --> F1["free_symbolic ⚠️"]
        S1 -.->|"might run after free!"| BAD["💥 Segfault"]
    end

    subgraph "With dependency (CORRECT)"
        A2["analyze"] --> S2["solve_with_symbol"] --> X2["x #40;result#41;"]
        X2 --> F2["free_symbolic#40;sym, dependency=x#41;"]
    end
```

## Example: Inside JIT

```python
@jax.jit
def dynamic_solve(Ai, Aj, Ax, b):
    # Create handle inside JIT (not ideal, but sometimes necessary)
    sym = klujax.analyze(Ai, Aj, 5)

    # Solve
    x = klujax.solve_with_symbol(Ai, Aj, Ax, b, sym)

    # CRITICAL: Free with dependency to ensure correct ordering
    klujax.free_symbolic(sym, dependency=x)

    return x
```

## When You Don't Need These

Outside JIT, handles clean up automatically:

```python
# This is fine — no need to call free_symbolic
symbolic = klujax.analyze(Ai, Aj, n_col)
x = klujax.solve_with_symbol(Ai, Aj, Ax, b, symbolic)
# symbolic is freed when garbage collected
```

Or use a context manager:

```python
with klujax.analyze(Ai, Aj, n_col) as symbolic:
    x = klujax.solve_with_symbol(Ai, Aj, Ax, b, symbolic)
# Freed here
```

!!! tip
    The best practice is to create handles **outside** JIT. Then you never need `free_symbolic` or `free_numeric` at all. See [Memory Management](../advanced/memory-management.md).


## Numeric allocations in compiled loops

Keep the symbolic owner outside compiled code. Each numeric allocation made by
`factor` inside `jit` or `vmap` requires exactly one runtime `free_numeric`:

```python
symbolic = klujax.analyze(Ai, Aj, n_col)

@jax.jit
def run(n_iter, values, b):
    def body(i, total):
        numeric = klujax.factor(Ai, Aj, values * (1 + 1e-6 * i), symbolic)
        x = klujax.solve_with_numeric(numeric, b, symbolic)
        y = klujax.tsolve_with_numeric(numeric, b, symbolic)
        klujax.free_numeric(numeric, dependency=(x, y))
        return total + x[0] + y[0]
    return jax.lax.fori_loop(0, n_iter, body, 0.0)
```

`dependency` must contain every consumer result; it can be an array or pytree.
Raw handles require this argument. Freeing and allocation are side effects, so
an unused status or loop-invariant matrix does not remove or hoist these calls.
The native free releases all pointers in a batched allocation. Mapping over a
shared numeric handle while freeing it is rejected; free it once after the
mapped consumers instead. Never use a raw handle after freeing it.

This form is supported and emits no deprecation warning. Eager `KLUNumeric`
owners still use automatic cleanup or `close()` after synchronizing consumers;
calling `free_numeric` on such an owner inside tracing is rejected.
