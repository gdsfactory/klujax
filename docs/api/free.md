---
title: free_symbolic / free_numeric
summary: Deprecated helpers for closing Python handle owners
---

# free_symbolic / free_numeric

```python
klujax.free_symbolic(symbolic, dependency=None)
klujax.free_numeric(numeric, dependency=None)
```

These deprecated helpers emit `DeprecationWarning`, call `close()` when passed
a Python handle owner, and return a scalar int32 zero. Prefer calling `close()`
or using the owner's context manager.

`dependency` is retained for compatibility and is currently ignored. These
helpers do not schedule native cleanup for raw arrays or handles created inside
JIT. Create owners outside JIT, or use `klujax.solve` to manage temporary
allocations internally.

Wait for dependent JAX results before closing an owner:

```python
with klujax.analyze(Ai, Aj, n_col) as symbolic:
    x = klujax.solve_with_symbol(Ai, Aj, Ax, b, symbolic)
    x.block_until_ready()
```

Repeated closes are harmless. Accessing a closed owner or passing a stale ID to
a native operation raises an error. See [Memory Management](../advanced/memory-management.md).
