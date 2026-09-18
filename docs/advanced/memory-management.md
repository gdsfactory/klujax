---
title: Memory Management
summary: Own and release native KLU handles safely
---

# Memory Management

`klujax.analyze` returns a `KLUSymbolic` owner, and `klujax.factor` returns a
`KLUNumeric` owner. `KLUHandleManager` is an alias for `KLUSymbolic`. These
objects own opaque native IDs, not memory addresses.

## Explicit lifetime

Create owners outside JIT and keep them alive until all dependent JAX work
finishes:

```python
with klujax.analyze(Ai, Aj, n_col) as symbolic:
    with klujax.factor(Ai, Aj, Ax, symbolic) as numeric:
        x = klujax.solve_with_numeric(numeric, b, symbolic)
        x.block_until_ready()
```

`close()` releases native allocations and clears the owner's stored IDs.
Repeated calls are harmless, including calls from different Python threads.
Garbage collection also closes owners, but context managers give deterministic
cleanup. Copying, deep copying and serializing owners raise `TypeError`.

## Native validation

Rust stores allocations behind opaque, never-reused `uint64` IDs. A stale or
fabricated ID returns an error instead of being dereferenced. The wrapper checks
handle kinds, real/complex scalar types, the analyzed matrix pattern, and RHS
buffer dimensions before calling KLU.

Operations hold native allocations alive while using them. Closing an owner
prevents new lookups; an operation that already acquired it may finish safely.
Calls sharing a symbolic handle are serialized. Independent symbolic handles
can execute concurrently.

## JIT and asynchronous work

Owners can be passed into JIT-compiled functions. Wait for the returned arrays
with `block_until_ready()` before closing the owners. A cached JIT computation
that retained an ID cannot use it after close.

Calling `analyze` or `factor` inside JIT returns arrays without Python owners.
The deprecated `free_symbolic` and `free_numeric` helpers only close Python
owners; they do not schedule cleanup of traced arrays. Prefer creating owners
outside JIT, or use `klujax.solve` for a complete solve whose temporary native
allocations are released internally.

## Handle properties

| Property | Meaning |
| --- | --- |
| `symbolic.raw` | Opaque symbolic ID; raises after close |
| `symbolic.handle` | The same symbolic owner |
| `numeric.as_list()` | Copy of the numeric IDs; raises after close |
| `numeric.size` | Number of numeric IDs; zero after close |

The IDs belong to the current process and loaded native library. They are not
portable pointers or serializable solver state.
