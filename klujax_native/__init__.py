"""Location of the compiled Rust ``klujax-ffi`` cdylib plus a pure-Python shim
of the former pybind11 module (``klujax_cpp``).

``klujax.py`` imports this module as ``klujax_cpp``; it provides:

- zero-argument capsule providers (``solve_f64()``, ``analyze()``, …) that wrap
  the corresponding ``extern "C"`` symbols from the cdylib with
  ``jax.ffi.pycapsule``;
- ``KLUSymbolic`` / ``KLUNumeric`` handle objects backed by the plain C-ABI
  ``klujax_free_*`` symbols.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

_NAME = {
    "darwin": "libklujax_ffi.dylib",
    "win32": "klujax_ffi.dll",
}.get(sys.platform, "libklujax_ffi.so")


def lib_path() -> Path:
    """Return the path to the compiled cdylib, searching packaged then target."""
    here = Path(__file__).resolve().parent
    packaged = here / _NAME
    if packaged.exists():
        return packaged
    root = here.parent
    for profile in ("release", "debug"):
        candidate = root / "target" / profile / _NAME
        if candidate.exists():
            return candidate
    msg = (
        f"could not find {_NAME} in {here} or target/; build it with `just rust-build`"
    )
    raise FileNotFoundError(msg)


_LIB: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    """Load (once) and return the cdylib."""
    global _LIB  # noqa: PLW0603
    if _LIB is None:
        _LIB = ctypes.CDLL(str(lib_path()))
        _LIB.klujax_version.restype = ctypes.c_char_p
        _LIB.klujax_free_symbolic.argtypes = [ctypes.c_uint64]
        _LIB.klujax_free_symbolic.restype = ctypes.c_int
        _LIB.klujax_free_numeric.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        _LIB.klujax_free_numeric.restype = ctypes.c_int
    return _LIB


#: XLA FFI targets exported by the cdylib.
TARGETS = [
    "dot_f64",
    "dot_c128",
    "solve_f64",
    "solve_c128",
    "solve_with_symbol_f64",
    "solve_with_symbol_c128",
    "tsolve_with_symbol_f64",
    "tsolve_with_symbol_c128",
    "factor_f64",
    "factor_c128",
    "refactor_f64",
    "refactor_c128",
    "refactor_and_solve_f64",
    "refactor_and_solve_c128",
    "solve_with_numeric_f64",
    "solve_with_numeric_c128",
    "tsolve_with_numeric_f64",
    "tsolve_with_numeric_c128",
    "free_numeric",
    "free_symbolic",
    "analyze",
]


def _capsule(name: str):
    """Wrap the cdylib symbol ``name`` as a JAX FFI target ``PyCapsule``."""
    import jax.ffi

    return jax.ffi.pycapsule(getattr(lib(), name))


def dot_f64():
    return _capsule("dot_f64")


def dot_c128():
    return _capsule("dot_c128")


def solve_f64():
    return _capsule("solve_f64")


def solve_c128():
    return _capsule("solve_c128")


def solve_with_symbol_f64():
    return _capsule("solve_with_symbol_f64")


def solve_with_symbol_c128():
    return _capsule("solve_with_symbol_c128")


def tsolve_with_symbol_f64():
    return _capsule("tsolve_with_symbol_f64")


def tsolve_with_symbol_c128():
    return _capsule("tsolve_with_symbol_c128")


def factor_f64():
    return _capsule("factor_f64")


def factor_c128():
    return _capsule("factor_c128")


def refactor_f64():
    return _capsule("refactor_f64")


def refactor_c128():
    return _capsule("refactor_c128")


def refactor_and_solve_f64():
    return _capsule("refactor_and_solve_f64")


def refactor_and_solve_c128():
    return _capsule("refactor_and_solve_c128")


def solve_with_numeric_f64():
    return _capsule("solve_with_numeric_f64")


def solve_with_numeric_c128():
    return _capsule("solve_with_numeric_c128")


def tsolve_with_numeric_f64():
    return _capsule("tsolve_with_numeric_f64")


def tsolve_with_numeric_c128():
    return _capsule("tsolve_with_numeric_c128")


def free_numeric():
    return _capsule("free_numeric")


def free_symbolic():
    return _capsule("free_symbolic")


def analyze():
    return _capsule("analyze")


class KLUSymbolic:
    """Symbolic analysis handle (owns a ``klu_symbolic*`` stored as ``u64``)."""

    __slots__ = ("_closed", "_raw")

    def __init__(self, raw: int) -> None:
        self._raw = int(raw)
        self._closed = False

    @property
    def raw(self) -> int:
        return self._raw

    @property
    def handle(self) -> KLUSymbolic:
        return self

    def close(self) -> None:
        if not self._closed:
            lib().klujax_free_symbolic(ctypes.c_uint64(self._raw))
            self._closed = True

    def __enter__(self) -> KLUSymbolic:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class KLUNumeric:
    """Numeric factorization handle (owns one ``klu_numeric*`` per batch)."""

    __slots__ = ("_closed", "_handles")

    def __init__(self, handles) -> None:
        self._handles = [int(h) for h in handles]
        self._closed = False

    @property
    def size(self) -> int:
        return len(self._handles)

    def as_list(self) -> list[int]:
        return list(self._handles)

    def close(self) -> None:
        if not self._closed and self._handles:
            arr = (ctypes.c_uint64 * len(self._handles))(*self._handles)
            lib().klujax_free_numeric(arr, len(self._handles))
            self._closed = True

    def __enter__(self) -> KLUNumeric:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
