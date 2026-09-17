#!/usr/bin/env python
"""Stage 1 smoke test: load the Rust cdylib with ctypes and verify the ABI surface.

This does not require JAX; it only checks that the shared library can be loaded
and that every XLA FFI handler symbol and C-ABI shim is exported.

Usage:
    python scripts/ffi_smoke.py [path/to/libklujax_ffi.{so,dylib,dll}]
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

HANDLER_SYMBOLS = [
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

CAPI_SYMBOLS = [
    "klujax_version",
    "klujax_free_symbolic",
    "klujax_free_numeric",
]


def default_lib() -> Path:
    root = Path(__file__).resolve().parent.parent
    names = {
        "darwin": "libklujax_ffi.dylib",
        "win32": "klujax_ffi.dll",
    }
    name = names.get(sys.platform, "libklujax_ffi.so")
    for profile in ("release", "debug"):
        candidate = root / "target" / profile / name
        if candidate.exists():
            return candidate
    msg = f"could not find {name} under target/release or target/debug"
    raise FileNotFoundError(msg)


def main() -> int:
    lib_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_lib()
    lib = ctypes.CDLL(str(lib_path))
    print(f"loaded: {lib_path}")

    missing = []
    for symbol in HANDLER_SYMBOLS + CAPI_SYMBOLS:
        if not hasattr(lib, symbol):
            missing.append(symbol)
    if missing:
        print(f"MISSING SYMBOLS: {missing}", file=sys.stderr)
        return 1

    lib.klujax_version.restype = ctypes.c_char_p
    version = lib.klujax_version().decode()
    print(f"klujax_version() -> {version}")

    print(
        f"OK: {len(HANDLER_SYMBOLS)} handlers + {len(CAPI_SYMBOLS)} C-ABI symbols present"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
