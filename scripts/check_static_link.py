#!/usr/bin/env python
"""Verify the built cdylib statically links SuiteSparse (no dynamic dependency).

Fails if the shared library dynamically references KLU / SuiteSparse. This is the
whole point of the `klu-sys` crate: KLU must be baked into the cdylib, with no
runtime `libklu` / `libsuitesparse`.

Usage:
    python scripts/check_static_link.py [path/to/libklujax_ffi.{so,dylib,dll}]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN = re.compile(
    r"(suitesparse|libamd\b|libcolamd\b|libbtf\b|libklu\b|\bklu\.)",
    re.IGNORECASE,
)


def default_lib() -> Path:
    root = Path(__file__).resolve().parent.parent
    name = {
        "darwin": "libklujax_ffi.dylib",
        "win32": "klujax_ffi.dll",
    }.get(sys.platform, "libklujax_ffi.so")
    for candidate in (
        root / "klujax_native" / name,
        root / "target" / "release" / name,
    ):
        if candidate.exists():
            return candidate
    msg = f"could not find {name} under klujax_native/ or target/release"
    raise FileNotFoundError(msg)


def dependencies(lib: Path) -> list[str]:
    if sys.platform == "darwin":
        out = subprocess.run(
            ["otool", "-L", str(lib)], capture_output=True, text=True, check=True
        ).stdout
    elif sys.platform == "win32":
        out = subprocess.run(
            ["dumpbin", "/dependents", str(lib)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    else:
        out = subprocess.run(
            ["ldd", str(lib)], capture_output=True, text=True, check=True
        ).stdout
    # First line is the library itself; also drop the self-referential install
    # name (which contains our own `libklujax_ffi`).
    self_name = lib.name
    return [line for line in out.splitlines()[1:] if self_name not in line]


def main() -> int:
    lib = Path(sys.argv[1]) if len(sys.argv) > 1 else default_lib()
    deps = dependencies(lib)
    offenders = [line.strip() for line in deps if FORBIDDEN.search(line)]

    if offenders:
        print(f"FAIL: {lib} dynamically references SuiteSparse/KLU:", file=sys.stderr)
        for line in offenders:
            print(f"  {line}", file=sys.stderr)
        return 1

    print(f"OK: {lib} statically links SuiteSparse (no dynamic klu/suitesparse).")
    for line in deps:
        print(f"  {line.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
