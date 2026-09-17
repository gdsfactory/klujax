"""Location of the compiled Rust `klujax-ffi` cdylib.

The shared library is copied here by `setup.py`'s `CargoBuildExt`. During
development, it may also be loaded directly from `target/release`.
"""

from __future__ import annotations

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
        f"could not find {_NAME} in {here} or target/; "
        "build it with `just rust-build`"
    )
    raise FileNotFoundError(msg)
