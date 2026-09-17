"""KLUJAX Setup.

Builds the Rust ``klujax-ffi`` cdylib — which statically links SuiteSparse's KLU
through the ``klu-sys`` crate — and installs it into the ``klujax_native``
package. There is no pybind11 / C++ extension.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution

ROOT = Path(__file__).resolve().parent


def _rust_lib_name() -> str:
    if sys.platform == "darwin":
        return "libklujax_ffi.dylib"
    if sys.platform == "win32":
        return "klujax_ffi.dll"
    return "libklujax_ffi.so"


class CargoBuildExt(build_ext):
    """Build the Rust cdylib and copy it into the ``klujax_native`` package."""

    def run(self) -> None:
        cargo = os.environ.get("CARGO", "cargo")
        if shutil.which(cargo) is None:
            msg = "cargo not found: install Rust (https://rustup.rs) to build klujax"
            raise RuntimeError(msg)
        subprocess.run(
            [cargo, "build", "--release", "-p", "klujax-ffi"],
            cwd=ROOT,
            check=True,
        )
        built = ROOT / "target" / "release" / _rust_lib_name()
        if not built.exists():
            msg = f"expected cargo artifact not found: {built}"
            raise FileNotFoundError(msg)
        dest_dir = ROOT / "klujax_native"
        dest_dir.mkdir(exist_ok=True)
        shutil.copy2(built, dest_dir / _rust_lib_name())
        super().run()


class BinaryDistribution(Distribution):
    """Mark the distribution as platform-specific.

    The wheel bundles a native cdylib, so it must not be tagged
    ``py3-none-any`` (otherwise it would be installed on incompatible
    platforms).
    """

    def has_ext_modules(self) -> bool:  # noqa: D102
        return True


setup(
    py_modules=["klujax"],
    packages=["klujax_native"],
    package_data={
        "klujax_native": ["*.so", "*.dylib", "*.dll"],
    },
    cmdclass={"build_ext": CargoBuildExt},
    distclass=BinaryDistribution,
)
