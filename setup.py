"""KLUJAX Setup.

The primary artifact is now the Rust `klujax-ffi` cdylib. The legacy C++
extension (`klujax_cpp`) is still built when its vendored dependencies are
present, so that the pre-migration golden corpus can be generated; it is removed
in Stage 6 of the Rust migration (see `work.md`).
"""

import os
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

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
            msg = (
                "cargo not found: install Rust (https://rustup.rs) to build "
                "klujax"
            )
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


# Legacy C++ extension ---------------------------------------------------------
# Only built when the vendored dependencies are present (see `just deps`).
_deps_present = all(
    (ROOT / dep).is_dir() for dep in ("suitesparse", "xla", "pybind11")
)
_build_cpp = _deps_present and os.environ.get("KLUJAX_BUILD_CPP", "0") == "1"

include_dirs = [
    "xla",
    os.path.join("pybind11", "include"),
    os.path.join("suitesparse", "SuiteSparse_config"),
    os.path.join("suitesparse", "AMD", "Include"),
    os.path.join("suitesparse", "COLAMD", "Include"),
    os.path.join("suitesparse", "BTF", "Include"),
    os.path.join("suitesparse", "KLU", "Include"),
]

suitesparse_sources = [
    os.path.join("suitesparse", "SuiteSparse_config", "SuiteSparse_config.c"),
    *glob(os.path.join("suitesparse", "AMD", "Source", "*.c")),
    *glob(os.path.join("suitesparse", "COLAMD", "Source", "*.c")),
    *glob(os.path.join("suitesparse", "BTF", "Source", "*.c")),
    *glob(os.path.join("suitesparse", "KLU", "Source", "*.c")),
]


def _cpp_extension() -> Extension:
    if sys.platform == "linux":  # gcc
        return Extension(
            name="klujax_cpp",
            sources=["klujax.cpp", *suitesparse_sources],
            include_dirs=include_dirs,
            extra_compile_args=["-std=c++17"],
            extra_link_args=["-static-libgcc", "-static-libstdc++"],
            language="c++",
        )
    if sys.platform == "win32":  # cl
        return Extension(
            name="klujax_cpp",
            sources=["klujax.cpp", *suitesparse_sources],
            include_dirs=include_dirs,
            extra_compile_args=["/std:c++17"],
            language="c++",
        )
    return Extension(  # darwin clang
        name="klujax_cpp",
        sources=["klujax.cpp", *suitesparse_sources],
        include_dirs=include_dirs,
        extra_compile_args=["-std=c++17"],
        language="c++",
    )


# Custom BuildExt to enable combined build of C and C++ files (clang), on top
# of the CargoBuildExt behaviour.
class BuildExt(CargoBuildExt):
    def build_extension(self, ext: Extension) -> None:
        sources = ext.sources
        c_sources = sorted([s for s in sources if s.endswith("c")])
        cpp_sources = sorted([s for s in sources if s not in c_sources])
        ext_path = self.get_ext_fullpath(ext.name)
        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))
        c_objects = self.compiler.compile(
            c_sources,
            output_dir=self.build_temp,
            macros=macros,
            include_dirs=ext.include_dirs,
            debug=self.debug,
            extra_postargs=[
                f
                for f in ext.extra_compile_args
                if f not in ["-std=c++17", "/std:c++17"]  # THIS IS OUR HACK
            ],
            depends=ext.depends,
        )
        cpp_objects = self.compiler.compile(
            cpp_sources,
            output_dir=self.build_temp,
            macros=macros,
            include_dirs=ext.include_dirs,
            debug=self.debug,
            extra_postargs=ext.extra_compile_args,
            depends=ext.depends,
        )
        objects = c_objects + cpp_objects
        extra_args = ext.extra_link_args or []
        self.compiler.link_shared_object(
            objects,
            ext_path,
            libraries=self.get_libraries(ext),
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=extra_args,
            export_symbols=self.get_export_symbols(ext),
            debug=self.debug,
            build_temp=self.build_temp,
            target_lang=ext.language,
        )


setup(
    py_modules=["klujax"],
    packages=["klujax_native"],
    package_data={
        "klujax_native": ["*.so", "*.dylib", "*.dll"],
    },
    ext_modules=[_cpp_extension()] if _build_cpp else [],
    cmdclass={"build_ext": BuildExt},
)
