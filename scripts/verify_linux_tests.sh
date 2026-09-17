#!/usr/bin/env bash
# Full Linux verification: build klujax from source with `pip install .` inside a
# Docker container and run the whole pytest suite (tests.py +
# tests_characterization) against the statically linked cdylib.
#
# Slower than scripts/verify_linux.sh (installs Python + JAX), but proves the
# Linux end-to-end path (ctypes load + XLA FFI + all 130 tests).
#
# Usage: scripts/verify_linux_tests.sh
# Override the image with KLUJAX_LINUX_IMAGE (default: rust:1.85-bookworm).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${KLUJAX_LINUX_IMAGE:-rust:1.85-bookworm}"

docker run --rm \
  -v "$ROOT":/src:ro \
  -v klujax_target:/target \
  "$IMAGE" bash -c '
    set -euo pipefail
    export PATH="/usr/local/cargo/bin:$PATH"
    export CARGO_TARGET_DIR=/target

    if [ ! -x /venv/bin/pip ]; then
      apt-get update -qq >/dev/null 2>&1
      apt-get install -y -qq python3 python3-venv python3-pip >/dev/null 2>&1
      python3 -m venv /venv
      /venv/bin/pip install -q --upgrade pip >/dev/null 2>&1
    fi

    rm -rf /work && mkdir -p /work
    tar -C /src -cf - \
      --exclude=./target --exclude=./.git --exclude=./.venv \
      --exclude=./dist --exclude=./site --exclude=./build \
      --exclude=./klujax_native/*.dylib --exclude=./klujax_native/*.so \
      --exclude=./klujax_native/*.dll . | tar -C /work -xf -

    cd /work
    /venv/bin/pip install -q . "jax[cpu]" numpy scipy pytest
    /venv/bin/python -c "import numpy as np, jax.numpy as jnp, klujax; print(\"solve:\", np.asarray(klujax.solve(jnp.array([0,1],jnp.int32), jnp.array([0,1],jnp.int32), jnp.array([2.,4.]), jnp.array([10.,20.]))))"
    /venv/bin/pytest -q
  '
