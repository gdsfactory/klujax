#!/usr/bin/env bash
# Cross-platform verification: build klujax-ffi on Linux (via Docker) and confirm
# the cdylib statically links SuiteSparse (no dynamic klu/suitesparse).
#
# The SuiteSparse submodule is bind-mounted read-only; build artifacts go to a
# named Docker volume so the host tree is untouched.
#
# Usage: scripts/verify_linux.sh
# Override the image with KLUJAX_LINUX_IMAGE (default: rust:1.85-bookworm).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${KLUJAX_LINUX_IMAGE:-rust:1.85-bookworm}"

docker run --rm \
  -v "$ROOT":/src:ro \
  -v klujax_target:/target \
  -e CARGO_TARGET_DIR=/target \
  "$IMAGE" bash -c '
    set -euo pipefail
    export PATH="/usr/local/cargo/bin:$PATH"
    cd /src
    echo "== cc: $(cc --version | head -1)"
    echo "== cargo: $(cargo --version)"
    cargo build --release -p klujax-ffi
    if ldd /target/release/libklujax_ffi.so | grep -iE "klu|suitesparse|libamd|libcolamd|libbtf"; then
      echo "FAIL: cdylib has a dynamic SuiteSparse dependency" >&2
      exit 1
    fi
    echo "OK: Linux cdylib statically links SuiteSparse"
    echo "handlers: $(nm -D /target/release/libklujax_ffi.so | grep -cE " T (solve|dot|factor|refactor|tsolve|free|analyze)")"
  '
