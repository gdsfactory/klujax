#!/usr/bin/env bash
# Downstream integration verification: run klujax as a SAX backend.
#
# SAX pins jax<0.10 and has a large dependency tree, so it gets its own venv
# rather than being a core test dependency.
#
# Usage: scripts/verify_sax.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${KLUJAX_SAX_VENV:-/tmp/klujax_sax_venv}"
PYTHON="${PYTHON:-python3.12}"

echo "== build wheel from current sources =="
WHEEL_DIR="$(mktemp -d)"
uv build --wheel --out-dir "$WHEEL_DIR" >/dev/null

echo "== create venv + install sax and the local klujax wheel =="
uv venv "$VENV" --python "$PYTHON" --clear >/dev/null
uv pip install --python "$VENV/bin/python" sax pytest >/dev/null
# Ensure our local build wins over the PyPI klujax pulled in by sax.
uv pip install --python "$VENV/bin/python" --force-reinstall --no-deps "$WHEEL_DIR"/*.whl >/dev/null

echo "== run the SAX integration tests =="
# -c /dev/null avoids the repo's pytest `pythonpath = .` (which would import the
# source tree instead of the installed wheel).
"$VENV/bin/python" -m pytest -c /dev/null -q -p no:cacheprovider \
  "$ROOT/tests_characterization/test_sax_integration.py"
