# klujax justfile

# List all available commands
list:
    just --list

# Initialize the vendored SuiteSparse submodule
submodule:
    git submodule update --init --recursive

# Build the Rust cdylib (release)
rust-build:
    cargo build --release -p klujax-ffi

# Run the Rust test suite
rust-test:
    cargo test --workspace

# Smoke-test the Rust cdylib ABI surface with ctypes
rust-smoke: rust-build
    python scripts/ffi_smoke.py

# Verify the cdylib statically links SuiteSparse (no dynamic klu dependency)
static-link-check: rust-build
    python scripts/check_static_link.py

# Enforce the unsafe-code budget (see work.md Stage 0)
unsafe-budget:
    tools/unsafe_budget.sh

# Run miri on the pure-Rust decode/COO-CSC subset (requires nightly + miri)
miri:
    cargo +nightly miri test -p klujax-ffi --lib

# Cross-platform verification: build + static-link check on Linux via Docker
verify-linux:
    bash scripts/verify_linux.sh

# Full Linux verification: build from source + run pytest in Docker
verify-linux-tests:
    bash scripts/verify_linux_tests.sh

# Downstream integration: run klujax as a SAX backend (dedicated venv)
verify-sax:
    bash scripts/verify_sax.sh

# Format the Rust code
rust-fmt:
    cargo fmt --all

# Lint the Rust code
rust-clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Remove Rust build artifacts
rust-clean:
    cargo clean

# Set up development environment (initializes the SuiteSparse submodule)
dev: submodule bver
    uv venv --python 3.13 --clear
    uv sync --all-extras --all-groups --upgrade
    uv run python setup.py build_ext --inplace

# Build distribution
dist:
    uv run python setup.py build sdist bdist_wheel

# Version bumping
[linux,macos]
bver:
    curl -LsSf https://github.com/flaport/bver/releases/latest/download/install.sh | sh

# Version bumping
[windows]
bver:
    powershell -ExecutionPolicy ByPass -c "irm https://github.com/flaport/bver/releases/latest/download/install.ps1 | iex"

# Build extension in place
inplace:
    uv run python setup.py build_ext --inplace

# Run tests
test:
    uv run pytest

# Regenerate the golden corpus (uses the Rust backend)
golden:
    PYTHONPATH=. uv run python tests_characterization/_generate_golden.py

# Run tests with coverage
coverage:
    uv run pytest --cov=klujax --cov-report=term-missing

# Build docs
docs:
    uv run mkdocs build

# Serve docs locally
serve:
    uv run mkdocs serve -a localhost:8080

# Clean build artifacts
clean:
    rm -rf .venv
    cargo clean
    find . -name "dist" | xargs rm -rf
    find . -name "build" | xargs rm -rf
    find . -name "builds" | xargs rm -rf
    find . -name "__pycache__" | xargs rm -rf
    find . -name "*.so" | xargs rm -rf
    find . -name "*.dylib" | xargs rm -rf
    find . -name "*.egg-info" | xargs rm -rf
    find . -name ".ipynb_checkpoints" | xargs rm -rf
    find . -name ".pytest_cache" | xargs rm -rf

# Clean everything
clean-all: clean
    rm uv.lock

bump version="patch":
    bver bump "{{ version }}"
