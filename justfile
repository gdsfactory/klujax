# klujax justfile

# List all available commands
list:
    just --list

# Build the Rust cdylib (release)
rust-build:
    cargo build --release -p klujax-ffi

# Run the Rust test suite
rust-test:
    cargo test --workspace

# Smoke-test the Rust cdylib ABI surface with ctypes
rust-smoke: rust-build
    python scripts/ffi_smoke.py

# Format the Rust code
rust-fmt:
    cargo fmt --all

# Lint the Rust code
rust-clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Remove Rust build artifacts
rust-clean:
    cargo clean

# Set up development environment (clones dependencies first)
dev: maybe-deps bver
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

# (Re-)initialize dependencies
deps: suitesparse xla pybind11

# Initialize missing dependencies only
maybe-deps:
    @if [ ! -d "suitesparse" ]; then just suitesparse; fi
    @if [ ! -d "xla" ]; then just xla; fi
    @if [ ! -d "pybind11" ]; then just pybind11; fi

# Build extension in place
inplace:
    uv run python setup.py build_ext --inplace

# Run tests
test:
    uv run pytest

# Regenerate the golden corpus (requires the C++ extension)
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

# Clone SuiteSparse
suitesparse:
    rm -rf suitesparse
    git clone --depth 1 --branch v7.5.0 https://github.com/DrTimothyAldenDavis/SuiteSparse suitesparse || true
    cd suitesparse && rm -rf .git

# Clone XLA
xla:
    rm -rf xla
    git clone https://github.com/openxla/xla xla
    cd xla && git checkout 05f004e8368c955b872126b1c978c60e33bbc5c8 && rm -rf .git

# Clone pybind11
pybind11:
    rm -rf pybind11
    git clone --depth 1 --branch v2.13.6 https://github.com/pybind/pybind11 pybind11
    cd pybind11 && rm -rf .git

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
    rm -rf suitesparse
    rm -rf xla
    rm -rf pybind11
    rm uv.lock

bump version="patch":
    bver bump "{{ version }}"
