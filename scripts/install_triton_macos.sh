#!/usr/bin/env bash
# Build Triton from source on macOS so TRITON_INTERPRET=1 works on CPU tensors.
# PyPI only ships Linux triton wheels, and the interpreter still needs libtriton.
# On a Linux/CUDA box, skip this and use the triton wheel that matches your torch.
set -euo pipefail

TRITON_TAG="${TRITON_TAG:-v3.9.0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPS_DIR="${DEPS_DIR:-$ROOT/.deps}"
PY="${PY:-$ROOT/.venv/bin/python}"
SRC="$DEPS_DIR/triton-$TRITON_TAG"

mkdir -p "$DEPS_DIR"
if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch "$TRITON_TAG" https://github.com/triton-lang/triton.git "$SRC"
fi

"$PY" -m pip install -r "$SRC/python/requirements.txt"

cd "$SRC"
# setup.py shells out to cmake/ninja; use the ones pip put in the venv.
export PATH="$(dirname "$PY"):$PATH"
# LLVM and toolchain downloads go under .deps instead of ~/.triton.
export TRITON_HOME="$DEPS_DIR/triton-home"
export TRITON_BUILD_PROTON=OFF
export TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF"
export MAX_JOBS="${MAX_JOBS:-10}"
"$PY" -m pip install --no-build-isolation -v .

# Post-build check: launch a kernel on a CPU tensor, not just import triton.
TRITON_INTERPRET=1 "$PY" - <<'EOF'
import torch
import triton
import triton.language as tl


@triton.jit
def add_one(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=offs < n) + 1.0, mask=offs < n)


x = torch.zeros(10)
add_one[(1,)](x, 10, BLOCK=16)
assert torch.equal(x, torch.ones(10)), x
print("triton", triton.__version__, "interpreter OK")
EOF
