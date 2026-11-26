#!/usr/bin/env bash
# Environment activation script for real2sim-layered (hosted on handzero PFS)

export REAL2SIM_ROOT="${REAL2SIM_ROOT:-/home/wangziyu-s/pfs/project/handzero/real2sim-layered}"
export UV_BIN="${UV_BIN:-$HOME/pfs/personal/tools/py311/bin/uv}"
export UV_PYTHON="${UV_PYTHON:-$HOME/pfs/personal/tools/py311/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11}"

export UV_CACHE_DIR="$REAL2SIM_ROOT/cache/uv"
export UV_PROJECT_ENVIRONMENT="$REAL2SIM_ROOT/.venv"
export UV_LINK_MODE="hardlink"
export HF_HOME="$REAL2SIM_ROOT/cache/hf"
export TORCH_HOME="$REAL2SIM_ROOT/cache/torch"
export TMPDIR="$REAL2SIM_ROOT/tmp"

export BLENDER_HOME="$REAL2SIM_ROOT/tools/blender-4.2.9-linux-x64"
export BLENDER_BIN="$BLENDER_HOME/blender"
export SYSLIBS_DIR="$REAL2SIM_ROOT/tools/syslibs"
export LD_LIBRARY_PATH="$SYSLIBS_DIR/usr/lib/x86_64-linux-gnu:$SYSLIBS_DIR/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

export PATH="$REAL2SIM_ROOT/.venv/bin:$BLENDER_HOME:$HOME/pfs/personal/tools/py311/bin:$PATH"
