#!/bin/bash
# Sourced by the FASTrack SLURM scripts to activate the Python environment on a
# compute node.  Edit once here, and every script picks it up.
#
# Two supported layouts:
#   1. venv (default when <repo>/.venv exists, or FASTRACK_VENV is set):
#      load the cluster's Python module, then activate the venv.  This is the
#      Sherlock setup (python/3.12.1 module + repo .venv; see hpc/README.md).
#   2. conda (fallback): source "$(conda info --base)/etc/profile.d/conda.sh"
#      && conda activate $CONDA_ENV.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTRACK_VENV="${FASTRACK_VENV:-$REPO_DIR/.venv}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12.1}"
CONDA_ENV="${CONDA_ENV:-fastrack-ridge}"

if [ -f "$FASTRACK_VENV/bin/activate" ]; then
    if command -v ml >/dev/null 2>&1; then
        ml purge 2>/dev/null
        ml "$PYTHON_MODULE"
    fi
    # shellcheck disable=SC1091
    source "$FASTRACK_VENV/bin/activate"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

echo "[env.sh] python: $(command -v python) ($(python --version 2>&1)); ffmpeg: $(command -v ffmpeg || echo none)"
