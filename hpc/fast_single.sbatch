#!/bin/bash
#SBATCH --job-name=fast
#SBATCH --output=fast_%j.out
#SBATCH --error=fast_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=possu
##SBATCH --account=<your_account>      # uncomment + set if your cluster requires an allocation
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=you@stanford.edu  # uncomment + set to get notified
#SBATCH --requeue
#
# Run the FAST analysis on ONE dataset directory.
#
# Usage:
#   sbatch hpc/fast_single.sbatch /scratch/users/ruijgrok/data/mydataset
#   sbatch hpc/fast_single.sbatch <datadir> --detector ridge-fast --export-trajectories
#   sbatch --cpus-per-task=32 hpc/fast_single.sbatch <datadir>     # override cores
#
# Outputs land in ./outputs/ under the directory you submit from (or set
# FAST_WORKDIR=/scratch/.../results to run there).
set -euo pipefail

DATADIR="${1:?usage: sbatch fast_single.sbatch <dataset_dir> [extra fast args]}"
shift || true

# Slurm runs a spooled copy of this script, so $0 is not in the repo: locate the
# original via the job's Command= (override with FASTRACK_HPC_DIR=<repo>/hpc).
if [ -n "${FASTRACK_HPC_DIR:-}" ]; then
    SCRIPT_DIR="$FASTRACK_HPC_DIR"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_PATH="$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -n 1)"
    case "$SCRIPT_PATH" in /*) ;; *) SCRIPT_PATH="${SLURM_SUBMIT_DIR}/${SCRIPT_PATH}" ;; esac
    SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
else
    SCRIPT_DIR="$(dirname "$(realpath "$0")")"
fi
source "$SCRIPT_DIR/env.sh"

if [ -n "${FAST_WORKDIR:-}" ]; then
    mkdir -p "$FAST_WORKDIR" && cd "$FAST_WORKDIR"
fi

echo "host=$(hostname)  cpus=${SLURM_CPUS_PER_TASK:-1}  datadir=$DATADIR  cwd=$(pwd)"
fast -d "$DATADIR" -j "${SLURM_CPUS_PER_TASK:-1}" "$@"
