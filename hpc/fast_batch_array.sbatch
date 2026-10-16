#!/bin/bash
#SBATCH --job-name=fast_batch
#SBATCH --output=fast_batch_%A_%a.out
#SBATCH --error=fast_batch_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=possu
##SBATCH --account=<your_account>      # uncomment + set if your cluster requires an allocation
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=you@stanford.edu  # uncomment + set to get notified
#SBATCH --requeue
#SBATCH --array=0-3                     # T tasks (indices 0..T-1); see sizing note below
#
# Process a whole dataset MANIFEST as a chunked job array.  Each array task runs
# fast-batch on a contiguous slice (shard) of the manifest, on its own node with
# its own per-shard state file -- so tasks run concurrently and never clobber
# each other.  Restartable: a requeued/re-submitted task skips datasets already
# marked done in its shard state.
#
# SIZING -- pick T = number of array tasks:
#   N=$(python hpc/manifest_count.py datasets.csv)   # number of datasets
#   choose T <= N (e.g. T=N for max concurrency, or fewer if N is large), then
#   edit "#SBATCH --array=0-$((T-1))" above.  NUM_SHARDS is derived from the
#   array range automatically, so you only set it in one place.
#
# Usage:
#   sbatch hpc/fast_batch_array.sbatch datasets.csv
#   sbatch hpc/fast_batch_array.sbatch datasets.csv --detector ridge-fast
set -euo pipefail

MANIFEST="${1:?usage: sbatch fast_batch_array.sbatch <manifest.csv> [extra fast-batch args]}"
shift || true

# Number of shards = size of the array range (keeps --array and sharding in sync).
NUM_SHARDS="${NUM_SHARDS:-$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))}"

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

echo "host=$(hostname)  shard=${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}  cpus=${SLURM_CPUS_PER_TASK:-1}  manifest=$MANIFEST  cwd=$(pwd)"
fast-batch "$MANIFEST" \
    -j "${SLURM_CPUS_PER_TASK:-1}" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --logdir "fastrack_batch_logs" \
    "$@"
