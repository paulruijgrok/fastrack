#!/bin/bash
# Launch a preemption-tolerant fast batch: one short Slurm task per movie on
# owners,normal, then a sweep job that resubmits stragglers and combines.
#
#   cd $SCRATCH/fastrack_runs/<run>          # run dir: manifest + config + data links
#   bash ~/code/fastrack/hpc/fast_movies_launch.sh datasets.csv
#
# One-shot: builds movies.tsv, submits the movie array and the first sweep
# (--dependency=afterany), and exits.  Nothing keeps running on the login node;
# requeue on preemption is Slurm's own (owners: PreemptMode=REQUEUE), and later
# rounds are submitted by fast_movies_sweep.sbatch from inside a job.
#
# Progress:   tail $FAST_RUN/controller.log ; ls $FAST_RUN/done | wc -l
# Finished:   $FAST_RUN/COMPLETE exists; results in $FAST_RUN/outputs/
# Re-launching in the same run dir only redoes movies without a done marker.
#
# Overrides (env): MOVIE_ARRAY_ARGS (extra sbatch args for the movie array,
# e.g. "--time=03:00:00"), MAX_ROUNDS, RETRY_TIME (see the sweep script).
set -euo pipefail

MANIFEST="${1:?usage: fast_movies_launch.sh <manifest.csv>}"
export FASTRACK_HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FAST_RUN="$(pwd)"

if [ -n "$(squeue --me -h -n fast_movie,fast_sweep -o '%i %Z' | awk -v d="$FAST_RUN" '$2 == d')" ]; then
    echo "fast_movie/fast_sweep jobs from $FAST_RUN are already queued; not launching twice." >&2
    exit 1
fi

source "$FASTRACK_HPC_DIR/env.sh"
python "$FASTRACK_HPC_DIR/manifest_movies.py" "$MANIFEST" > movies.tsv.new
mv movies.tsv.new movies.tsv
N=$(wc -l < movies.tsv)
mkdir -p done movie_logs movie_work combine_logs

# shellcheck disable=SC2086
arr=$(sbatch --parsable --array="0-$((N - 1))" ${MOVIE_ARRAY_ARGS:-} \
    -o "movie_logs/fast_movie_%A_%a.log" "$FASTRACK_HPC_DIR/fast_movie_array.sbatch")
sweep=$(sbatch --parsable --dependency="afterany:$arr" --export=ALL,SWEEP_ROUND=1 \
    -o "fast_sweep_%j.log" "$FASTRACK_HPC_DIR/fast_movies_sweep.sbatch")
echo "[$(date -Is)] [launch] $N movies from $MANIFEST: movie array $arr, sweep $sweep" | tee -a controller.log
