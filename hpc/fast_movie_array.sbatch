#!/bin/bash
#SBATCH --job-name=fast_movie
#SBATCH --partition=owners,normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --requeue
#SBATCH --open-mode=append
#
# MOVIE STAGE of the preemption-tolerant batch: one array task = one movie
# (row SLURM_ARRAY_TASK_ID of $FAST_RUN/movies.tsv).  Submitted by
# fast_movies_launch.sh and fast_movies_sweep.sbatch -- not by hand.
#
# WHY PER MOVIE. `owners` jobs are preempted at any time.  A per-slide task
# (fast-batch) runs for hours and always forces a full recompute, so a preempted
# slide starts over.  Here a task is one movie (laptop: most 6-15 min, worst
# 1 h 24 m), and `fast` runs WITHOUT -f: detection (per-frame filXYs) and links
# are cached in the movie folder, so a requeued task resumes from the cache.
#
# RESUBMISSION. Preemption on owners is PreemptMode=REQUEUE, and --requeue makes
# Slurm put the task back in the queue itself.  Anything that fails for another
# reason (timeout, node failure) has no done marker and is resubmitted by
# fast_movies_sweep.sbatch.
#
# Sizing (2026-10-16 trial, taka slide_04): ~5.5 GB peak, ~20% CPU efficiency
# on 8 cores -- linking is single-threaded and dominates, so 4 CPUs is plenty.
#
# Done marker: $FAST_RUN/done/<index>, written only when `fast` exits 0.
set -euo pipefail

: "${FAST_RUN:?FAST_RUN (run dir) must be set by the launcher}"
SCRIPT_DIR="${FASTRACK_HPC_DIR:?FASTRACK_HPC_DIR must be set by the launcher}"
source "$SCRIPT_DIR/env.sh"

IDX="$SLURM_ARRAY_TASK_ID"
IFS=$'\t' read -r _ NAME BASE CONFIG MOVIE < <(awk -F'\t' -v i="$IDX" '$1 == i' "$FAST_RUN/movies.tsv")
[ -n "${MOVIE:-}" ] || { echo "no row $IDX in $FAST_RUN/movies.tsv" >&2; exit 1; }

if [ -e "$FAST_RUN/done/$IDX" ]; then
    echo "movie $IDX already done; nothing to do"
    exit 0
fi

CFG_ARGS=()
[ "$CONFIG" != "-" ] && CFG_ARGS=(--config "$CONFIG")

# Per-movie outputs/ are scratch; the real outputs come from the combine step.
WORK="$FAST_RUN/movie_work/$IDX"
mkdir -p "$WORK" "$FAST_RUN/done"
cd "$WORK"

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID} restarts=${SLURM_RESTART_COUNT:-0} idx=$IDX dataset=$NAME"
echo "movie=$MOVIE"
fast -d "$MOVIE" "${CFG_ARGS[@]}" -j "${SLURM_CPUS_PER_TASK:-1}"

echo "$(date -Is) job=${SLURM_JOB_ID} host=$(hostname) $MOVIE" > "$FAST_RUN/done/$IDX"
echo "[$(date -Is)] movie $IDX done"
