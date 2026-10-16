#!/bin/bash
#SBATCH --job-name=fast_sweep
#SBATCH --partition=owners,normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH --open-mode=append
#
# SWEEP + COMBINE STAGE of the preemption-tolerant batch.  Runs after the movie
# array has left the queue (--dependency=afterany, set by the submitter).
#
#   1. Movies without a done marker (timed out, node failure, ... -- preemption
#      itself is already requeued by Slurm) are resubmitted as a new array,
#      followed by a new sweep that depends on it.  Retry rounds get
#      RETRY_TIME (default 04:00:00), in case the miss was a timeout.  Capped at
#      MAX_ROUNDS (default 5); then this job fails and lists what is missing.
#   2. When every movie is done: `fast -d <dataset>` once per dataset, without
#      -f, from $FAST_RUN.  Detection and links come from the movie caches, so
#      this only redoes paths/statistics and writes outputs/*/combined/.
#      Then $FAST_RUN/COMPLETE is written.
#
# This replaces a login-node controller loop: Slurm does the waiting, through
# job dependencies, and nothing runs on the login node after the launch.
set -euo pipefail

: "${FAST_RUN:?FAST_RUN (run dir) must be set by the launcher}"
SCRIPT_DIR="${FASTRACK_HPC_DIR:?FASTRACK_HPC_DIR must be set by the launcher}"
ROUND="${SWEEP_ROUND:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-5}"
RETRY_TIME="${RETRY_TIME:-04:00:00}"
LOG="$FAST_RUN/controller.log"
log() { echo "[$(date -Is)] [sweep $ROUND] $*" | tee -a "$LOG"; }

# Collapses sorted ints into Slurm --array syntax: (0 1 2 5 7 8) -> 0-2,5,7-8
join_ranges() {
    local start="" prev="" out=""
    for n in "$@"; do
        if [ -z "$start" ]; then start=$n; prev=$n
        elif [ "$n" -eq $((prev + 1)) ]; then prev=$n
        else out+="${out:+,}$start"; [ "$start" != "$prev" ] && out+="-$prev"; start=$n; prev=$n
        fi
    done
    [ -n "$start" ] && { out+="${out:+,}$start"; [ "$start" != "$prev" ] && out+="-$prev"; }
    echo "$out"
}

missing=()
while IFS=$'\t' read -r idx _; do
    [ -e "$FAST_RUN/done/$idx" ] || missing+=("$idx")
done < "$FAST_RUN/movies.tsv"
total=$(wc -l < "$FAST_RUN/movies.tsv")

if [ "${#missing[@]}" -gt 0 ]; then
    spec=$(join_ranges "${missing[@]}")
    if [ "$ROUND" -ge "$MAX_ROUNDS" ]; then
        log "GIVING UP: ${#missing[@]}/$total movies still not done after $MAX_ROUNDS rounds: $spec -- check movie_logs/"
        exit 1
    fi
    log "${#missing[@]}/$total movies not done ($spec); resubmitting with --time=$RETRY_TIME"
    cd "$FAST_RUN"
    arr=$(sbatch --parsable --array="$spec" --time="$RETRY_TIME" \
        -o "movie_logs/fast_movie_%A_%a.log" "$SCRIPT_DIR/fast_movie_array.sbatch")
    nxt=$(sbatch --parsable --dependency="afterany:$arr" --export=ALL,SWEEP_ROUND=$((ROUND + 1)) \
        -o "fast_sweep_%j.log" "$SCRIPT_DIR/fast_movies_sweep.sbatch")
    log "movie array $arr, next sweep $nxt"
    exit 0
fi

log "all $total movies done; combining per dataset"
source "$SCRIPT_DIR/env.sh"
cd "$FAST_RUN"
cut -f2-4 movies.tsv | sort -u | while IFS=$'\t' read -r name base config; do
    cfg=(); [ "$config" != "-" ] && cfg=(--config "$config")
    log "combine $name"
    fast -d "$base" "${cfg[@]}" -j "${SLURM_CPUS_PER_TASK:-1}" > "combine_logs/${name}.log" 2>&1
done
date -Is > "$FAST_RUN/COMPLETE"
log "campaign complete -- outputs under $FAST_RUN/outputs"
