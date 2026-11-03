#!/bin/bash
#SBATCH --job-name=fast_preview
#SBATCH --partition=owners,normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --requeue
#
# PREVIEW BATCH: `fast-preview` on several folders in one job, gathered into one
# browsable place.  Preview takes seconds per movie (Taka: ~6 s on 2 cores), so
# one job runs everything; a preempted job simply reruns from the start.
#
#   PREVIEW_OUT=$SCRATCH/fastrack_runs/<run> \
#     sbatch -o "$PREVIEW_OUT/fast_preview_%j.log" hpc/fast_preview_batch.sbatch <folder>...
#
# Each <folder> is a slide (or a single movie); extra fast-preview flags can go
# in PREVIEW_ARGS (e.g. "--config config.toml -n 40").  Output:
#   $PREVIEW_OUT/preview_summary_all.csv     every movie, with a `folder` column
#   $PREVIEW_OUT/dashboards/<folder>__<movie>_preview.png   (all dashboards, flat)
#   $PREVIEW_OUT/per_slide/<folder>/         fast-preview's own output (json, csv, png)
set -euo pipefail

: "${PREVIEW_OUT:?set PREVIEW_OUT to the output directory}"
[ "$#" -gt 0 ] || { echo "usage: sbatch fast_preview_batch.sbatch <folder>..." >&2; exit 1; }
# Slurm runs a spooled copy of this script; find the repo via the job's Command=.
SCRIPT_DIR="${FASTRACK_HPC_DIR:-$(dirname "$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')")}"
source "$SCRIPT_DIR/env.sh"

mkdir -p "$PREVIEW_OUT/per_slide" "$PREVIEW_OUT/dashboards"
for d in "$@"; do
    name=$(basename "$d")
    echo "[$(date -Is)] preview $name"
    # shellcheck disable=SC2086
    fast-preview "$d" -o "$PREVIEW_OUT/per_slide/$name" -j "${SLURM_CPUS_PER_TASK:-1}" \
        ${PREVIEW_ARGS:-} < /dev/null || echo "[$(date -Is)] fast-preview failed on $name" >&2
done

python - "$PREVIEW_OUT" <<'EOF'
import csv, glob, os, shutil, sys
out = sys.argv[1]
rows, header = [], None
for csv_path in sorted(glob.glob(os.path.join(out, "per_slide", "*", "preview_summary.csv"))):
    folder = os.path.basename(os.path.dirname(csv_path))
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        h = next(r)
        header = header or ["folder"] + h
        rows += [[folder] + row for row in r]
    for png in glob.glob(os.path.join(os.path.dirname(csv_path), "*_preview.png")):
        # fast-preview names a movie relative to the folder's parent, so the
        # folder is usually in the name already (slide_04__WT_1_preview.png).
        base = os.path.basename(png)
        if not base.startswith(folder + "__"):
            base = folder + "__" + base
        shutil.copy2(png, os.path.join(out, "dashboards", base))
with open(os.path.join(out, "preview_summary_all.csv"), "w", newline="") as f:
    w = csv.writer(f)
    if header:
        w.writerow(header)
    w.writerows(rows)
print("%d movies -> %s/preview_summary_all.csv; %d dashboards"
      % (len(rows), out, len(os.listdir(os.path.join(out, "dashboards")))))
EOF
echo "[$(date -Is)] done"
