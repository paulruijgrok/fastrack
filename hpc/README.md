# Running FASTrack on a SLURM cluster

Thin `sbatch` wrappers around the `fast` and `fast-batch` commands. They follow
the cluster's usual idiom (conda activation via `conda info --base`,
`--partition=possu`, `$SLURM_CPUS_PER_TASK` passed to the analysis), so the only
things you normally edit are the resource lines and `env.sh`.

The analysis is CPU-bound and parallel both *within* a dataset (per-frame
multiprocessing, `-j`) and *across* datasets (one shard per array task). For why
this is the right model — and what a GPU would and wouldn't buy you — see
[`docs/gpu_feasibility.md`](../docs/gpu_feasibility.md).

## One-time setup

1. Install FASTrack into a Python environment on the cluster.

   **Sherlock (Stanford) — module Python + repo `.venv`** (what `env.sh` uses by
   default). Run the install inside a job, not on a login node:
   ```bash
   srun -p dev -t 00:45:00 -c 4 --mem=16G --pty bash
   ml purge; ml python/3.12.1                     # don't `ml devel` again: it's sticky
   export PIP_CACHE_DIR=$SCRATCH/.cache/pip       # keep pip's cache out of $HOME
   python3 -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install --prefer-binary -e '.[plus,ridge-fast,batch,test]'
   # compute nodes have no libGL -> use the headless OpenCV build (same `cv2`)
   pip uninstall -y opencv-python && pip install --prefer-binary opencv-python-headless
   # Sherlock's ffmpeg modules lack libx264 -> use imageio-ffmpeg's static build
   pip install --prefer-binary imageio-ffmpeg
   ln -sf "$(python -c 'import imageio_ffmpeg as f; print(f.get_ffmpeg_exe())')" .venv/bin/ffmpeg
   pytest -q                                      # expect all pass (a few optional skips)
   ```
   Why these steps: Sherlock runs CentOS 7 (glibc 2.17), so the newest
   numpy/opencv wheels don't apply and pip would try (and fail) to compile them
   from source. `--prefer-binary` picks the newest version that has a compatible
   wheel. The shared group conda base isn't writable, which is why we skip conda.

   **Elsewhere — conda:**
   ```bash
   conda create -n fastrack-ridge python=3.11 -y && conda activate fastrack-ridge
   pip install -e '.[ridge-fast,batch]'        # drop extras you don't need
   ```
2. `hpc/env.sh` activates `<repo>/.venv` (after `ml $PYTHON_MODULE`) if it
   exists, and otherwise falls back to conda env `$CONDA_ENV` (default
   `fastrack-ridge`). You can override `FASTRACK_VENV`, `PYTHON_MODULE`, or
   `CONDA_ENV` in the environment.
3. In each `.sbatch`, set `--partition` / `--account` for your allocation and,
   optionally, uncomment the `--mail-user` line.

Defaults baked in: **16 CPUs, 32 GB, 8 h** per task — bump them for big datasets.

## One dataset

```bash
sbatch hpc/fast_single.sbatch /scratch/users/$USER/data/mydataset
# extra fast args pass straight through:
sbatch hpc/fast_single.sbatch <datadir> --detector ridge-fast --export-trajectories
sbatch --cpus-per-task=32 hpc/fast_single.sbatch <datadir>     # more cores
```

## Many datasets (job array)

Make a manifest (see `../datasets.example.csv`) — one row per dataset, columns
`base_dir`, optional `config`, optional `name`. Then size and submit the array:

```bash
N=$(python hpc/manifest_count.py datasets.csv)     # number of datasets
# choose T <= N array tasks, edit  #SBATCH --array=0-(T-1)  in the script, then:
sbatch hpc/fast_batch_array.sbatch datasets.csv
```

Each array task processes a contiguous **shard** of the manifest with its own
state file (`fastrack_batch_logs/batch_state_shard<i>.json`), so tasks run
concurrently without clobbering each other. `NUM_SHARDS` is derived from the
`--array` range automatically — you only set the range in one place.

## Many movies on a preemptible partition (Sherlock `owners`)

A `fast-batch` task is a whole dataset (hours) and always forces a full
recompute, so on a preemptible partition a preempted task starts over. The
`fast_movies_*` scripts split the work into **one short task per movie**:

```bash
cd $SCRATCH/fastrack_runs/<run>            # manifest + config (+ data symlink) here
bash ~/code/fastrack/hpc/fast_movies_launch.sh datasets.csv
tail $SCRATCH/fastrack_runs/<run>/controller.log     # progress; COMPLETE when done
```

- `fast_movie_array.sbatch` — one array task per movie (`movies.tsv`, built by
  `manifest_movies.py`) on `owners,normal`, 2 h, 4 CPUs, 12 GB. Runs `fast`
  **without `-f`**: detection and links are cached in the movie folder (written
  atomically), so a requeued task resumes. Preemption is requeued by Slurm itself
  (`--requeue`, owners `PreemptMode=REQUEUE`). Writes `done/<index>` on success.
- `fast_movies_sweep.sbatch` — runs after the array (`afterany`). Resubmits
  movies without a done marker plus a new sweep (`RETRY_TIME` 4 h,
  `MAX_ROUNDS` 5), then runs `fast -d <dataset>` per dataset from the cache to
  write `outputs/*/combined/` (~2 min per slide), and `COMPLETE`.
- Nothing runs on the login node after the launch: no controller loop, Slurm
  dependencies do the waiting. Re-launching in the same run dir only redoes
  movies without a done marker.

## Restart / resume

Both scripts use `--requeue`, and `fast-batch` skips datasets already completed
successfully (per-shard state). So a preempted, timed-out, or re-submitted job
picks up where it left off — just `sbatch` the same command again. Use
`--force` (passed through) to recompute everything, or `--retry-failed` to retry
only the datasets that errored.

## Where things go

- **Analysis outputs:** `./outputs/...` under the directory you submit from, or
  set `FAST_WORKDIR=/scratch/users/$USER/results` to run there instead.
- **fast-batch logs:** `./fastrack_batch_logs/` — a run log plus a per-dataset
  log capturing the full pipeline output (ffmpeg/OpenCV included).
- **SLURM logs:** `fast_%j.out` / `fast_batch_%A_%a.out` in the submission dir.

## Quick checklist before a big overnight array

```bash
# 1. dry-run the checks over the whole list (no heavy work)
fast-batch datasets.csv --preflight-only --smoke
# 2. size the array from the dataset count, set --array in the script
python hpc/manifest_count.py datasets.csv
# 3. submit
sbatch hpc/fast_batch_array.sbatch datasets.csv
```
