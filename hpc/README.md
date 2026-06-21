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

1. Install FASTrack into a conda env on the cluster (login node is fine):
   ```bash
   conda create -n fastrack-ridge python=3.11 -y && conda activate fastrack-ridge
   pip install -e '.[ridge-fast,batch]'        # drop extras you don't need
   ```
2. Edit `hpc/env.sh` if your env name isn't `fastrack-ridge` (set `CONDA_ENV`) or
   if conda comes from a module (`module load ...`).
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
