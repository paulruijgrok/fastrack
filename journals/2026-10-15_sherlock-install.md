# 2026-10-15 — Installing FASTrack on Sherlock (Stanford HPC)

Mirror of the Notion session log:
<https://app.notion.com/p/3fbdaf1f293b811a998ffaccfa679470>
(hub: **Fastrack** —
<https://app.notion.com/p/39edaf1f293b81179192ea41a1110d6b>).

## TL;DR

- Goal of the Sherlock checkout (`~/code/fastrack`): test the pipeline with
  distributed computing on a cluster without being memory-bound. The code stays
  in sync with `github.com:paulruijgrok/fastrack` (also developed on the laptop).
- **Installed** into a repo `.venv` built from Sherlock's `python/3.12.1` module
  (not conda). Extras: `plus, ridge-fast, batch, test`. **`pytest`: 136 passed,
  5 skipped** on a compute node.
- Three Sherlock-specific fixes were needed: `--prefer-binary` (CentOS 7 /
  glibc 2.17), `opencv-python-headless` (no libGL on compute nodes), and
  imageio-ffmpeg's static ffmpeg (the cluster ffmpeg modules lack libx264).
- `hpc/env.sh` now activates `<repo>/.venv` when it exists and falls back to
  conda otherwise. `hpc/README.md` documents the Sherlock install.

## Detailed narrative

1. **Why conda failed.** The `conda` on PATH is the shared group base
   `/home/groups/possu/miniconda`, and `~/.condarc` points `envs_dirs` at
   `/home/groups/possu/christian/miniconda3/envs`. Neither is ours to write to.
   Sherlock's guidance is to use a module Python plus a venv anyway.
2. **Module Python.** `ml spider python` → `python/3.12.1` (needs `devel`,
   which is already sticky-loaded). Running `ml devel python/3.12.1` silently
   left the system Python 3.6 active, so load only `ml python/3.12.1`.
3. **numpy source build failed** (`no such instruction: endbr64`; the old
   binutils on CentOS 7). The latest numpy has no manylinux2014 wheel.
   `pip install --prefer-binary` resolved numpy 2.2.6, opencv 5.0.0.93,
   scipy 1.16.3, scikit-image 0.25.2, numba 0.67.0 / llvmlite 0.49.0, and
   ridge-detector 0.1.4 + ridge-detector-fast 0.1.0 (git), all as wheels.
4. **`ImportError: libGL.so.1`** on compute nodes from `opencv-python`. Swapped
   to `opencv-python-headless` (same `cv2`). There are no cv2 GUI calls in
   `src/` or `tools/`.
5. **`Unknown encoder 'libx264'`** (one failing test,
   `test_movie.py::test_writer_moves_movie_to_extra_fname`). None of
   `ffmpeg/4.0, 4.2.1, 5.0, 7.1` has libx264. Installed `imageio-ffmpeg` and
   symlinked its static binary to `.venv/bin/ffmpeg`. The writer uses
   `shutil.which("ffmpeg")`, so it is picked up whenever the venv is active.
6. All installs ran in `dev`-partition jobs (`srun -p dev -t 00:45:00 -c 4
   --mem=16G`), with the pip cache in `$SCRATCH/.cache/pip`. `.venv` is 633 MB in
   `$HOME` (4.4 / 15 GB used before).
7. `hpc/env.sh` was verified on a compute node under `set -euo pipefail` (as
   the sbatch scripts source it): it finds the venv python and ffmpeg, and
   `cv2`, `numba`, and `ridge_detector` import.

Exact install commands: see `hpc/README.md` → *One-time setup* → Sherlock.

## Later the same day: preparing the taka redo

Decisions from the user: push the `hpc/` changes straight to `main` (no
Sherlock branch), keep code development on Sherlock minimal, and redo
`fast-batch datasets.taka.csv` on the **normal** partition (CPU only).

1. **Data.** The test data is in `$SCRATCH/fastrack_data` (`taka/`,
   `loaded_motility/`, `unloaded_motility/`). `examples/` in the repo is now a
   symlink to it, so `datasets.taka.csv` and the golden tests work unchanged.
   The symlink is ignored in `.git/info/exclude`, because the `.gitignore`
   pattern `examples/` matches only real directories.
2. **Preflight smoke test** (`fast-batch datasets.taka.csv --preflight-only
   --smoke -j 4`, dev job 48050486) completed.
3. **Laptop cache moved aside.** Detection reuses `filXYs_*.npy` even with
   `-f`. The laptop run's preprocessed-ridge outputs (2,388 files:
   `filXYs_ridge_ppf89a1917*`, `links_ridge_ppf89a1917*`,
   `preprocess_model_f89a1917.npz`, plots and velocity txts) were moved into
   `Pos0/_laptop_run_2026-10-09/` in each movie folder. That location was wrong:
   discovery treats any folder with `filXYs*` files as a movie, so the backups
   doubled the taka count (76 instead of 38). The backups were later moved out of
   the data tree to `$SCRATCH/fastrack_runs/laptop_run_2026-10-09/<slide>/<movie>/Pos0/`
   (38 dirs, 2,388 files). Discovery now finds 38 movies (8 / 7 / 19 / 4).
4. **Tests with data present.** Once `examples/` is populated, the golden-master
   tests run. On a dev job the suite then took more than 10 min (job 48050683,
   TIMEOUT). A rerun with 8 CPUs was refused by the dev QOS
   (`QOSMaxCpuPerUserLimit`; 4 CPUs is the limit). The suite was rerun with 4
   CPUs: **139 passed, 1 failed, 2 skipped** (11 min). The failure,
   `test_stack_pipeline.py::test_stack_equals_frames_with_matched_timing`, was
   caused by the data: copying from the Mac left 8,258 AppleDouble `._*` files (4 KB
   each), and `discovery.py` treats `stacks/.../.__1.tif` as stack movies (8
   stacks vs 4 frame movies). With the user's approval, all `._*` files under
   `$SCRATCH/fastrack_data` were deleted. Rerun (dev job 48081065): **140 passed,
   2 skipped** (optomerge not installed), 14 min, max RSS 0.7 GB.
5. **opencv.** The venv has only `opencv-python-headless` 5.0.0.93 installed.
   `pyproject.toml` still lists `opencv-python`; it was left unchanged to keep
   the laptop setup as it is (see open items).

## Files / commits

| File | Change |
| --- | --- |
| `hpc/env.sh` | venv-first activation (`FASTRACK_VENV`, `PYTHON_MODULE`), conda fallback, prints python/ffmpeg in use |
| `hpc/README.md` | Sherlock install recipe and the reasons behind each step |
| `journals/2026-10-15_sherlock-install.md` | this journal |
| `.venv/` (git-ignored) | Python 3.12.1 venv, 633 MB |

## Open items / next steps

- Future Mac → Sherlock copies: exclude AppleDouble files (`rsync --exclude='._*'`,
  or `COPYFILE_DISABLE=1` for tar). Optionally make `discovery.py` skip dotfiles.

- `pyproject.toml` still pins `opencv-python`. Consider an extra or a note so
  that `pip install -e .` on a headless cluster doesn't pull in the GUI build.
  (Re-running `pip install -e .` here may reinstall `opencv-python` alongside
  headless; if `libGL` errors return, uninstall it again.)
- The sbatch scripts use `--partition=possu`; the user wants `normal` for
  this project. Override with `sbatch -p normal` or change the scripts.
- **Submitted: taka redo, array job `48083434` (tasks 0-3, one slide each)** — failed in 8 s (spooled-script `env.sh` path); superseded by the per-movie run, see `2026-10-16_owners-per-movie-batch.md`.
  Run dir `$SCRATCH/fastrack_runs/taka_redo_2026-10-15/` (contains copies of
  `datasets.taka.csv` and `config.taka.toml` plus an `examples` symlink to the
  data). Command: `sbatch -p normal -t 12:00:00 --array=0-3
  hpc/fast_batch_array.sbatch datasets.taka.csv`, with 16 CPUs / 32 GB per task.
  Laptop reference: slide_03 took 3 h 31 m. Next session: `sacct -j 48083434`,
  `seff` per task, then compare `outputs/*/combined/MEAN_values.txt` with the laptop run.
- (Plan, kept for reference.) **The taka redo on `normal`.** Run `fast-batch datasets.taka.csv` (array
  or a single job) on `normal` with an explicit time limit, from a run dir under
  `$SCRATCH/fastrack_runs/`. Measure memory per task with `seff`, then compare
  against the laptop results in `$SCRATCH/fastrack_runs/laptop_run_2026-10-09/`.
- `~/.bashrc` auto-activates the possu conda base. This is harmless for jobs
  (the venv is prepended), but don't stack `.venv` on a conda env interactively.
