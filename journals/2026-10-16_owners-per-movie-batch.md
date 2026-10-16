# 2026-10-16 — Taka redo on `owners`: per-movie, preemption-tolerant batch

(hub: **Fastrack** — <https://app.notion.com/p/39edaf1f293b81179192ea41a1110d6b>)

## TL;DR

- The 2026-10-15 array job `48083434` failed in 8 s: Slurm runs a spooled copy
  of the batch script, so `$(dirname $0)/env.sh` did not exist. Fixed in
  `fast_batch_array.sbatch` / `fast_single.sbatch` (locate the script via
  `scontrol show job` → `Command=`, or `FASTRACK_HPC_DIR`).
- At the user's request, the run moved to the preemptible **`owners`** partition
  (41 idle nodes vs 0 on `normal`). A per-slide `fast-batch` task (up to 3.5 h, always forced) restarts from
  zero when preempted, so a new **per-movie** workflow was added:
  `hpc/fast_movies_launch.sh` → `fast_movie_array.sbatch` (1 task = 1 movie) →
  `fast_movies_sweep.sbatch` (resubmit stragglers, then combine per slide).
- Cache writes (`filXYs`, `links`) are now **atomic** (`fastrack.io.atomic`), so
  a preempted task cannot leave a truncated cache file for its requeue to trip on.
- Trial on slide_04 (4 movies), including a simulated preemption
  (`scontrol requeue`): resumed from cache, combine 1.5 min, condition means
  **E11K 1279 / WT 1402 nm/s = laptop run exactly**. `pytest`: 140 passed, 2 skipped.
- **Full run launched:** `$SCRATCH/fastrack_runs/taka_movies_2026-10-16/`,
  movie array `48153343` (38 tasks), sweep `48153345`.

## Detailed narrative

1. **Why the 10-15 job failed.** `fast_batch_48083434_*.err`:
   `/var/spool/slurmd/job…/env.sh: No such file or directory`. The previous
   session drafted the `scontrol`-based fix and verified it with a 2-task probe
   array. It then started pytest on the login node and was cut off. This session
   ran pytest in a job instead (`48131799`: 140 passed, 2 skipped, 20 min on 4 CPUs).
2. **Resubmitted and switched to owners.** The 10-15 command was resubmitted
   (`48131798`), then switched with `scontrol update Partition=owners,normal`.
   The user then pointed out that owners jobs are preempted, so tasks must be
   short and must be resubmitted until they complete (in the style of
   dx_binder_design's `*_resubmit_until_done.sh`).
3. **Not a login-node controller.** dx_binder_design's controller is a `nohup`
   loop on the login node that polls `squeue`. Sherlock's rules prohibit that
   (unattended, outlives the session, polls the scheduler). Instead:
   - preemption → Slurm requeues by itself (`scontrol show partition owners`:
     `PreemptMode=REQUEUE`; cluster `JobRequeue=1`; scripts set `--requeue`);
   - other failures (timeout, node failure) → `fast_movies_sweep.sbatch`, run with
     `--dependency=afterany` on the array. It resubmits the indices without a
     `done/<i>` marker plus a new sweep (retry time 4 h, at most 5 rounds).
     When every movie is done, it runs `fast -d <slide>` per slide.
4. **Why per movie works.** Without `-f`, `gliding.run` skips detection for
   frames that are already in the cache (`store.has`) and loads `links<tag>.npy`
   (`Motility.read_frame_links`), both stored in the movie folder. A requeued
   task therefore resumes, and the per-slide combine only redoes
   paths/statistics. `fast-batch`, in contrast, sets `force_analysis=True`.
5. **Atomic cache writes.** `np.save` wrote directly to the final name, so a
   kill during the write could leave a truncated `.npy`, which the requeue would
   then fail to load on every round. New `save_npy_atomic` (temp `*.npy.tmp` +
   `os.replace`) is used in `NpyFilamentStore.write`, `Frame.save_filXYs` and
   `Motility.save_links`. File contents are unchanged (golden masters pass).
   `preprocess.py` already wrote its model atomically.
6. **Trial** (`$SCRATCH/fastrack_runs/taka_movies_trial_2026-10-16/`, slide_04):
   - Movie array `48133963`; task 2 requeued by hand after 39 cached frames.
     It restarted (`restarts=1`) and loaded the cached frames. (Cached frames are
     re-saved with identical content; this is pre-existing and harmless.)
   - Movies took 13–28 min. `seff`: 5.5 GB peak, CPU efficiency 20 % of 8 cores
     (linking is single-threaded) → movie tasks resized to **4 CPUs / 12 GB**.
   - Sweep `48134053` ran on owners: all 4 movies done → combine → `COMPLETE`.
   - Comparison with the laptop run (2026-10-09): movie means are
     1293.4 / 1265.0 (E11K) and 1460.9 / 1344.7 (WT), giving condition means 1279 / 1402
     (laptop: 1279 / 1402). The requeued movie's velocity files are
     byte-identical to the laptop's. The other movies have the same row counts
     with a few rows differing (7–20 values/movie, presumably opencv 5 /
     numpy 2.2 on Sherlock vs the laptop stack).
7. **Clean-up before the full run.** `48131798` was cancelled. Task 0 had
   completed slide_01, task 1 had partly done slide_02, and task 2 had evidently
   been preempted once (slide_03 frames from 10:17) before it was held. Because
   that run used the old non-atomic writes, its 990 `*f89a1917*` cache files were
   **moved** (deletion was not permitted) to
   `$SCRATCH/fastrack_runs/cancelled_48131798_cache_2026-10-16/`, outside the
   data tree. Older caches from July (`filXYs_ridge*`, another recipe) in
   slide_01 were left in place. The slide_04 trial caches were kept and are
   reused.
8. **Full run.** `cd $SCRATCH/fastrack_runs/taka_movies_2026-10-16 && bash
   ~/code/fastrack/hpc/fast_movies_launch.sh datasets.taka.csv` → 38 movies
   (8 / 7 / 19 / 4), array `48153343`, sweep `48153345`.

## Files / commits

| File | Change |
| --- | --- |
| `hpc/fast_batch_array.sbatch`, `hpc/fast_single.sbatch` | find `env.sh` from the spooled script (`FASTRACK_HPC_DIR` / `scontrol` `Command=`) |
| `hpc/fast_movies_launch.sh` (new) | one-shot launcher: `movies.tsv`, movie array + first sweep, duplicate-launch guard |
| `hpc/fast_movie_array.sbatch` (new) | one movie per task, `owners,normal`, 2 h, 4 CPU / 12 GB, `--requeue`, done marker |
| `hpc/fast_movies_sweep.sbatch` (new) | resubmit stragglers (≤5 rounds) → per-slide combine → `COMPLETE` |
| `hpc/manifest_movies.py` (new) | manifest → one row per movie (same parser + discovery as fast-batch / fast) |
| `src/fastrack/io/atomic.py` (new) + `npy_store.py`, `frame.py`, `motility.py` | atomic `.npy` cache writes |
| `hpc/README.md` | “Many movies on a preemptible partition” section |
| `journals/2026-10-15_sherlock-install.md` | records the 10-15 submission (failed; see this journal) |

## Open items / next steps

- **The recipe this run uses is superseded.** A parallel laptop session
  (`2026-10-16_ridge-snr-filter`, Notion) found heavy noise over-detection with
  the current taka config. The fix is on branch `detect-snr-filter` (`fe35a63`,
  not pushed yet): new `config.taka.toml` (60/120, `min_len 8`, `min_snr 3`) and
  ridge params in the filXYs cache key. That branch touches `core/frame.py`,
  `core/motility.py` and `pipelines/gliding.py`, so it needs a rebase onto `0297717`
  (atomic writes) before merging. Once it's on `main`, pull on Sherlock and rerun
  with `fast_movies_launch.sh` in a new run dir. The new cache tag means no
  stale caches are reused.
- **Check the full run:** `tail $SCRATCH/fastrack_runs/taka_movies_2026-10-16/controller.log`;
  `COMPLETE` there means done. Then `sacct -j 48153343` (look at
  `restarts`/preemptions and the slide_03 E11K FG20_1 movie: 1 h 24 m on the laptop,
  with a 2 h task limit and 4 h on retry), and compare `outputs/*/combined/MEAN_values.txt` with the
  laptop table (2026-10-09 journal).
- Per-row differences vs the laptop (step 6) are presumably library versions;
  confirm by checking versions if the slide-level numbers drift.
- `$SCRATCH/fastrack_runs/cancelled_48131798_cache_2026-10-16/` (990 files) and
  `taka_redo_2026-10-15/` can be removed once the run is accepted.
- Cached frames are re-saved on every run (`_extract_frame` always calls
  `save_frame`); this could be skipped when the frame was loaded from cache.
- Still open from 10-15: `pyproject.toml` pins `opencv-python` (headless on the
  cluster); `discovery.py` could skip dotfiles.
