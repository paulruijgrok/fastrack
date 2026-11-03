# 2026-10-20 — Taka rerun on Sherlock: SNR recipe + overlay movies

(hub: **Fastrack** — <https://app.notion.com/p/39edaf1f293b81179192ea41a1110d6b>)

## TL;DR

- Pulled `aba20f0` (preview mode, test-warning fixes, `wire_frame_links`
  gap-closing from d=2). Reinstalled in a dev job and re-applied the
  opencv-headless fix. `python -m pytest -q` in job `48664662`: **178 passed,
  2 skipped** (same as the Mac).
- **Full taka rerun, all 38 movies**, with `config.taka.toml` (contrast 60/120,
  `min_len 8`, `min_snr 3`) plus `[runtime] overlay_movie = true`:
  `$SCRATCH/fastrack_runs/taka_movies_2026-10-20/`. The movie array was gated on
  pytest (`afterok`). Movie tasks took **≤ 4 min each** (all on `normal`) and
  were all done 9 min after starting. Old recipe on 10-16: up to 1 h 30 m.
- **Bug found and fixed** (`f4969d5`): the sweep's combine loop reads datasets
  from a pipe, and the ffmpeg run for the overlays ate its stdin. Sweep 1
  combined only slide_01/03 and still wrote `COMPLETE`. Fix: `fast … < /dev/null`.
  Re-ran the sweep: all 4 slides combined.
- Outputs: 4 `combined/MEAN_values.txt`, **38 `*_overlay_tracks.mp4`** under
  `outputs/slide_0*__…__det_ridge__pp_f89a1917/`. Numbers match the 10-16 cloud
  run (`outputs/taka_2026-10-16_snr/` on the Mac) within a few points/nm/s.

## Detailed narrative

1. **Catch-up.** The 10-16 old-recipe run (`taka_movies_2026-10-16/`) was
   complete. `origin/main` was 14 commits ahead (laptop work: `fast-preview`,
   warning fixes, tracking changes in `greedy.py` / `motility.py` / `frame.py`,
   `filterwarnings = error::RuntimeWarning:fastrack` in `pyproject.toml`).
2. **Install.** `srun -p dev -c 2`: `pip install -e .` brought back GUI
   `opencv-python` again, so I uninstalled both opencv packages and reinstalled
   `opencv-python-headless==5.0.0.93`. Then checked that the imports work
   (`cv2 5.0.0`, `fastrack.pipelines.preview`) and that the venv's ffmpeg has
   libx264.
3. **Run setup.** The run dir has a copy of `datasets.taka.csv`, the
   `examples -> $SCRATCH/fastrack_data` link, and `config.taka.toml` from the
   repo with `[runtime] overlay_movie = true` appended. Overlays are rendered by
   every `fast` call: once in each movie task (into the scratch
   `movie_work/<i>/outputs/`, harmless duplicate) and again in the combine
   (into `outputs/`). The PNG frames are written in the movie folder and removed
   after encoding (checked: no `overlay_*.png` left in the data tree).
4. **Launch.**
   ```bash
   cd $SCRATCH/fastrack_runs/taka_movies_2026-10-20
   MOVIE_ARRAY_ARGS="--time=02:00:00 --dependency=afterok:48664662" \
     SBATCH_TIMELIMIT=03:00:00 bash ~/code/fastrack/hpc/fast_movies_launch.sh datasets.taka.csv
   ```
   Movie array `48664687`, sweep `48664688`. `SBATCH_TIMELIMIT` raised the
   sweep's 1 h limit in case overlay rendering slowed the combine. That turned
   out not to be needed: an overlay is 60 frames, about 3 MB, and takes seconds.
5. **Timing.** pytest 12:06 → 12:26; movie tasks 12:26 → 12:35 (max 4 min,
   task 11). Sweep 1 combine 12:35 → 12:42. Re-combine (`48668513`)
   12:46 → 12:50. About 45 min wall from submission, 1 h 45 min total job time.
6. **The stdin bug.** `controller.log` went from `combine taka_slide_01`
   straight to `taka_slide_03`, then `campaign complete`. Cause: in
   `cut … | sort -u | while read …; do fast …; done`, ffmpeg (started inside
   `fast` for the overlays) reads stdin and consumed the remaining lines. The
   10-16 run had no overlays, so it never hit this. Fix in
   `hpc/fast_movies_sweep.sbatch`: `< /dev/null` on the `fast` call. The other
   `while read` loop in the script only runs builtins. Recovery: moved the false
   `COMPLETE` to `COMPLETE.partial_sweep1`, logged a `[manual]` line in
   `controller.log`, and resubmitted the sweep with `SWEEP_ROUND=1`. Without
   `-f`, `fast` skips a slide that already has `MEAN_values.txt`
   (`gliding.py:245`), so slides 01/03 kept their 12:37/12:42 outputs (their
   re-run combine logs are empty files). Slides 02/04 were combined fresh.
7. **Results** (per condition; points summed, MVEL-filtered and filtered length
   points-weighted over movies, stuck % = mean over movies):

| condition | movies | points | MVEL-filtered | stuck % | length nm |
|---|---|---|---|---|---|
| s01 E11K_300nM | 2 | 5078 | 1397 | 2.4 | 1458 |
| s01 E11K_500nM | 2 | 4147 | 1418 | 1.1 | 1677 |
| s01 WT_300nM | 2 | 1249 | 1459 | 7.4 | 2857 |
| s01 WT_500nM | 2 | 1244 | 1586 | 3.3 | 2627 |
| s02 E11K_150nM | 2 | 1730 | 791 | 33.1 | 1353 |
| s02 E11K_250nM | 4 | 12960 | 1383 | 4.5 | 2065 |
| s02 WT_250nM | 1 | 186 | 1419 | 15.1 | 3739 |
| s03 E11K_250nM_FG10 | 4 | 13651 | 1305 | 4.5 | 1727 |
| s03 E11K_250nM_FG20 | 4 | 3421 | 1464 | 3.0 | 1409 |
| s03 WT_500nM_FG10 | 4 | 6141 | 1560 | 6.2 | 2043 |
| s03 WT_500nM_FG20 | 4 | 4311 | 1846 | 5.5 | 1938 |
| s03 test | 3 | 657 | 1543 | 10.0 | 2449 |
| s04 E11K_250nM_FG10 | 2 | 3328 | 1103 | 12.4 | 2039 |
| s04 WT_500nM_FG10 | 2 | 1767 | 1245 | 15.1 | 2210 |

   Point counts are within ±8 of the 10-16 cloud run and stuck % is identical.
   MVEL-filtered differs by 0–30 nm/s. The averaging here is not necessarily
   the cloud table's, and the tracking changes from the pull also contribute.

## Addendum: preview-mode batch on the same 38 movies

- New `hpc/fast_preview_batch.sbatch`: `fast-preview` on several folders in one
  job (preview is seconds per movie, so no array), then gathers everything into
  one browsable place: `preview_summary_all.csv` (all movies, `folder` column)
  and `dashboards/` (all dashboard PNGs, flat). Tested on slide_04 in a dev job
  (4 movies, 1 min).
- Run: job `48673221` on `normal`, **3 min 33 s** for all 38 movies (max 10 s per
  movie on 8 cores), default preview settings (they equal the Taka config:
  80.65 nm, ridge 60/120, `min_len 8`, `min_snr 3`; 20 frames, bin 2).
  ```bash
  R=$SCRATCH/fastrack_runs/taka_preview_2026-10-20; E="$SCRATCH/fastrack_data/taka/04302026 Motility Experiment"
  PREVIEW_OUT=$R sbatch -o "$R/fast_preview_%j.log" hpc/fast_preview_batch.sbatch "$E"/slide_0{1,2,3,4}
  ```
- Verdicts: 28 GOOD, 4 MARGINAL, 6 POOR. POOR: slide_01 WT_500nM_1 and
  slide_02 WT_250nM_1 (floppy), slide_02 E11K_150nM_1/2 (stuck; 34.6 % on _2,
  matching the full run's 33 %), slide_03 test_2 (focus), test_3 (flatness).
  MARGINAL: all on floppy % (slide_01 E11K_300nM_1, WT_300nM_2, WT_500nM_2;
  slide_03 WT_500nM_FG20_4).
- Left over: `$SCRATCH/fastrack_runs/preview_test_2026-10-20/` (slide_04 test
  output; delete when convenient).

## Files / commits

| item | what |
|---|---|
| `aba20f0` | pulled (laptop: preview mode, warning fixes, tracking tweaks) |
| `f4969d5` | `hpc/fast_movies_sweep.sbatch`: combine `fast` gets `< /dev/null` |
| `$SCRATCH/fastrack_runs/taka_movies_2026-10-20/` | run dir; `outputs/`, `controller.log`, `COMPLETE` |
| `$SCRATCH/fastrack_runs/pytest_48664662.log` | 178 passed, 2 skipped |
| `hpc/fast_preview_batch.sbatch` | preview batch job script (addendum) |
| `$SCRATCH/fastrack_runs/taka_preview_2026-10-20/` | preview run: `preview_summary_all.csv`, `dashboards/`, `per_slide/` |
| jobs | preview `48673221`; pytest `48664662`; movies `48664687`; sweeps `48664688`, `48668513` |

## Open items / next steps

- `io/movie/ffmpeg.py`: run ffmpeg with `stdin=subprocess.DEVNULL` (or
  `-nostdin`) so any caller inside a read loop is safe. This is a laptop-side
  code change.
- Overlays are rendered twice per movie (movie task + combine). That's cheap
  (seconds), but it could be skipped in the movie task if it matters.
- `pyproject.toml` still pins GUI `opencv-python`; every `pip install -e .` on
  Sherlock needs the headless swap.
- `$SCRATCH` is purged after 90 days: copy the overlays/results to the Mac or Oak
  if they are to be kept.
- Still open from before: pixel size for Taka's setup; review the
  slide_02 E11K_150nM overlays (33 % stuck); `test_*` movies in slide_03.
- Sherlock's `CLAUDE.md` now says `python -m pytest -q`; the hpc pytest one-liner
  in this journal uses that.
