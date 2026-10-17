# 2026-10-16 — Ridge over-detection on taka data: parameter sweep + signal-to-noise filter

Mirror of the Notion session log: *2026/10/16 Ridge over-detection: SNR filter*
(hub: **Fastrack** — <https://app.notion.com/p/39edaf1f293b81179192ea41a1110d6b>).

## TL;DR

- The 2026-10-09 batch run's slow linking came from **over-detection of
  background noise**: ~90 % of the ~2000 ridges/frame on dim movies were curly
  lines through the background (greedy linking is O(n²) in ridges/frame → up to
  84 min for one movie).
- Parameter sweep (32 combos, 4 movies): `low/high_contrast 60/120` + `min_len 8`
  fixes the moderately dim movies and *also* recovers short real filaments that
  `min_len 12` was dropping (~25 % of bright filaments).
- **Root cause that parameters can't fix:** the per-movie stretch maps each
  movie's own 99.9th pct to 255, so the same camera noise lands at **~0.5–5 grey
  levels** across movies (stretch ceilings 42–211). Fixed 8-bit thresholds
  can't work for all movies.
- **Fix (code, branch `detect-snr-filter`, commit `fe35a63`, not merged/pushed):**
  the preprocessing model measures each movie's pixel noise; the ridge detector's
  new `min_snr` drops ridges below `min_snr` × noise. `min_snr = 3` validated on
  the dimmest and brightest movies. Result is nearly independent of the contrast
  thresholds. Also: ridge params now in the filXYs cache key.
- Full pipeline on 4 movies: **84 min → 109 s** (worst movie), 27 min → 174 s
  (dimmest). Velocities within a length bin agree with the old settings to ~2 %;
  plain means drop because more short (slower) filaments are found.

## What was done and why

1. **Diagnosis from the cached detections** (`filXYs_ridge_ppf89a1917*.npy` of
   slide_03 E11K_250nM_FG20_1 = "bad", slide_01 E11K_300nM_2 = "good"): bad
   874–3624 ridges/frame (median 2295), wildly fluctuating frame to frame; 53 %
   < 20 px; bad movie's 90th-pct ridge intensity ~18/255 vs ~108 for good.
   Overlays: in a 250 px crop of the bad movie ~10 real filaments, ~150 noise
   lines. Per-contour mean intensity is bimodal (noise < ~30/255; filaments 40–250).
2. **Sweep** (`line_widths` [3,5]/[3] × 8 low/high pairs × `min_len` 8/12, frames
   0/30/59; metric: recall of bright reference objects (smoothed > 30, ≥ 8 px),
   fraction of noise-like ridges, time). `high_contrast` is the main lever;
   `min_len 12` costs ~25 % recall. Best: `[3,5]`, 60/120, `min_len 8`
   (bad 1883 → 127 ridges, recall 0.75 → 0.93; good 1146 → 546, recall 0.77 → 0.95).
3. **End-to-end** (`fast` with 60/120/ml8) on bad + good: bad 110 s (vs 84 min),
   points 29 → 873. Good movie MVEL-filtered 1575 → 1392 nm/s — traced to
   population change: velocity rises with length (0.65–0.97 µm: 1052 nm/s …
   2.5–4 µm: 1762), and within bins old/new agree to ~2 %.
4. **Generalisation check:** stretch ceilings across all 38 movies range 42–211
   (test_* up to 1807). On the dimmest (slide_02 E11K_150nM_1, hi 42) 60/120 still
   gave 1229–2457 ridges/frame, nearly all noise. Measured noise: 4.9 grey levels
   there vs 0.5 on the brightest → thresholds in stretched units are ~10× apart.
5. **SNR filter**, prototyped then implemented: SNR = (mean smoothed intensity
   along centerline − frame median) / pixel noise. Noise mode at SNR ~1–2 in all
   movies; real filaments well above. Noise estimator went through three
   versions: 5×5-median residual MAD (biased low ~0.74× for white noise) →
   whole-frame MAD of the correction (inflated by crowding / slow structure) →
   **frame-difference MAD / √2** (final: static structure cancels, crowding-robust,
   unbiased on white noise). Calibrated on the final estimator: `min_snr 3`
   (5 starts dropping real dim filaments on the good movie: 488 → 380).

## Results (full `fast` runs, new config, 2 cores in the cloud dev env)

| movie (stretch ceiling) | old time (Mac batch) | new time | points-filtered | MVEL-filtered nm/s | MVEL ≥ 1.3 µm | stuck % |
|---|---|---|---|---|---|---|
| s03 E11K_250nM_FG20_1 (69) | 84 min | 109 s | 29 → 1027 | 1783 → 1559 | 1701 | 3.2 |
| s01 E11K_300nM_2 (120) | 6 min | 194 s | 778 → 3781 | 1575 → 1392 | 1542 | 2.7 |
| s02 E11K_150nM_1 (42) | 27 min | 174 s | 71 → 694 | 1271 → 859 | 980 | 15.5 |
| s02 E11K_250nM_4 (211) | 2.2 min | 173 s | 1920 → 3262 | 1434 → 1380 | 1453 | 4.1 |

Figures (in `docs/assets/`): `taka_snr_01_param_compare.png` (current vs 40/120 vs
60/120 overlays), `taka_snr_02_min_snr_overlays.png` (min_snr 0/3/5 on the
dimmest, good and brightest movies), `taka_snr_03_snr_histograms.png`.

## Commands

```bash
# get the branch (bundle written to outputs/_stage/ by Claude)
cd ~/Documents/Claude/Projects/FASTrack
git fetch outputs/_stage/detect-snr-filter.bundle detect-snr-filter:detect-snr-filter
git checkout detect-snr-filter
conda activate fastrack-ridge && pip install -e '.[ridge,batch]'   # no new deps
python -m pytest -q                       # full suite incl. golden on the Mac

# one movie, new settings (config.taka.toml now has 60/120, min_len 8, min_snr 3)
fast -d '<dir containing one movie>' -f --config config.taka.toml
# whole dataset
fast-batch datasets.taka.csv --force
```

## Files / commits

| Path | What |
|---|---|
| branch `detect-snr-filter`, commit `fe35a63` (on `main` `e679b64`) | everything below |
| `src/fastrack/preprocess.py` | `estimate_noise_sd`, `PreprocessModel.noise_sd`, `output_noise_sd` |
| `src/fastrack/core/input/{base,preprocessed}.py` | `FrameSource.noise_sd()`; refit of models without noise |
| `src/fastrack/core/{frame,motility}.py` | `Frame.noise_sd` set on read |
| `src/fastrack/core/detection/ridge.py` | `min_snr`, `contour_snr`, `estimate_frame_noise` |
| `src/fastrack/{config.py,cli/__init__.py}` | `[ridge] min_snr`, `--ridge-min-snr` |
| `src/fastrack/pipelines/gliding.py` | `detection_tags()`; ridge params in cache key |
| `src/fastrack/pipelines/directional.py` | noise passed to FASTplus filament detection |
| `tests/test_snr_filter.py` (new, 13 tests) | estimator, model, plumbing, filter, settings, cache key, e2e |
| `config.taka.toml`, `config.example.toml`, README, `docs/preprocessing.md`, CHANGELOG | config + docs |
| `journals/2026-10-16_ridge-snr-filter.md`, `docs/assets/taka_snr_*.png` | this journal + figures (untracked) |

## Open items / next steps

- **Run the full suite on the Mac** (golden + stack tests skip in the cloud),
  then merge `detect-snr-filter` into `main` and push; commit this journal + figures.
- **Re-run the whole taka set** with the new `config.taka.toml` (expect minutes
  per movie instead of up to 84 min); `--force` or a fresh cache is automatic now
  that ridge params are in the cache key.
- **Dim 150 nM movies**: many short round puncta that are slow/stuck (MVEL 859,
  15.5 % stuck). Likely debris/aggregates — consider an aspect-ratio/shape filter,
  or report by length bin. Needs a look at the overlays with Taka.
- **Velocity reporting**: with more short filaments the raw mean drops; report a
  velocity-vs-length plateau (`fit_function` exp/uyeda) or length-matched means.
- Output-tree name does not include detector params (only the cache does).
- Greedy linker: pure-Python O(n²) candidate search; a KD-tree gate would make
  it robust to dense fields.
- Pre-existing unused imports in config.py / frame.py / motility.py /
  directional.py — separate cleanup (check for legacy re-exports first).
- Still open from before: pixel size for Taka's setup; `fast` vs `lima`;
  `test_*` movies in slide_03; `docs/skills/` untracked.

## Update (afternoon): median SNR + full taka run with overlay movies

- **SNR signal switched from mean to median** along the centerline (commit
  `fa1edeb` on `detect-snr-filter`), so an isolated spike (cosmic ray / hot
  pixel) on a borderline noise ridge cannot lift it over `min_snr`. Counts at
  `min_snr 3` vs the mean: 133 → 121 (dimmest), 95 → 93, 496 → 497, 457 → 459;
  threshold kept at 3. New test: a 3000-count hot pixel passes with the mean,
  not with the median. Suite: 149 passed, 6 skipped (cloud).
- **Full run, all 38 movies** (slides 01–04, `config.taka.toml` + overlay
  movies, 2 cores in the cloud dev env): **84 min total** (Oct 9: 6.7 h).
  Results copied to `outputs/taka_2026-10-16_snr/` (separate folder because the
  output-tree name does not include detector params and would otherwise
  overwrite the Oct 9 trees): per-slide trees with `combined/MEAN_values.txt`,
  38 `*_overlay_tracks.mp4`, `summary_per_movie.csv`, logs, `run.toml`.

| condition | movies | points | MVEL-filtered | MVEL ≥ 1.3 µm | stuck % | mean length nm |
|---|---|---|---|---|---|---|
| s01 E11K_300nM | 2 | 5078 | 1403 | 1549 | 2.4 | 1452 |
| s01 E11K_500nM | 2 | 4146 | 1410 | 1512 | 1.1 | 1718 |
| s01 WT_300nM | 2 | 1253 | 1486 | 1489 | 7.4 | 2849 |
| s01 WT_500nM | 2 | 1244 | 1588 | 1588 | 3.3 | 2661 |
| s02 E11K_150nM | 2 | 1730 | 801 | 910 | 33.1 | 1359 |
| s02 E11K_250nM | 4 | 12963 | 1384 | 1457 | 4.5 | 2068 |
| s02 WT_250nM | 1 | 186 | 1419 | 1419 | 15.1 | 3739 |
| s03 E11K_250nM_FG10 | 4 | 13653 | 1310 | 1432 | 4.5 | 1725 |
| s03 E11K_250nM_FG20 | 4 | 3425 | 1458 | 1646 | 3.0 | 1423 |
| s03 WT_500nM_FG10 | 4 | 6140 | 1571 | 1647 | 6.2 | 2130 |
| s03 WT_500nM_FG20 | 4 | 4303 | 1871 | 2020 | 5.5 | 1923 |
| s03 test | 3 | 657 | 1485 | 1541 | 10.0 | 2509 |
| s04 E11K_250nM_FG10 | 2 | 3329 | 1099 | 1155 | 12.4 | 2038 |
| s04 WT_500nM_FG10 | 2 | 1763 | 1244 | 1296 | 15.1 | 2210 |

Observations: at matched length (≥ 1.3 µm) WT > E11K on slides 03/04
(FG10 +215, FG20 +374, s04 +141 nm/s), not on slide 01; WT FG20 > WT FG10.
E11K FG20 now has real statistics (3425 points vs 104 on Oct 9). slide_02
E11K_150nM_2 is 51 % stuck (puncta) — review its overlay. WT filaments longer
throughout.

Open items added: pull `fa1edeb` on the Mac (`git pull
outputs/_stage/detect-snr-filter.bundle detect-snr-filter`), run the suite,
merge, push; delete `outputs/_stage/` (~4 GB of transfer files).
