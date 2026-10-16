# 2026-10-09 — Full taka batch run with in-pipeline `--preprocess` (slides 01–04)

Mirror of the Notion session log:
<https://app.notion.com/p/3fadaf1f293b81619613e4fa485c74ef>
(hub: **Fastrack** — <https://app.notion.com/p/39edaf1f293b81179192ea41a1110d6b>).
Journal written 2026-10-15.

## TL;DR

- Ran **all 38 taka movies** (`examples/taka/04302026 Motility Experiment/slide_01..04`)
  through `fast` on branch `preprocess-flatfield` with the in-pipeline
  `--preprocess` stage + ridge detector, unattended via `fast-batch`.
  **4/4 slides OK, 24 149 s (~6.7 h)** on the Mac.
- **Validation passed:** slide_01 via `--preprocess` reproduces the July
  script-route results to rounding (e.g. E11K_300nM_1 MVEL-filtered 1613.97 vs
  1614.22 nm/s; points-filtered within ±2 per movie). This was the open gate for
  merging `preprocess-flatfield` (still needs `tests/test_stack_pipeline.py` on the Mac).
- Added `config.taka.toml` (ridge + `[preprocess]` recipe) and
  `datasets.taka.csv` (one dataset per slide).
- **Runtime is dominated by greedy linking** (1–2k filaments/frame). Most movies
  6–15 min; the slide_03 E11K FG20 movies were pathological (one 1 h 24 m) yet kept
  only 29–56 filtered points each → heavy over-detection there.

## What changed and why

The 2026-09-15 session left one gate before merging the in-pipeline preprocessing
branch: run the full slide_01 set end to end on the Mac (through linking, which
exceeded the VM call cap) and compare with the July script-route table. Rather
than one slide, all four slides were run so the new data (slides 02–04) got
analysed at the same time.

- **`config.taka.toml`** (new, repo root): `[analysis] detection_algorithm =
  "ridge"`; `[ridge] line_widths=[3,5], low_contrast=18, high_contrast=50,
  min_len=12` (identical to the July run); `[preprocess] enabled=true` with the
  defaults (median bg, flat_sigma 50, denoise 0.8, clip 99.9). `pixel_size_nm =
  80.65` carries a TODO — still unconfirmed for Taka's setup.
- **`datasets.taka.csv`** (new): fast-batch manifest, one row per slide, all
  using `config.taka.toml`. Paths are relative to the repo (Mac-local data under
  `examples/`, which is git-ignored) — a cluster run needs its own manifest.
- Recipe hash for this config: `pp_f89a1917` (output-tree suffix and filXYs cache tag).

## Commands

```bash
conda activate fastrack-ridge            # Python 3.11 (tomllib) + numba for ridge
cd ~/Documents/Claude/Projects/FASTrack
git checkout preprocess-flatfield
fast-batch datasets.taka.csv --preflight-only   # 4/4 ready
fast-batch datasets.taka.csv                    # 15:44 -> 22:27, SUMMARY {'done': 4}
# single slide equivalent:
fast -d 'examples/taka/04302026 Motility Experiment/slide_01' -f --config config.taka.toml
```

Logs: `fastrack_batch_logs/batch_20261009_154425.log` (+ per-slide
`taka_slide_0N.log`). Outputs:
`outputs/slide_0N__pt_none__n_5__ymax_1500__p_5__fx_none__det_ridge__pp_f89a1917/combined/MEAN_values.txt`.

## Timing

| Slide | Movies | Wall time |
|---|---|---|
| 01 | 8 | 1 h 23 m |
| 02 | 7 | 1 h 10 m |
| 03 | 19 (incl. 3 `test_*`) | 3 h 31 m |
| 04 | 4 | 38 m |

Detection + preprocessing ≈ 1 min/movie; the rest is linking. Slide_03 E11K
FG10 movies took 4–6 min each; E11K FG20_1 took 1 h 24 m, FG20_2/3 ~27 min.

## Results (MVEL-filtered, nm/s; per-condition unweighted mean of movies)

| Slide | Condition | E11K | WT |
|---|---|---|---|
| 01 | 300/500 nM | 1593 | 1607 |
| 02 | 250 nM | 1456 | 1544 (1 movie, 126 pts) |
| 02 | 150 nM | 1271 / 1154 (71 / 59 pts) | — |
| 03 | FG10 (E11K 250 nM, WT 500 nM) | 1488 | 1718 |
| 03 | FG20 | 1741 (29–56 pts/movie — unreliable) | 2015 |
| 04 | FG10 | 1279 | 1402 |

Observations:
- WT > E11K by ~120–270 nm/s on slides 03/04; no difference on slide 01.
  WT FG20 > WT FG10 (~2.0 vs ~1.7 µm/s).
- WT filaments longer on every slide (slide 01: WT 2.5–3.3 µm vs E11K 1.9–2.2 µm;
  smaller gap in FG sets).
- Slide 04 slower overall with more stuck (6–17 % vs ~0–6 %) → slide-to-slide
  variation; compare within a slide.
- Thin/odd movies: slide_02 E11K_150nM_2 56 % stuck; WT_250nM_1 126 pts;
  slide_03 E11K FG20 ×4 (29–56 pts).
- slide_03 `test_1/2/3` (15–30 frames) are pooled in slide_03's combined output
  (protein label `test`).

## Files / commits

| Path | What |
|---|---|
| `config.taka.toml` (new) | taka ridge + preprocess config |
| `datasets.taka.csv` (new) | fast-batch manifest, slides 01–04 |
| `journals/2026-10-09_taka-full-batch-preprocess.md` (new) | this journal |
| `.gitignore` | ignore `journals/.obsidian/` |
| branch `preprocess-flatfield` (`6164659`, `c77b6d9`) | code under test; unchanged this session |

## Open items / next steps

- **Merge + push.** Full suite run on the Mac 2026-10-15 in `fastrack-ridge`
  (`python -m pytest -q`): **139 passed, 3 skipped** (6.5 min), incl.
  `tests/test_stack_pipeline.py` and both golden masters. `preprocess-flatfield`
  fast-forwards `main`; merge and push `main` (local `main` was also 1 commit ahead of `origin/main`, `56ebc22`).
  Needed before cluster runs (clusters clone from GitHub).
- **Cluster processing** — next session: use `hpc/` (`fast_single.sbatch`,
  `fast_batch_array.sbatch`, `env.sh`); needs a cluster-side manifest pointing at
  the data's cluster path and a ridge-capable env (Python ≥3.11 for TOML configs).
- **Over-detection / linking cost:** inspect an E11K FG20 detection overlay; try
  higher `low_contrast` / `min_len` (the greedy linker is O(n²) in filaments/frame).
- **Pixel size** (`80.65 nm`) unconfirmed for Taka's objective/camera — all
  velocities scale with it. `utrophin(nM)` still prints 0 (folder names not parsed).
- Decide whether `test_*` movies belong in slide_03 combined results.
- `fast` vs `lima` (if nM is a utrophin load); WT vs E11K length check.
- Batch runner: a resume restarts a whole slide (per-dataset granularity), and a
  mistyped manifest path gives a raw traceback — both minor.
- Untracked `docs/skills/` still undecided.
