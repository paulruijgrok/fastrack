# 2026-09-15 — In-pipeline preprocessing (`--preprocess`) on branch `preprocess-flatfield`

Mirror of the Notion session log: *2026/09/15 In-pipeline preprocessing (option A) on a branch*.

## TL;DR

- Decided the cross-pipeline preprocessing strategy: **A** (opt-in stage at the
  frame-source seam) with **C** (shared `fastrack.preprocess` module also backing
  the standalone tool) and **E** (`model_frames=N` sampled fit) as the fallback for
  stacks too large to hold in memory. Built on a new branch, `preprocess-flatfield`
  (one commit, `6164659`, not merged, not pushed).
- `fast --preprocess` and `fastplus --preprocess` now run raw low-count movies end
  to end. Pixels and ridge detections are **identical** to the script route on real
  taka data (20 frames checked, max pixel diff 0, detections byte-identical).
- Golden master passes for both cache layouts (default path unchanged). Full suite
  135 passed / 6 skipped in the dev container; the new `tests/test_preprocess.py`
  adds 12 tests.
- Also this session: committed the July journal edits on `main` (`56ebc22`) and
  saved a project recap doc to the Claude project.

## What changed and why

The standalone `tools/preprocess_lowsnr.py` (option B, July) proved the recipe
but forces a two-step workflow with duplicated movies. The stage now lives inside
the pipelines:

- **`src/fastrack/preprocess.py`** — the maths, factored out of the script:
  `estimate_background_gain`, `apply_correction`, `correct_stack`,
  `stretch_to_uint`, `preprocess_stack`, plus a fit-once / apply-per-frame
  `PreprocessModel` (background, gain field, stretch ceiling; `.npz` save/load),
  `fit_model` / `fit_model_from_reader`, `sample_indices`, recipe hashing and
  `params_from_settings`.
- **`src/fastrack/core/input/preprocessed.py`** — `PreprocessedSource` (kind
  `"preprocessed"`) wrapping any `FrameSource`; `read()` returns the corrected
  8-bit frame `*257`, i.e. exactly what the pipeline sees reading the script's
  output from disk. Resolves the whole-stack problem: the parent fits the model
  once (`ensure_model()`), saves `preprocess_model_<hash>.npz` in the work dir,
  and pool workers rebuild the source from the descriptor and load it.
- **`FrameSource.read_raw()`** on `mm_dir` (cv2 `IMREAD_UNCHANGED`) and
  `tiff_stack` — native-depth reads bypassing the 16→8→16 quantisation.
- **`config.py`** — `[preprocess]` section (`enabled`, `background`,
  `model_frames`, `flat_sigma`, `denoise_sigma`, `clip_hi_pct`);
  `to_run_kwargs` / `to_directional_kwargs` carry `preprocess` (`None` when off,
  so nothing changes for existing runs).
- **CLI** — `--preprocess/--no-preprocess`, `--pp-background`, `--pp-model-frames`,
  `--pp-flat-sigma`, `--pp-denoise-sigma`, `--pp-clip-hi-pct` on `fast` and
  `fastplus` (shared `_add_preprocess_args`).
- **gliding.py** — wraps the source descriptor, fits the model up front, suffixes
  the output tree `__pp_<hash>` and the filXYs cache tag `_pp<hash>`.
- **directional.py** — fits/applies the model on the split filament stack (head
  channel left raw); recipe folded into the filament detection-cache key.
- **Entropy quality gate** — needs no change: the full-range stretch makes the
  absolute-contrast threshold pass (tested bad → good).
- **tools/preprocess_lowsnr.py** — imports the shared functions; dropped its own
  copies and an unused `multiprocessing` import.
- **Docs** — new `docs/preprocessing.md`; README section + config list;
  `docs/fastplus.md` note; `docs/preprocessing_notes.md` status ("implemented");
  `config.example.toml` `[preprocess]`; CHANGELOG.

## Environment notes

The Cowork sandbox on the Mac is a Linux VM (Python 3.10, 4 cores, ~3 GB, calls
capped at ~3 min, background processes killed between calls). Set up
`$HOME/fvenv` there with `pip install -e '.[plus,ridge]' pytest`. Heavy work was
done in a cloud clone (via `git bundle`) and shipped back as a bundle:

```bash
# Mac side (VM)
git fetch preprocess-flatfield.bundle +preprocess-flatfield:refs/heads/preprocess-flatfield
git checkout preprocess-flatfield
```

## Verification

```bash
pytest -q                                   # cloud: 135 passed, 6 skipped
pytest -q "tests/test_golden.py::test_outputs_match_baseline[per-frame]"   # Mac VM: pass, 170 s
pytest -q "tests/test_golden.py::test_outputs_match_baseline[per-movie]"   # Mac VM: pass, 164 s
pytest -q tests/test_input.py tests/test_preprocess.py tests/test_preprocess_lowsnr.py  # Mac VM: pass
```

Real-data equivalence (20-frame copy of `taka/…/slide_01/E11K_300nM_1` in VM scratch):

```bash
fast -d ~/taka_20 -f --preprocess --detector ridge --ridge-line-widths 3 5 \
     --ridge-low-contrast 18 --ridge-high-contrast 50 --ridge-min-len 12 -j 4
python tools/preprocess_lowsnr.py -i ~/taka_20 -o ~/taka_20_pp -j 2
fast -d ~/taka_20_pp -f --detector ridge --ridge-line-widths 3 5 \
     --ridge-low-contrast 18 --ridge-high-contrast 50 --ridge-min-len 12 -j 4
# compare filXYs_ridge_pp<hash>NNN.npy vs filXYs_ridgeNNN.npy and the frames
```

Result: max |pipeline frame − script frame| = 0 on all 20 frames (raw legacy read:
2 grey levels; corrected: 256); ridge detections byte-identical on all 20 frames
(1.1k–2.6k filaments per frame).

Not completed in the VM: `tests/test_stack_pipeline.py` (two exact-rank full
runs) and the linking stage of the taka run (greedy linker on ~2k filaments/frame
exceeds the per-call cap) — both need a run on the Mac proper.

## Files / commits

| Path | What |
|---|---|
| branch `preprocess-flatfield`, commit `6164659` | everything below |
| `src/fastrack/preprocess.py` (new) | shared maths + `PreprocessModel` |
| `src/fastrack/core/input/preprocessed.py` (new) | `PreprocessedSource` |
| `src/fastrack/core/input/{base,mm_dir,tiff_stack,__init__}.py` | `read_raw`, registration |
| `src/fastrack/config.py`, `src/fastrack/cli/__init__.py` | `[preprocess]`, `--pp-*` |
| `src/fastrack/pipelines/{gliding,directional}.py` | wiring, cache/output keys |
| `tools/preprocess_lowsnr.py` | delegates to the module |
| `tests/test_preprocess.py` (new, 12 tests) | module, source, settings/CLI, gate, end-to-end |
| `docs/preprocessing.md` (new), README, `docs/fastplus.md`, `docs/preprocessing_notes.md`, `config.example.toml`, CHANGELOG | docs |
| `main`: `56ebc22` | July journal edits committed |

## Open items / next steps

- **Run the full 8-movie taka slide_01 set on the Mac** with `--preprocess` through
  linking and compare velocities / lengths with the July script-route table; also
  run `tests/test_stack_pipeline.py` there. Then merge the branch (PR description
  is in the commit message).
- Filament counts on corrected frames are 1.1k–2.6k per frame with min_len 12 —
  worth a look at whether the thresholds over-detect (the greedy linker is O(n²)).
- `model_frames` is a sampled fit; a true rolling background for slow drift is not
  implemented.
- Head channel in FASTplus is left raw; add a switch if LoG head detection needs it.
- Batch `--smoke` preflight still detects on the raw frame 0.
- Untracked `docs/skills/` (duplicated skill snapshot) still sits in the working
  tree — decide whether to commit or delete.
- July items still open: `pixel_size_nm` calibration for Taka's setup; `fast` vs
  `lima`; WT vs E11K length; repin optomerge to a tag.
