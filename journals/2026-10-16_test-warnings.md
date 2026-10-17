# 2026-10-16 — Test-suite warnings: four causes, one dead/broken fit removed

Mirror of the Notion session log: *2026/10/16 Test-suite warnings cleanup*
(hub: **Fastrack** — <https://app.notion.com/p/39edaf1f293b81179192ea41a1110d6b>).

## TL;DR

- The full suite printed 48 warnings of four types. All traced and fixed on
  branch `fix-warnings` (commit `133a7b0`). Mac: **158 passed, 3 skipped, 0
  warnings** (golden masters pass → outputs unchanged).
- **Real bug found (no reported number affected):** `plot_length_velocity`
  ran a length-weighted coupling fit on every movie whose histogram binned nm
  lengths into µm bins → every weight 1/0 = ∞ → fit returned its initial guess
  (700, 200, 500), all residuals NaN. Its results were never used, plotted or
  returned. Removed as dead code; the `fit_function` fit (exp/uyeda) is separate
  and unchanged. **The original FAST code (`Old code/FASTrack-master/FAST/motility.py`,
  `plot_length_velocity`) has the identical bug** (nm data, µm edges; `digitize`
  uses µm correctly, so only the counts are wrong) and likewise never used the
  results — it was never functional there either. The length histogram in the
  figure was always correct (it bins `1E-3 * length`).
- pytest now treats any `RuntimeWarning` raised from `fastrack` as an error
  (`pyproject.toml [tool.pytest.ini_options]`).

## The four warnings

| warning | cause | real problem? | fix |
|---|---|---|---|
| `analysis/fitting.py:63` invalid divide | weighted coupling fit: nm vs µm histogram → ∞ weights → `err/weights` = ∞/∞ | bug, in dead code | removed the block in `viz/plots.py`; residuals computed directly (`model − data`) in both fit functions |
| `core/motility.py:284` divide by zero | `wire_frame_links` at `d=1` pairs a path end with a path starting in the **same** frame (dt = 0) | no — already rejected (inf/NaN fail the comparisons) | loop now starts at `d=2` (the `d=1` pass could never link); `dt <= 0` guard kept for duplicate timestamps |
| `core/frame.py:164` invalid scalar divide | quality gate on a blank frame (0/0) | no — NaN already gave "bad" | explicit blank-frame guard |
| skimage "Bad rank filter performance" | exact 16-bit rank path (`fast_rank=False`, used by golden tests) | no — documented cost of the exact path | `_rank_percentile()` silences that message only |

## Verification

- Cloud: 154 passed, 6 skipped, 0 warnings (was 4). New
  `tests/test_warning_fixes.py` (5 tests) + 3 existing tests fail on the old
  code (verified by stashing `src/`) and pass now.
- Mac (full suite incl. golden + stack-pipeline): 158 passed, 3 skipped, no
  warnings summary.

## Commands

```bash
git fetch outputs/_stage/fix-warnings.bundle fix-warnings:fix-warnings
git checkout fix-warnings && python -m pytest -q
git checkout main && git merge --ff-only fix-warnings && git push origin main
```

## Files / commits

| Path | What |
|---|---|
| `fix-warnings` / `133a7b0` (on `main` `e91b2e0`) | all below |
| `src/fastrack/viz/plots.py` | dead weighted coupling fit + unused histograms removed; `weighted` documented as ignored |
| `src/fastrack/analysis/fitting.py` | direct residuals |
| `src/fastrack/core/motility.py` | gap closing starts at `d=2` (commit `ed09817`); dt ≤ 0 guard kept for duplicate timestamps |
| `src/fastrack/core/frame.py` | blank-frame guard; `_rank_percentile` |
| `pyproject.toml` | `filterwarnings = ["error::RuntimeWarning:fastrack"]` |
| `tests/test_warning_fixes.py` | 5 regression tests |

## Open items / next steps

- If a length-weighted plateau velocity is wanted, re-implement the weighted
  coupling fit correctly (µm histogram) — it was never functional.
- Unused locals `num_points_mobile` / `num_points_all` in `viz/plots.py`; other
  pre-existing unused imports (see 2026-10-16 SNR journal).
