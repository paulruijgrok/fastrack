# Design notes: optional ridge-detector filament detection

Status: **plain tier implemented** (upstream `ridge-detector`, no numba). The
optimized/numba `ridge-fast` tier remains future work. Captures the plan and the
as-built integration of a ridge-detection filament detector as an *optional*
(non-default) choice, so the default install stays lean.

## As built (plain tier)

- `src/fastrack/core/detection/ridge.py` — `RidgeLineDetector(Detector)`,
  registered as `"ridge"`, lazily importing `ridge_detector` (clear error +
  `pip install 'fastrack[ridge]'` hint if missing). `_contours_to_filxys` maps
  each upstream `Line` (`.row`/`.col`/`.width_l`/`.width_r`) to a FASTrack
  `filXYs` entry `[contour(int N×2), width, density, midpoint]`, then
  `frame.filXY2filaments()` does the rest. Contours are rounded to integer
  pixels for full downstream compatibility (sub-pixel revisit = future work).
- `pyproject.toml`: `[project.optional-dependencies] ridge = ["ridge-detector>=0.1.3"]`.
- Selection: `fast --detector ridge` plus `--ridge-*` params; programmatically
  `Settings(analysis.detection_algorithm='ridge')` + `RidgeSettings`.
- **Detector isolation:** non-entropy detectors write to their own output tree
  (`...__det_ridge/`) and their own filXYs cache (`filXYs_ridge%03d.npy`), so
  entropy and ridge never collide or reuse each other's cache. Entropy paths and
  cache filenames are byte-identical to before (golden-master safe).
- Quality gate routes through `detector.assess_quality` for non-entropy
  detectors; entropy keeps its exact percentile gate.
- Tests: `tests/test_ridge.py` (registry + mapping run dependency-free; the
  end-to-end test `importorskip`s the stack + `ridge_detector`).

## Background

Two packages are in play:

- **Upstream `ridge-detector`** (`lxfhfut/ridge-detector`, on PyPI, MIT): a
  multi-scale Steger curvilinear-structure detector. Import name `ridge_detector`.
- **The optimized fork** (`paulruijgrok/ridge-detector`, branch `optimize-eigh`):
  currently a working area (scripts + the upstream vendored as a git submodule)
  whose key file `optimized_detector.py` defines
  `OptimizedRidgeDetector(RidgeDetector)` — a drop-in subclass that overrides
  `apply_filtering`, `compute_line_width`, `compute_contours` with an analytical
  2×2 `eigh`, float32 arrays, and a Numba-JIT ridge tracer. Same I/O as upstream,
  faster. **Adds a `numba` dependency.**

## API contract (what the detector exposes)

```python
det = RidgeDetector(            # or OptimizedRidgeDetector
    line_widths=[3],           # scales to detect (installed version uses line_widths, plural)
    low_contrast=50,
    high_contrast=150,
    min_len=10,
    max_len=0,                 # 0 = no upper limit
    dark_line=False,           # False = bright ridges on dark background (gliding assays)
    estimate_width=True,
    # upstream also: extend_line, correct_pos
)
det.detect_lines(image)        # accepts a FILE PATH or a numpy array  <-- pass frame.img
```

After `detect_lines`, results live on the object:

- `det.contours` — list of `Line` objects, each with:
  - `.num` — point count
  - `.row`, `.col` — sub-pixel centerline coordinates
  - `.width_l`, `.width_r` — per-point half-widths (with `estimate_width=True`)
  - `.intensity` — per-point intensity
  - `.angle`, `.response` — orientation and ridge strength
- `det.junctions` — list of `Junction` objects

## Mapping into FASTrack's detector seam

The `Detector` interface only requires that `detect(frame)` populate
`frame.filXYs` and call `frame.filXY2filaments()`. Each `frame.filXYs` entry is
`[contour, width, density, midpoint]`. Mapping per ridge `Line`:

- `contour` = `np.column_stack([Line.row, Line.col])` (round to int if needed)
- `width`   = `mean(Line.width_l + Line.width_r)`  → `fil_width`
- `density` = `mean(Line.intensity)` (or sample `frame.img` along the contour) → `fil_density`
- `midpoint`= `contour[len(contour)//2 - 1]`

Then `frame.filXY2filaments()` builds proper `Filament` objects (computing
`fil_length`, `cm`, etc.), and the existing tracking / stats / plotting work
unchanged. This is a richer fit than the entropy detector (sub-pixel positions +
a measured width).

`RidgeDetector(Detector)` lives in `src/fastrack/core/detection/ridge.py`,
registered via `@DETECTORS.register("ridge")`, with the `ridge_detector` import
done lazily inside `__init__` so the default install never imports it.

## Dependencies — why "optional" matters

- **Upstream `ridge-detector` declares 0 install deps in its metadata**, but the
  code needs OpenCV, scikit-image, numpy, scipy (already FASTrack deps) **and
  `numba`** — `ridge_detector/utils.py` does `from numba import jit` at import
  time. numba is undeclared upstream, so `pip install ridge-detector` does *not*
  pull it; the `[ridge]` extra adds `numba` explicitly.
- **`numba` (→ `llvmlite` + bundled LLVM, tens of MB, platform-specific) is the
  only heavy addition**, and it is required even by the *plain* upstream
  detector. This is precisely why the ridge detector is opt-in: keeping it out of
  the default install keeps numba/LLVM out too.
- (The optimized fork also uses numba, for its own JIT tracer — same heavy dep,
  no additional install burden beyond what the plain package already needs.)

### Proposed two-tier optional extra

```toml
[project.optional-dependencies]
ridge      = ["ridge-detector>=0.1.3"]                 # plain detector; no heavy new deps
ridge-fast = ["ridge-detector>=0.1.3", "<optimized-pkg>", "numba>=0.59"]
```

- `pip install fastrack[ridge]` → plain `RidgeDetector`.
- `pip install fastrack[ridge-fast]` → `OptimizedRidgeDetector` (+ numba).

Default `pip install fastrack` stays exactly as lean as today.

## Integration details to handle at implementation time

- **Lazy import + clear error.** Register `ridge` always; import `ridge_detector`
  (and the optimized package / numba) only inside `RidgeDetector.__init__`,
  raising a helpful "install fastrack[ridge]" message if missing.
- **Cache key must include the detector.** `filXYs%03d.npy` and the output dir
  name don't encode the detector; fold `detection_algorithm` into `main_out_dir`
  (and/or the cache filename) so entropy/ridge runs don't reuse each other's
  cached filaments.
- **Quality gate.** Route the pipeline's gate through `detector.assess_quality`
  so ridge can define its own (or default to "good").
- **Config.** Add ridge params (line widths, contrasts, min/max length,
  dark_line, estimate_width) as a `RidgeSettings` group or `detection_params`
  dict; `Motility.get_detector()` passes the right kwargs per algorithm.
- **Pin the upstream version** the optimized code was written against (the fork's
  submodule commit). The optimized code imports semi-internal symbols from
  `ridge_detector.utils` / `ridge_detector.constants`, so API drift between
  upstream releases could break it.
- **Tests.** `tests/test_ridge.py` uses `pytest.importorskip("ridge_detector")`
  so it auto-skips on the default install; assert the `Filament` contract holds
  on a tiny synthetic line image. Capture a *separate* ridge golden baseline once
  validated (its numbers differ from entropy's).

---

# Making the optimized ridge-detector pip-installable

The fork is not installable today because it relies on a **git submodule** and
local script imports, with no `pyproject.toml`. Steps to turn it into a clean,
installable package that FASTrack can depend on:

1. **Pick a distinct distribution + import name.** `ridge-detector` (dist) and
   `ridge_detector` (import) are taken by upstream. Use e.g. distribution
   `ridge-detector-fast` and import package `ridge_detector_fast`.

2. **Drop the git submodule; depend on upstream instead.** Remove the vendored
   submodule and declare `ridge-detector` as a normal dependency. The
   `from ridge_detector import ...` imports then resolve to the installed PyPI
   package. Pin to the version matching the submodule commit the optimized code
   targets (verify the imported `utils`/`constants` symbols exist there).

3. **Adopt a src layout:**
   ```
   ridge-detector-fast/
   ├── pyproject.toml
   ├── src/ridge_detector_fast/
   │   ├── __init__.py        # exports OptimizedRidgeDetector
   │   └── detector.py        # current optimized_detector.py
   └── tests/test_detector.py
   ```

4. **Write `pyproject.toml`.** Because upstream declares no deps, declare the full
   runtime set here:
   ```toml
   [build-system]
   requires = ["setuptools>=64", "wheel"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "ridge-detector-fast"
   version = "0.1.0"
   requires-python = ">=3.9"
   dependencies = [
       "ridge-detector>=0.1.3",   # pin to the matching upstream version
       "numba>=0.59",
       "numpy>=1.23",
       "scipy>=1.9",
       "opencv-python>=4.6",
       "scikit-image>=0.20",
   ]

   [tool.setuptools]
   package-dir = { "" = "src" }

   [tool.setuptools.packages.find]
   where = ["src"]
   ```

5. **Add `__init__.py`** exporting the public API:
   ```python
   from .detector import OptimizedRidgeDetector
   __all__ = ["OptimizedRidgeDetector"]
   ```

6. **De-hardcode the scripts.** `detect_stack.py` / `test_ridge.py` contain
   absolute `/Users/...` paths — move them to `examples/` or parametrize, and
   make the test use a tiny bundled image or `importorskip`.

7. **Verify locally:**
   ```bash
   pip install -e .
   python -c "from ridge_detector_fast import OptimizedRidgeDetector; print('ok')"
   pytest -q
   python -m build        # optional: build a wheel
   ```

8. **Make it consumable by FASTrack.** Either publish to PyPI, or have FASTrack's
   `ridge-fast` extra reference it by git tag:
   ```toml
   ridge-fast = [
       "ridge-detector>=0.1.3",
       "ridge-detector-fast @ git+https://github.com/paulruijgrok/ridge-detector-fast@v0.1.0",
       "numba>=0.59",
   ]
   ```
   (PEP 621 allows a direct git reference inside an extra.)

### Gotchas

- **Name clash:** don't reuse `ridge-detector` / `ridge_detector`; pick new names.
- **numba builds on first call** (`cache=True` mitigates); large wheel — exactly
  why it lives behind the `ridge-fast` extra, not the default install.
- **Upstream private API:** the optimized code pulls semi-internal symbols
  (`Line`, `Junction`, `LinesUtil`, `compute_gauss_mask_*`, `dirtab`, kernels…)
  from `ridge_detector.utils`/`.constants` — pin upstream and add a smoke test
  that imports them so an upstream update can't silently break the fork.
