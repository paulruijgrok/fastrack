# FASTrack (Python 3 port)

A clean, modern Python 3 port of Tural Aksel's **FAST** / **FASTrack** package —
*Fast Actin filament Spud Trekker* — for automated analysis of in-vitro actin
gliding-assay movies (Aksel et al., *Cell Reports*, 2015).

The numerical algorithm is preserved verbatim from the original Python 2.7 code.
This port only updates it to run on current scientific-Python stacks and
replaces two external dependencies:

- **Parallelism:** the original spawned per-frame Python subprocesses through the
  `ppss` bash tool. This port uses Python's built-in `multiprocessing` instead —
  no external tool to install.
- **Movie encoding:** `avconv` is replaced by `ffmpeg`.

## What's included

| Module | Replaces | Purpose |
|--------|----------|---------|
| `fastrack/motility.py` | `FAST/motility.py` | Core image-processing + filament-tracking algorithm |
| `fastrack/pipeline.py` | `bin/fast` | Analysis driver (directory walk, parallel frame extraction, linking, plots, combined stats) |
| `fastrack/lima.py` | `bin/lima` | Loaded in-vitro motility assay (stop-model fit, force parameter `Ks`) |
| `fastrack/stack2tifs.py` | `bin/stack2tifs` | TIFF-stack → micro-manager frame-file conversion |
| `fastrack/cli.py` | argparse front-ends | Console entry points `fast`, `lima`, `stack2tifs` |
| `fastrack/plotparams.py` | `FAST/plotparams.py` | Shared Matplotlib styling (headless Agg backend) |

## Test data not included

The example movies used below (`examples/`) and the original upstream sources
(`Old code/`) are **not** committed to this repository — the gliding-assay
dataset alone is well over a gigabyte. To reproduce the validation runs, obtain
the original test dataset and place it under `examples/` with the same layout
(e.g. `examples/unloaded_motility/micromanager_tifs/...`). The original code and
data are available from Tural Aksel's upstream project at
https://github.com/turalaksel/FAST.

## Install (on your Mac)

```bash
cd "/Users/paulruijgrok/Documents/Claude/Projects/FASTrack"
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the runtime dependencies (numpy, scipy, scikit-image,
opencv-python, matplotlib, imageio) and the three command-line tools `fast`,
`lima`, and `stack2tifs`.

For movie generation (`fast -m`) you also need ffmpeg:

```bash
brew install ffmpeg
```

## Run on the test dataset

The unloaded-motility example contains α-cardiac and β-cardiac myosin movies.
From the project root:

```bash
fast -d examples/unloaded_motility/micromanager_tifs
```

Results are written under `outputs/<dataset>__pt_none__n_5__ymax_1500__p_5__fx_none/`,
including per-movie `*_length_velocity.png` plots and combined
`combined/MEAN_values.txt` / `combined/SEM_values.txt`.

Useful flags (defaults match the original): `-px 80.65` (pixel size nm),
`-n 5` (frames averaged), `-p 5` (min path length), `-maxd 2016.25`
(max inter-frame distance nm), `-minv 80` (stuck threshold nm/s),
`-j N` (worker processes), `-f` (force re-analysis), `-m` (make tracking movie),
`-v` (verbose).

To convert the original stacks to frame files first (if needed):

```bash
stack2tifs -d examples/unloaded_motility/stacks
```

For the loaded-motility (utrophin) series, run `fast` on that tree, then:

```bash
lima -d outputs/<loaded_dataset_output_dir>
```

## Validate against the paper

The key qualitative check from Aksel et al. 2015: **α-cardiac myosin glides
filaments substantially faster than β-cardiac** (roughly 1.5–2.5×). Open the
combined `MEAN_values.txt` and compare the `top-vel-5` / `MVEL` columns for the
`alpha_*` vs `beta_*` rows. Expected order-of-magnitude: β-cardiac top velocities
in the ~500–900 nm/s range, with α-cardiac clearly higher. The α > β ordering is
the primary parity criterion.

## Tests

```bash
pip install -e .[test]
pytest -q
```

`tests/test_smoke.py` covers the pure-numeric helpers (Gaussian fitting,
Uyeda length–velocity model, coupling-velocity fit, contour distance, binning).

## Notes / things to review

- **`make_frame_links` partner recovery.** The original recovered the accepted
  partner filament by relying on `filament2` being the last loop variable after
  the candidate loop (a latent bug: the link's scores describe the sorted-best
  candidate, but its partner identity/geometry came from whatever filament was
  iterated last). By default this port fixes that, recovering the partner by
  label so identity and scores agree:
  `filament2 = self.frame2.filaments[int(link_candidates[0, 0])]`.

  To reproduce the published results bit-for-bit, pass `--legacy-linking` to
  `fast` (or set `legacy_linking=True` on a `Motility` object / in
  `pipeline.run(...)`), which restores the original leftover-loop-variable
  behaviour. Running the dataset both ways lets you measure how much the fix
  actually moves the velocities:

  ```bash
  fast -d examples/unloaded_motility/micromanager_tifs            # corrected (default)
  fast -d examples/unloaded_motility/micromanager_tifs --legacy-linking -f
  ```

  (Use `-f` on the second run to force re-analysis rather than reusing cached
  links.)
- The image pipeline (scipy/scikit-image/opencv) could not be executed in the
  build sandbox, so end-to-end numbers must be produced on your Mac using the
  steps above.
- All Matplotlib output uses the headless `Agg` backend, so plots render without
  a display server.

## Credits and license

The original FAST / FASTrack algorithm and code were written by **Tural Aksel**
(turalaksel@gmail.com) and published in Aksel et al., "Ensemble force changes
that result from human cardiac myosin mutations and a small-molecule effector,"
*Cell Reports*, 2015. Upstream: https://github.com/turalaksel/FAST.

This repository is a Python 3 port that preserves Aksel's numerical algorithm
and modernizes the surrounding code. It is distributed under the same MIT license
as the original; see [`LICENSE`](LICENSE), which retains Tural Aksel's copyright.
