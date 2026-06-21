# GPU acceleration feasibility for FASTrack

**Status: report only — no GPU code.** This documents where FASTrack spends its
time, what a GPU could and couldn't accelerate, the porting effort and risks, and
a concrete experiment plan if you decide to pursue it.

## TL;DR / recommendation

For this workload, **CPU job arrays are the right tool now** (see `hpc/`), and a
GPU is a *research investigation*, not a quick win:

- The analysis is **embarrassingly parallel** at two levels — per-frame within a
  dataset (`-j` multiprocessing) and per-dataset across the cluster (one array
  shard per task). That already gives near-linear throughput on CPUs, which are
  plentiful and need no code changes.
- A GPU would only speed up the **per-frame detection compute**, which is a
  *fraction* of wall-clock (the rest — linking, plotting, I/O, encoding — stays
  on CPU), so Amdahl's law caps the end-to-end gain.
- The functions that dominate detection (local **percentile/rank filters** and
  **watershed**) do **not** have drop-in GPU equivalents in cuCIM (as of the
  26.x series — verify against the API list for your version), so a GPU port
  means writing and validating custom CUDA/CuPy kernels, with uncertain payoff
  and a hardware requirement to test.

Spend effort on more CPU cores / more array tasks before GPU.

## Where the time actually goes

Profiled hotspots (per frame), by detector:

**Entropy detector (default).** The dominant cost is **two radius-15 local
percentile (rank) filters** in `entropy_clusters` (and `check_picture_quality`),
followed by **Otsu thresholding**, **watershed** decomposition, and
**skeletonization**. scikit-image's rank filters maintain a per-pixel local
histogram whose size scales with the number of grey levels — which is why the
8-bit `fast_rank` path is ~1.8× faster than the 16-bit path. This rank-filter
step is the single biggest CPU consumer.

**Ridge detector (`ridge` / `ridge-fast`).** Gaussian derivative convolutions →
a 2×2 Hessian eigendecomposition per pixel → sub-pixel contour tracing. The
`ridge-fast` fork already optimizes this on CPU (analytical 2×2 eig, float32,
OpenCV separable filters, a numba-compiled tracer) for ~4× over the reference.
The contour-tracing inner loop is inherently **sequential** (it walks along each
ridge), which is the least GPU-friendly part.

Everything after detection — frame-to-frame linking, path building, the
matplotlib plots, CSV/movie output — is CPU/IO work that a GPU does not touch.

## What a GPU could accelerate (and how cleanly)

Tooling options: **cuCIM/CuPy** (a GPU reimplementation of much of scikit-image),
**CuPy** custom kernels / `cupyx.scipy.ndimage`, and **numba.cuda**.

| Stage | GPU path | Portability |
|---|---|---|
| Gaussian / separable convolutions (ridge) | cuCIM / CuPy / `cupyx.scipy.ndimage` | **Good** — well supported |
| Otsu threshold, morphology, connected components | cuCIM | **Good** — ported in cuCIM |
| 2×2 Hessian eigendecomposition (ridge) | CuPy elementwise / numba.cuda | **Good** — embarrassingly parallel per pixel |
| **Local percentile / rank filter (entropy hotspot)** | no cuCIM equivalent; custom CuPy/CUDA kernel | **Poor** — must implement + validate yourself |
| **Watershed (entropy)** | limited/!absent in cuCIM | **Poor** — would need a GPU watershed |
| Sub-pixel contour tracing (ridge) | sequential walk | **Poor** — not GPU-friendly |
| Linking / plotting / movie / CSV | n/a | stays on CPU |

So the *ridge* detector is the more GPU-amenable of the two (its heavy parts are
convolutions + per-pixel eigen-math), while the *entropy* detector's two biggest
costs (rank filter, watershed) are exactly the parts cuCIM doesn't give you for
free.

## Why CPU arrays usually win here

- **Throughput scales by adding tasks, not cores-per-task.** With N datasets you
  can run N shards concurrently; total throughput is bounded by cluster CPU
  availability, not by any single device. A pile of CPU cores typically beats one
  GPU on $/throughput for parallel-per-item batch work — and it's available today
  with zero code change.
- **Small frames are transfer-bound.** Gliding-assay frames are ~1000×1000; the
  per-frame GPU compute is short relative to host↔device transfer, so naive
  per-frame offload can be dominated by copies unless you batch many frames into
  GPU memory and keep them resident.
- **Amdahl ceiling.** If detection is, say, ~60% of wall-clock (as measured for
  ridge vs. the full pipeline), even an *infinitely* fast detector caps the
  end-to-end speedup at ~2.5×; the linking/plotting/I/O remainder is unaffected.

## Costs and risks of a GPU port

- **New kernels to write and maintain** for the rank filter (and possibly
  watershed) — the parts with no library support.
- **Numerical parity:** the golden-master test must still pass. float32 GPU math
  and different reductions can perturb results (we already saw last-digit drift
  with the float32 `ridge-fast`); re-validation is mandatory.
- **Dependency weight:** cuCIM/CuPy pin to CUDA/driver versions and only run on
  GPU nodes — it would have to be an optional extra, like `ridge`.
- **Hardware to validate:** needs a GPU node to develop and benchmark against.

## If you pursue it: a staged experiment plan

1. **Measure first.** Use `tools/compare_detectors.py --time-detection` and
   `tools/profile_frame.py` to confirm what fraction of wall-clock detection
   actually is on your data and which detector you'll run in production. If
   detection isn't dominant, stop — GPU can't help much.
2. **Pick the amenable target:** prototype the **ridge** path (convolutions +
   2×2 eig) in CuPy, processing a *stack* of frames resident on the GPU, and
   benchmark **including** host↔device transfer against `ridge-fast` on CPU.
3. **Only if step 2 shows a real win**, tackle the entropy hotspot: implement the
   local percentile/rank filter as a CuPy/CUDA kernel and benchmark a full frame
   end-to-end.
4. **Validate** against the golden master (`tests/test_golden.py`) and gate the
   GPU detector behind an optional extra + a registry entry (same seam as
   `ridge`/`ridge-fast`), so it never affects the default install.
5. **Compare the realistic alternative**: the same effort spent enabling more
   array tasks / more CPU partitions, to make sure GPU actually wins on
   throughput-per-effort.

## Verify for your stack

cuCIM's covered-function list changes between releases; before relying on any row
above, check the API for your installed version
(`https://docs.rapids.ai/api/cucim/stable/`) — particularly whether
`cucim.skimage.filters.rank` and `cucim.skimage.segmentation.watershed` exist.

## Sources

- [Accelerating Scikit-Image API with cuCIM — NVIDIA Technical Blog](https://developer.nvidia.com/blog/cucim-rapid-n-dimensional-image-processing-and-i-o-on-gpus/)
- [cuCIM documentation (RAPIDS)](https://docs.rapids.ai/api/cucim/stable/)
- [RAPIDS cuCIM: Porting scikit-image Code to the GPU — Quansight](https://quansight.com/post/rapids-cucim-porting-scikit-image-code-to-the-gpu/)
- [skimage.filters.rank — scikit-image documentation](https://scikit-image.org/docs/stable/api/skimage.filters.rank.html)
