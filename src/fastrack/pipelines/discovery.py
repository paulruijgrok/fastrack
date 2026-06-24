"""Discover the movies under a dataset directory.

A *movie* is either a micro-manager **frame folder** (a leaf directory of
``img_******NNN_<tail>_000.tif`` frames, optionally with a cached ``filXYs``)
or a multi-page **TIFF stack** file.  Both map to the same ``(top_root, exp)``
identity so downstream grouping, naming, and per-condition statistics are
identical whether a movie was supplied as split frames or as a stack:

    .../slide_2/alpha_0.04mg_ml/_2/         (frame folder)  -> top_root=.../alpha_0.04mg_ml, exp=_2
    .../slide_2/alpha_0.04mg_ml/_2.tif      (stack file)    -> top_root=.../alpha_0.04mg_ml, exp=_2

so a stack and its pre-split frames produce matching output names.
"""
import os
import re

# A micro-manager per-frame file: img_<digits>_<tail>_000.tif
_MM_FRAME_RE = re.compile(r"^img_\d+_.*_000\.tif$")


def _is_stack_candidate(files):
    """``.tif`` files in this dir that are NOT micro-manager per-frame files."""
    return sorted(
        f for f in files
        if f.lower().endswith(".tif") and not _MM_FRAME_RE.match(f)
    )


def discover_movies(main_dir, input_format="auto"):
    """Return a list of movie dicts found under ``main_dir`` (sorted, stable).

    Each movie: ``{"kind": "mm"|"stack", "top_root", "exp", "input"}`` where
    ``input`` is the frame folder (mm) or the stack file (stack).  ``top_root``
    and ``exp`` are the grouping identity described in the module docstring.

    ``main_dir`` may also be a single ``.tif`` stack file.  ``input_format``
    overrides the per-folder auto-detection: ``"frames"`` treats every leaf with
    ``.tif`` as one micro-manager movie; ``"stack"`` treats every ``.tif`` as its
    own stack.
    """
    # A single stack file passed directly.
    if os.path.isfile(main_dir) and main_dir.lower().endswith((".tif", ".tiff")):
        return [{
            "kind": "stack",
            "top_root": os.path.dirname(main_dir) or ".",
            "exp": os.path.splitext(os.path.basename(main_dir))[0],
            "input": main_dir,
        }]

    movies = []
    for root, _subdirs, files in os.walk(main_dir):
        tifs = [f for f in files if f.lower().endswith(".tif")]
        mm_frames = [f for f in tifs if _MM_FRAME_RE.match(f)]
        has_filxys = any(f.startswith("filXYs") for f in files)

        if input_format == "frames":
            treat_as_mm = bool(tifs or has_filxys)
        elif input_format == "stack":
            treat_as_mm = False
        else:                                  # auto
            treat_as_mm = bool(mm_frames or (has_filxys and not tifs))

        if treat_as_mm:
            if tifs or has_filxys:
                movies.append({
                    "kind": "mm",
                    "top_root": os.path.dirname(root),
                    "exp": os.path.basename(root),
                    "input": root,
                })
        else:
            # each .tif here is its own stack movie (all of them when forced)
            stacks = (sorted(tifs) if input_format == "stack"
                      else _is_stack_candidate(files))
            for t in stacks:
                movies.append({
                    "kind": "stack",
                    "top_root": root,
                    "exp": os.path.splitext(t)[0],
                    "input": os.path.join(root, t),
                })
    movies.sort(key=lambda m: (m["top_root"], m["exp"]))
    return movies


def group_by_top_root(movies):
    """Group discovered movies by ``top_root`` (preserving sorted order)."""
    groups = {}
    for m in movies:
        groups.setdefault(m["top_root"], []).append(m)
    return groups
