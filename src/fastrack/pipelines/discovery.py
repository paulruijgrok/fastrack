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


def discover_movies(main_dir):
    """Return a list of movie dicts found under ``main_dir`` (sorted, stable).

    Each movie: ``{"kind": "mm"|"stack", "top_root", "exp", "input"}`` where
    ``input`` is the frame folder (mm) or the stack file (stack).  ``top_root``
    and ``exp`` are the grouping identity described in the module docstring.
    """
    movies = []
    for root, _subdirs, files in os.walk(main_dir):
        tifs = [f for f in files if f.lower().endswith(".tif")]
        mm_frames = [f for f in tifs if _MM_FRAME_RE.match(f)]
        has_filxys = any(f.startswith("filXYs") for f in files)

        if mm_frames or (has_filxys and not tifs):
            # one micro-manager movie = this leaf folder
            movies.append({
                "kind": "mm",
                "top_root": os.path.dirname(root),
                "exp": os.path.basename(root),
                "input": root,
            })
        else:
            # each non-mm .tif here is its own stack movie
            for t in _is_stack_candidate(files):
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
