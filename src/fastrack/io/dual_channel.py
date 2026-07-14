"""Two-channel (polarity-labelled) movie ingestion (FASTplus, req. 5).

Prepares a two-colour, polarity-labelled movie for the directional pipeline by
splitting it into a head stack and a filament stack (both ``(T, H, W)``). Two
input modes are supported:

1. **Pre-registered RGB** (default) -- an already-aligned RGB / multi-channel
   movie in which one colour channel holds the filaments and another the
   point-like "heads". FASTplus just reads and splits it; no extra dependency.

2. **Raw, spatially-packed** (``register=True``) -- the two channels occupy
   different regions of one camera frame (e.g. filaments top-half / heads
   bottom-half). These are aligned + merged into an RGB movie in-process by the
   optional **optomerge** package (its feature aligner, built for point-vs-line
   channels), then split. ``channel_order`` names the spatial layout.

optomerge is an **optional** dependency: if ``register=True`` but optomerge is
not installed, FASTplus warns and falls back to treating ``path`` as an
already-registered RGB movie (mode 1). Install ``fastrack[plus-register]`` to
enable in-process preprocessing, or align separately with optomerge's batch
driver (see ``docs/fastplus.md``).

Memory: the movie is read once and the two channels are returned as contiguous
copies of a single colour each (uint-preserving), not as float stacks.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np

_CH = {"red": 0, "green": 1, "blue": 2}


def parse_channel_map(spec: str) -> Dict[str, str]:
    """Parse 'red=heads,green=filaments' -> {'heads':'red','filaments':'green'}."""
    out: Dict[str, str] = {}
    for part in str(spec).split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        colour, role = (s.strip().lower() for s in part.split("=", 1))
        out[role] = colour
    return out


class TwoChannelMovie:
    """Load and split a two-channel polarity-labelled movie (RGB or raw-packed)."""

    def __init__(self, path: str, head_channel: str = "red",
                 filament_channel: str = "green", channel_map: str = "",
                 register: bool = False, channel_order: str = "auto",
                 max_shift: float = 30.0, register_frames: Optional[int] = None,
                 rgb_bitdepth: int = 8):
        self.path = path
        if channel_map:
            m = parse_channel_map(channel_map)
            head_channel = m.get("heads", head_channel)
            filament_channel = m.get("filaments", filament_channel)
        self.head_channel = head_channel
        self.filament_channel = filament_channel
        # register=True -> treat `path` as a raw spatially-packed movie and merge
        # it into an aligned RGB with optomerge; False -> `path` is already RGB.
        self.register_channels = register
        self.channel_order = channel_order          # optomerge spatial layout
        self.max_shift = max_shift                  # feature-aligner search bound (px)
        self.register_frames = register_frames      # frames used for detection (None=all)
        self.rgb_bitdepth = rgb_bitdepth            # 8 or 16, for the merged RGB
        self._stack: Optional[np.ndarray] = None      # (T, H, W, C)
        self.registered = False

    # ------------------------------------------------------------------ #
    def load(self) -> "TwoChannelMovie":
        """Populate ``self._stack`` as an ``(T, H, W, C)`` RGB movie.

        When ``register=True`` the raw movie is aligned + merged via optomerge;
        otherwise (or if optomerge is unavailable) ``path`` is read as an
        already-registered RGB movie.
        """
        if self.register_channels:
            merged = self._merge_with_optomerge()
            if merged is not None:
                self._stack = merged
                self.registered = True
                return self
            # optomerge unavailable -> fall back to reading path as RGB.
        self._stack = self._read_rgb(self.path)
        return self

    # ------------------------------------------------------------------ #
    def _read_rgb(self, path: str) -> np.ndarray:
        """Read an already-aligned RGB / multi-channel movie as ``(T, H, W, C)``."""
        try:
            import tifffile
            arr = tifffile.imread(path)
        except ImportError:
            import imageio.v3 as iio
            arr = np.asarray(iio.imread(path))
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):      # single RGB frame
            arr = arr[None]
        if arr.ndim != 4:
            raise ValueError(
                "expected a (T, H, W, C) RGB movie, got shape %r" % (arr.shape,))
        return arr

    # ------------------------------------------------------------------ #
    def _merge_with_optomerge(self) -> Optional[np.ndarray]:
        """Align + merge the raw spatially-packed movie into ``(T, H, W, 3)`` RGB.

        Drives optomerge's ``MergePipeline`` with the feature aligner and returns
        the merged RGB as a uint stack matching a disk-saved RGB bit-for-bit
        (red=heads, green=filaments). Returns ``None`` (and warns) if optomerge
        is not installed, so ``load`` can fall back to the pre-registered path.
        """
        try:
            from optomerge import MergePipeline, Settings
        except Exception:
            warnings.warn(
                "register=True but 'optomerge' is not installed; cannot preprocess "
                "raw spatially-packed movies. Falling back to reading %r as an "
                "already-registered RGB movie. Install fastrack[plus-register] to "
                "enable in-process registration (or align separately with "
                "optomerge; see docs/fastplus.md)." % self.path,
                RuntimeWarning,
            )
            return None

        settings = Settings().with_overrides(
            method="feature", channel_order=self.channel_order,
            projection_frames=self.register_frames, max_shift=self.max_shift,
        )
        pipe = MergePipeline(source=self.path, rgb_bitdepth=self.rgb_bitdepth,
                             **settings.to_pipeline_kwargs())
        rgb = pipe.run(output=None)          # open + calibrate + merge, in-memory
        arr = rgb.to_array()                 # (H, W, 3, T) float in ~[0, 1]

        # Match optomerge's own float->uint conversion (save_rgb_tiff) so the
        # in-memory result is identical to a saved-then-reloaded aligned RGB.
        scale = 255 if self.rgb_bitdepth == 8 else 65535
        dtype = np.uint8 if self.rgb_bitdepth == 8 else np.uint16
        scaled = (np.clip(arr, 0.0, 1.0) * scale).astype(dtype)
        return np.moveaxis(scaled, -1, 0)    # (H, W, 3, T) -> (T, H, W, 3)

    # ------------------------------------------------------------------ #
    def register(self) -> "TwoChannelMovie":
        """Ensure the movie is loaded (this performs the merge when register=True)."""
        if self._stack is None:
            self.load()
        return self

    # ------------------------------------------------------------------ #
    def split(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(head_stack, filament_stack)`` as ``(T, H, W)`` arrays."""
        if self._stack is None:
            self.load()
        heads = np.ascontiguousarray(self._stack[..., _CH[self.head_channel]])
        fils = np.ascontiguousarray(self._stack[..., _CH[self.filament_channel]])
        return heads, fils

    def release(self) -> None:
        """Drop the in-memory stack (call after splitting large movies)."""
        self._stack = None
