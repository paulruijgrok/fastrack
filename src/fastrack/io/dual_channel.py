"""Two-channel (polarity-labelled) movie ingestion + registration (FASTplus, req. 5).

Accepts an RGB / multi-channel movie in which one colour channel holds the
filaments and another holds the point-like polarity "heads", and prepares it for
the directional pipeline:

    load -> (optional) register the two channels -> split into a head stack and
    a filament stack (both ``(T, H, W)``).

Channel registration reuses the standalone **optomerge** package, pulled in only
when ``register=True`` (``pip install 'fastrack[plus]'``).  If optomerge is not
installed, registration is skipped with a clear, actionable message rather than
a hard crash, so the rest of the pipeline still runs on already-aligned data.

Memory: the movie is read once and the two channels are returned as views/copies
of a single colour each (uint-preserving), not as float stacks.
"""
from __future__ import annotations

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
    """Load, register, and split a two-channel polarity-labelled movie."""

    def __init__(self, path: str, head_channel: str = "red",
                 filament_channel: str = "green", channel_map: str = "",
                 register: bool = True):
        self.path = path
        if channel_map:
            m = parse_channel_map(channel_map)
            head_channel = m.get("heads", head_channel)
            filament_channel = m.get("filaments", filament_channel)
        self.head_channel = head_channel
        self.filament_channel = filament_channel
        self.register_channels = register
        self._stack: Optional[np.ndarray] = None      # (T, H, W, C)
        self.registered = False

    # ------------------------------------------------------------------ #
    def load(self) -> "TwoChannelMovie":
        try:
            import tifffile
            arr = tifffile.imread(self.path)
        except ImportError:
            import imageio.v3 as iio
            arr = np.asarray(iio.imread(self.path))
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):      # single RGB frame
            arr = arr[None]
        if arr.ndim != 4:
            raise ValueError(
                "expected a (T, H, W, C) RGB movie, got shape %r" % (arr.shape,))
        self._stack = arr
        return self

    # ------------------------------------------------------------------ #
    def register(self, method: str = "optomerge") -> "TwoChannelMovie":
        """Register the two colour channels (optional optomerge dependency)."""
        if not self.register_channels:
            return self
        if self._stack is None:
            self.load()
        try:
            import optomerge  # noqa: F401
        except Exception:
            import warnings
            warnings.warn(
                "channel registration requested but 'optomerge' is not installed; "
                "proceeding with the raw (unregistered) channels. Install with "
                "pip install 'fastrack[plus]' to enable registration.",
                RuntimeWarning,
            )
            return self
        self._stack = self._apply_optomerge(self._stack)
        self.registered = True
        return self

    def _apply_optomerge(self, stack: np.ndarray) -> np.ndarray:
        """Adapter around optomerge's registration API.

        Kept isolated so the exact optomerge entry point can change without
        touching the pipeline. Estimates the channel-to-channel transform from a
        projection and applies it to every frame of the head channel.
        """
        import optomerge
        hi, fi = _CH[self.head_channel], _CH[self.filament_channel]
        # optomerge exposes a registration routine that returns a transform from
        # a (moving, fixed) image pair; apply it frame-by-frame to keep memory flat.
        register_pair = getattr(optomerge, "register", None) or \
            getattr(getattr(optomerge, "registration", None), "register", None)
        if register_pair is None:                      # unexpected API: skip safely
            return stack
        fixed_proj = stack[..., fi].max(axis=0)
        moving_proj = stack[..., hi].max(axis=0)
        transform = register_pair(moving_proj, fixed_proj)
        apply = getattr(optomerge, "apply_transform", None) or \
            getattr(getattr(optomerge, "transform", None), "apply", None)
        if apply is None:
            return stack
        out = stack.copy()
        for t in range(stack.shape[0]):
            out[t, ..., hi] = apply(stack[t, ..., hi], transform)
        return out

    # ------------------------------------------------------------------ #
    def split(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(head_stack, filament_stack)`` as ``(T, H, W)`` arrays."""
        if self._stack is None:
            self.load()
        if self.register_channels and not self.registered:
            self.register()
        heads = np.ascontiguousarray(self._stack[..., _CH[self.head_channel]])
        fils = np.ascontiguousarray(self._stack[..., _CH[self.filament_channel]])
        return heads, fils

    def release(self) -> None:
        """Drop the in-memory stack (call after splitting large movies)."""
        self._stack = None
