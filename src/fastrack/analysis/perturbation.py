"""External-perturbation (LED switch) timing for kinetic fitting (FASTplus).

The directional kinetic fit needs to know *when* the external perturbation
(typically a blue-light pulse train) switched on and off during a movie.  This
module resolves that schedule from several sources, in a unified representation,
and computes the per-cycle rise/decay segments used by
:class:`~fastrack.analysis.kinetics.KineticModelFitter`.

A :class:`Perturbation` is a list of *switch events*, each a (frame, state)
pair where ``state > 0`` means the perturbation is ON after that switch.  This
generalizes naturally to N on/off cycles.

SOURCES (resolved per movie, in precedence order by default = "auto")
---------------------------------------------------------------------
1. **Sidecar file** next to the movie: ``<stem>.perturb.toml`` (or .json/.yaml).
   The easiest, most explicit option::

       [perturbation]
       switch_frames = [98, 298]     # or switch_times_s = [13.29, 40.41]
       states        = [0, 1]        # optional; >0 = ON. default: alternate from OFF

2. **Legacy v0.1 LED files**: a per-frame times file ``<base>.csv`` (one
   acquisition time in ms per row) plus ``<base> led.csv`` (row 0 = switch
   times, row 1 = LED voltages).  Reproduces the original ``find_switch_signal_
   files`` / ``get_switch_frames`` logic.  ``<base>`` is the movie name with a
   trailing " RGB" stripped.

3. **Config / CLI**: explicit ``switch_frames`` or ``perturbation_times_s``
   applied to every movie (for datasets with a fixed, known schedule).

Frames are the stable quantity across replicate movies of identical format;
conversion to seconds uses the acquisition frame interval.  Only stdlib + numpy
are required (tomllib is stdlib on 3.11+; PyYAML is used only if a .yaml sidecar
is supplied and the package is installed).
"""
from __future__ import annotations

import csv
import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Perturbation:
    """Ordered external-perturbation switch events for one movie."""
    switch_frames: List[int] = field(default_factory=list)   # frame index of each switch
    states: List[float] = field(default_factory=list)        # state AFTER each switch (>0 = ON)
    source: str = "none"                                      # provenance, for logging
    initial_state: float = 0.0                               # state before the first switch

    # ------------------------------------------------------------------ #
    def __bool__(self) -> bool:
        return len(self.switch_frames) > 0

    def switch_times_s(self, frame_interval_s: float) -> List[float]:
        """Switch times in seconds = frame index * acquisition interval."""
        return [f * frame_interval_s for f in self.switch_frames]

    def on_off_frames(self) -> List[Tuple[int, int]]:
        """Return (on_frame, off_frame) pairs for each complete lit cycle."""
        pairs, on = [], None
        prev = self.initial_state
        for f, s in zip(self.switch_frames, self.states):
            if s > 0 and prev <= 0:
                on = f
            elif s <= 0 and prev > 0 and on is not None:
                pairs.append((on, f)); on = None
            prev = s
        return pairs

    def segments(self, n_frames: int, frame_interval_s: float) -> List[dict]:
        """Rise/decay segments implied by the switches (handles N cycles).

        A segment is emitted only at a *real* state transition (redundant
        markers that don't change state are ignored).  Each segment dict has
        ``t0_s`` (onset), ``end_s`` (next switch or trace end), ``model``
        ("exp_rise" when turning ON, "exp_decay" when turning OFF), and ``state``.
        """
        out = []
        prev = self.initial_state
        frames = list(self.switch_frames) + [n_frames]
        for i, s in enumerate(self.states):
            f0 = self.switch_frames[i]
            f1 = frames[i + 1]
            if (s > 0) != (prev > 0):                       # real transition
                out.append({
                    "t0_s": f0 * frame_interval_s,
                    "end_s": f1 * frame_interval_s,
                    "model": "exp_rise" if s > 0 else "exp_decay",
                    "state": s, "onset_frame": int(f0),
                })
            prev = s
        return out


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def from_frames(frames: Sequence[int], states: Optional[Sequence[float]] = None,
                source: str = "frames", initial_state: float = 0.0) -> Perturbation:
    frames = [int(round(f)) for f in frames]
    if states is None:                                       # alternate ON/OFF from OFF
        states = [1.0 if (i % 2 == 0) else 0.0 for i in range(len(frames))]
    return Perturbation(switch_frames=frames, states=[float(s) for s in states],
                        source=source, initial_state=float(initial_state))


def from_times_s(times_s: Sequence[float], frame_interval_s: float,
                 states: Optional[Sequence[float]] = None,
                 source: str = "times_s") -> Perturbation:
    if not frame_interval_s or frame_interval_s <= 0:
        frame_interval_s = 1.0
    frames = [int(round(t / frame_interval_s)) for t in times_s]
    return from_frames(frames, states, source=source)


def from_led_csv(times_csv: str, led_csv: str, offset: int = 0) -> Optional[Perturbation]:
    """Reproduce the v0.1 ``get_switch_frames`` logic from the LED files."""
    if not (os.path.isfile(times_csv) and os.path.isfile(led_csv)):
        return None
    with open(times_csv) as f:
        times = [float(r[0]) for r in csv.reader(f) if r]
    times = times[1:]                                       # original drops first frame
    if not any(times):
        return None
    times = np.asarray(times)

    with open(led_csv) as f:
        rows = [r for r in csv.reader(f) if r]
    if len(rows) < 2:
        return None
    switch_times = np.asarray([float(x) for x in rows[0]])
    voltages = [float(x) for x in rows[1]]
    if not np.any(switch_times):
        return None

    # exact match (same clock); fall back to nearest within half a frame
    isin = np.isin(times, switch_times)
    frames = list(np.where(isin)[0])
    if len(frames) < len(switch_times):
        half = 0.5 * float(np.median(np.diff(times))) if len(times) > 1 else 0.0
        frames = []
        for st in switch_times:
            j = int(np.argmin(np.abs(times - st)))
            if abs(times[j] - st) <= half:
                frames.append(j)
    if offset > 0 and frames and offset < (len(times) - frames[-1]):
        frames = [f + offset for f in frames]

    # row 1 is the LED state DURING the interval BEFORE each switch; voltages[0]
    # is therefore the movie's initial state (matches the v0.1 comments:
    # switch_voltages[0] > 0  <=>  "video starts with LED on").
    states_before = list(voltages[:len(frames)])
    # trim to whole rise/decay cycles, exactly as the original did
    if len(frames) % 2 != 0 and states_before:
        if states_before[0] > 0.0:        # starts ON -> drop last
            frames, states_before = frames[:-1], states_before[:-1]
        elif states_before[-1] == 0.0:    # ends ON -> drop first
            frames, states_before = frames[1:], states_before[1:]
    if not frames:
        return None
    initial_state = float(states_before[0])
    # state AFTER each switch = toggle of the state before it (the LED flips)
    states_after = [0.0 if v > 0 else 1.0 for v in states_before]
    return Perturbation(switch_frames=[int(f) for f in frames],
                        states=states_after, source="led-csv",
                        initial_state=initial_state)


def from_sidecar(path: str) -> Optional[Perturbation]:
    """Load a ``.perturb.toml`` / ``.json`` / ``.yaml`` sidecar."""
    if not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    data = None
    if ext == ".toml":
        try:
            import tomllib                       # Python 3.11+
        except ModuleNotFoundError:
            try:
                import tomli as tomllib          # 3.8-3.10 backport
            except ModuleNotFoundError:
                warnings.warn("no TOML parser (need Python 3.11+ or 'tomli') for %s" % path)
                return None
        with open(path, "rb") as f:
            data = tomllib.load(f)
    elif ext == ".json":
        import json
        with open(path) as f:
            data = json.load(f)
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
        except Exception:
            warnings.warn("PyYAML not installed; cannot read %s" % path)
            return None
        with open(path) as f:
            data = yaml.safe_load(f)
    if not data:
        return None
    p = data.get("perturbation", data)
    states = p.get("states")
    init = float(p.get("initial_state", 0.0))
    if "switch_frames" in p:
        pert = from_frames(p["switch_frames"], states, source="sidecar:%s" % os.path.basename(path),
                           initial_state=init)
    elif "switch_times_s" in p:
        dt = float(p.get("frame_interval_s", 1.0))
        pert = from_times_s(p["switch_times_s"], dt, states,
                            source="sidecar:%s" % os.path.basename(path))
        pert.initial_state = init
    else:
        return None
    return pert


# --------------------------------------------------------------------------- #
# Per-movie resolver
# --------------------------------------------------------------------------- #
def _legacy_base(movie_path: str) -> str:
    """Movie path -> v0.1 base name (extension and trailing ' RGB' removed)."""
    stem = os.path.splitext(movie_path)[0]
    if stem.endswith(" RGB"):
        stem = stem[:-4]
    elif stem.endswith("_RGB"):
        stem = stem[:-4]
    return stem


def resolve(movie_path: str, *, source: str = "auto",
            config_switch_frames: Optional[Sequence[int]] = None,
            config_times_s: Optional[Sequence[float]] = None,
            config_states: Optional[Sequence[float]] = None,
            frame_interval_s: float = 1.0,
            verbose: bool = False) -> Perturbation:
    """Resolve the perturbation schedule for one movie.

    ``source``: "auto" (sidecar -> led.csv -> config), "sidecar", "led-csv",
    "config", or "none".
    """
    if source == "none":
        return Perturbation(source="none")

    def _try_sidecar():
        stem = os.path.splitext(movie_path)[0]
        bases = [stem]
        lb = _legacy_base(movie_path)
        if lb != stem:
            bases.append(lb)
        for b in bases:
            for ext in (".perturb.toml", ".perturb.json", ".perturb.yaml", ".perturb.yml"):
                pert = from_sidecar(b + ext)
                if pert:
                    return pert
        return None

    def _try_led():
        base = _legacy_base(movie_path)
        times_csv, led_csv = base + ".csv", base + " led.csv"
        return from_led_csv(times_csv, led_csv)

    def _try_config():
        if config_switch_frames:
            return from_frames(config_switch_frames, config_states, source="config")
        if config_times_s:
            return from_times_s(config_times_s, frame_interval_s, config_states,
                                source="config")
        return None

    order = {"auto": [_try_sidecar, _try_led, _try_config],
             "sidecar": [_try_sidecar], "led-csv": [_try_led],
             "config": [_try_config]}.get(source, [_try_sidecar, _try_led, _try_config])

    for fn in order:
        pert = fn()
        if pert:
            if verbose:
                print("[fastplus]     perturbation: %s -> frames %s states %s"
                      % (pert.source, pert.switch_frames, pert.states))
            return pert
    return Perturbation(source="none")
