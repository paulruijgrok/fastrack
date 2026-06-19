"""Layered configuration for a motility analysis run.

Settings are grouped by concern -- ``hardware`` (microscope/camera),
``analysis`` (algorithm parameters and strategy selection), ``plotting``
(presentation), and ``runtime`` (execution) -- so that different setups can be
composed from small, independent pieces.

The :meth:`Settings.from_sources` constructor is the seam for that composition:
it merges any number of layers (built-in defaults -> base file -> hardware
overlay -> experiment overlay -> CLI overrides), each later layer overriding the
earlier ones.  Layers are plain nested dicts here; loading them from TOML/YAML
is a thin adapter that lives on top of this (see :meth:`Settings.from_toml`),
and adding it later touches only this module -- not the pipeline or the CLI.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, Mapping


@dataclass
class HardwareSettings:
    pixel_size_nm: float = 80.65
    #: Max inter-frame travel allowed for a filament (nm); the linking gate.
    max_inter_frame_distance_nm: float = 2016.25


@dataclass
class AnalysisSettings:
    num_frames_ave: int = 5
    min_path_length: int = 5
    percent_tolerance: int = 500
    #: Average path velocity below which a filament is "stuck" (nm/s).
    stuck_velocity_nm_s: float = 80.0
    overlap_score_cutoff: float = 0.4
    log_area_score_cutoff: float = 1.0
    diff_log_area_score_cutoff: float = 0.5
    fit_function: str = "none"
    # Pluggable strategy selection (looked up in the registries).
    detection_algorithm: str = "entropy"
    tracking_algorithm: str = "greedy"
    legacy_linking: bool = False
    fast_rank: bool = True
    morph_contrast: bool = False


@dataclass
class PlottingSettings:
    ymax: int = 1500
    xmax: int = 10000
    maxvel_color: str = "b"


@dataclass
class RuntimeSettings:
    force_analysis: bool = False
    recalculate: bool = False
    make_movie: bool = False
    overlay_movie: bool = False
    #: filXYs cache layout: "per-frame" (one .npy/frame, default) or "per-movie".
    cache_layout: str = "per-frame"
    #: write a tidy per-movie trajectory CSV (and optionally the contour geometry).
    export_trajectories: bool = False
    export_contours: bool = False
    nprocs: Any = None  # None -> all cores
    verbose: bool = False


@dataclass
class RidgeSettings:
    """Parameters for the optional ridge detector (analysis.detection_algorithm='ridge')."""
    line_widths: list = field(default_factory=lambda: [3])
    low_contrast: float = 50.0
    high_contrast: float = 150.0
    min_len: float = 10.0
    max_len: float = 0.0
    dark_line: bool = False
    estimate_width: bool = True


@dataclass
class OverlaySettings:
    """Styling for the overlay movie (fast --overlay-movie)."""
    fps: float = 10.0                 # playback frame rate
    frame_label: bool = True          # show frame number (bottom-left)
    time_label: bool = True           # show mm:ss time (bottom-right)
    frame_interval_s: float = 1.0     # seconds/frame fallback when no metadata
    font_scale: float = 0.6


_SECTIONS = {
    "hardware": HardwareSettings,
    "analysis": AnalysisSettings,
    "plotting": PlottingSettings,
    "runtime": RuntimeSettings,
    "ridge": RidgeSettings,
    "overlay": OverlaySettings,
}


@dataclass
class Settings:
    hardware: HardwareSettings = field(default_factory=HardwareSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    plotting: PlottingSettings = field(default_factory=PlottingSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    ridge: RidgeSettings = field(default_factory=RidgeSettings)
    overlay: OverlaySettings = field(default_factory=OverlaySettings)

    # ------------------------------------------------------------------ #
    # Construction / layering
    # ------------------------------------------------------------------ #
    @classmethod
    def from_sources(cls, *layers: Mapping[str, Mapping[str, Any]]) -> "Settings":
        """Build settings by merging nested-dict ``layers`` left-to-right.

        Each layer looks like ``{"hardware": {...}, "analysis": {...}}`` and may
        set only the keys it cares about; later layers win.  Unknown keys raise,
        to catch typos in config files early.
        """
        merged: Dict[str, Dict[str, Any]] = {name: {} for name in _SECTIONS}
        for layer in layers:
            if not layer:
                continue
            for section, values in layer.items():
                if section not in _SECTIONS:
                    raise KeyError("Unknown settings section: %r" % section)
                valid = {f.name for f in fields(_SECTIONS[section])}
                for key, value in dict(values).items():
                    if key not in valid:
                        raise KeyError("Unknown %s setting: %r" % (section, key))
                    merged[section][key] = value
        return cls(**{name: _SECTIONS[name](**vals) for name, vals in merged.items()})

    def with_overrides(self, **flat: Any) -> "Settings":
        """Return a copy with flat ``field=value`` overrides.

        Field names are unique across sections, so callers (e.g. the CLI) can
        pass them flat without knowing which section they live in.  ``None``
        values are ignored, so unset CLI flags don't clobber defaults.
        """
        index = {}
        for name, sect_cls in _SECTIONS.items():
            for f in fields(sect_cls):
                index[f.name] = name
        updates: Dict[str, Dict[str, Any]] = {name: {} for name in _SECTIONS}
        for key, value in flat.items():
            if value is None:
                continue
            if key not in index:
                raise KeyError("Unknown setting: %r" % key)
            updates[index[key]][key] = value
        new_sections = {
            name: replace(getattr(self, name), **updates[name]) for name in _SECTIONS
        }
        return replace(self, **new_sections)

    @classmethod
    def from_toml(cls, *paths: str) -> "Settings":
        """Load and merge TOML config files (requires Python 3.11+ tomllib)."""
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - py<3.11
            raise RuntimeError(
                "TOML config requires Python 3.11+ (tomllib). Use from_sources(dict)."
            )
        layers = []
        for path in paths:
            with open(path, "rb") as f:
                layers.append(tomllib.load(f))
        return cls.from_sources(*layers)

    # ------------------------------------------------------------------ #
    # Adapters
    # ------------------------------------------------------------------ #
    def to_run_kwargs(self) -> Dict[str, Any]:
        """Flatten to the keyword arguments accepted by ``pipeline.run``."""
        hw, an, pl, rt = self.hardware, self.analysis, self.plotting, self.runtime
        return {
            "pixel_size": hw.pixel_size_nm,
            "max_velocity": hw.max_inter_frame_distance_nm,
            "num_frames_ave": an.num_frames_ave,
            "min_path_length": an.min_path_length,
            "percent_tolerance": an.percent_tolerance,
            "min_velocity": an.stuck_velocity_nm_s,
            "overlap_score_cutoff": an.overlap_score_cutoff,
            "log_area_score_cutoff": an.log_area_score_cutoff,
            "diff_log_area_score_cutoff": an.diff_log_area_score_cutoff,
            "fit_function": an.fit_function,
            "detection_algorithm": an.detection_algorithm,
            "detection_params": (asdict(self.ridge)
                                 if an.detection_algorithm in ("ridge", "ridge-fast") else {}),
            "tracking_algorithm": an.tracking_algorithm,
            "legacy_linking": an.legacy_linking,
            "fast_rank": an.fast_rank,
            "morph_contrast": an.morph_contrast,
            "plot_ymax": pl.ymax,
            "plot_xmax": pl.xmax,
            "maxvel_color": pl.maxvel_color,
            "force_analysis": rt.force_analysis,
            "recalculate": rt.recalculate,
            "make_movie": rt.make_movie,
            "overlay_movie": rt.overlay_movie,
            "cache_layout": rt.cache_layout,
            "export_trajectories": rt.export_trajectories,
            "export_contours": rt.export_contours,
            "overlay_fps": self.overlay.fps,
            "overlay_frame_label": self.overlay.frame_label,
            "overlay_time_label": self.overlay.time_label,
            "overlay_frame_interval_s": self.overlay.frame_interval_s,
            "overlay_font_scale": self.overlay.font_scale,
            "nprocs": rt.nprocs,
            "verbose": rt.verbose,
        }

    def as_dict(self) -> Dict[str, Any]:
        """Nested-dict view (round-trips through ``from_sources``)."""
        return asdict(self)
