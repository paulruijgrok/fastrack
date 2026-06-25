"""FASTplus: directional (polarity-aware) gliding-motility analysis.

This subpackage adds *signed* velocity scoring to FASTrack by using a second
fluorescence channel that marks one polar end ("head") of each filament.  The
pieces are deliberately small and numpy-only so they can be unit-tested and
reused independently of the heavy image-processing core:

* :mod:`~fastrack.polarity.spot`            -- head detection / track records
* :mod:`~fastrack.polarity.datamodel`       -- PolarFilament, DirectionalPath
* :mod:`~fastrack.polarity.association`     -- head <-> filament association
* :mod:`~fastrack.polarity.disambiguation`  -- one-head-on-one-end inclusion gate
* :mod:`~fastrack.polarity.scoring`         -- signed-velocity scoring

The orchestration that wires these together with the detectors / head tracker
lives in :mod:`fastrack.pipelines.directional`.
"""
from .association import HeadFilamentAssociator
from .datamodel import (BOTH_ENDS, MIDDLE, NONE, PLUS_END, DirectionalPath,
                        PolarFilament)
from .disambiguation import PolarityClassifier
from .scoring import DirectionalScorer
from .spot import SpotRecord, SpotTable

__all__ = [
    "SpotRecord", "SpotTable",
    "PolarFilament", "DirectionalPath",
    "PLUS_END", "BOTH_ENDS", "MIDDLE", "NONE",
    "HeadFilamentAssociator", "PolarityClassifier", "DirectionalScorer",
]
