"""Polarity disambiguation: the unambiguous-labelling inclusion gate (FASTplus).

A filament is included in directional analysis only if it is *unambiguously*
polarity-labelled: exactly **one** head, sitting on exactly **one** tip.  All
other cases are excluded:

* heads on **both** tips        -> ``BOTH_ENDS``
* head(s) only in the **middle** -> ``MIDDLE``
* **no** head                    -> ``NONE``

For an included filament the marked tip becomes the plus-end and the opposite
tip the minus-end, fixing the intrinsic polar axis used for signed scoring.
Consumes the ``_regions`` / ``_tips`` annotations produced by
:class:`~fastrack.polarity.association.HeadFilamentAssociator`.  numpy only.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .datamodel import BOTH_ENDS, MIDDLE, NONE, PLUS_END, PolarFilament


class PolarityClassifier:
    """Classify polar filaments and keep only the unambiguously labelled ones."""

    def classify(self, pf: PolarFilament) -> str:
        regions = getattr(pf, "_regions", {"tip0": [], "tip1": [], "middle": []})
        n0, n1, nm = len(regions["tip0"]), len(regions["tip1"]), len(regions["middle"])

        if n0 == 0 and n1 == 0 and nm == 0:
            pf.classification = NONE
        elif n0 > 0 and n1 > 0:
            pf.classification = BOTH_ENDS
        elif nm > 0 and n0 == 0 and n1 == 0:
            pf.classification = MIDDLE
        elif (n0 == 1 and n1 == 0 and nm == 0) or (n1 == 1 and n0 == 0 and nm == 0):
            # exactly one head on exactly one tip -> unambiguous
            tip0, tip1 = getattr(pf, "_tips", (None, None))
            if n0 == 1:
                plus, minus, head = tip0, tip1, regions["tip0"][0]
            else:
                plus, minus, head = tip1, tip0, regions["tip1"][0]
            pf.plus_end_xy = np.asarray(plus, float)
            pf.minus_end_xy = np.asarray(minus, float)
            pf.head_ids = [head.track_id] if head.track_id is not None else []
            pf.classification = PLUS_END
        else:
            # e.g. one tip head + a middle head, or two heads on one tip -> ambiguous
            pf.classification = MIDDLE if (n0 + n1) == 0 else BOTH_ENDS
        return pf.classification

    def classify_all(self, polar_filaments: List[PolarFilament]) -> List[PolarFilament]:
        for pf in polar_filaments:
            self.classify(pf)
        return polar_filaments

    def filter(self, polar_filaments: List[PolarFilament]) -> List[PolarFilament]:
        """Return only the unambiguously polarity-labelled filaments."""
        return [pf for pf in self.classify_all(polar_filaments) if pf.is_unambiguous]

    @staticmethod
    def counts(polar_filaments: List[PolarFilament]) -> dict:
        """Tally of classifications (useful for QC / reporting)."""
        out = {PLUS_END: 0, BOTH_ENDS: 0, MIDDLE: 0, NONE: 0}
        for pf in polar_filaments:
            out[pf.classification] = out.get(pf.classification, 0) + 1
        return out
