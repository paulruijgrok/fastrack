"""FASTrack: Fast Actin filament Spud Trekker.

Modernized Python 3 port of Tural Aksel's FAST/FASTrack package for automated
analysis of in-vitro actin-gliding motility assays (Aksel et al., Cell Reports,
2015).

This release reorganizes the code into logical sub-packages:

    fastrack.config      - layered Settings (hardware / analysis / runtime)
    fastrack.datamodel   - Filament records and the cross-frame FilamentTable
    fastrack.core        - Frame/Island/Filament, Link/Path, Motility driver
    fastrack.core.detection - pluggable filament-detection strategies
    fastrack.core.tracking  - pluggable frame-to-frame linking strategies
    fastrack.analysis    - fitting, velocity metrics, geometry helpers
    fastrack.io          - image reading, stores, export, movie writers
    fastrack.viz         - Matplotlib styling and plots
    fastrack.pipelines   - orchestration (gliding, loaded/LIMA)
    fastrack.cli         - console entry points (fast, lima, stack2tifs)

The top-level import is kept lightweight (no scipy/scikit-image pulled in just
to read ``fastrack.__version__``); import the sub-packages you need.  A
``fastrack.motility`` compatibility shim re-exports the pre-refactor names.
"""
__version__ = "3.0.0"

__all__ = ["__version__"]
