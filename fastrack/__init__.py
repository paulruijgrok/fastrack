"""FASTrack: Fast Actin filament Spud Trekker.

Modernized Python 3 port of Tural Aksel's FAST/FASTrack package for automated
analysis of in-vitro actin-gliding motility assays (Aksel et al., Cell Reports,
2015).

Public submodules:
    motility   - core image-processing and tracking algorithm
    pipeline   - the ``fast`` analysis driver (multiprocessing-based)
    lima       - loaded in-vitro motility assay analysis
    stack2tifs - TIFF stack -> frame-file conversion utility
    plotparams - shared Matplotlib styling (headless Agg backend)
"""
__version__ = "2.0.0"

__all__ = ["motility", "pipeline", "lima", "stack2tifs", "plotparams"]
