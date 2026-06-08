"""Shared morphological structuring elements and constants.

These were module-level globals in the original ``motility.py``; they are
collected here so the core classes can share them without duplicating the
construction.
"""
import numpy as np
from skimage.morphology import disk

# Global structuring elements.
# ``square(n)`` (deprecated in scikit-image 0.26) is exactly an n-by-n array of
# uint8 ones, so we build it directly with numpy -- identical output, no
# deprecation, and independent of the scikit-image version.
sqr_1 = np.ones((1, 1), dtype=np.uint8)
sqr_2 = np.ones((2, 2), dtype=np.uint8)
sqr_3 = np.ones((3, 3), dtype=np.uint8)
disk_1 = disk(1)
disk_2 = disk(2)
disk_3 = disk(3)

# Very small number used to keep logarithms finite.
ZERO = 1e-100
