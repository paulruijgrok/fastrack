"""Export a :class:`~fastrack.datamodel.FilamentTable` to tabular formats.

CSV is implemented with the standard library (no extra dependency).  Parquet /
HDF5 exporters can be added the same way; each consumes ``table.to_rows()`` so
the export format is decoupled from how filaments are computed or stored.
"""
import csv

from ..datamodel import FilamentTable

# Stable column order for the tidy per-filament table.
_COLUMNS = [
    "frame",
    "label",
    "n_points",
    "length",
    "density",
    "width",
    "area",
    "end2end",
    "midpoint_x",
    "midpoint_y",
    "path_id",
]


def to_csv(table: FilamentTable, path: str) -> None:
    """Write one row per filament to ``path`` as CSV."""
    rows = table.to_rows()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
