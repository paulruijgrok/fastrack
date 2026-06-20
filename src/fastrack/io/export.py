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


# --------------------------------------------------------------------------- #
# Trajectory export -- the rich, downstream-friendly deliverable
# --------------------------------------------------------------------------- #
# Tidy "long" table: one row per filament per frame.  Group by (movie, path_id)
# to recover a full trajectory.  Positions/lengths in nm, time in seconds.
TRAJECTORY_COLUMNS = [
    "movie", "path_id", "frame", "time_s",
    "length_nm", "x_nm", "y_nm", "cm_x_nm", "cm_y_nm",
    "velocity_nm_s", "stuck", "n_points",
]

# Optional skeleton geometry: one row per contour point (joins on movie/path_id/frame).
CONTOUR_COLUMNS = ["movie", "path_id", "frame", "point", "x_nm", "y_nm"]


def write_rows_csv(rows, path: str, columns) -> int:
    """Write an iterable of dict ``rows`` to ``path`` as CSV; return the count.

    ``extrasaction='ignore'`` keeps a stray extra key from breaking the write.
    """
    return _rows_csv(rows, path, columns, mode="w", header=True)


def append_rows_csv(rows, path: str, columns, write_header: bool) -> int:
    """Append ``rows`` to ``path`` (writing the header only if ``write_header``).

    Used to accumulate one combined all-movies CSV across the run without holding
    every movie's rows in memory at once.
    """
    return _rows_csv(rows, path, columns, mode="a", header=write_header)


def _rows_csv(rows, path, columns, mode, header) -> int:
    n = 0
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        if header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
            n += 1
    return n
