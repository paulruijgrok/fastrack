#!/usr/bin/env python3
"""Expand a fast-batch manifest into one row per movie (for fast_movie_array.sbatch).

    python hpc/manifest_movies.py datasets.csv > movies.tsv

Columns (tab-separated, no header): index, dataset name, dataset base_dir,
config (or "-"), movie input (frame folder or stack file).  All paths are
absolute.  Uses the same manifest parser and movie discovery as fast-batch and
fast, so the movie set is exactly what `fast -d <base_dir>` would analyse.
"""
import sys

from fastrack.pipelines.batch import read_manifest
from fastrack.pipelines.discovery import discover_movies

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: manifest_movies.py <manifest.csv|.tsv|.xlsx>")
    idx = 0
    for spec in read_manifest(sys.argv[1]):
        movies = discover_movies(spec.base_dir)
        if not movies:
            sys.exit("no movies found under %s (dataset %s)" % (spec.base_dir, spec.name))
        for m in movies:
            print("\t".join([str(idx), spec.name, spec.base_dir, spec.config or "-", m["input"]]))
            idx += 1
