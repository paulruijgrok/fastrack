#!/usr/bin/env python3
"""Print the number of datasets in a fast-batch manifest.

Use it to size the job array in fast_batch_array.sbatch:

    N=$(python hpc/manifest_count.py datasets.csv)
    # then set  #SBATCH --array=0-$((T-1))  with T <= N
"""
import sys

from fastrack.pipelines.batch import read_manifest

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: manifest_count.py <manifest.csv|.tsv|.xlsx>")
    print(len(read_manifest(sys.argv[1])))
