#!/bin/bash
# Sourced by the FASTrack SLURM scripts to activate the Python environment on a
# compute node.  Edit CONDA_ENV (and add any `module load` lines your cluster
# needs) once here, and every script picks it up.
#
# Matches the conda-activation idiom used elsewhere on the cluster:
#   source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate <env>

CONDA_ENV="${CONDA_ENV:-fastrack-ridge}"

# module load python/3.11   # uncomment if your cluster provides conda via a module
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
