"""The example library (configs/ and datasets/) stays loadable.

Every config must load through the same code path as ``--config`` (so a typo in
a section or key name fails here), the perturbation template must parse as a
sidecar, and every manifest must parse with each referenced config present.
The movie data under ``examples/`` is not in git, so dataset directories are
not required to exist.
"""
import glob
import os

import pytest

tomllib = pytest.importorskip("tomllib")

from fastrack.config import Settings  # noqa: E402
from fastrack.pipelines.batch import read_manifest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = sorted(glob.glob(os.path.join(ROOT, "configs", "*.toml")))
MANIFESTS = sorted(glob.glob(os.path.join(ROOT, "datasets", "*.csv")))
SIDECARS = [p for p in CONFIGS if os.path.basename(p).startswith("perturb.")]


def test_library_is_not_empty():
    assert len(CONFIGS) >= 4 and len(MANIFESTS) >= 3


@pytest.mark.parametrize("path", [p for p in CONFIGS if p not in SIDECARS],
                         ids=os.path.basename)
def test_config_loads_like_dash_dash_config(path):
    Settings.from_toml(path)


@pytest.mark.parametrize("path", SIDECARS, ids=os.path.basename)
def test_perturbation_template_parses(path):
    from fastrack.analysis.perturbation import from_sidecar
    p = from_sidecar(path)
    assert p is not None and len(p.switch_frames) >= 1


@pytest.mark.parametrize("path", MANIFESTS, ids=os.path.basename)
def test_manifest_parses_and_configs_exist(path):
    specs = read_manifest(path)
    assert specs, "no datasets in %s" % path
    for spec in specs:
        # resolved against the manifest's folder, so they must land in the repo
        assert os.path.commonpath([ROOT, spec.base_dir]) == ROOT
        if spec.config:
            assert os.path.isfile(spec.config), "%s: missing config %s" % (path, spec.config)
            assert os.path.dirname(spec.config) == os.path.join(ROOT, "configs")


def test_no_configs_or_manifests_left_at_the_repo_root():
    stray = [f for f in os.listdir(ROOT)
             if (f.startswith("config") and f.endswith(".toml")) or
             (f.startswith("datasets") and f.endswith(".csv"))]
    assert not stray, "move these into configs/ or datasets/: %s" % stray
