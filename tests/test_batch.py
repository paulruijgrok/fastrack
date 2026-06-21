"""Tests for the unattended batch runner (no image stack needed).

manifest parsing, the input signature, pre-flight checks and the state file are
all importable/runnable without scipy/skimage because batch.py imports the
gliding pipeline lazily.
"""
import os

import pytest

from fastrack.pipelines import batch


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").close()


def _make_dataset(root, name, n_tifs=2):
    """A dataset folder with one movie leaf containing n_tifs frames."""
    movie = os.path.join(root, name, "slide_1", "_1")
    for i in range(n_tifs):
        _touch(os.path.join(movie, "img_000000%03d_MMStack_000.tif" % i))
    return os.path.join(root, name)


def test_read_manifest_csv_resolves_and_aliases(tmp_path):
    base = _make_dataset(str(tmp_path / "data"), "ds1")
    man = tmp_path / "list.csv"
    man.write_text(
        "# a comment line\n"
        "Name,Directory,Config\n"
        "first,data/ds1,\n"
        "\n"                                  # blank row ignored
        ",data/ds1,\n"                        # no name -> derived, deduped
    )
    specs = batch.read_manifest(str(man))
    assert [s.name for s in specs] == ["first", "ds1"]
    assert specs[0].base_dir == os.path.normpath(str(tmp_path / "data" / "ds1"))
    assert specs[0].config is None
    assert os.path.isdir(base)


def test_read_manifest_dedupes_names(tmp_path):
    _make_dataset(str(tmp_path), "ds")
    man = tmp_path / "l.tsv"
    man.write_text("name\tbase_dir\nx\tds\nx\tds\n")
    specs = batch.read_manifest(str(man))
    assert [s.name for s in specs] == ["x", "x#1"]


def test_manifest_requires_base_column(tmp_path):
    man = tmp_path / "bad.csv"
    man.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError):
        batch.read_manifest(str(man))


def test_signature_changes_when_inputs_change(tmp_path):
    base = _make_dataset(str(tmp_path), "ds", n_tifs=2)
    spec = batch.DatasetSpec(name="ds", base_dir=base)
    s1 = batch.signature(spec)
    assert batch.signature(spec) == s1            # stable
    _touch(os.path.join(base, "slide_1", "_1", "img_000000099_MMStack_000.tif"))
    assert batch.signature(spec) != s1            # new frame -> new signature


def test_preflight_flags_missing_dir_and_missing_tifs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                    # keep ./outputs out of the repo
    missing = batch.DatasetSpec(name="m", base_dir=str(tmp_path / "nope"))
    assert any("not found" in p for p in batch.preflight(missing))

    empty = tmp_path / "empty"
    empty.mkdir()
    probs = batch.preflight(batch.DatasetSpec(name="e", base_dir=str(empty)))
    assert any("no movie folders" in p for p in probs)


def test_preflight_ok_and_missing_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = _make_dataset(str(tmp_path), "ds")
    assert batch.preflight(batch.DatasetSpec(name="ds", base_dir=base)) == []
    probs = batch.preflight(
        batch.DatasetSpec(name="ds", base_dir=base, config=str(tmp_path / "nope.toml")))
    assert any("config file not found" in p for p in probs)


def test_state_roundtrip(tmp_path):
    p = str(tmp_path / "state.json")
    assert batch.load_state(p) == {"datasets": {}}
    st = {"datasets": {"ds": {"status": "done", "signature": "abc"}}}
    batch.save_state(p, st)
    assert batch.load_state(p) == st


def _fake_gliding(monkeypatch):
    """Install a stub gliding whose run() succeeds unless the path says 'bad'."""
    import sys
    import types as _t
    mod = _t.ModuleType("fastrack.pipelines.gliding")
    mod.calls = []

    def run(main_dir=None, **kw):
        mod.calls.append((main_dir, kw))
        if "bad" in str(main_dir):
            raise RuntimeError("boom")
        if "exit" in str(main_dir):
            raise SystemExit("Directory doesn't exist. Program is exiting.")
    mod.run = run
    monkeypatch.setitem(sys.modules, "fastrack.pipelines.gliding", mod)
    return mod


def test_run_batch_continues_persists_and_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    g = _fake_gliding(monkeypatch)
    ok = _make_dataset(str(tmp_path), "ok_ds")
    bad = _make_dataset(str(tmp_path), "bad_ds")
    ex = _make_dataset(str(tmp_path), "exit_ds")
    man = tmp_path / "m.csv"
    man.write_text("name,base_dir\nok,%s\nbad,%s\nexitcase,%s\n" % (ok, bad, ex))
    logdir = str(tmp_path / "logs")

    out = batch.run_batch(str(man), logdir=logdir)
    # the batch forces the pipeline so a pre-existing outputs/ tree can't no-op it
    assert all(kw.get("force_analysis") is True for _md, kw in g.calls)
    # one success, two failures -- and the run did NOT stop at the first failure
    # (SystemExit from gliding is caught too, not allowed to kill the batch).
    assert out["results"] == {"ok": "done", "bad": "failed", "exitcase": "failed"}

    state = batch.load_state(os.path.join(logdir, "batch_state.json"))["datasets"]
    assert state["ok"]["status"] == "done" and "signature" in state["ok"]
    assert state["bad"]["status"] == "failed"

    # resume: success skipped; failures skipped unless --retry-failed
    out2 = batch.run_batch(str(man), logdir=logdir)
    assert out2["results"]["ok"] == "skipped"
    assert out2["results"]["bad"] == "skipped_failed"

    out3 = batch.run_batch(str(man), logdir=logdir, retry_failed=True)
    assert out3["results"]["bad"] == "failed"      # retried (and failed again)
    assert out3["results"]["ok"] == "skipped"      # still skipped (done)


def test_sharding_splits_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    g = _fake_gliding(monkeypatch)
    rows = ["name,base_dir"]
    for i in range(5):
        rows.append("ds%d,%s" % (i, _make_dataset(str(tmp_path), "ds%d" % i)))
    man = tmp_path / "m.csv"
    man.write_text("\n".join(rows) + "\n")

    # 5 datasets, 2 shards -> ceil(5/2)=3 then 2; per-shard state files
    out0 = batch.run_batch(str(man), logdir=str(tmp_path / "logs"),
                           num_shards=2, shard_index=0)
    out1 = batch.run_batch(str(man), logdir=str(tmp_path / "logs"),
                           num_shards=2, shard_index=1)
    assert list(out0["results"]) == ["ds0", "ds1", "ds2"]
    assert list(out1["results"]) == ["ds3", "ds4"]
    assert os.path.isfile(str(tmp_path / "logs" / "batch_state_shard0.json"))
    assert os.path.isfile(str(tmp_path / "logs" / "batch_state_shard1.json"))


def test_preflight_only_does_not_process(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fake_gliding(monkeypatch)
    base = _make_dataset(str(tmp_path), "ds")
    man = tmp_path / "m.csv"
    man.write_text("name,base_dir\nds,%s\n" % base)
    out = batch.run_batch(str(man), logdir=str(tmp_path / "logs"), preflight_only=True)
    assert out["results"] == {}                    # nothing processed
    assert not os.path.isfile(os.path.join(str(tmp_path / "logs"), "batch_state.json"))
