"""Tests for movie discovery (frame folders vs TIFF stacks)."""
import os

from fastrack.pipelines.discovery import discover_movies, group_by_top_root


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").close()


def test_discovers_frame_folder_as_one_movie(tmp_path):
    cond = tmp_path / "slide_2" / "alpha_0.04mg_ml" / "_2"
    for i in range(3):
        _touch(str(cond / ("img_000000%03d__000.tif" % i)))
    movies = discover_movies(str(tmp_path))
    assert len(movies) == 1
    m = movies[0]
    assert m["kind"] == "mm" and m["exp"] == "_2"
    assert m["top_root"].endswith(os.path.join("slide_2", "alpha_0.04mg_ml"))
    assert m["input"] == str(cond)


def test_discovers_each_stack_file_as_a_movie(tmp_path):
    cond = tmp_path / "slide_2" / "alpha_0.04mg_ml"
    _touch(str(cond / "_1.tif"))
    _touch(str(cond / "_2.tif"))
    movies = discover_movies(str(tmp_path))
    assert [m["exp"] for m in movies] == ["_1", "_2"]
    assert all(m["kind"] == "stack" for m in movies)
    assert all(m["top_root"] == str(cond) for m in movies)
    assert movies[1]["input"] == str(cond / "_2.tif")


def test_stack_and_frame_trees_share_identity(tmp_path):
    # the same condition supplied two ways -> same (relative top_root, exp)
    frames = tmp_path / "frames" / "slide_2" / "alpha" / "_2"
    _touch(str(frames / "img_000000000__000.tif"))
    stacks = tmp_path / "stacks" / "slide_2" / "alpha"
    _touch(str(stacks / "_2.tif"))

    def ident(m):
        return (m["top_root"].split(os.sep)[-2:], m["exp"])
    mm = [ident(m) for m in discover_movies(str(tmp_path / "frames"))]
    st = [ident(m) for m in discover_movies(str(tmp_path / "stacks"))]
    assert mm == st


def test_input_format_override(tmp_path):
    cond = tmp_path / "slide_2" / "alpha_0.04mg_ml"
    _touch(str(cond / "_1.tif"))
    _touch(str(cond / "_2.tif"))
    # auto and stack both treat each .tif as a stack movie
    assert [m["kind"] for m in discover_movies(str(tmp_path))] == ["stack", "stack"]
    assert [m["kind"] for m in discover_movies(str(tmp_path), "stack")] == ["stack", "stack"]
    # frames forces the folder to be one micro-manager movie
    forced = discover_movies(str(tmp_path), "frames")
    assert [(m["kind"], m["exp"]) for m in forced] == [("mm", "alpha_0.04mg_ml")]


def test_single_tif_file_as_input(tmp_path):
    f = tmp_path / "movie" / "_2.tif"
    _touch(str(f))
    movies = discover_movies(str(f))
    assert len(movies) == 1
    assert movies[0]["kind"] == "stack" and movies[0]["exp"] == "_2"
    assert movies[0]["input"] == str(f)


def test_group_by_top_root(tmp_path):
    cond = tmp_path / "alpha_0.04mg_ml"
    _touch(str(cond / "_1.tif"))
    _touch(str(cond / "_2.tif"))
    groups = group_by_top_root(discover_movies(str(tmp_path)))
    assert list(groups) == [str(cond)]
    assert [m["exp"] for m in groups[str(cond)]] == ["_1", "_2"]
