"""Saved runs: provenance, round trips, and refusing what cannot be traced."""
import json
import math

import numpy as np
import pandas as pd
import pytest

import runlog as R


def _save(root, name="exp", **kw):
    kw.setdefault("synthetic", True)
    return R.save_run(name, {"seed": 0}, {"acc": 0.5}, root=root, **kw)


def test_round_trip(tmp_path):
    table = pd.DataFrame({"a": [1, 2], "b": [0.5, 0.25]})
    path = R.save_run("exp", {"seed": np.int64(3), "dims": ("x", "y")},
                      {"acc": np.float32(0.75), "per": np.array([1.0, 2.0])},
                      {"table": table, "note": "hello", "extra": {"k": [1, 2]}},
                      synthetic=False, root=tmp_path)
    run = R.load_run(path)
    assert run.config == {"seed": 3, "dims": ["x", "y"]}
    assert run.metrics == {"acc": 0.75, "per": [1.0, 2.0]}
    assert run.info["synthetic"] is False and run.info["name"] == "exp"
    pd.testing.assert_frame_equal(run.table("table"), table)
    assert set(run.artifacts) == {"table", "note", "extra"}
    assert (path / "note.txt").read_text() == "hello"


def test_records_the_commit_of_this_repository(tmp_path):
    info = R.load_run(_save(tmp_path)).info
    commit, dirty = R.git_state()
    assert info["git_commit"] == commit and len(commit) == 40
    assert info["git_dirty"] == dirty


def test_without_git_the_commit_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNLOG_COMMIT", "abc123")
    assert R.git_state(tmp_path) == ("abc123", None)


def test_nan_is_saved_as_null_not_as_invalid_json(tmp_path):
    path = R.save_run("exp", {}, {"p": float("nan"), "q": np.inf}, synthetic=True,
                      root=tmp_path)
    text = (path / "metrics.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"p": None, "q": None}


def test_synthetic_flag_is_required_and_must_be_a_bool(tmp_path):
    with pytest.raises(TypeError):
        R.save_run("exp", {}, {}, root=tmp_path)
    with pytest.raises(TypeError):
        R.save_run("exp", {}, {}, synthetic="yes", root=tmp_path)


def test_tables_inside_metrics_are_refused(tmp_path):
    with pytest.raises(TypeError, match="artifacts"):
        R.save_run("exp", {}, {"t": pd.DataFrame({"a": [1]})}, synthetic=True,
                   root=tmp_path)


def test_a_failed_save_leaves_nothing_behind(tmp_path):
    with pytest.raises(TypeError):
        R.save_run("exp", {}, {"bad": object()}, synthetic=True, root=tmp_path)
    assert not any((tmp_path / "exp").iterdir())
    assert R.list_runs(root=tmp_path).empty


@pytest.mark.parametrize("name", ["../up", "a/b", "", ".hidden"])
def test_bad_run_names_are_refused(tmp_path, name):
    with pytest.raises(ValueError):
        R.save_run(name, {}, {}, synthetic=True, root=tmp_path)


def test_artifacts_cannot_overwrite_the_run_files(tmp_path):
    with pytest.raises(ValueError):
        R.save_run("exp", {}, {}, {"metrics": {"a": 1}}, synthetic=True, root=tmp_path)


def test_runs_saved_together_do_not_collide(tmp_path):
    paths = {_save(tmp_path) for _ in range(5)}
    assert len(paths) == 5
    assert len(R.list_runs("exp", root=tmp_path)) == 5


def test_load_by_name_gives_the_latest(tmp_path):
    _save(tmp_path)
    last = R.save_run("exp", {}, {"acc": 0.9}, synthetic=True, root=tmp_path)
    assert R.load_run("exp", root=tmp_path).path == last


def test_missing_run_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        R.load_run("nothing", root=tmp_path)


def test_list_runs_covers_every_name_and_skips_unsaved_folders(tmp_path):
    _save(tmp_path, "a")
    _save(tmp_path, "b", synthetic=False)
    (tmp_path / "o2" / "old_layout").mkdir(parents=True)
    runs = R.list_runs(root=tmp_path)
    assert list(runs["name"]) == ["a", "b"]
    assert list(runs["synthetic"]) == [True, False]


def test_environment_sets_the_default_root(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNLOG_ROOT", str(tmp_path / "elsewhere"))
    path = R.save_run("exp", {}, {}, synthetic=True)
    assert path.parent.parent == tmp_path / "elsewhere"


def test_existing_files_are_copied_with_their_suffix(tmp_path):
    src = tmp_path / "fig.png"
    src.write_bytes(b"\x89PNG")
    path = R.save_run("exp", {}, {}, {"figure": src}, synthetic=True,
                      root=tmp_path / "r")
    assert (path / "figure.png").read_bytes() == b"\x89PNG"


def test_dataclasses_are_saved_as_their_fields(tmp_path):
    from dataclasses import dataclass

    @dataclass
    class C:
        x: float = math.pi
    run = R.load_run(R.save_run("exp", C(), {}, synthetic=True, root=tmp_path))
    assert run.config == {"x": math.pi}


def test_code_changes_ignore_results_and_untracked_documents():
    status = (" M haptic/descriptor.py\n?? PLAN.docx\n?? new_script.py\n"
              " M results/o1_gate/x/run.json\n?? notes/\n")
    assert R._code_changes(status) == ["haptic/descriptor.py", "new_script.py", "notes/"]
    assert R._code_changes("?? PLAN.docx\n") == []


def test_source_digest_changes_with_code_and_ignores_line_endings(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "pkg" / "a.py").write_bytes(b"x = 1\n")
    (tmp_path / "data" / "big.py").write_bytes(b"ignored\n")
    d1 = R.source_digest(tmp_path)
    (tmp_path / "data" / "big.py").write_bytes(b"still ignored\n")
    assert R.source_digest(tmp_path) == d1
    (tmp_path / "pkg" / "a.py").write_bytes(b"x = 1\r\n")
    assert R.source_digest(tmp_path) == d1
    (tmp_path / "pkg" / "a.py").write_bytes(b"x = 2\n")
    assert R.source_digest(tmp_path) != d1


def test_runs_record_source_digest(tmp_path):
    info = R.load_run(R.save_run("t", {}, {}, synthetic=True, root=tmp_path)).info
    assert len(info["source_sha256"]) == 64
    assert isinstance(info["dirty_paths"], list)
