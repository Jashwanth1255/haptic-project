import pytest


@pytest.fixture(autouse=True)
def _runs_go_to_tmp(tmp_path, monkeypatch):
    """No test may write a run into the repository's results/."""
    monkeypatch.setenv("RUNLOG_ROOT", str(tmp_path / "saved_runs"))
