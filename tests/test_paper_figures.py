"""Paper figures must draw from saved runs and record which ones."""
import json

import pytest

import make_paper_figures as F
import runlog


def test_every_figure_names_its_source_runs(tmp_path, monkeypatch):
    # read the repository's committed runs; conftest points RUNLOG_ROOT at a temp dir
    monkeypatch.setenv("RUNLOG_ROOT", str(runlog.REPO / runlog.RESULTS_DIR))
    for name in ("o1_real_analysis", "o2_cluster_scans", "o2_cluster_scan_scale",
                 "velocity_all_materials"):
        try:
            runlog.load_run(name)
        except FileNotFoundError:
            pytest.skip(f"no saved {name} run")
    monkeypatch.setattr(F, "OUT", tmp_path)
    assert F.main() == 0
    sources = json.loads((tmp_path / "SOURCES.json").read_text(encoding="utf-8"))
    assert len(sources) == 4
    for fig, runs in sources.items():
        assert (tmp_path / f"{fig}.png").exists() and (tmp_path / f"{fig}.pdf").exists()
        for name, run_id in runs.items():
            assert runlog.load_run(name).info["run_id"] == run_id
