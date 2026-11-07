"""The paper's headline numbers must match the saved runs they cite.

Each check formats the run's value the way main.tex prints it and requires that
string to appear in the manuscript. Editing a number by hand, or rerunning an
analysis without updating the paper, makes this fail.
"""
from pathlib import Path

import pytest

import runlog

TEX = Path(__file__).resolve().parents[1] / "paper" / "main.tex"


@pytest.fixture(autouse=True)
def _repo_results(monkeypatch):
    monkeypatch.setenv("RUNLOG_ROOT", str(runlog.REPO / runlog.RESULTS_DIR))


def _run(name):
    try:
        return runlog.load_run(name)
    except FileNotFoundError:
        pytest.skip(f"run {name} not saved")


@pytest.fixture(scope="module")
def tex():
    return TEX.read_text(encoding="utf-8")


def test_gate(tex):
    m = _run("o1_gate").metrics
    assert f"{m['raw']['silhouette']:.3f}" in tex
    assert f"{m['raw']['n_recordings']:,}" in tex


def test_cnn(tex):
    r = _run("o1_learned_baseline")
    assert f"{r.metrics['direction']['accuracy_mean']:.3f}" in tex
    assert f"{r.metrics['repeat']['accuracy_mean']:.3f}" in tex
    assert r.info["run_id"].replace("_", r"\_") in tex


def test_velocity_confirmation(tex):
    m = _run("velocity_confirm_cloth").metrics
    assert m["confirmed"] is True
    assert f"{m['cloth_median']:.3f}" in tex
    assert f"{m['n_cells']}" in tex
    assert f"{m['holm'][0]:.1e}".replace("e-0", r"\times10^{-") in tex


def test_velocity_all(tex):
    f = _run("velocity_all_materials").metrics["frequency_exponent"]
    assert f"{f['cloth_median']:.3f}" in tex
    assert f"{f['other_median']:.3f}" in tex
    assert f"{100 * f['share_within_025_of_0']:.0f}\\%" in tex


def test_scans(tex):
    m = _run("o2_cluster_scans").metrics["minc"]
    assert f"{100 * m['fabric_fraction']:.2f}\\%" in tex
    assert f"({m['called']['fabric']}/{m['n_crops']:,})" in tex
