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


def test_looks_feel(tex):
    t = _run("looks_feel").metrics["tests"]
    assert f"r=+{t['all_free']['r']:.3f}" in tex
    assert f"Holm $p={t['all_free']['holm_p']:.4f}" in tex
    assert f"{t['cloth_only']['r']:.3f}".replace("-", "$-") in tex


def test_audio_vs_accel(tex):
    m = _run("audio_vs_accel").metrics
    assert m["confirmatory_test"]["prediction_held"] is True
    for k in ("audio_full", "audio_band", "accel_rerun"):
        assert f"{100 * m['inputs'][k]['accuracy_mean']:.1f}\\%" in tex
    assert f"p={m['confirmatory_test']['wilcoxon_greater_p']:.4f}" in tex


def test_velocity_warp(tex):
    m = _run("velocity_warp").metrics
    for k in ("k0", "k_confirmed", "k1"):
        assert f"{100 * m['conditions'][k]['accuracy_mean']:.1f}\\%" in tex
    t = m["tests"]
    assert t["k_confirmed_vs_k0"]["prediction_held"] is True
    assert t["k_confirmed_vs_k1"]["prediction_held"] is False
    assert f"Holm $p={t['k_confirmed_vs_k0']['holm_p']:.3f}" in tex
    assert f"$p={t['k_confirmed_vs_k1']['p']:.3f}" in tex


def test_benchmark(tex):
    c = _run("benchmark118_cnn").metrics["results"]
    lin = _run("benchmark118_linear").metrics["results"]
    for task in ("material_id", "category"):
        for split in ("direction", "repeat"):
            assert f"{c[f'{task}|{split}|cnn']['accuracy_mean']:.3f}" in tex
            for fs in ("raw4", "bands", "bands+raw4"):
                assert f"{lin[f'{task}|{split}|{fs}']['accuracy']:.3f}" in tex
    assert f"{100 * c['material_id|direction|cnn']['accuracy_mean']:.1f}\\%" in tex
