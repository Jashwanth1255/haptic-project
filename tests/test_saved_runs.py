"""Each experiment entry point saves a run, marked synthetic where it is."""
import pandas as pd

import make_o3_synthetic as O3
import make_o4_table as O4
import runlog
from haptic import cli


def test_gate_saves_a_synthetic_run(tmp_path):
    assert cli.main(["gate", "--synthetic", "--seconds", "0.4", "--repeats", "2",
                     "--out", str(tmp_path / "d.csv")]) == 0
    run = runlog.load_run("o1_gate_synthetic")
    assert run.info["synthetic"] is True
    assert set(run.metrics) >= {"raw", "normalised", "n_out_of_band"}
    assert run.metrics["raw"]["thresholds"]["min_separable"] == 8
    assert {"descriptors", "raw_anova", "normalised_per_velocity"} <= set(run.artifacts)


def test_gate_can_skip_saving(tmp_path):
    cli.main(["gate", "--synthetic", "--seconds", "0.4", "--repeats", "2",
              "--out", str(tmp_path / "d.csv"), "--no-save"])
    assert runlog.list_runs().empty


def test_o3_saves_every_table_flat_and_synthetic():
    multi = pd.DataFrame([[1.0, 2.0]], columns=pd.MultiIndex.from_tuples(
        [("top1", "mean"), ("top1", "std")]))
    verdicts = pd.DataFrame({"verdict": ["conditioning helps", "no clear difference",
                                         "conditioning helps"]})
    path = O3.save({"coupling0.8_closed_summary": multi,
                    "velocity_generator": verdicts}, {"coupling0.8": ["x"]})
    run = runlog.load_run(path)
    assert run.info["synthetic"] is True
    assert list(run.table("coupling0.8_closed_summary").columns) == ["top1_mean", "top1_std"]
    assert run.metrics["velocity_generator_verdicts"] == {
        "conditioning helps": 2, "no clear difference": 1}
    assert run.metrics["dropped"] == {"coupling0.8": ["x"]}


def test_o4_saves_its_table(monkeypatch):
    rows = [{"material": "m", "velocity": 20.0, "scheme": s, "tau_ms": t,
             "lsd_db": 1.0, "wasserstein_hz": 2.0, "level_error": 0.1,
             "renderable": r}
            for s in "ABCD" for t in O4.TAUS_MS for r in (True, False)]
    monkeypatch.setattr(O4, "rows", lambda: pd.DataFrame(rows))
    assert O4.main() == 0
    run = runlog.load_run("o4_fidelity_synthetic")
    assert run.info["synthetic"] is True
    assert run.metrics["median_renderable"]["B"]["wasserstein_hz"] == 2.0
    assert len(run.table("fidelity")) == len(rows)
