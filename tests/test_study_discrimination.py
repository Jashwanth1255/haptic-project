"""Stage 3 discrimination. A simulated observer that obeys the differencing model
has known sensitivity per scheme: the analysis must recover it, and must not find
an effect where there is none."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import runlog
from study import discrimination as S
from study import reliability as R

ROOT = Path(__file__).parent.parent
FABRICS = ("a", "b", "c", "d")


def _rates(d, k):
    f = 2 * norm.cdf(-k / np.sqrt(2))
    h = norm.cdf((d - k) / np.sqrt(2)) + norm.cdf((-d - k) / np.sqrt(2))
    return h, f


@pytest.mark.parametrize("d", [0.5, 1.5, 3.0])
def test_dprime_inverts_the_differencing_model(d):
    h, f = _rates(d, k=1.2)
    assert S.dprime_same_different(h, f) == pytest.approx(d, abs=1e-6)


def test_no_sensitivity_when_hits_do_not_beat_false_alarms():
    assert S.dprime_same_different(0.3, 0.3) == 0.0
    assert S.dprime_same_different(0.2, 0.4) == 0.0


def test_yes_no_formula_understates_same_different_sensitivity():
    h, f = _rates(2.0, k=1.5)
    assert S.dprime_yes_no(h, f) < S.dprime_same_different(h, f)


def test_log_linear_rate_is_never_0_or_1():
    assert 0 < S.rate(0, 6) < S.rate(6, 6) < 1


def _observer(true_d: dict, n_participants=24, reps=6, seed=0, devices=("Pixel 7",)):
    """Each trial: two noisy samples; respond "different" if they differ by more
    than k. Different fabrics differ by d in perceived position."""
    rng = np.random.default_rng(seed)
    k, rows = 1.0, []
    for p in range(n_participants):
        for scheme, d in true_d.items():
            for _ in range(reps):
                for i, fa in enumerate(FABRICS):
                    for fb in FABRICS[i:]:
                        same = fa == fb
                        delta = 0.0 if same else d
                        x = rng.normal(0, 1) - rng.normal(delta, 1)
                        rows.append({"participant": f"p{p}",
                                     "device_model": devices[p % len(devices)],
                                     "block": "discrimination", "scheme": scheme,
                                     "fabric_a": fa, "fabric_b": fb,
                                     "same": "true" if same else "false",
                                     "response_different":
                                         "true" if abs(x) > k
                                         else "false"})
    return pd.DataFrame(rows)


def test_recovers_which_schemes_beat_the_control():
    df = S._clean(_observer({"A": 2.0, "B": 0.0, "D": 0.0}))
    pp = S.per_participant(df)
    table, overall = S.compare_schemes(pp)
    t = table.set_index("scheme")
    assert t.loc["A", "verdict"] == "better than control"
    assert t.loc["B", "verdict"] == "no clear difference from control"
    assert overall["friedman_p"] < 0.05
    assert pp[pp["scheme"] == "D"]["dprime"].mean() < 0.5


def test_pairs_use_their_schemes_same_trials_as_false_alarms():
    df = S._clean(_observer({"A": 2.0, "D": 0.0}, n_participants=4))
    pairs = S.per_pair(df)
    assert len(pairs) == 2 * 6
    fa = pairs.groupby("scheme")["false_alarm_rate"].nunique()
    assert (fa == 1).all()


def test_device_breakdown_has_one_row_per_model_and_scheme():
    df = S._clean(_observer({"A": 1.0, "D": 0.0}, devices=("Pixel 7", "Galaxy A14")))
    dev = S.by_device(S.per_participant(df))
    assert set(dev["device_model"]) == {"Pixel 7", "Galaxy A14"} and len(dev) == 4


def test_tracking_finds_dprime_that_follows_rating_gaps():
    ratings = pd.DataFrame({"roughness": [1.0, 2.0, 4.0, 7.0],
                            "warmth": [3.0, 3.1, 2.9, 3.0]}, index=list(FABRICS))
    rows = []
    for i, a in enumerate(FABRICS):
        for b in FABRICS[i + 1:]:
            rows.append({"scheme": "A", "fabric_a": a, "fabric_b": b,
                         "dprime": abs(ratings.loc[a, "roughness"] - ratings.loc[b, "roughness"])})
    t = S.tracks_attributes(pd.DataFrame(rows), ratings).set_index("attribute")
    assert t.loc["roughness", "spearman"] == pytest.approx(1.0)
    assert abs(t.loc["warmth", "spearman"]) < 1.0


def test_bad_answer_values_are_refused():
    df = _observer({"A": 1.0}, n_participants=1).assign(same="maybe")
    with pytest.raises(ValueError, match="true or false"):
        S._clean(df)


def test_cli_saves_a_run(tmp_path):
    df = _observer({"A": 2.0, "B": 1.0, "C": 0.5, "D": 0.0}, n_participants=8, reps=2)
    paths = []
    for p, g in df.groupby("participant"):
        paths.append(str(tmp_path / f"{p}.csv"))
        g.to_csv(paths[-1], index=False)
    ratings = tmp_path / "r.csv"
    pd.DataFrame({"fabric": FABRICS, "roughness": [1, 3, 5, 7]}).to_csv(ratings, index=False)
    assert S.main([*paths, "--name", "test", "--fabric-ratings", str(ratings)]) == 0
    run = runlog.load_run("study_discrimination_test")
    assert set(run.metrics["verdicts"]) == {"A", "B", "C"}
    assert "attribute_tracking" in run.artifacts


def test_instrument_exports_the_columns_the_analysis_reads():
    html = (ROOT / "study" / "rating_instrument.html").read_text(encoding="utf-8")
    cols = re.search(r"const COLUMNS = \[(.*?)\];", html, re.S).group(1)
    exported = re.findall(r'"(\w+)"', cols)
    needed = set(R.COLUMNS) | {"scheme", "fabric_a", "fabric_b", "same", "response_different"}
    assert needed <= set(exported)


def test_placeholder_stimuli_say_so():
    js = (ROOT / "study" / "stimuli.js").read_text(encoding="utf-8")
    assert "const PLACEHOLDER = true;" in js and "PATTERNS" in js
