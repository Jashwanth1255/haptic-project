"""Rater reliability: the ICC matches the published example, agreement and noise
are told apart, and click-through participants are flagged."""
import numpy as np
import pandas as pd
import pytest

import runlog
from study import reliability as R

SHROUT_FLEISS = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
                 [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]


def test_icc_matches_shrout_and_fleiss_1979_table():
    r = R.icc2(SHROUT_FLEISS)
    assert r["icc2_1"] == pytest.approx(0.29, abs=0.005)
    assert r["icc2_k"] == pytest.approx(0.62, abs=0.005)


def test_identical_raters_agree_perfectly():
    col = np.arange(10.0)
    r = R.icc2(np.c_[col, col, col])
    assert r["icc2_1"] == pytest.approx(1.0) and r["icc2_k"] == pytest.approx(1.0)


def test_incomplete_matrix_is_refused():
    with pytest.raises(ValueError):
        R.icc2([[1, np.nan], [2, 3]])


def _sessions(n_participants=8, n_images=12, noise=None, seed=0, rt=4000):
    """Signal attribute 'roughness' is shared; 'warmth' is each rater's own noise."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(1, 7, n_images)
    rows = []
    for p in range(n_participants):
        for i in range(n_images):
            for rep in (0, 1) if i < 4 else (0,):
                for attr, value in (("roughness", truth[i] + rng.normal(0, 0.5)),
                                    ("warmth", rng.uniform(1, 7))):
                    rows.append({"participant": f"p{p}", "device_model": "Pixel 7",
                                 "block": "rating", "image_id": f"img{i}",
                                 "attribute": attr,
                                 "rating": int(np.clip(round(value), 1, 7)),
                                 "repeat": rep, "rt_ms": rt})
    return pd.DataFrame(rows)


def test_shared_attribute_is_reliable_and_noise_is_flagged():
    icc = R.icc_by_attribute(_sessions()).set_index("attribute")
    assert icc.loc["roughness", "icc2_k"] > 0.9
    assert icc.loc["warmth", "icc2_k"] < R.ICC_MIN
    assert bool(icc.loc["warmth", "unreliable"]) and not icc.loc["roughness", "unreliable"]


def test_images_missing_a_rater_are_dropped_and_counted():
    df = _sessions()
    df = df[~((df["participant"] == "p0") & (df["image_id"] == "img5"))]
    icc = R.icc_by_attribute(df).set_index("attribute")
    assert icc.loc["roughness", "n_images_dropped"] == 1
    assert icc.loc["roughness", "n_targets"] == 11


def test_intra_rater_uses_the_repeats():
    ir = R.intra_rater(_sessions())
    rough = ir[ir["attribute"] == "roughness"]
    assert (rough["n_repeated"] == 4).all()
    assert rough["spearman"].median() > 0.5


def test_fast_and_straight_line_participants_are_flagged():
    df = _sessions()
    fast = df["participant"] == "p1"
    df.loc[fast, "rt_ms"] = 600
    line = df["participant"] == "p2"
    df.loc[line, "rating"] = 4
    f = R.flags(df).set_index("participant")
    assert "median time" in f.loc["p1", "flags"]
    assert "one value" in f.loc["p2", "flags"]
    assert f.loc["p3", "flags"] == ""


def test_duplicated_ratings_are_refused(tmp_path):
    df = _sessions(n_participants=2)
    path = tmp_path / "s.csv"
    pd.concat([df, df.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicated"):
        R.read_sessions([path])


def test_discrimination_rows_are_left_out(tmp_path):
    df = _sessions(n_participants=2)
    extra = df.iloc[:3].assign(block="discrimination")
    path = tmp_path / "s.csv"
    pd.concat([df, extra]).to_csv(path, index=False)
    assert len(R.read_sessions([path])) == len(df)


def test_cli_saves_a_run(tmp_path, capsys):
    df = _sessions()
    paths = []
    for p, g in df.groupby("participant"):
        paths.append(tmp_path / f"{p}.csv")
        g.to_csv(paths[-1], index=False)
    assert R.main([*map(str, paths), "--name", "test"]) == 0
    run = runlog.load_run("study_reliability_test")
    assert run.metrics["unreliable"] == ["warmth"]
    assert run.metrics["n_participants"] == 8
    assert run.table("mean_ratings").shape == (12, 3)
