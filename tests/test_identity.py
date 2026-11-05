"""Exploratory identity analysis: it must find identity that is there, and not
find identity that is not."""
import numpy as np
import pandas as pd

from haptic import identity as I


def _recordings(signal: bool, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for mid in range(102, 108):
        centre = rng.normal(size=4) * (3 if signal else 0)
        for direction in range(0, 360, 45):
            for rep in range(4):
                v = centre + rng.normal(size=4)
                rows.append({"material_id": mid, "direction_deg": direction,
                             "velocity_mm_s": 40.0, "force_n": 1.0,
                             "frequency_hz": np.exp(5 + 0.3 * v[0]), "intensity": np.exp(v[1]),
                             "pattern": 0.5 + 0.05 * v[2], "duration_ms": np.exp(3 + v[3])})
    return pd.DataFrame(rows)


def test_identity_that_is_there_is_found():
    r = I.identify(_recordings(True), I.RAW4)
    assert r["accuracy"] > 0.8 and r["n_folds"] == 8


def test_no_identity_gives_chance():
    accs = [I.identify(_recordings(False, seed=s), I.RAW4)["accuracy"] for s in range(5)]
    assert abs(np.mean(accs) - 1 / 6) < 0.08


def test_band_features_follow_the_spectrum():
    fs = 6000
    t = np.arange(0, 2, 1 / fs)
    # gravity on Z, as in the real recordings, keeps the vector norm linear in the tone
    low = np.column_stack([0 * t, 0 * t, -1 + 0.05 * np.sin(2 * np.pi * 40 * t)])
    high = np.column_stack([0 * t, 0 * t, -1 + 0.05 * np.sin(2 * np.pi * 700 * t)])
    a, b = I.band_features(low, fs), I.band_features(high, fs)
    assert a["band02"] > b["band02"] and b["band14"] > a["band14"]


def test_variance_shares_sum_to_one_and_find_the_factor():
    df = _recordings(True)
    s = I.variance_shares(df, ["intensity"]).iloc[0]
    parts = s.drop("dimension")
    assert abs(parts.sum() - 1) < 1e-9 and parts["material_id"] > 0.5
