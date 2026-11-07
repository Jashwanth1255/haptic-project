"""Round 4 helpers on invented data (scaffolding, not results)."""
import numpy as np
import pandas as pd

from haptic import round4 as R4


def test_icc_perfect_and_null():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    assert R4.icc3_1(a, a) > 0.999
    assert abs(R4.icc3_1(a, rng.normal(size=500))) < 0.15
    # consistency ICC ignores a constant offset between repeats
    assert R4.icc3_1(a, a + 5) > 0.999


def test_icc_matches_signal_to_noise():
    rng = np.random.default_rng(1)
    s = rng.normal(size=20000)
    icc = R4.icc3_1(s + rng.normal(size=s.size), s + rng.normal(size=s.size))
    assert abs(icc - 0.5) < 0.03          # var(s) / (var(s) + var(e))


def test_whiten_removes_shared_condition_effect():
    rows = []
    for m in range(5):
        for v in (20, 40, 60):
            rows.append({"material_id": m, "velocity_mm_s": v, "force_n": 1.0,
                         "direction_deg": 0, "f": m + 10 * v})
    out = R4.whiten(pd.DataFrame(rows), ["f"])
    # after whitening only the material offset remains, identical at every velocity
    assert out.groupby("material_id").f.nunique().eq(1).all()
    assert np.allclose(out.groupby("velocity_mm_s").f.median(), 0)


def _rep_table(signal_sd, noise_sd, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(40):
        mu = rng.normal(0, signal_sd)
        for d in range(0, 360, 45):
            for r in (0, 1):
                rows.append({"material_id": m, "velocity_mm_s": 40.0, "force_n": 1.0,
                             "direction_deg": d, "repeat": 2 * (d // 45) + r,
                             "f": mu + rng.normal(0, noise_sd)})
    return pd.DataFrame(rows)


def test_reliability_pairs_repeats_by_parity():
    r = R4.reliability(_rep_table(1.0, 0.1), "f")
    assert r["n_cells"] == 40 * 8 and r["icc3_1"] > 0.95 and r["split_half_sb"] > 0.95
    r0 = R4.reliability(_rep_table(0.0, 1.0), "f")
    assert r0["icc3_1"] < 0.2
