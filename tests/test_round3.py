"""Round 3 helpers on invented data (scaffolding, not results)."""
import numpy as np
import pandas as pd
import pytest

from haptic import identity as I
from haptic import round3 as R3


def _table(axis_effect: float, n_mat=4, seed=0):
    """Features depend on material plus, if axis_effect > 0, on the stroke axis
    (θ mod 180): opposite strokes match, perpendicular ones do not."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_mat):
        base = rng.normal(0, 3, len(R3.FEATURES))
        axis = rng.normal(0, 1, len(R3.FEATURES))
        for v in (20, 40):
            for f in (0.5, 1.0):
                for r in (0, 1):
                    for d in range(0, 360, 45):
                        a = np.cos(np.deg2rad(2 * d))      # same for θ and θ+180
                        x = base + axis_effect * a * axis + rng.normal(0, 0.1, base.size)
                        rows.append({"material_id": m, "velocity_mm_s": v, "force_n": f,
                                     "repeat": r, "direction_deg": d,
                                     **dict(zip(R3.FEATURES, x))})
    return pd.DataFrame(rows)


def test_opposite_closer_when_axis_matters():
    out = R3.direction_distances(_table(axis_effect=2.0))
    assert (out.perp_minus_opp > 0).all()
    assert (out.opp < out.perp).all()


def test_no_axis_effect_gives_no_gap():
    out = R3.direction_distances(_table(axis_effect=0.0))
    assert out.perp_minus_opp.abs().max() < 0.05
    # the noise floor is comparable to the pair distances when nothing differs
    assert np.allclose(out.repeat, out.opp, rtol=0.3)


def test_missing_direction_drops_only_its_pairs():
    df = _table(axis_effect=2.0)
    df = df[~((df.material_id == 0) & (df.direction_deg == 0))]
    out = R3.direction_distances(df)
    assert np.isfinite(out.loc[0, ["opp", "perp", "diag"]].to_numpy()).all()


def test_envelope_features_find_the_modulation_band():
    t = np.arange(500) / 1000
    feats = R3.envelope_features(1 + 0.8 * np.sin(2 * np.pi * 8 * t))
    bands = np.array(feats[:-1])
    k = np.searchsorted(R3.MOD_BANDS, 8.0) - 1
    assert bands.argmax() == k
    assert len(feats) == 13


def test_render_pair_keeps_slow_modulation_and_loses_fast():
    fs = 6000
    t = np.arange(int(1.0 * fs)) / fs
    carrier = np.sin(2 * np.pi * 200 * t)
    for mod_hz, expect_kept in ((5, True), (80, False)):
        x = (1 + 0.9 * np.sin(2 * np.pi * mod_hz * t)) * carrier
        intended, rendered = R3.render_pair(x, fs, intensity=0.5, intensity_ref=1.0)
        assert intended.size == rendered.size == R3.ENV_MS
        c = np.corrcoef(intended - intended.mean(), rendered - rendered.mean())[0, 1]
        assert (c > 0.5) == expect_kept


def test_render_pair_rejects_short_window():
    with pytest.raises(ValueError):
        R3.render_pair(np.random.default_rng(0).normal(size=1000), 6000, 0.5, 1.0)


def _force_table(force_shift: float, seed=0):
    rng = np.random.default_rng(seed)
    cols = list(R3.FEATURES)
    rows = []
    centres = rng.normal(0, 1.5, (5, len(cols)))
    shift = rng.normal(0, 1, len(cols))
    for m in range(5):
        for f in (0.5, 1.0):
            for r in (0, 1):
                for _ in range(20):
                    x = centres[m] + (force_shift * shift if f == 1.0 else 0) + rng.normal(0, 0.6, len(cols))
                    rows.append({"material_id": m, "force_n": f, "repeat": r, **dict(zip(cols, x))})
    return pd.DataFrame(rows), cols


def test_force_transfer_drops_when_force_moves_features():
    df, cols = _force_table(force_shift=3.0)
    r = R3.force_transfer(df, cols)
    assert r["cross"]["accuracy"] < r["same"]["accuracy"] - 0.2


def test_force_transfer_flat_without_force_effect():
    df, cols = _force_table(force_shift=0.0)
    r = R3.force_transfer(df, cols)
    assert abs(r["cross"]["accuracy"] - r["same"]["accuracy"]) < 0.1


def test_force_transfer_needs_two_forces():
    df, cols = _force_table(0.0)
    with pytest.raises(ValueError):
        R3.force_transfer(df[df.force_n == 0.5], cols)


def test_features_constant_matches_identity():
    assert R3.FEATURES[:-1] == I.BAND_COLUMNS


def test_global_repeat_numbering_is_reduced_to_parity():
    df = _table(axis_effect=2.0)
    df["repeat"] = df["repeat"] + 2 * (df.direction_deg // 45)   # dataset-style 0-15
    out = R3.direction_distances(df)
    assert np.isfinite(out[["opp", "perp", "diag"]].to_numpy()).all()
    assert (out.perp_minus_opp > 0).all()
