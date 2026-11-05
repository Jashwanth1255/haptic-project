"""Objective 3 velocity experiment.

The hand-built frames obey or break the relations exactly. The `recordings` tests
are SYNTHETIC: conditioning wins them by construction, so passing is a check on the
code. The velocity_free test is the important one: it shows the experiment can
return "conditioning hurts".
"""
import numpy as np
import pandas as pd
import pytest

from haptic import association as A
from haptic import descriptor as D
from haptic import synth_o3 as S
from haptic import velocity as Y

VELOCITIES = (20.0, 30.0, 40.0, 50.0, 60.0)


def _frame(obeys=True, n_materials=8, repeats=4, seed=0, noise=0.0):
    """Exact descriptors. `obeys` False freezes them at the reference velocity."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_materials):
        lam, rough, burst = 0.2 + 0.1 * m, 1 + 0.2 * m, 0.5 + 0.3 * m
        for v in VELOCITIES:
            u = v if obeys else D.REF_VELOCITY_MM_S
            for r in range(repeats):
                j = 1 + noise * rng.normal()
                d = D.Descriptor(material_id=m, velocity_mm_s=v, force_n=1.0,
                                 frequency_hz=j * u / lam,
                                 intensity=rough * np.sqrt(u / 40), pattern=0.5,
                                 duration_ms=burst * 1000 / u)
                rows.append({**d.as_dict(), "material_name": f"m{m}", "repeat": r})
    return D.normalise_frame(pd.DataFrame(rows)).assign(in_band=True)


@pytest.fixture(scope="module")
def generator():
    return S.recordings(repeats=4, seconds=1.2)


@pytest.fixture(scope="module")
def velocity_free():
    return S.recordings(repeats=4, seconds=1.2, velocity_free=True)


def test_conditioning_is_exact_when_the_relations_hold():
    e = Y.run(_frame())
    assert e["err_conditioned"].max() == pytest.approx(0, abs=1e-9)
    assert e[e["dimension"] == "frequency_hz"]["err_agnostic"].max() > 1


def test_agnostic_is_nearly_exact_when_nothing_moves():
    e = Y.run(_frame(obeys=False))
    assert e["err_agnostic"].max() == pytest.approx(0, abs=1e-9)
    assert e[e["dimension"] == "frequency_hz"]["err_conditioned"].max() > 1


def test_verdicts_go_both_ways():
    good = Y.compare(Y.run(_frame(n_materials=16)))
    bad = Y.compare(Y.run(_frame(obeys=False, n_materials=16)))
    moved = good[good["dimension"] != "pattern"]
    assert (moved[moved["velocity_mm_s"] != 40]["verdict"] == "conditioning helps").all()
    assert "conditioning hurts" in set(bad["verdict"])


def test_pattern_has_no_velocity_relation_so_no_difference():
    good = Y.compare(Y.run(_frame()))
    assert set(good[good["dimension"] == "pattern"]["verdict"]) == {"no clear difference"}


def test_small_material_counts_cannot_reach_significance():
    """Holm over 20 tests needs p < 0.0025, and 8 pairs bottom out at 0.0078. This
    records why the material count has to be reported with any null verdict."""
    assert Y.smallest_holm_p(n_materials=8, n_tests=20) > Y.ALPHA
    assert Y.smallest_holm_p(n_materials=16, n_tests=20) < Y.ALPHA
    t = Y.compare(Y.run(_frame(n_materials=8)))
    assert "conditioning helps" not in set(t["verdict"])


def test_improvement_is_a_percentage_reduction():
    e = pd.DataFrame({"material_id": range(8), "velocity_mm_s": 20.0, "force_n": 1.0,
                      "dimension": "intensity", "err_agnostic": 2.0,
                      "err_conditioned": np.linspace(0.9, 1.1, 8)})
    assert Y.compare(e)["improvement_pct"].iloc[0] == pytest.approx(50.0)


def test_too_few_materials_gets_no_verdict():
    e = pd.DataFrame({"material_id": range(3), "velocity_mm_s": 20.0, "force_n": 1.0,
                      "dimension": "intensity", "err_agnostic": 2.0,
                      "err_conditioned": [1.0, 1.1, 1.2]})
    assert Y.compare(e)["verdict"].iloc[0] == "too few materials"


def test_reference_and_query_share_no_recording():
    ref, query = Y.split_repeats(_frame())
    assert set(ref["repeat"]).isdisjoint(query["repeat"])
    assert len(ref) and len(query)


def test_one_repeat_cannot_be_split():
    with pytest.raises(ValueError, match="repeats"):
        Y.split_repeats(_frame(repeats=1))


def test_query_noise_does_not_reach_the_reference():
    f = _frame()
    q = f["repeat"] >= 2
    f.loc[q & (f["material_id"] == 0), "frequency_hz"] *= 3
    e = Y.run(f)
    row = e[(e["material_id"] == 0) & (e["dimension"] == "frequency_hz")]
    assert row["err_conditioned"].max() > 1
    other = e[(e["material_id"] == 1) & (e["dimension"] == "frequency_hz")]
    assert other["err_conditioned"].max() == pytest.approx(0, abs=1e-9)


def test_out_of_band_recordings_are_excluded_and_counted():
    f = _frame()
    f.loc[(f["material_id"] == 0) & (f["velocity_mm_s"] == 20), "in_band"] = False
    e = Y.run(f)
    assert e[(e["material_id"] == 0) & (e["velocity_mm_s"] == 20)].empty
    assert Y.excluded(f).loc[20.0, "excluded"] == 4


def test_agnostic_weights_velocities_equally():
    """Dropping recordings at one velocity must not bias the mean towards the rest."""
    ref = pd.DataFrame({"velocity_mm_s": [20.0, 20.0, 20.0, 60.0],
                        "frequency_hz": 100.0, "intensity": 1.0, "pattern": 0.5,
                        "duration_ms": 10.0})
    ref.loc[3, "frequency_hz"] = 300.0
    assert Y._agnostic(ref)["frequency_hz"] == pytest.approx(200.0)


def test_one_outlier_recording_cannot_set_the_reference():
    """A rare spurious envelope peak gives a duration ten times too long."""
    ref = pd.DataFrame({"velocity_mm_s": 40.0, "frequency_hz": 100.0, "intensity": 1.0,
                        "pattern": 0.5, "duration_ms": [18.0, 18.0, 341.0]})
    assert Y._agnostic(ref)["duration_ms"] == 18.0


def test_substitute_is_used_in_place_of_the_material():
    f = _frame()
    wrong = Y.run(f, substitute={m: (m + 1) % 8 for m in range(8)})
    assert wrong["err_conditioned"].max() > 1


def test_excluding_the_query_velocity_removes_it_from_the_reference():
    f = _frame(obeys=False)
    f.loc[f["velocity_mm_s"] == 20, "frequency_hz"] *= 5
    held = Y.run(f, exclude_query_velocity=True)
    kept = Y.run(f)
    assert not held["err_agnostic"].equals(kept["err_agnostic"])


def test_generator_recovers_its_own_relation(generator):
    """By construction. This checks the code, and is not evidence about fabric."""
    t = Y.compare(Y.run(generator))
    f = t[(t["dimension"] == "frequency_hz") & (t["velocity_mm_s"].isin([20.0, 60.0]))]
    assert (f["mae_conditioned"] < f["mae_agnostic"]).all()


def test_experiment_reports_failure_when_relations_are_false(velocity_free):
    t = Y.compare(Y.run(velocity_free))
    edges = t[t["velocity_mm_s"].isin([20.0, 60.0])
              & t["dimension"].isin(["frequency_hz", "duration_ms"])]
    assert (edges["mae_conditioned"] > edges["mae_agnostic"]).all()
    assert "conditioning hurts" in set(t["verdict"])


def test_warning_is_in_the_docstring():
    assert "NO RESULT" in Y.__doc__ and "by construction" in Y.__doc__
    assert A.DIMS == D.Descriptor.CORE


def _scaling_frame(exponent: float):
    rows = []
    for mid in (1, 2, 3):
        lam = 0.5 * mid
        for v in (20.0, 30.0, 40.0, 50.0, 60.0):
            for rep in range(3):
                f = 100 * (v / 40) ** exponent * mid
                rows.append({"material_id": mid, "force_n": 1.0, "velocity_mm_s": v,
                             "repeat": rep, "in_band": True, "frequency_hz": f,
                             "intensity": 0.1 * v / 40, "pattern": 0.5,
                             "duration_ms": 1000 / f * 5, "wavelength_mm": v / f,
                             "roughness_index": 0.1, "burst_spacing_mm": v * 5 / f,
                             "lam": lam})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("exponent", [1.0, 0.0])
def test_scaling_exponent_recovers_the_built_in_slope(exponent):
    from haptic.velocity import scaling_exponents
    s = scaling_exponents(_scaling_frame(exponent))
    assert np.allclose(s["frequency_hz"], exponent, atol=1e-9)
    assert np.allclose(s["duration_ms"], -exponent, atol=1e-9)
    assert np.allclose(s["intensity"], 1.0, atol=1e-9)


def test_normalisation_removes_variation_only_when_f_scales_with_v():
    from haptic.velocity import variation_across_velocity
    ok = variation_across_velocity(_scaling_frame(1.0))
    bad = variation_across_velocity(_scaling_frame(0.0))
    assert (ok["wavelength_mm"] < 1e-9).all() and (ok["frequency_hz"] > 0.2).all()
    assert (bad["wavelength_mm"] > 0.2).all() and (bad["frequency_hz"] < 1e-9).all()
