"""Verification for the Objective 1 gate.

The gate must pass when materials really are distinct and fail when they are
not. A gate that always passes is worse than no gate.
"""
import numpy as np
import pandas as pd
import pytest

from haptic import descriptor as D
from haptic import separability as S
from haptic import signals as G


def _frame(materials, velocities=(40.0,), forces=(1.0,), repeats=4, seconds=1.2):
    recs = G.synth_dataset(materials=materials, velocities=velocities,
                           forces=forces, repeats=repeats, seconds=seconds)
    ds = D.extract_many(recs)
    df = pd.DataFrame([d.as_dict() for d in ds])
    df["material_name"] = [r["material_name"] for r in recs]
    return D.normalise_frame(df)


DISTINCT = (
    G.SyntheticMaterial("a-fine-smooth",   0.15, 0.20, 0.6, 0.5),
    G.SyntheticMaterial("b-mid-periodic",  0.45, 0.85, 1.5, 0.9),
    G.SyntheticMaterial("c-coarse-rough",  0.90, 0.60, 2.6, 1.4),
    G.SyntheticMaterial("d-pile-stoch",    0.28, 0.08, 1.1, 1.9),
)

IDENTICAL = (
    G.SyntheticMaterial("x1", 0.30, 0.50, 1.0, 0.9),
    G.SyntheticMaterial("x2", 0.30, 0.50, 1.0, 0.9),
    G.SyntheticMaterial("x3", 0.30, 0.50, 1.0, 0.9),
)


def test_distinct_materials_separate():
    df = _frame(DISTINCT)
    assert S.cluster_quality(df) > S.SILHOUETTE_MIN


def test_identical_materials_collapse():
    """Three copies of the same material must be reported as one class."""
    df = _frame(IDENTICAL)
    res = S.run_gate(df)
    assert res.effective_classes == 1, (
        f"identical materials were not merged: {res.effective_classes} classes")
    assert not res.passed


def test_anova_detects_real_differences():
    aov = S.anova_by_dimension(_frame(DISTINCT))
    assert (aov["p_holm"] < S.ALPHA).any()
    assert aov["omega_squared"].max() > 0.1


def test_anova_finds_nothing_in_identical_materials():
    """Omega-squared, not eta-squared.

    With 3 identical materials x 4 recordings, eta2 reaches ~0.25 from sampling
    noise alone -- its null expectation is (k-1)/(n-1). Omega-squared removes
    that bias and sits near zero, which is the honest answer.
    """
    aov = S.anova_by_dimension(_frame(IDENTICAL, repeats=8))
    sig = aov[aov["p_holm"] < S.ALPHA]
    assert sig.empty, f"false positive on identical materials:\n{aov.to_string()}"


def test_pairwise_ranks_closest_pair_first():
    mats = DISTINCT + (G.SyntheticMaterial("a-twin", 0.15, 0.20, 0.6, 0.5),)
    pairs = S.pairwise_separation(_frame(mats))
    top = {pairs.iloc[0]["material_a"], pairs.iloc[0]["material_b"]}
    assert top == {"a-fine-smooth", "a-twin"}, f"closest pair was {top}"


def test_pooling_velocities_degrades_raw_but_not_normalised():
    """The project's central measurement problem, as a test.

    Raw descriptors move with sliding velocity, so pooling velocities smears
    each material across the space. The normalised descriptors should survive.
    """
    df = _frame(DISTINCT, velocities=G.VELOCITIES_MM_S, forces=(0.5, 1.0),
                repeats=3)
    df = df[df["in_band"]]
    raw = S.cluster_quality(df, D.Descriptor.CORE)
    norm = S.cluster_quality(df, D.NORMALISED_CORE)
    assert norm > raw + 0.2, f"normalisation did not help: raw={raw:.3f} norm={norm:.3f}"


def test_duplicate_columns_raise_rather_than_mislead():
    df = _frame(DISTINCT)
    bad = pd.concat([df, df[["pattern"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate columns"):
        S.anova_by_dimension(bad)


def test_gate_summary_renders():
    assert "OBJECTIVE 1 GATE" in S.run_gate(_frame(DISTINCT)).summary()
