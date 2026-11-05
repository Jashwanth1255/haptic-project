"""The pre-registered group test, on invented descriptors whose answer is known."""
import re
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from haptic import groups as G
from haptic.descriptor import NORMALISED_CORE


def _frame(centres: dict[int, np.ndarray], per=10, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for mid, c in centres.items():
        for _ in range(per):
            rows.append({"material_id": mid,
                         **dict(zip(NORMALISED_CORE, c + noise * rng.normal(size=4)))})
    return pd.DataFrame(rows)


def _clustered(groups_that_cluster, seed=0):
    """Materials in a clustering group sit near their group's centre; the rest are
    scattered at random."""
    rng = np.random.default_rng(seed)
    centre = {g: rng.normal(scale=5, size=4) for g in ("pile", "weave", "drape")}
    out = {}
    for mid, g in G.ASSIGNMENT.items():
        if g in groups_that_cluster:
            out[mid] = centre[g] + 0.3 * rng.normal(size=4)
        else:
            out[mid] = rng.normal(scale=5, size=4)
    return _frame(out, seed=seed)


def test_assignment_matches_the_preregistration_table():
    text = Path(__file__).parents[1].joinpath("PREREGISTRATION.md").read_text(encoding="utf-8")
    section = text.split("## Cluster cloth group assignment", 1)[1]
    found = {}
    for group in ("Pile", "Weave", "Drape", "Neither"):
        row = re.search(rf"^\| {group} \| (.+?) \| \d+ \|$", section, re.M).group(1)
        for mid in re.findall(r"\b(1\d\d)\b", row):
            found[int(mid)] = group.lower()
    assert found == G.ASSIGNMENT
    assert set(G.ASSIGNMENT) == set(range(102, 118))


def test_a_clustering_group_is_detected_and_the_permutation_is_exact():
    res = G.run(_clustered({"weave", "drape"}))
    weave = res.set_index("group").loc["weave"]
    assert weave.tested and weave.ratio < 1 and weave.p < 0.01
    assert weave.n_permutations == comb(13, 7)
    assert res.set_index("group").loc["drape"].clusters


def test_scattered_groups_do_not_pass():
    ps = []
    for seed in range(20):
        res = G.run(_clustered(set(), seed=seed)).set_index("group")
        ps.append(res.loc["weave", "p"])
    assert np.mean(np.array(ps) < 0.05) <= 0.15
    assert np.median(ps) > 0.2


def test_pile_is_untested_and_the_confirmatory_verdict_cannot_be_supported():
    res = G.run(_clustered({"pile", "weave", "drape"}))
    pile = res.set_index("group").loc["pile"]
    assert not pile.tested and np.isnan(pile.p) and pile.clusters is None
    v = G.verdict(res)
    assert v["confirmatory"].startswith("untested") and "supported" not in v["confirmatory"]
    assert v["drape"].startswith("drape clusters")


def test_holm_is_over_the_family_of_two_even_with_pile_untested():
    res = G.run(_clustered({"weave"})).set_index("group")
    assert res.loc["weave", "p_holm"] == pytest.approx(min(1.0, 2 * res.loc["weave", "p"]))


def test_a_failed_weave_is_reported_as_failed():
    v = G.verdict(G.run(_clustered(set(), seed=3)))
    assert "failed" in v["confirmatory"]


def test_scale_of_one_dimension_does_not_dominate():
    """Standardising first: a dimension in large units must not set the distances."""
    df = _clustered({"weave"})
    df2 = df.assign(burst_spacing_mm=df["burst_spacing_mm"] * 1000)
    a = G.run(df).set_index("group")["ratio"]
    b = G.run(df2).set_index("group")["ratio"]
    assert np.allclose(a.dropna(), b.dropna())


def test_sensitivity_reports_every_reading():
    s = G.sensitivity(_clustered({"weave"}))
    assert set(s.reading) == {"pile+weave+drape (registered reading)", "pile+weave only",
                              "all 16, neither included"}
    only = s[(s.reading == "pile+weave only") & (s.group == "weave")].iloc[0]
    assert only.tested and only.n_pool == 9
