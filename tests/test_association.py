"""Objective 3 retrieval protocols. Constructed and SYNTHETIC inputs only: passing
shows the protocols behave, and says nothing about fabric."""
import numpy as np
import pandas as pd
import pytest

from haptic import association as A
from haptic import descriptor as D
from haptic import separability as P
from haptic import synth_o3 as S

N = 16


def _views(n=N, v=8, seed=0, noise=0.05, pairs=None, provenance="synthetic"):
    """Each material is one basis direction. Descriptors grow with the index, so
    the nearest neighbour of material i in descriptors is i +/- 1."""
    rng = np.random.default_rng(seed)
    base = np.eye(n)
    for i in range(n - 1):
        base[i, i + 1] = 0.5
    emb = base[:, None, :] + rng.normal(0, noise, (n, v, n))
    desc = np.arange(n)[:, None] * np.array([10.0, 1.0, 0.1, 5.0]) + 1.0
    m, t = pairs if pairs else (np.arange(n), np.arange(n))
    k = max(m) + 1, max(t) + 1
    mp, tp = np.full((n, v, k[0]), 0.1), np.full((n, v, k[1]), 0.1)
    mp[np.arange(n), :, m] = 0.9
    tp[np.arange(n), :, t] = 0.9
    return A.Views(tuple(f"m{i}" for i in range(n)), desc, emb, mp, tp, provenance)


# ---------------------------------- ties ------------------------------------ #

def test_topk_clear_win_and_loss():
    s = np.array([0.9, 0.5, 0.1])
    assert A.topk_credit(s, 0, 1) == 1 and A.topk_credit(s, 2, 1) == 0
    assert A.topk_credit(s, 1, 2) == 1


def test_tied_truth_gets_fractional_credit_not_a_win():
    s = np.array([0.5, 0.5, 0.5, 0.5])
    assert A.topk_credit(s, 2, 1) == pytest.approx(0.25)
    assert A.topk_credit(s, 2, 3) == pytest.approx(0.75)


def test_tie_behind_a_winner():
    s = np.array([0.9, 0.5, 0.5])
    assert A.topk_credit(s, 1, 1) == 0
    assert A.topk_credit(s, 1, 2) == pytest.approx(0.5)


def test_substitute_averages_a_tie():
    d = np.array([[1.0], [3.0], [9.0]])
    assert A.substitute(np.array([0.5, 0.5, 0.1]), d) == pytest.approx([2.0])


def test_identical_class_pairs_score_at_chance():
    """Every material in one pair: the lookup knows nothing and must say so."""
    pairs = (np.zeros(N, int), np.zeros(N, int))
    v = _views(pairs=pairs)
    r = A.closed_set(v, "class_pair")
    assert r["top1"].mean() == pytest.approx(1 / N)
    assert r["top3"].mean() == pytest.approx(3 / N)


# ------------------------------- closed-set --------------------------------- #

@pytest.mark.parametrize("strategy", A.STRATEGIES)
def test_closed_set_finds_clean_materials(strategy):
    r = A.closed_set(_views(), strategy)
    assert (r["top1"] == 1.0).all()
    assert len(r) == 4


def test_cosine_falls_to_chance_on_noise():
    r = A.closed_set(_views(noise=50.0), "cosine", n_folds=4)
    assert r["top1"].mean() < 3 / N


def test_top3_is_at_least_top1():
    r = A.closed_set(_views(noise=0.8), "cosine")
    assert (r["top3"] >= r["top1"]).all()


def test_every_view_is_a_query_exactly_once():
    folds = A.view_folds(8, 4)
    assert sorted(np.concatenate(folds)) == list(range(8))


def test_database_cannot_see_query_views():
    """Corrupt the fold's query views: the database must not change."""
    v = _views()
    ref = np.arange(2, 8)
    a = A.build_database(v, np.arange(N), ref)
    emb = v.embeddings.copy()
    emb[:, :2] = 999.0
    b = A.build_database(A.Views(v.names, v.descriptors, emb, v.material_probs,
                                 v.texture_probs, "synthetic"), np.arange(N), ref)
    assert np.array_equal(a.embeddings, b.embeddings)


def test_too_few_views_for_the_folds():
    with pytest.raises(ValueError, match="folds"):
        A.view_folds(3, 4)


# -------------------------------- open-set ---------------------------------- #

def test_held_out_material_is_never_its_own_substitute():
    """With near-perfect embeddings, its own descriptor would give zero error."""
    r = A.open_set(_views(), "cosine")
    assert (r[r["dimension"] == "frequency_hz"]["mae"] > 0).all()


def test_open_set_mae_by_hand():
    """Clean basis embeddings: material 0 has one neighbour, 1, so frequency is out
    by 10. A middle material ties its two neighbours, which is covered next."""
    r = A.open_set(_views(noise=0.0), "cosine")
    end = r[(r["fold"] == 0) & (r["dimension"] == "frequency_hz")]
    assert end["mae"].iloc[0] == pytest.approx(10.0)


def test_tied_neighbours_are_averaged():
    r = A.open_set(_views(noise=0.0), "cosine")
    mid = r[(r["fold"] == 7) & (r["dimension"] == "frequency_hz")]
    assert mid["mae"].iloc[0] == pytest.approx(0.0)


def test_one_fold_per_material_per_dimension():
    r = A.open_set(_views(), "cosine")
    assert r["fold"].nunique() == N and len(r) == N * 4


def test_scaling_cannot_see_the_held_out_material():
    v = _views()
    far = v.descriptors.copy()
    far[3] *= 1000
    r = A.open_set(A.Views(v.names, far, v.embeddings, v.material_probs,
                           v.texture_probs, "synthetic"), "cosine")
    held = r[(r["fold"] == 3) & (r["dimension"] == "pattern")].iloc[0]
    base = v.descriptors[np.delete(np.arange(N), 3), 2].std(ddof=1)
    assert held["std_mae"] == pytest.approx(held["mae"] / base)


def test_random_baseline_is_the_mean_error_over_substitutes():
    r = A.open_set(_views(), "cosine")
    row = r[(r["fold"] == 0) & (r["dimension"] == "frequency_hz")].iloc[0]
    assert row["mae_random_baseline"] == pytest.approx(np.mean(np.arange(1, N) * 10.0))


def test_substitutes_never_map_a_material_to_itself():
    s = A.substitutes(_views(), "cosine")
    assert all(k != v for k, v in s.items())


def test_summaries_report_spread_not_just_means():
    r = A.summarise_open(A.open_set(_views(), "cosine"))
    assert ("std_mae", "std") in r.columns and ("std_mae", "min") in r.columns
    c = A.summarise_closed(A.closed_set(_views(noise=0.8), "cosine"), N)
    assert ("top1", "std") in c.columns
    assert c[("chance", "top1")].iloc[0] == pytest.approx(1 / N)


def test_split_statement_is_explicit():
    s = A.split_statement(_views(), 4)
    assert "16 materials" in s and "leave-one-material-out" in s and "synthetic" in s


# ------------------------------- bad input ---------------------------------- #

def test_provenance_is_required_to_be_known():
    with pytest.raises(ValueError, match="provenance"):
        _views(provenance="maybe")


def test_nan_descriptor_is_refused():
    v = _views()
    d = v.descriptors.copy()
    d[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        A.Views(v.names, d, v.embeddings, v.material_probs, v.texture_probs, "real")


def test_misaligned_rows_are_refused():
    v = _views()
    with pytest.raises(ValueError, match="rows"):
        A.Views(v.names, v.descriptors[:-1], v.embeddings, v.material_probs,
                v.texture_probs, "real")


def test_paired_test():
    a = np.array([3.0, 4.1, 5.2, 6.3, 7.4, 8.5, 9.6, 10.7])
    assert A.paired_test(a, a)["p"] == 1.0
    assert A.paired_test(a, a - 1 - np.arange(8) / 10)["p"] < 0.05
    with pytest.raises(ValueError):
        A.paired_test(a, a[:-1])


def test_holm_keeps_nan_and_is_monotone():
    p = P.holm_adjust([0.01, np.nan, 0.04, 0.03])
    assert np.isnan(p[1])
    assert p[[0, 3, 2]].tolist() == pytest.approx([0.03, 0.06, 0.06])


# --------------------------- simulated embeddings ---------------------------- #

def _sim(df, cfg):
    """Descriptors from the fabrics' own parameters, fast enough for many seeds."""
    mats, texs = S.simulated_class_probs(cfg)
    return A.Views(tuple(m.name for m in S.FABRICS), df, S.simulated_embeddings(cfg),
                   mats, texs, "synthetic")


@pytest.fixture(scope="module")
def table():
    return S._latent()[:, :4].copy()


def test_there_are_16_fictional_materials_with_class_pairs():
    assert len(S.FABRICS) == 16
    assert {m.name for m in S.FABRICS} == set(S.FABRIC_CLASSES)


def test_coupled_embeddings_beat_random_substitutes(table):
    r = A.open_set(_sim(table, S.ViewConfig(coupling=0.95, view_noise=0.1)), "cosine")
    assert r["std_mae"].mean() < 0.8 * r["std_mae_random_baseline"].mean()


def test_uncoupled_embeddings_do_not_beat_random_substitutes(table):
    """The failure this experiment must be able to report. Across seeds, since one
    draw can get lucky."""
    gaps = []
    for seed in range(12):
        r = A.open_set(_sim(table, S.ViewConfig(coupling=0.0, seed=seed)), "cosine")
        gaps.append(r["std_mae_random_baseline"].mean() - r["std_mae"].mean())
    assert abs(np.mean(gaps)) < 2 * np.std(gaps) / np.sqrt(len(gaps)) + 0.05


def test_class_pairs_collide_as_expected(table):
    v = _sim(table, S.ViewConfig(classifier_accuracy=1.0))
    db = A.build_database(v, np.arange(16), np.arange(8))
    pairs = set(zip(db.material_class.tolist(), db.texture_class.tolist()))
    assert len(pairs) < 16


def test_reports_carry_the_warning():
    assert "NO RESULT" in A.__doc__ and "not a result" in A.SYNTHETIC_BANNER
    assert "SCAFFOLDING" in S.__doc__


def test_denormalise_inverts_normalise():
    d = D.Descriptor(material_id=0, velocity_mm_s=30.0, force_n=0.5, frequency_hz=150.0,
                     intensity=1.2, pattern=0.4, duration_ms=80.0)
    n = D.normalise(d)
    back = D.denormalise(n["wavelength_mm"], n["roughness_index"], n["pattern"],
                         n["burst_spacing_mm"], 30.0, 0.5)
    for k, v in back.items():
        assert v == pytest.approx(getattr(d, k))


def test_denormalise_moves_with_velocity():
    a = D.denormalise(0.5, 1.0, 0.4, 2.0, 20.0, 1.0)
    b = D.denormalise(0.5, 1.0, 0.4, 2.0, 40.0, 1.0)
    assert b["frequency_hz"] == 2 * a["frequency_hz"]
    assert b["duration_ms"] == a["duration_ms"] / 2
    assert b["pattern"] == a["pattern"]


def test_frame_has_the_columns_retrieval_needs():
    assert isinstance(A.DIMS, tuple) and set(A.DIMS) <= set(pd.Index(D.Descriptor.CORE))
