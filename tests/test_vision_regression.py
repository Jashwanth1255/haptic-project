"""Small-sample ridge regression: it must find real signal, report none in noise,
and never let a held-out group into its own training rows."""
import numpy as np
import pandas as pd
import pytest

from vision import regression as G


def _data(n=40, p=6, signal=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = signal * X[:, 0] + 0.3 * rng.normal(size=n)
    return X, y


def test_finds_a_real_linear_signal():
    X, y = _data()
    t = G.evaluate([X], y, ["y"], n_permutations=99)
    assert t.loc[0, "r2"] > 0.6 and t.loc[0, "spearman"] > 0.8
    assert t.loc[0, "p_permutation"] == pytest.approx(0.01)
    assert t.loc[0, "mae"] < t.loc[0, "mae_mean_predictor"]


def test_reports_nothing_in_noise():
    rng = np.random.default_rng(1)
    r2, p = [], []
    for seed in range(5):
        X, y = rng.normal(size=(30, 50)), rng.normal(size=30)
        t = G.evaluate([X], y, ["y"], n_permutations=49, seed=seed)
        r2.append(t.loc[0, "r2"])
        p.append(t.loc[0, "p_permutation"])
    assert np.mean(r2) < 0.05
    assert np.mean(np.array(p) < 0.05) <= 0.2


def test_constant_features_score_exactly_the_mean_predictor():
    y = np.random.default_rng(0).normal(size=12)
    pred, base = G.predict_out_of_fold([np.ones((12, 3))], y)
    np.testing.assert_allclose(pred, base)
    assert G.r2_out_of_sample(y, pred[:, 0], base[:, 0]) == pytest.approx(0.0)


def test_leaving_out_groups_stops_a_group_predicting_itself():
    """Each material has 3 images. A one-hot material code predicts its own target
    perfectly if the material's other images are in training, and not at all
    once the whole material is held out."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 3)
    X = np.eye(10)[groups]
    y = rng.normal(size=10)[groups]
    leaky = G.evaluate([X], y, ["y"], n_permutations=0)
    held = G.evaluate([X], y, ["y"], groups=groups, n_permutations=0)
    assert leaky.loc[0, "r2"] > 0.5
    assert held.loc[0, "r2"] < 0.05
    assert held.loc[0, "n_groups"] == 10


def test_blocks_carry_equal_weight_whatever_their_width():
    rng = np.random.default_rng(0)
    wide, narrow = rng.normal(size=(20, 400)), rng.normal(size=(20, 2))
    a, _ = G._design([wide, narrow], [wide[:1], narrow[:1]])
    assert np.sum(a[:, :400].var(0)) == pytest.approx(np.sum(a[:, 400:].var(0)))


def test_group_permutation_needs_one_target_per_group():
    groups = np.repeat(np.arange(4), 2)
    y = np.arange(8.0)
    with pytest.raises(ValueError, match="one target per group"):
        G.evaluate([np.random.default_rng(0).normal(size=(8, 2))], y, ["y"],
                   groups=groups, n_permutations=5)


def test_too_few_groups_is_an_error():
    with pytest.raises(ValueError, match="3 groups"):
        G.predict_out_of_fold([np.zeros((4, 1))], np.arange(4.0), groups=[0, 0, 1, 1])


def test_ablation_labels_each_feature_set():
    X, y = _data(n=20)
    t = G.ablation({"a": [X[:, :1]], "b": [X[:, 1:]], "a+b": [X[:, :1], X[:, 1:]]},
                   np.c_[y, y], ["y1", "y2"], n_permutations=0)
    assert list(t["features"].unique()) == ["a", "b", "a+b"]
    r2 = t.groupby("features")["r2"].mean()
    assert r2["a"] > r2["b"]


def test_ceiling_is_the_square_root_of_icc():
    t = pd.DataFrame({"target": ["roughness", "warmth"], "spearman": [0.6, 0.1]})
    out = G.against_ceiling(t, pd.Series({"roughness": 0.81, "warmth": -0.2}))
    assert out.loc[0, "ceiling"] == pytest.approx(0.9)
    assert out.loc[0, "fraction_of_ceiling"] == pytest.approx(0.6 / 0.9)
    assert np.isnan(out.loc[1, "ceiling"])


def test_zero_shot_baseline_aligns_by_image():
    idx = list("abcdef")
    ratings = pd.DataFrame({"roughness": [1, 2, 3, 4, 5, 6]}, index=idx)
    scores = pd.DataFrame({"roughness": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]},
                          index=idx[::-1])
    t = G.zero_shot_baseline(scores, ratings)
    assert t.loc[0, "spearman_zero_shot"] == pytest.approx(1.0)
