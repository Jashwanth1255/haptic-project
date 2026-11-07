"""Mantel test: must detect a real relation, reject an absent one, and with
strata must not credit structure the strata share. Scaffolding data only."""
import numpy as np

from haptic import rsa


def _points(n, seed):
    return np.random.default_rng(seed).normal(size=(n, 5))


def test_identical_geometry_is_significant():
    x = _points(20, 0)
    d = rsa.euclidean_distance(x)
    noisy = rsa.euclidean_distance(x + np.random.default_rng(1).normal(scale=0.1, size=x.shape))
    r = rsa.mantel(d, noisy, n_perm=499)
    assert r["r"] > 0.8 and r["p"] < 0.01


def test_unrelated_geometry_is_not_significant():
    r = rsa.mantel(rsa.euclidean_distance(_points(20, 2)),
                   rsa.euclidean_distance(_points(20, 3)), n_perm=499)
    assert r["p"] > 0.05


def test_strata_remove_shared_group_structure():
    # Two matrices that agree only on which group an item is in.
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(4), 6)
    centres = rng.normal(scale=10, size=(4, 3))
    a = centres[groups] + rng.normal(size=(24, 3))
    b = centres[groups] + rng.normal(size=(24, 3))
    da, db = rsa.euclidean_distance(a), rsa.euclidean_distance(b)
    free = rsa.mantel(da, db, n_perm=499)
    within = rsa.mantel(da, db, n_perm=499, strata=groups)
    assert free["p"] < 0.01
    assert within["p"] > 0.05


def test_strata_keep_a_within_group_relation():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(3), 8)
    x = rng.normal(size=(24, 3))
    da = rsa.euclidean_distance(x)
    db = rsa.euclidean_distance(x + rng.normal(scale=0.1, size=x.shape))
    assert rsa.mantel(da, db, n_perm=499, strata=groups)["p"] < 0.01


def test_cosine_distance_basics():
    x = np.array([[1.0, 0], [2.0, 0], [0, 1.0]])
    d = rsa.cosine_distance(x)
    assert np.allclose(np.diag(d), 0) and np.isclose(d[0, 1], 0) and np.isclose(d[0, 2], 1)
