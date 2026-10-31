"""Mantel test for Objective 3, on constructed matrices only.

No real or synthetic-signal descriptors appear here. Every matrix is built so the
right answer is known in advance.
"""
import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from haptic import crossmodal as X


def _points(k, rng):
    return squareform(pdist(rng.normal(size=(k, 2))))


def _confusion(s, per_class=1.0):
    """Confusion matrix whose symmetrised off-diagonal is exactly `s`.

    Every row sums to `per_class`, so normalising leaves the entries alone.
    """
    c = np.array(s, dtype=float) * per_class
    np.fill_diagonal(c, 0)
    np.fill_diagonal(c, per_class - c.sum(axis=1))
    assert (c >= 0).all()
    return c


def _dist_and_conf(k=12, seed=0):
    rng = np.random.default_rng(seed)
    d = _points(k, rng)
    return d, _confusion(0.5 * np.exp(-d) / k)


def test_perfect_positive_is_recovered_and_significant():
    """Confusability a strictly decreasing function of distance: rho is exactly 1."""
    d, c = _dist_and_conf()
    r = X.mantel_test(c, d, n_permutations=999)
    assert r.rho == pytest.approx(1.0)
    assert r.p_value == pytest.approx(1 / 1000)


def test_perfect_negative_is_recovered():
    d, _ = _dist_and_conf()
    c = _confusion(0.5 * d / d.max() / len(d))
    r = X.mantel_test(c, d, n_permutations=999)
    assert r.rho == pytest.approx(-1.0)
    assert r.p_value < 0.01


def test_rho_equals_scipy_on_the_off_diagonal_with_ties():
    rng = np.random.default_rng(3)
    k = 10
    d = _points(k, rng)
    s = rng.integers(0, 4, (k, k)) * 0.002     # heavily tied, like real confusions
    s = (s + s.T) / 2
    r = X.mantel_test(_confusion(s), d, n_permutations=9)
    upper = np.triu_indices(k, 1)
    assert r.rho == pytest.approx(spearmanr(s[upper], -d[upper]).statistic)


def test_noisy_real_relationship_is_detected():
    rng = np.random.default_rng(1)
    k = 14
    d = _points(k, rng)
    noise = rng.normal(0, 0.15, (k, k))
    s = np.exp(-d) + (noise + noise.T) / 2
    s = 0.5 * (s - s.min()) / np.ptp(s) / k
    r = X.mantel_test(_confusion(s), d, n_permutations=999)
    assert 0.3 < r.rho < 0.99
    assert r.p_value < 0.01


def test_null_case_comes_back_non_significant():
    """Confusions and distances built independently of each other."""
    rng = np.random.default_rng(7)
    k = 16
    d = _points(k, rng)
    other = _points(k, rng)
    r = X.mantel_test(_confusion(0.5 * np.exp(-other) / k), d, n_permutations=999)
    assert r.p_value > 0.05
    assert abs(r.rho) < 0.2


def test_false_positive_rate_is_near_alpha():
    """Across many null datasets the test must reject about 5% of the time.
    A wrong null distribution fails here even if one null case happens to pass."""
    rng = np.random.default_rng(11)
    k, runs = 10, 300
    rejected = 0
    for i in range(runs):
        d = _points(k, rng)
        other = _points(k, rng)
        r = X.mantel_test(_confusion(0.5 * np.exp(-other) / k), d,
                          n_permutations=199, seed=i)
        rejected += r.p_value < 0.05
    assert 0.02 <= rejected / runs <= 0.09, f"rejected {rejected}/{runs}"


def test_n_counts_pairs_not_cells():
    d, c = _dist_and_conf(k=8)
    r = X.mantel_test(c, d, n_permutations=9)
    assert r.n == 28 and r.n_materials == 8


def test_p_value_floor_and_ceiling():
    d, c = _dist_and_conf()
    r = X.mantel_test(c, d, n_permutations=99)
    assert r.p_value == pytest.approx(0.01)
    assert X.mantel_test(c, d, n_permutations=99, alternative="less").p_value == 1.0


def test_one_sided_alternatives_split_correctly():
    d, c = _dist_and_conf()
    assert X.mantel_test(c, d, n_permutations=99, alternative="greater").p_value < 0.05
    assert X.mantel_test(c, d, n_permutations=99, alternative="less").p_value > 0.9


def test_same_seed_same_answer():
    d, c = _dist_and_conf()
    a = X.mantel_test(c, d, n_permutations=99, seed=5)
    assert a == X.mantel_test(c, d, n_permutations=99, seed=5)


def test_symmetrise_by_hand_with_unequal_supports():
    """i->j is 2/10, j->i is 3/5. Raw counts would give 2.5, and let the larger
    class dominate."""
    s = X.symmetrise_confusion([[8, 2], [3, 2]])
    assert s[0, 1] == pytest.approx((0.2 + 0.6) / 2)
    assert s[0, 1] == s[1, 0]


def _balanced_counts(k, m, rng):
    """Integer counts with every row and column summing to `m`: a sum of random
    permutation matrices, which is how equal-support test splits look."""
    c = np.zeros((k, k))
    for _ in range(m):
        c[np.arange(k), rng.permutation(k)] += 1
    return c


def test_confusion_direction_does_not_matter():
    rng = np.random.default_rng(2)
    k = 9
    d = _points(k, rng)
    c = _balanced_counts(k, 40, rng)
    assert not np.array_equal(c, c.T)
    assert X.mantel_test(c, d, n_permutations=99).rho == pytest.approx(
        X.mantel_test(c.T, d, n_permutations=99).rho)


def test_tied_pairs_stay_tied_after_averaging():
    """Regression. 1+2 and 0+3 errors out of 40 are the same pair score, but
    float averaging can split them by one bit and rank them apart. The exact
    integer sums give the right answer to compare with."""
    rng = np.random.default_rng(2)
    k = 12
    d = _points(k, rng)
    c = _balanced_counts(k, 40, rng)
    upper = np.triu_indices(k, 1)
    exact = spearmanr((c + c.T)[upper], -d[upper]).statistic
    assert X.mantel_test(c, d, n_permutations=9).rho == pytest.approx(exact)


def test_diagonals_do_not_enter_the_correlation():
    d, c = _dist_and_conf()
    base = X.mantel_test(c, d, n_permutations=9).rho
    noisy = d.copy()
    np.fill_diagonal(noisy, 123.0)
    assert X.mantel_test(c, noisy, n_permutations=9).rho == pytest.approx(base)


def test_scaling_counts_changes_nothing():
    d, c = _dist_and_conf()
    assert X.mantel_test(c * 40, d, n_permutations=9).rho == pytest.approx(
        X.mantel_test(c, d, n_permutations=9).rho)


def test_scaling_distance_changes_nothing():
    d, c = _dist_and_conf()
    assert X.mantel_test(c, d * 7, n_permutations=9).rho == pytest.approx(
        X.mantel_test(c, d, n_permutations=9).rho)


GOOD = np.ones((4, 4)) - np.eye(4)


@pytest.mark.parametrize("conf, dist, match", [
    (np.ones((4, 4)), np.ones((5, 5)), "differ in size"),
    (np.ones((4, 3)), GOOD, "square"),
    (np.ones((4, 4)), np.arange(16.0).reshape(4, 4), "symmetric"),
    (np.ones((4, 4)), -GOOD, "negative"),
    (-np.ones((4, 4)), GOOD, "negative"),
    (np.ones((3, 3)), np.ones((3, 3)), "at least 4"),
    (np.eye(4) * 0, GOOD, "no images"),
    (np.ones((4, 4)) * np.nan, GOOD, "non-finite"),
    (np.ones((4, 4)), GOOD, "constant"),
])
def test_bad_input_raises(conf, dist, match):
    with pytest.raises(ValueError, match=match):
        X.mantel_test(conf, dist, n_permutations=9)


def test_constant_confusability_is_undefined():
    d, _ = _dist_and_conf(k=6)
    with pytest.raises(ValueError, match="confusability is constant"):
        X.mantel_test(np.eye(6) * 40, d, n_permutations=9)


def test_docstring_carries_the_warning():
    assert "DO NOT RUN UNTIL OBJECTIVE 1" in X.__doc__
