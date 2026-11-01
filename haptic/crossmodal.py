"""Does visual confusability predict haptic similarity?

**OBJECTIVE 3 ANALYSIS. DO NOT RUN UNTIL OBJECTIVE 1 HAS PRODUCED REAL DESCRIPTORS.**
The Cluster cloth descriptors do not exist yet. Running this on anything derived
from `signals.py` would only recover what the generator was told, and must never
be reported as a finding.

If two materials are easily mistaken for each other by the visual classifier, are
they also close in haptic descriptor space? Pairs of materials are the unit here,
which is what makes an ordinary significance test wrong: the N(N-1)/2 pairs share
materials, so they are not independent, and a p-value from `spearmanr` is too
small. The Mantel test instead permutes material labels, moving rows and columns
together, which keeps that dependence intact under the null.

Sign convention: `rho` is computed against haptic *similarity*, so a positive
value means more visually confusable pairs are more haptically alike.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

MIN_MATERIALS = 4       # 3 materials have 6 orderings: p could never be small
ALTERNATIVES = ("two-sided", "greater", "less")
TIE_DECIMALS = 12       # rates below this are rounding noise, not a difference


@dataclass(frozen=True)
class MantelResult:
    rho: float              # Spearman, off-diagonal, confusability vs similarity
    p_value: float          # Mantel permutation p
    n: int                  # material pairs, N(N-1)/2
    n_materials: int
    n_permutations: int
    alternative: str


def symmetrise_confusion(confusion) -> np.ndarray:
    """Symmetric confusability matrix from a confusion matrix of counts.

    Rows are normalised first, so a class with more test images does not look
    more confusable. Cell (i, j) of the result is then the mean of the two error
    rates i->j and j->i. The diagonal is returned as it stands and means accuracy,
    not confusability: callers must leave it out, as `mantel_test` does.
    """
    c = np.asarray(confusion, dtype=float)
    _check_square(c, "confusion")
    if (c < 0).any():
        raise ValueError("confusion holds a negative count")
    support = c.sum(axis=1, keepdims=True)
    if (support == 0).any():
        raise ValueError("confusion has a class with no images")
    rates = c / support
    return (rates + rates.T) / 2


def _check_square(m: np.ndarray, name: str) -> None:
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {m.shape}")
    if not np.isfinite(m).all():
        raise ValueError(f"{name} holds a non-finite value")


def _check_labels(confusion_labels, distance_labels, size: int) -> None:
    """Refuse anything but the same N unique names in the same order.

    Mismatches raise rather than being reordered: a misaligned matrix is a bug
    upstream, and the correlation it produces would look perfectly plausible.
    """
    if confusion_labels is None or distance_labels is None:
        raise ValueError("labels are required: bare arrays cannot be aligned")
    a, b = [str(x) for x in confusion_labels], [str(x) for x in distance_labels]
    for name, labels in (("confusion", a), ("distance", b)):
        if len(labels) != size:
            raise ValueError(f"{name}_labels has {len(labels)} names for "
                             f"{size} materials")
        if len(set(labels)) != size:
            raise ValueError(f"{name}_labels holds duplicate names")
    if set(a) != set(b):
        raise ValueError(f"labels mismatch: {sorted(set(a) ^ set(b))}")
    if a != b:
        at = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        raise ValueError(f"labels are in a different order, first at position "
                         f"{at}: {a[at]!r} vs {b[at]!r}")


def mantel_test(confusion, distance, *, confusion_labels, distance_labels,
                n_permutations: int = 9999, alternative: str = "two-sided",
                seed: int = 0) -> MantelResult:
    """Spearman correlation between confusability and haptic similarity.

    `confusion` is a matrix of counts and `distance` a symmetric haptic distance
    matrix. Each must come with its material labels, which have to be identical
    in content and order; there is no way to call this without them. Diagonals
    are excluded: a material's distance to itself and its accuracy say nothing
    about a pair.

    Ties are expected, since most confusion cells are zero, and get average
    ranks. The p-value counts the observed labelling as one permutation, so it
    can never be 0; its floor is 1 / (n_permutations + 1).
    """
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    conf = symmetrise_confusion(confusion)
    dist = np.asarray(distance, dtype=float)
    _check_square(dist, "distance")
    if conf.shape != dist.shape:
        raise ValueError(f"matrices differ in size: {conf.shape} vs {dist.shape}")
    if not np.allclose(dist, dist.T):
        raise ValueError("distance must be symmetric")
    if (dist < 0).any():
        raise ValueError("distance holds a negative value")
    k = len(dist)
    _check_labels(confusion_labels, distance_labels, k)
    if k < MIN_MATERIALS:
        raise ValueError(f"need at least {MIN_MATERIALS} materials, got {k}")

    upper = np.triu_indices(k, 1)
    # Rounded because (1/40 + 2/40)/2 and (0/40 + 3/40)/2 can differ in the last
    # bit. Those pairs are tied, and rankdata would otherwise order them by noise.
    a, b = np.round(conf[upper], TIE_DECIMALS), -dist[upper]
    for name, v in (("confusability", a), ("distance", b)):
        if np.ptp(v) == 0:
            raise ValueError(f"{name} is constant off the diagonal: rho is undefined")

    # Pearson on ranks is Spearman. Permuting labels only reorders the same
    # ranks, so they are computed once.
    ra = _standardise(rankdata(a))
    rb = _standardise(rankdata(b))
    ranks = np.zeros((k, k))
    ranks[upper] = rb
    ranks += ranks.T

    rho = float(ra @ rb)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_permutations):
        p = rng.permutation(k)
        r = ra @ ranks[np.ix_(p, p)][upper]
        hits += _as_extreme(r, rho, alternative)

    return MantelResult(rho=rho, p_value=(hits + 1) / (n_permutations + 1),
                        n=len(a), n_materials=k,
                        n_permutations=n_permutations, alternative=alternative)


def _standardise(v: np.ndarray) -> np.ndarray:
    v = v - v.mean()
    return v / np.linalg.norm(v)


def _as_extreme(r: float, observed: float, alternative: str) -> bool:
    # The tolerance keeps a relabelling that reproduces `observed` from being
    # missed through rounding, which would understate p.
    eps = 1e-12
    if alternative == "greater":
        return r >= observed - eps
    if alternative == "less":
        return r <= observed + eps
    return abs(r) >= abs(observed) - eps
