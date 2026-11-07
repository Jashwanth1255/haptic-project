"""Representational similarity: does one distance matrix track another?

Used for pre-registered analysis A (PREREGISTRATION.md, "Does looks predict
feel?"): visual distance between materials against haptic distance.

`mantel` correlates the upper triangles of two square distance matrices with
Spearman's rho, and builds the null by permuting the rows and columns of one
matrix together. With `strata`, labels are permuted only within each stratum
(for example, material category), so structure the strata share cannot produce
the correlation.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def upper(d: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(len(d), 1)
    return d[i, j]


def _pearson_of_ranks(rx: np.ndarray, ry: np.ndarray) -> float:
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


def _permutation(n: int, strata, rng) -> np.ndarray:
    if strata is None:
        return rng.permutation(n)
    strata = np.asarray(strata)
    p = np.arange(n)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        p[idx] = rng.permutation(idx)
    return p


def mantel(a: np.ndarray, b: np.ndarray, n_perm: int = 9999, strata=None,
           seed: int = 0) -> dict:
    """One-sided (r > 0) Spearman Mantel test. p counts the observed value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError("need two square matrices of the same size")
    n = len(a)
    i, j = np.triu_indices(n, 1)
    ra = rankdata(a[i, j])
    r_obs = _pearson_of_ranks(ra, rankdata(b[i, j]))
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        p = _permutation(n, strata, rng)
        bp = b[np.ix_(p, p)]
        if _pearson_of_ranks(ra, rankdata(bp[i, j])) >= r_obs:
            ge += 1
    return {"r": r_obs, "p": (ge + 1) / (n_perm + 1), "n_items": n, "n_perm": n_perm,
            "stratified": strata is not None}


def cosine_distance(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return np.clip(1.0 - x @ x.T, 0.0, 2.0)


def euclidean_distance(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    sq = (x ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0.0))
