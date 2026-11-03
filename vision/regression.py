"""Ridge regression for small samples: Stage 1 (images to perception) and Stage 2
(perception and haptics).

With 16 materials and up to thousands of embedding dimensions, an in-sample fit
means nothing. Every score here is out of sample:

- Leave one group out (one image, or one material). Standardisation, block
  weighting and the ridge penalty are all fitted on the training rows only. The
  penalty is chosen by efficient leave-one-out inside the training rows.
- R² is against the **mean predictor computed on the same training rows**, so 0
  means "no better than ignoring the input", and it can go negative.
- A permutation test shuffles targets across groups and refits everything. Its p
  is (1 + exceedances) / (1 + permutations).

Feature blocks (an embedding and five ratings, say) are standardised and then
scaled by 1/sqrt(width), so a wide block cannot drown a narrow one.

A Spearman correlation against human ratings is only interpretable beside the
ratings' own reliability. `against_ceiling` puts sqrt(ICC(2,k)) beside it: under
classical test theory, no predictor can be expected to correlate with the mean
ratings more strongly than that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

ALPHAS = np.logspace(-3, 5, 17)
N_PERMUTATIONS = 999


def _blocks(blocks) -> list[np.ndarray]:
    out = [np.asarray(b, dtype=float) for b in blocks]
    out = [b[:, None] if b.ndim == 1 else b for b in out]
    n = {b.shape[0] for b in out}
    if len(n) != 1:
        raise ValueError(f"blocks disagree on row count: {sorted(n)}")
    if not all(np.isfinite(b).all() for b in out):
        raise ValueError("non-finite feature values")
    return out


def _design(train: list[np.ndarray], test: list[np.ndarray]):
    """Standardise each column on the training rows, weight each block equally."""
    xs_tr, xs_te = [], []
    for tr, te in zip(train, test):
        mu, sd = tr.mean(0), tr.std(0)
        sd[sd == 0] = 1.0
        w = 1 / np.sqrt(tr.shape[1])
        xs_tr.append((tr - mu) / sd * w)
        xs_te.append((te - mu) / sd * w)
    return np.hstack(xs_tr), np.hstack(xs_te)


def _folds(n: int, groups):
    g = np.arange(n) if groups is None else np.asarray(groups)
    if len(g) != n:
        raise ValueError("one group label per row")
    levels = pd.unique(g)
    if len(levels) < 3:
        raise ValueError("need at least 3 groups to leave one out")
    return [(np.flatnonzero(g != lv), np.flatnonzero(g == lv)) for lv in levels]


def predict_out_of_fold(blocks, Y, groups=None, alphas=ALPHAS):
    """(ridge predictions, mean-predictor predictions), both (n, targets)."""
    X = _blocks(blocks)
    Y = np.asarray(Y, dtype=float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    if Y.shape[0] != X[0].shape[0]:
        raise ValueError("features and targets disagree on row count")
    pred, base = np.empty_like(Y), np.empty_like(Y)
    for tr, te in _folds(len(Y), groups):
        a, b = _design([x[tr] for x in X], [x[te] for x in X])
        mu = Y[tr].mean(0)
        model = RidgeCV(alphas=alphas, alpha_per_target=True).fit(a, Y[tr] - mu)
        pred[te] = model.predict(b).reshape(len(te), -1) + mu
        base[te] = mu
    return pred, base


def r2_out_of_sample(y, pred, base) -> float:
    sse, ss0 = np.sum((y - pred) ** 2), np.sum((y - base) ** 2)
    return float(1 - sse / ss0) if ss0 > 0 else float("nan")


def _scores(Y, pred, base, names) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(names):
        y = Y[:, j]
        rho = spearmanr(y, pred[:, j]).statistic if np.ptp(pred[:, j]) > 0 else np.nan
        rows.append({"target": name, "r2": r2_out_of_sample(y, pred[:, j], base[:, j]),
                     "spearman": float(rho),
                     "mae": float(np.abs(y - pred[:, j]).mean()),
                     "mae_mean_predictor": float(np.abs(y - base[:, j]).mean())})
    return pd.DataFrame(rows)


def _permute(Y, groups, rng):
    """Shuffle targets across groups. Groups must each carry one target row, or
    the shuffle would break the structure the test assumes."""
    if groups is None:
        return Y[rng.permutation(len(Y))]
    g = np.asarray(groups)
    levels = pd.unique(g)
    first = {lv: np.flatnonzero(g == lv)[0] for lv in levels}
    for lv in levels:
        if not np.allclose(Y[g == lv], Y[first[lv]]):
            raise ValueError("a permutation test across groups needs one target per group")
    shuffled = dict(zip(levels, rng.permutation(levels)))
    return np.vstack([Y[first[shuffled[lv]]] for lv in g])


def evaluate(blocks, Y, names, groups=None, n_permutations: int = N_PERMUTATIONS,
             seed: int = 0, alphas=ALPHAS) -> pd.DataFrame:
    """Out-of-sample R², Spearman and MAE per target, with permutation p for R²."""
    Y = np.asarray(Y, dtype=float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    names = list(names)
    if len(names) != Y.shape[1]:
        raise ValueError("one name per target column")
    pred, base = predict_out_of_fold(blocks, Y, groups, alphas)
    table = _scores(Y, pred, base, names)
    if n_permutations:
        rng = np.random.default_rng(seed)
        exceed = np.zeros(len(names))
        for _ in range(n_permutations):
            Yp = _permute(Y, groups, rng)
            pp, bp = predict_out_of_fold(blocks, Yp, groups, alphas)
            null = [r2_out_of_sample(Yp[:, j], pp[:, j], bp[:, j]) for j in range(len(names))]
            exceed += np.asarray(null) >= table["r2"].to_numpy()
        table["p_permutation"] = (1 + exceed) / (1 + n_permutations)
    table["n"] = len(Y)
    table["n_groups"] = len(pd.unique(np.arange(len(Y)) if groups is None else groups))
    return table


def ablation(sources: dict[str, list], Y, names, groups=None, **kw) -> pd.DataFrame:
    """The same targets predicted from each named combination of feature blocks,
    e.g. {"embedding": [E], "perception": [P], "embedding+perception": [E, P]}."""
    return pd.concat([evaluate(blocks, Y, names, groups, **kw).assign(features=label)
                      for label, blocks in sources.items()], ignore_index=True)


def zero_shot_baseline(scores: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Spearman between CLIP zero-shot scores and mean ratings. Nothing is fitted,
    so no cross-validation is needed. Rows are aligned by index."""
    common = scores.index.intersection(ratings.index)
    if len(common) < 3:
        raise ValueError("fewer than 3 images in common")
    return pd.DataFrame([{"target": a, "spearman_zero_shot":
                          float(spearmanr(scores.loc[common, a],
                                          ratings.loc[common, a]).statistic)}
                         for a in ratings.columns if a in scores.columns])


def against_ceiling(table: pd.DataFrame, icc: pd.Series) -> pd.DataFrame:
    """Add each target's ICC(2,k) and its correlation ceiling sqrt(ICC). A
    negative ICC gives no ceiling: the raters did not agree at all."""
    out = table.copy()
    out["icc2k"] = out["target"].map(icc)
    out["ceiling"] = np.sqrt(out["icc2k"].clip(lower=0)).where(out["icc2k"] > 0)
    out["fraction_of_ceiling"] = out["spearman"] / out["ceiling"]
    return out
