"""How much fabric identity is in the acceleration at all? EXPLORATORY.

Not pre-registered. Added on 2026-11-04 after the Objective 1 gate failed on real
cloth, to separate two explanations: the four-number descriptor discards identity
that the signal holds, or the signal holds little. It does not replace the gate,
whose verdict stands.

Material identification, cross-validated with the **sliding direction held out**:
eight folds, one per direction, so a recording is never classified by a model that
saw its own direction. Three feature sets are compared on the same folds:

    raw4         the four Objective 1 descriptors (log of the positive ones)
    norm4        the four velocity-normalised descriptors
    bands        log power in N_BANDS log-spaced bands over 20-1000 Hz, as shares of
                 the band total, plus log RMS: the spectrum's shape and level

The classifier is a standardised multinomial logistic regression. Chance is 1/16.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

from . import descriptor as D

N_BANDS = 16
BAND_EDGES = np.geomspace(D.BAND_LOW_HZ, D.BAND_HIGH_HZ, N_BANDS + 1)
BAND_COLUMNS = tuple(f"band{k:02d}" for k in range(N_BANDS))
RAW4 = ("frequency_hz", "intensity", "pattern", "duration_ms")
NORM4 = D.NORMALISED_CORE


def band_features(accel: np.ndarray, fs: float) -> dict:
    """Log share of 20-1000 Hz power in each band, and log RMS in the band."""
    x = D.bandpass(D.to_magnitude(accel), fs)
    f, p = welch(x, fs, nperseg=min(2048, len(x)))
    inside = (f >= BAND_EDGES[0]) & (f < BAND_EDGES[-1])
    total = p[inside].sum()
    out = {}
    for k, (lo, hi) in enumerate(zip(BAND_EDGES[:-1], BAND_EDGES[1:])):
        out[BAND_COLUMNS[k]] = float(np.log(p[(f >= lo) & (f < hi)].sum() / total + 1e-12))
    out["log_rms"] = float(np.log(np.sqrt(np.mean(x ** 2)) + 1e-12))
    return out


def _matrix(df: pd.DataFrame, columns) -> np.ndarray:
    x = df[list(columns)].astype(float).to_numpy()
    positive = (x > 0).all(axis=0) & (np.asarray(columns) != "pattern")
    x[:, positive] = np.log(x[:, positive])
    return x


def identify(df: pd.DataFrame, columns, groups: str = "direction_deg",
             label: str = "material_id") -> dict:
    """Leave-one-`groups`-value-out accuracy, its spread over folds, and the
    per-material recall pooled over folds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x, y, g = _matrix(df, columns), df[label].to_numpy(), df[groups].to_numpy()
    pred = np.empty_like(y)
    per_fold = []
    for held in np.unique(g):
        test = g == held
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        model.fit(x[~test], y[~test])
        pred[test] = model.predict(x[test])
        per_fold.append(float((pred[test] == y[test]).mean()))
    recall = pd.Series(pred == y).groupby(y).mean()
    return {"accuracy": float((pred == y).mean()), "fold_mean": float(np.mean(per_fold)),
            "fold_std": float(np.std(per_fold, ddof=1)), "n_folds": len(per_fold),
            "chance": 1 / len(np.unique(y)), "n": len(y),
            "recall": recall.to_dict(), "predictions": pred}


def variance_shares(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Share of each dimension's variance (type II sums of squares) taken by
    material, direction, their interaction, velocity, force and the residual.
    Positive dimensions are logged first."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    rows = []
    for col in columns:
        y = df[col].astype(float)
        if col != "pattern" and (y > 0).all():
            y = np.log(y)
        extra = "".join(f" + C({c})" for c in ("velocity_mm_s", "force_n")
                        if df[c].nunique() > 1)
        fit = smf.ols(f"y ~ C(material_id) * C(direction_deg){extra}",
                      data=df.assign(y=y.to_numpy())).fit()
        ss = anova_lm(fit, typ=2)["sum_sq"]
        share = (ss / ss.sum()).rename(lambda s: s.replace("C(", "").replace(")", ""))
        rows.append({"dimension": col, **share.to_dict()})
    return pd.DataFrame(rows)
