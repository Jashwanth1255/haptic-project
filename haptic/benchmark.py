"""Whole-dataset identification benchmark, linear models (analysis D, round 2).

Plan fixed in PREREGISTRATION.md ("Benchmark and findings round 2", D) before
any result: logistic regression on 4 raw descriptors, on 16 log bands + RMS, and
on both; tasks material and category; splits direction held out (8 folds) and
repeat 0 -> 1. Reported: accuracy, macro-F1, top-5, per-class recall, chance and
majority-class baseline. Descriptive: no predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import identity as I

FEATURE_SETS = {
    "raw4": I.RAW4,
    "bands": (*I.BAND_COLUMNS, "log_rms"),
    "bands+raw4": (*I.BAND_COLUMNS, "log_rms", *I.RAW4),
}


def splits(df: pd.DataFrame, split: str):
    if split == "direction":
        g = df.direction_deg.to_numpy()
        return [(g != d, g == d, str(d)) for d in np.unique(g)]
    if split == "repeat":
        r = df["repeat"].to_numpy() % 2
        return [(r == 0, r == 1, "repeat0->1")]
    raise ValueError(split)


def linear(df: pd.DataFrame, columns, label: str, split: str, top_k: int = 5,
           max_iter: int = 3000) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x = I._matrix(df, columns)
    y = df[label].to_numpy()
    classes = np.unique(y)
    pred = np.empty_like(y)
    hit_k = np.zeros(len(y), bool)
    tested = np.zeros(len(y), bool)
    per_fold = []
    for train, test, name in splits(df, split):
        if not test.any() or not train.any():
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, C=1.0))
        model.fit(x[train], y[train])
        proba = model.predict_proba(x[test])
        cls = model.classes_
        pred[test] = cls[proba.argmax(1)]
        k = min(top_k, len(cls))
        top = cls[np.argsort(-proba, 1)[:, :k]]
        hit_k[test] = (top == y[test][:, None]).any(1)
        tested |= test
        per_fold.append({"fold": name, "n": int(test.sum()),
                         "accuracy": float((pred[test] == y[test]).mean())})
    if not tested.any():
        return None
    yt, pt = y[tested], pred[tested]
    recall = pd.Series(pt == yt).groupby(yt).mean()
    counts = pd.Series(y).value_counts()
    folds = pd.DataFrame(per_fold)
    return {"accuracy": float((pt == yt).mean()),
            "macro_f1": float(f1_score(yt, pt, average="macro")),
            f"top{min(top_k, len(classes))}": float(hit_k[tested].mean()),
            "fold_std": float(folds.accuracy.std(ddof=1)) if len(folds) > 1 else None,
            "n": int(tested.sum()), "n_classes": int(len(classes)),
            "chance": 1 / len(classes), "majority_baseline": float(counts.max() / counts.sum()),
            "recall": {str(k): float(v) for k, v in recall.items()},
            "folds": per_fold}
