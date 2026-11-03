"""Objective 2's headline: how much is lost moving from DTD to product photographs?

Two people label each product photograph with one of the 12 pre-registered textile
classes, independently. Agreement is reported as Cohen's kappa. Only photographs
both labellers agree on are scored, and the disagreements are counted, not hidden.

Every DTD fold model scores the same photographs, and each is compared with its own
fold's 12-way restricted accuracy on the DTD test split. The gap is therefore paired
per model, and reported as mean and spread across the folds, overall and per class.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .textile import TEXTILE_CLASSES

MIN_PHOTOS = 50         # the set the project committed to
SMALL_CLASS = 5         # per-class recall on fewer photos than this is flagged


def read_labels(path: str | Path) -> pd.Series:
    """A labeller's CSV with columns `image,label`. Labels must be textile classes."""
    df = pd.read_csv(path, dtype=str).apply(lambda c: c.str.strip())
    if list(df.columns) != ["image", "label"]:
        raise ValueError(f"{path}: columns must be image,label, got {list(df.columns)}")
    if df["image"].duplicated().any():
        raise ValueError(f"{path}: duplicated images "
                         f"{sorted(df.loc[df['image'].duplicated(), 'image'])}")
    bad = sorted(set(df["label"]) - set(TEXTILE_CLASSES))
    if bad:
        raise ValueError(f"{path}: labels outside the 12 textile classes: {bad}")
    return df.set_index("image")["label"]


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0 or a.size != b.size:
        raise ValueError("need two equal, non-empty label lists")
    labels = np.union1d(a, b)
    observed = float((a == b).mean())
    expected = float(sum((a == c).mean() * (b == c).mean() for c in labels))
    return float("nan") if expected == 1.0 else (observed - expected) / (1 - expected)


def agreement(a: pd.Series, b: pd.Series) -> dict:
    common = a.index.intersection(b.index)
    x, y = a[common], b[common]
    return {"n_labelled_a": len(a), "n_labelled_b": len(b), "n_both": len(common),
            "only_a": sorted(a.index.difference(b.index)),
            "only_b": sorted(b.index.difference(a.index)),
            "n_agreed": int((x == y).sum()),
            "raw_agreement": float((x == y).mean()) if len(common) else float("nan"),
            "cohen_kappa": cohen_kappa(x, y) if len(common) else float("nan")}


def agreed(a: pd.Series, b: pd.Series) -> pd.Series:
    common = a.index.intersection(b.index)
    keep = common[(a[common] == b[common]).to_numpy()]
    return a[keep].sort_index()


def dtd_reference(results: str | Path, fold: int) -> tuple[float, pd.DataFrame]:
    """A fold's 12-way restricted accuracy and per-class table on the DTD test split."""
    folder = Path(results) / f"dtd_fold{fold}_seed0" / "textile12"
    m = json.loads((folder / "textile_metrics.json").read_text())
    if m.get("fold", fold) != fold:     # fold 1 was saved before the key existed
        raise ValueError(f"{folder} holds fold {m['fold']}, not {fold}")
    return m["restricted"]["top1"], pd.read_csv(folder / "restricted_per_class.csv")


def score_photos(model, items, classes, device, num_workers: int = 0) -> M.ClassificationReport:
    """12-way restricted scores on photographs. `items` are (path, index into the
    model's class list)."""
    from . import data as V
    from .textile import logits_of, score_subset
    ds = V.ImageList(items, V.transforms_for(train=False))
    logits, y = logits_of(model, ds, device, num_workers=num_workers)
    return score_subset(logits, y, classes)["restricted"]


def gap_tables(photo: dict[int, M.ClassificationReport],
               reference: dict[int, tuple[float, pd.DataFrame]]
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per fold, per class). Gap is photo minus DTD, so negative means a loss."""
    folds, per_class = [], []
    for fold, rep in sorted(photo.items()):
        top1, ref = reference[fold]
        folds.append({"fold": fold, "dtd_top1": top1, "photo_top1": rep.top1,
                      "gap_top1": rep.top1 - top1, "photo_macro_f1": rep.macro_f1})
        pc = rep.per_class.set_index("class")
        r = ref.set_index("class")
        for c in TEXTILE_CLASSES:
            per_class.append({"fold": fold, "class": c,
                              "n_photos": int(pc.loc[c, "support"]),
                              "dtd_recall": float(r.loc[c, "recall"]),
                              "photo_recall": float(pc.loc[c, "recall"])})
    pcl = pd.DataFrame(per_class)
    pcl["gap_recall"] = pcl["photo_recall"] - pcl["dtd_recall"]
    summary = (pcl.groupby("class", sort=False)
               .agg(n_photos=("n_photos", "first"), dtd_recall=("dtd_recall", "mean"),
                    photo_recall=("photo_recall", "mean"),
                    gap_mean=("gap_recall", "mean"), gap_std=("gap_recall", "std"))
               .reset_index().sort_values("gap_mean"))
    summary["few_photos"] = summary["n_photos"] < SMALL_CLASS
    return pd.DataFrame(folds), summary
