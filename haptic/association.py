"""Objective 3 -- associate an image with a haptic descriptor.

**NO RESULT FROM THIS MODULE IS VALID UNTIL REAL CLUSTER DESCRIPTORS EXIST.** At
present it only runs on `synth_o3`, whose images and descriptors are both invented,
and such a run proves that the code works and nothing about fabric.
`Views.provenance` carries this into every report. Objective 1 gates this analysis.

Two strategies:

    class pair   a material's (MINC class, DTD class) is looked up. A query scores
                 each material as P(its material class) x P(its texture class).
    cosine       cosine similarity between the query's ResNet-50 embedding and each
                 material's mean reference embedding.

Two protocols, which answer different questions:

    closed-set   every material is in the database. Did it retrieve the right one?
                 Views are split into `n_folds`; each fold in turn is the query set
                 and all others build the database, so query and database never
                 share a view. Top-1 and top-3, per fold.
    open-set     leave-one-material-out, one fold per material. The test material
                 is absent from the database. How wrong is the nearest substitute?
                 MAE per dimension against the held-out truth, per fold, next to
                 two baselines: the mean of the database, and a random substitute.

The folds of each protocol share materials or views, so the std across folds is
spread between folds and not a standard error.

Materials with the same class pair are indistinguishable to the lookup. On real
cloth this will happen often, since MINC is likely to call nearly all of it
"fabric". Ties are never broken in the strategy's favour: top-k gives fractional
credit, and a tied substitute is the mean of the tied descriptors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .descriptor import Descriptor

DIMS = Descriptor.CORE
SYNTHETIC_BANNER = "SYNTHETIC -- scaffolding only, not a result. Valid only on real descriptors."
TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Views:
    """Everything known per material, over a common set of views.

    names          n material names
    descriptors    (n, 4) ground truth, in DIMS order
    embeddings     (n, views, d)
    material_probs (n, views, material classes)
    texture_probs  (n, views, texture classes)
    provenance     "synthetic" or "real"
    """
    names: tuple[str, ...]
    descriptors: np.ndarray
    embeddings: np.ndarray
    material_probs: np.ndarray
    texture_probs: np.ndarray
    provenance: str

    def __post_init__(self):
        if self.provenance not in ("synthetic", "real"):
            raise ValueError(f"unknown provenance: {self.provenance}")
        n = len(self.names)
        for name, a in (("descriptors", self.descriptors), ("embeddings", self.embeddings),
                        ("material_probs", self.material_probs),
                        ("texture_probs", self.texture_probs)):
            if len(a) != n:
                raise ValueError(f"{name} has {len(a)} rows for {n} materials")
        if not np.isfinite(self.descriptors).all():
            raise ValueError("descriptors hold a NaN: drop those materials and "
                             "report them, as for out-of-band recordings")

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def n_views(self) -> int:
        return self.embeddings.shape[1]


@dataclass(frozen=True)
class Database:
    keep: np.ndarray            # material indices present
    descriptors: np.ndarray
    embeddings: np.ndarray      # (m, d), unit length
    material_class: np.ndarray
    texture_class: np.ndarray


def build_database(views: Views, materials, reference_views) -> Database:
    """A database from these materials and views only: nothing else can leak in."""
    keep = np.asarray(materials)
    ref = np.asarray(reference_views)
    emb = views.embeddings[keep][:, ref].mean(axis=1)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def majority(probs):
        votes = probs[keep][:, ref].argmax(axis=2)
        return np.array([np.bincount(row, minlength=probs.shape[2]).argmax() for row in votes])

    return Database(keep, views.descriptors[keep], emb,
                    majority(views.material_probs), majority(views.texture_probs))


def cosine_scores(views: Views, material: int, view: int, db: Database) -> np.ndarray:
    q = views.embeddings[material, view]
    return db.embeddings @ (q / np.linalg.norm(q))


def class_pair_scores(views: Views, material: int, view: int, db: Database) -> np.ndarray:
    return (views.material_probs[material, view][db.material_class]
            * views.texture_probs[material, view][db.texture_class])


STRATEGIES: dict[str, Callable] = {"class_pair": class_pair_scores, "cosine": cosine_scores}


def topk_credit(scores: np.ndarray, true_index: int, k: int) -> float:
    """1 if the truth is in the top k, with fractional credit across a tie: the
    chance a random tie-break would have put it there."""
    s = scores[true_index]
    better = int((scores > s + TIE_TOLERANCE).sum())
    tied = int((np.abs(scores - s) <= TIE_TOLERANCE).sum())
    return float(np.clip((k - better) / tied, 0.0, 1.0))


def substitute(scores: np.ndarray, descriptors: np.ndarray) -> np.ndarray:
    """Top-scoring descriptor. A tie returns the mean of the tied descriptors."""
    return descriptors[scores >= scores.max() - TIE_TOLERANCE].mean(axis=0)


def view_folds(n_views: int, n_folds: int) -> list[np.ndarray]:
    if not 2 <= n_folds <= n_views:
        raise ValueError(f"cannot make {n_folds} folds from {n_views} views")
    return np.array_split(np.arange(n_views), n_folds)


def closed_set(views: Views, strategy: str, n_folds: int = 4) -> pd.DataFrame:
    """Per-fold top-1 and top-3, with every material in the database."""
    rows = []
    every = np.arange(views.n)
    for f, query in enumerate(view_folds(views.n_views, n_folds)):
        db = build_database(views, every, np.setdiff1d(np.arange(views.n_views), query))
        credits = [(topk_credit(STRATEGIES[strategy](views, m, v, db), m, 1),
                    topk_credit(STRATEGIES[strategy](views, m, v, db), m, 3))
                   for m in every for v in query]
        rows.append({"strategy": strategy, "fold": f, "n_queries": len(credits),
                     "top1": float(np.mean([c[0] for c in credits])),
                     "top3": float(np.mean([c[1] for c in credits]))})
    return pd.DataFrame(rows)


def open_set(views: Views, strategy: str) -> pd.DataFrame:
    """Leave-one-material-out. One row per fold (material) per dimension.

    Standardised errors divide by the spread of the database that fold can see,
    so the held-out material does not influence its own scaling.
    """
    all_views = np.arange(views.n_views)
    rows = []
    for m in range(views.n):
        db = build_database(views, np.delete(np.arange(views.n), m), all_views)
        truth = views.descriptors[m]
        retrieved = np.mean([np.abs(substitute(STRATEGIES[strategy](views, m, v, db),
                                               db.descriptors) - truth)
                             for v in all_views], axis=0)
        mean_base = np.abs(db.descriptors.mean(axis=0) - truth)
        random_base = np.abs(db.descriptors - truth).mean(axis=0)
        scale = db.descriptors.std(axis=0, ddof=1)
        for i, dim in enumerate(DIMS):
            rows.append({"strategy": strategy, "fold": m, "held_out": views.names[m],
                         "dimension": dim, "mae": retrieved[i],
                         "mae_mean_baseline": mean_base[i],
                         "mae_random_baseline": random_base[i],
                         "std_mae": retrieved[i] / scale[i],
                         "std_mae_mean_baseline": mean_base[i] / scale[i],
                         "std_mae_random_baseline": random_base[i] / scale[i]})
    return pd.DataFrame(rows)


def substitutes(views: Views, strategy: str) -> dict[int, int]:
    """Each material's most-often retrieved substitute, with it held out."""
    out = {}
    all_views = np.arange(views.n_views)
    for m in range(views.n):
        db = build_database(views, np.delete(np.arange(views.n), m), all_views)
        picks = [int(np.argmax(STRATEGIES[strategy](views, m, v, db))) for v in all_views]
        out[m] = int(db.keep[np.bincount(picks).argmax()])
    return out


def summarise_closed(folds: pd.DataFrame, n_materials: int) -> pd.DataFrame:
    out = folds.groupby("strategy")[["top1", "top3"]].agg(["mean", "std", "min", "max"])
    out[("chance", "top1")] = 1 / n_materials
    out[("chance", "top3")] = min(3 / n_materials, 1.0)
    return out


def summarise_open(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, min and max across folds, per strategy and dimension."""
    cols = ["std_mae", "std_mae_mean_baseline", "std_mae_random_baseline", "mae"]
    return folds.groupby(["strategy", "dimension"], sort=False)[cols].agg(
        ["mean", "std", "min", "max"])


def paired_test(a, b) -> dict:
    """Wilcoxon signed-rank on fold-paired errors, and the mean difference a - b."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples differ in length")
    d = a - b
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d, zero_method="wilcox").pvalue)
    return {"n": int(d.size), "mean_difference": float(d.mean()), "p": p}


def split_statement(views: Views, n_folds: int) -> str:
    per_fold = len(view_folds(views.n_views, n_folds)[0])
    return (f"Closed-set: {views.n} materials, {views.n_views} views each, {n_folds} "
            f"view folds; about {per_fold} views query and the rest build the "
            f"database. Open-set: leave-one-material-out, {views.n} folds, with "
            f"{views.n - 1} materials in the database and {views.n_views} query "
            f"views per fold. Provenance: {views.provenance}.")
