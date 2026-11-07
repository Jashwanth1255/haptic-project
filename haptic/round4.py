"""Round 4 helpers. Plan fixed in PREREGISTRATION.md ("Round 4", commit a86cb06).

I  rig whitening: subtract each feature's cross-material median at the same
   (velocity, force, direction).
K  test-retest reliability: ICC(3,1) over cells, and Spearman-Brown corrected
   split-half reliability of per-material means.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONDITION = ["velocity_mm_s", "force_n", "direction_deg"]


def whiten(df: pd.DataFrame, columns, by=CONDITION) -> pd.DataFrame:
    """Each column minus its median over all materials at the same condition."""
    out = df.copy()
    med = df.groupby(by)[list(columns)].transform("median")
    out[list(columns)] = df[list(columns)].to_numpy() - med.to_numpy()
    return out


def icc3_1(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(3,1), two-way mixed, consistency, single measure, for two raters
    (repeat 0, repeat 1) over n targets (cells)."""
    x = np.column_stack([a, b]).astype(float)
    x = x[np.isfinite(x).all(1)]
    n, k = x.shape
    grand = x.mean()
    ss_rows = k * ((x.mean(1) - grand) ** 2).sum()
    ss_cols = n * ((x.mean(0) - grand) ** 2).sum()
    ss_err = ((x - grand) ** 2).sum() - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    return float((ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err))


def spearman_brown(r: float) -> float:
    return float(2 * r / (1 + r))


def paired_repeats(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """One row per (material, velocity, force, direction) with both repeats."""
    d = df.assign(rep=df["repeat"] % 2)
    w = d.pivot_table(index=["material_id", *CONDITION], columns="rep", values=column,
                      aggfunc="first")
    return w.dropna()


def reliability(df: pd.DataFrame, column: str) -> dict:
    w = paired_repeats(df, column)
    icc = icc3_1(w[0].to_numpy(), w[1].to_numpy())
    means = w.groupby(level="material_id").mean()
    r = float(np.corrcoef(means[0], means[1])[0, 1])
    return {"icc3_1": icc, "n_cells": len(w), "split_half_r": r,
            "split_half_sb": spearman_brown(r), "n_materials": len(means)}
