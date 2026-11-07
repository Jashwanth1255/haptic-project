"""Round 6, analysis P: whitening spectrograms before the CNN.
Plan fixed in PREREGISTRATION.md ("Round 6", commit 869b910).

For each (velocity, force, direction), take the per-bin median over all
recordings of each recording's time-mean log spectrum, and subtract it from every
frame of every recording at that condition. No labels are used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONDITION = ["velocity_mm_s", "force_n", "direction_deg"]


def condition_medians(specs, meta: pd.DataFrame, by=CONDITION) -> dict:
    means = np.stack([np.asarray(s, np.float64).mean(0) for s in specs])
    out = {}
    for key, idx in meta.reset_index(drop=True).groupby(by).indices.items():
        out[key] = np.median(means[idx], axis=0)
    return out


def whiten_specs(specs, meta: pd.DataFrame, medians: dict | None = None, by=CONDITION):
    """Spectrograms minus their condition's median spectrum. `medians` (from
    `condition_medians`) may come from a larger set than `specs`."""
    if medians is None:
        medians = condition_medians(specs, meta, by)
    meta = meta.reset_index(drop=True)
    keys = list(meta[by].itertuples(index=False, name=None))
    return [(np.asarray(s, np.float64) - medians[k]).astype(np.float32)
            for s, k in zip(specs, keys)]
