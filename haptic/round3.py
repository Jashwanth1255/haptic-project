"""Round 3 analyses E, F, H. Plan fixed in PREREGISTRATION.md ("Round 3"),
commit d3b3bf9, before any of these quantities was computed.

E  direction structure: are opposite strokes (θ, θ+180°) closer than
   perpendicular ones (θ, θ+90°)?
F  what survives the on/off channel: fabric identity from the intended vs
   the rendered (scheme B, actuator lag) amplitude envelope.
H  force transfer: train at one force, test at the same or the other force.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import benchmark as B
from . import identity as I
from . import render as R

FEATURES = (*I.BAND_COLUMNS, "log_rms")
CELL = ["material_id", "velocity_mm_s", "force_n", "repeat"]
OPPOSITE = [(d, d + 180) for d in (0, 45, 90, 135)]
PERPENDICULAR = [(d, (d + 90) % 360) for d in range(0, 360, 45)]
DIAGONAL = [(d, (d + s) % 360) for d in range(0, 360, 45) for s in (45, 135)]


def standardise(df: pd.DataFrame, columns=FEATURES) -> np.ndarray:
    x = df[list(columns)].to_numpy(float)
    sd = x.std(0)
    sd[sd == 0] = 1.0
    return (x - x.mean(0)) / sd


# --------------------------------------------------------------------------- E

def _pair_mean(vecs: dict, pairs) -> float:
    d = [np.linalg.norm(vecs[a] - vecs[b]) for a, b in pairs if a in vecs and b in vecs]
    return float(np.mean(d)) if d else np.nan


def direction_distances(df: pd.DataFrame, columns=FEATURES) -> pd.DataFrame:
    """Per material: mean distance between opposite, perpendicular and diagonal
    direction pairs within a (material, velocity, force, repeat) cell, and the
    repeat-to-repeat distance at the same direction as a noise floor."""
    # the dataset numbers repeats 0-15 across directions; the repeat within a
    # direction is the parity, as in benchmark.splits
    df = df.reset_index(drop=True).assign(repeat=lambda d: d["repeat"] % 2)
    x = standardise(df, columns)
    rows = []
    for key, idx in df.groupby(CELL).indices.items():
        vecs = {int(df.direction_deg[i]): x[i] for i in idx}
        rows.append({"material_id": key[0], "opp": _pair_mean(vecs, OPPOSITE),
                     "perp": _pair_mean(vecs, PERPENDICULAR),
                     "diag": _pair_mean(vecs, DIAGONAL)})
    rep = []
    for key, idx in df.groupby(["material_id", "velocity_mm_s", "force_n",
                                "direction_deg"]).indices.items():
        if len(idx) >= 2:
            rep.append({"material_id": key[0], "repeat": float(np.linalg.norm(x[idx[0]] - x[idx[1]]))})
    out = pd.DataFrame(rows).groupby("material_id").mean()
    out["repeat"] = pd.DataFrame(rep).groupby("material_id")["repeat"].mean()
    out["perp_minus_opp"] = out["perp"] - out["opp"]
    return out


# --------------------------------------------------------------------------- F

ENV_MS = 500
TICK_MS = 10
MOD_BANDS = np.geomspace(1.0, 50.0, 13)      # 12 bands


def envelope_features(env: np.ndarray, fs: float = R.ENV_FS_HZ) -> np.ndarray:
    """Log modulation power in 12 log bands, 1-50 Hz, then log mean level."""
    env = np.asarray(env, float)
    x = (env - env.mean()) * np.hanning(env.size)
    f = np.fft.rfftfreq(env.size, 1 / fs)
    p = np.abs(np.fft.rfft(x)) ** 2
    bands = [p[(f >= lo) & (f < hi)].sum() for lo, hi in zip(MOD_BANDS[:-1], MOD_BANDS[1:])]
    scale = max(env.mean(), 1e-12) ** 2 * env.size
    return np.log(np.asarray(bands) / scale + 1e-10).tolist() + [np.log(max(env.mean(), 1e-12))]


def render_pair(x: np.ndarray, fs: float, intensity: float, intensity_ref: float):
    """Intended envelope (first ENV_MS) and the achievable one under scheme B."""
    env = R.signal_envelope(x, fs)[: ENV_MS]
    if env.size < ENV_MS:
        raise ValueError(f"steady window too short for {ENV_MS} ms")
    cfg = R.RenderConfig(intensity_ref=intensity_ref, duration_ms=ENV_MS, tick_ms=TICK_MS)
    d = _Level(intensity)
    pattern = R.encode_b(d, cfg, envelope=env)
    # the intended envelope carries the same level the pattern is aiming for
    level = min(intensity / intensity_ref, 1.0)
    intended = env / max(env.mean(), 1e-12) * level
    return intended, R.achievable_envelope(pattern)[: ENV_MS]


class _Level:
    """The one field `encode_b` reads through `render._level`."""
    def __init__(self, intensity: float):
        self.intensity = float(intensity)


def fold_accuracies(df: pd.DataFrame, columns, label: str = "material_id") -> pd.Series:
    r = B.linear(df, columns, label, "direction")
    return pd.Series({f["fold"]: f["accuracy"] for f in r["folds"]}), r


# --------------------------------------------------------------------------- H

def force_transfer(df: pd.DataFrame, columns, label: str = "material_id",
                   max_iter: int = 3000) -> dict:
    """Train on force A repeat 0; test on repeat 1 at A (same) and at B (cross);
    both ways round, predictions pooled. Returns per-class recall and accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x = df[list(columns)].to_numpy(float)
    y = df[label].to_numpy()
    force = df.force_n.to_numpy()
    rep = df["repeat"].to_numpy() % 2
    forces = np.unique(force)
    if len(forces) != 2:
        raise ValueError(f"need exactly two forces, got {forces}")
    out = {"same": [], "cross": []}
    for a in forces:
        b = forces[forces != a][0]
        train = (force == a) & (rep == 0)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter))
        model.fit(x[train], y[train])
        for kind, f in (("same", a), ("cross", b)):
            test = (force == f) & (rep == 1)
            out[kind].append(pd.DataFrame({"y": y[test], "hit": model.predict(x[test]) == y[test]}))
    res = {}
    for kind in ("same", "cross"):
        t = pd.concat(out[kind])
        res[kind] = {"accuracy": float(t.hit.mean()), "n": len(t),
                     "recall": t.groupby("y").hit.mean()}
    return res
