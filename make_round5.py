"""Round 5, analyses N (whitening and the O1 gate) and O (velocity invariance
after whitening). Plan: PREREGISTRATION.md, "Round 5" (commit acc5a79).

    python make_round5.py --features data/round2_features/features.csv

Saves runs `gate_whitened` and `velocity_invariance`.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler

import runlog
from haptic import benchmark as B
from haptic import round4 as R4
from haptic import separability as S

CLOTH = range(102, 118)
BANDS = list(B.FEATURE_SETS["bands"])
DESC = ["log_frequency", "log_intensity", "pattern", "log_duration"]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["log_frequency"] = np.log(d.frequency_hz)
    d["log_intensity"] = np.log(d.intensity)
    d["log_duration"] = np.log(d.duration_ms)
    return d


def gate(df: pd.DataFrame, dims) -> dict:
    r = S.run_gate(df, tuple(dims))
    return {"silhouette": float(r.silhouette), "collapsed_pairs": len(r.collapsed_pairs),
            "effective_classes": int(r.effective_classes), "passed": bool(r.passed)}


def sample_silhouettes(df: pd.DataFrame, dims) -> np.ndarray:
    x = StandardScaler().fit_transform(df[list(dims)].to_numpy(float))
    return silhouette_samples(x, df.material_id.to_numpy())


def sign_flip_p(diff: np.ndarray, n_perm: int = 10_000, seed: int = 0) -> float:
    """One-sided paired permutation test of mean(diff) > 0 by random sign flips."""
    rng = np.random.default_rng(seed)
    obs = diff.mean()
    flips = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    null = (flips * diff).mean(1)
    return float((1 + np.sum(null >= obs)) / (n_perm + 1))


def analysis_n(df: pd.DataFrame, path: str) -> None:
    white = R4.whiten(df, [*BANDS, *DESC])
    raw_c = df[df.material_id.isin(CLOTH)].reset_index(drop=True)
    white_c = white[white.material_id.isin(CLOTH)].reset_index(drop=True)
    out = {}
    for name, dims in (("descriptors4", DESC), ("bands17", BANDS)):
        out[name] = {"raw": gate(raw_c, dims), "whitened": gate(white_c, dims)}
        print(f"[N] {name:12s} raw sil {out[name]['raw']['silhouette']:+.3f} "
              f"eff {out[name]['raw']['effective_classes']}  | whitened sil "
              f"{out[name]['whitened']['silhouette']:+.3f} eff {out[name]['whitened']['effective_classes']} "
              f"collapsed {out[name]['whitened']['collapsed_pairs']}/120")
    diff = sample_silhouettes(white_c, BANDS) - sample_silhouettes(raw_c, BANDS)
    p = sign_flip_p(diff)
    metrics = {"preregistration": "PREREGISTRATION.md: Round 5, N (commit acc5a79)",
               "n_recordings": len(raw_c), "gates": out,
               "test": {"statistic": "mean per-recording silhouette, whitened - raw, bands17",
                        "mean_diff": float(diff.mean()), "p": p, "predicted": "whitened > raw",
                        "prediction_held": bool(p < 0.05)},
               "thresholds": {"silhouette": S.SILHOUETTE_MIN, "effective_classes": S.MIN_SEPARABLE}}
    print(f"[N] bands17 mean silhouette gain {diff.mean():+.3f}, p {p:.4f}")
    run = runlog.save_run("gate_whitened", {"features": path}, metrics, synthetic=False)
    print("saved", run)


def velocity_sd(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Per material and feature: SD across velocities of the per-velocity mean."""
    m = df.groupby(["material_id", "velocity_mm_s"])[list(cols)].mean()
    return m.groupby(level="material_id").std(ddof=1)


def analysis_o(df: pd.DataFrame, path: str) -> None:
    white = R4.whiten(df, BANDS)
    raw_sd = velocity_sd(df, BANDS).median(1)
    white_sd = velocity_sd(white, BANDS).median(1)
    p = float(wilcoxon(raw_sd - white_sd, alternative="greater").pvalue)
    cat = df.groupby("material_id").category.first()
    by_cat = pd.DataFrame({"raw": raw_sd, "whitened": white_sd, "category": cat}).groupby("category").median()
    metrics = {"preregistration": "PREREGISTRATION.md: Round 5, O (commit acc5a79)",
               "n_materials": int(raw_sd.size),
               "median_velocity_sd_raw": float(raw_sd.median()),
               "median_velocity_sd_whitened": float(white_sd.median()),
               "n_lower_after": int((white_sd < raw_sd).sum()), "p": p,
               "predicted": "whitened < raw", "prediction_held": bool(p < 0.05),
               "by_category": by_cat.round(4).to_dict(orient="index")}
    print(f"[O] velocity SD (median over bands, then materials): raw {raw_sd.median():.3f} "
          f"whitened {white_sd.median():.3f}  lower in {metrics['n_lower_after']}/118  p {p:.3g}")
    print(by_cat.round(3).to_string())
    run = runlog.save_run("velocity_invariance", {"features": path}, metrics,
                          {"per_material": pd.DataFrame({"raw": raw_sd, "whitened": white_sd,
                                                         "category": cat}).reset_index()},
                          synthetic=False)
    print("saved", run)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_round5")
    ap.add_argument("--features", default="data/round2_features/features.csv")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.features)
    df = _prepare(df[df.in_band.astype(bool)].reset_index(drop=True))
    analysis_n(df, a.features)
    analysis_o(df, a.features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
