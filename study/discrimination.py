"""Stage 3: can people tell the rendered vibrations apart?

    python -m study.discrimination exports/*.csv --name pilot [--fabric-ratings r.csv]

Trials come from the instrument's "discrimination" block: two vibrations of one
scheme, either the same fabric twice or two different fabrics, answered "same" or
"different".

**d' uses the differencing model for same-different designs** (Macmillan and
Creelman, Detection Theory, ch. 9). Saying "different" on a different pair is a
hit, and on a same pair a false alarm. Under that model

    F = 2 Phi(-k / sqrt 2)
    H = Phi((d' - k) / sqrt 2) + Phi((-d' - k) / sqrt 2)

so k comes from F and d' is solved from H. The yes/no formula z(H) - z(F) is
reported beside it as `dprime_yes_no`, but it is not the right model for this
design and understates sensitivity. Rates of 0 or 1 use the log-linear correction,
(count + 0.5) / (n + 1).

Scheme D shuffles B's pulses and carries no texture: it is the control. A scheme
is only evidence of rendering if participants discriminate it better than D,
tested per participant with Wilcoxon signed-rank and Holm-corrected.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import friedmanchisquare, norm, spearmanr, wilcoxon

from haptic.separability import holm_adjust

from .reliability import read_sessions

CONTROL = "D"
ALPHA = 0.05
DPRIME_MAX = 8.0


def rate(count, n) -> float:
    """Log-linear corrected proportion, never exactly 0 or 1."""
    if n <= 0:
        return float("nan")
    return (count + 0.5) / (n + 1)


def dprime_same_different(h: float, f: float) -> float:
    """Differencing-model d'. Zero when hits do not exceed false alarms."""
    if not (0 < f < 1 and 0 < h < 1):
        return float("nan")
    k = -np.sqrt(2) * norm.ppf(f / 2)
    hit = lambda d: norm.cdf((d - k) / np.sqrt(2)) + norm.cdf((-d - k) / np.sqrt(2))
    if h <= hit(0.0) + 1e-12:      # hit(0) equals f up to rounding
        return 0.0
    if h >= hit(DPRIME_MAX):
        return DPRIME_MAX
    return float(brentq(lambda d: hit(d) - h, 0.0, DPRIME_MAX))


def dprime_yes_no(h: float, f: float) -> float:
    return float(norm.ppf(h) - norm.ppf(f))


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("same", "response_different"):
        df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False})
        if df[c].isna().any():
            raise ValueError(f"column {c} must be true or false")
    return df


def _score(g: pd.DataFrame, f_trials: pd.DataFrame) -> dict:
    diff = g[~g["same"]]
    h = rate(diff["response_different"].sum(), len(diff))
    f = rate(f_trials["response_different"].sum(), len(f_trials))
    return {"n_different": len(diff), "n_same": len(f_trials), "hit_rate": h,
            "false_alarm_rate": f, "dprime": dprime_same_different(h, f),
            "dprime_yes_no": dprime_yes_no(h, f)}


def per_participant(df: pd.DataFrame) -> pd.DataFrame:
    """d' per participant and scheme, over all of that scheme's fabric pairs."""
    rows = []
    for (p, s), g in df.groupby(["participant", "scheme"]):
        rows.append({"participant": p, "device_model": g["device_model"].iloc[0],
                     "scheme": s, **_score(g, g[g["same"]])})
    return pd.DataFrame(rows)


def per_pair(df: pd.DataFrame) -> pd.DataFrame:
    """d' per fabric pair and scheme, pooled over participants. Each pair's false
    alarms are that scheme's same trials, since a same trial has no pair."""
    rows = []
    for s, g in df.groupby("scheme"):
        same = g[g["same"]]
        for (a, b), pg in g[~g["same"]].groupby(["fabric_a", "fabric_b"]):
            rows.append({"scheme": s, "fabric_a": a, "fabric_b": b,
                         "n_participants": pg["participant"].nunique(),
                         **_score(pg, same)})
    return pd.DataFrame(rows)


def compare_schemes(pp: pd.DataFrame, control: str = CONTROL) -> tuple[pd.DataFrame, dict]:
    """Each scheme against the control, paired by participant, Holm-corrected;
    plus a Friedman test across all schemes."""
    wide = pp.pivot(index="participant", columns="scheme", values="dprime").dropna()
    if control not in wide.columns:
        raise ValueError(f"no trials for control scheme {control}")
    rows = []
    for s in [c for c in wide.columns if c != control]:
        d = wide[s] - wide[control]
        p = wilcoxon(wide[s], wide[control]).pvalue if (d != 0).sum() >= 1 else 1.0
        rows.append({"scheme": s, "n_participants": len(wide),
                     "mean_dprime": wide[s].mean(), "mean_dprime_control": wide[control].mean(),
                     "mean_difference": d.mean(), "p": float(p)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p"].to_numpy())
    out["verdict"] = np.where(out["p_holm"] >= ALPHA, "no clear difference from control",
                              np.where(out["mean_difference"] > 0, "better than control",
                                       "worse than control"))
    fried = (friedmanchisquare(*[wide[c] for c in wide.columns])
             if wide.shape[1] >= 3 and len(wide) >= 2 else None)
    return out, {"friedman_statistic": float(fried.statistic) if fried else None,
                 "friedman_p": float(fried.pvalue) if fried else None,
                 "n_participants": len(wide)}


def by_device(pp: pd.DataFrame) -> pd.DataFrame:
    """LRA and ERM motors feel different: report d' per phone model."""
    return (pp.groupby(["device_model", "scheme"])
            .agg(n_participants=("participant", "nunique"), mean_dprime=("dprime", "mean"))
            .reset_index())


def tracks_attributes(pairs: pd.DataFrame, fabric_ratings: pd.DataFrame,
                      control: str = CONTROL) -> pd.DataFrame:
    """Does a pair's d' grow with how differently its two fabrics were rated?
    Spearman across pairs, per scheme and attribute. `fabric_ratings` has one row
    per fabric (index) and one column per attribute (mean rating)."""
    rows = []
    for s, g in pairs[pairs["scheme"] != control].groupby("scheme"):
        g = g[g["fabric_a"].isin(fabric_ratings.index) & g["fabric_b"].isin(fabric_ratings.index)]
        for attr in fabric_ratings.columns:
            gap = (fabric_ratings.loc[g["fabric_a"], attr].to_numpy()
                   - fabric_ratings.loc[g["fabric_b"], attr].to_numpy())
            ok = len(g) >= 3 and np.ptp(gap) > 0 and g["dprime"].nunique() > 1
            r = spearmanr(np.abs(gap), g["dprime"]) if ok else None
            rows.append({"scheme": s, "attribute": attr, "n_pairs": len(g),
                         "spearman": float(r.statistic) if r else np.nan,
                         "p": float(r.pvalue) if r else np.nan})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="study.discrimination")
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--name", default="pilot", help="run name suffix")
    ap.add_argument("--fabric-ratings",
                    help="CSV: fabric, then one mean-rating column per attribute")
    a = ap.parse_args(argv)

    import runlog

    df = _clean(read_sessions(a.csv, block="discrimination"))
    pp, pairs = per_participant(df), per_pair(df)
    schemes, overall = compare_schemes(pp)
    devices = by_device(pp)
    print(f"{df['participant'].nunique()} participants, {len(df)} trials\n")
    print("d' per scheme against the control (differencing model):")
    print(schemes.round(3).to_string(index=False))
    print(f"\nFriedman across schemes: p = {overall['friedman_p']}")
    print("\nby phone model:")
    print(devices.round(3).to_string(index=False))
    artifacts = {"per_participant": pp, "per_pair": pairs, "schemes": schemes,
                 "devices": devices}
    if a.fabric_ratings:
        ratings = pd.read_csv(a.fabric_ratings, index_col=0)
        track = tracks_attributes(pairs, ratings)
        print("\nd' against rating differences:")
        print(track.round(3).to_string(index=False))
        artifacts["attribute_tracking"] = track
    saved = runlog.save_run(
        f"study_discrimination_{a.name}",
        {"csv": a.csv, "control": CONTROL, "fabric_ratings": a.fabric_ratings,
         "model": "same-different differencing", "correction": "log-linear"},
        {**overall, "n_trials": len(df),
         "mean_dprime": pp.groupby("scheme")["dprime"].mean().to_dict(),
         "verdicts": dict(zip(schemes["scheme"], schemes["verdict"]))},
        artifacts, synthetic=False)
    print(f"\nsaved run {saved}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
