"""Round 3, analyses E (direction structure) and H (force transfer).

    python make_round3.py --features data/round2_features/features.csv

Plan and predictions: PREREGISTRATION.md, "Round 3", commit d3b3bf9.
Saves runs `direction_structure` and `force_transfer`.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import runlog
from haptic import benchmark as B
from haptic import identity as I
from haptic import round3 as R3
from haptic.separability import holm_adjust

CLOTH = range(102, 118)


def _wilcoxon(x, alternative):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(wilcoxon(x, alternative=alternative).pvalue), len(x)


def analysis_e(df: pd.DataFrame, features_path: str) -> None:
    dist = R3.direction_distances(df)
    cloth = dist.loc[dist.index.isin(CLOTH)]
    other = dist.loc[~dist.index.isin(CLOTH)]
    p1, n1 = _wilcoxon(cloth.perp_minus_opp, "greater")
    p2, n2 = _wilcoxon(other.perp_minus_opp, "two-sided")
    holm = holm_adjust(np.array([p1, p2]))
    summary = lambda d: {k: float(d[k].median()) for k in ("opp", "perp", "diag", "repeat", "perp_minus_opp")}
    metrics = {
        "preregistration": "PREREGISTRATION.md: Round 3, E (commit d3b3bf9)",
        "n_recordings": len(df),
        "cloth": {**summary(cloth), "n_materials": n1,
                  "n_positive": int((cloth.perp_minus_opp > 0).sum()),
                  "p": p1, "holm_p": float(holm[0]), "significant": bool(holm[0] < 0.05),
                  "predicted": "perp > opp", "prediction_held": bool(holm[0] < 0.05)},
        "other": {**summary(other), "n_materials": n2,
                  "n_positive": int((other.perp_minus_opp > 0).sum()),
                  "p": p2, "holm_p": float(holm[1]), "significant": bool(holm[1] < 0.05),
                  "predicted": "none (two-sided)"},
    }
    tab = dist.reset_index()
    tab["category"] = tab.material_id.map(df.groupby("material_id").category.first())
    by_cat = tab.groupby("category")[["opp", "perp", "diag", "repeat", "perp_minus_opp"]].median()
    metrics["by_category_median"] = by_cat.round(4).to_dict(orient="index")
    for k in ("cloth", "other"):
        m = metrics[k]
        print(f"[E] {k:5s} opp {m['opp']:.3f} perp {m['perp']:.3f} diag {m['diag']:.3f} "
              f"repeat {m['repeat']:.3f}  perp>opp {m['n_positive']}/{m['n_materials']}  "
              f"p {m['p']:.2g} holm {m['holm_p']:.2g}")
    run = runlog.save_run("direction_structure", {"features": features_path,
                                                  "columns": list(R3.FEATURES)},
                          metrics, {"distances": tab}, synthetic=False)
    print("saved", run)


def analysis_h(df: pd.DataFrame, features_path: str) -> None:
    cols = list(B.FEATURE_SETS["bands+raw4"])
    level_free = df.copy()
    for c in I.BAND_COLUMNS:
        level_free[c] = level_free[c] - level_free["log_rms"]
    tests, desc, tables = {}, {}, {}
    for name, part in (("cloth", df[df.material_id.isin(CLOTH)]), ("all118", df)):
        r = R3.force_transfer(part, cols)
        rec = pd.DataFrame({"same": r["same"]["recall"], "cross": r["cross"]["recall"]})
        p, n = _wilcoxon(rec["same"] - rec["cross"], "greater")
        tests[name] = {"same_accuracy": r["same"]["accuracy"], "cross_accuracy": r["cross"]["accuracy"],
                       "n_same": r["same"]["n"], "n_cross": r["cross"]["n"],
                       "n_classes": n, "n_cross_lower": int((rec.cross < rec.same).sum()),
                       "chance": 1 / n, "p": p}
        tables[f"recall_{name}"] = rec.reset_index(names="material_id")
        lf = R3.force_transfer(level_free[level_free.material_id.isin(part.material_id.unique())],
                               [*I.BAND_COLUMNS, "pattern", "frequency_hz"])
        desc[name] = {"same_accuracy": lf["same"]["accuracy"], "cross_accuracy": lf["cross"]["accuracy"]}
    holm = holm_adjust(np.array([tests["cloth"]["p"], tests["all118"]["p"]]))
    for (k, t), h in zip(tests.items(), holm):
        t.update(holm_p=float(h), significant=bool(h < 0.05), predicted="cross < same",
                 prediction_held=bool(h < 0.05))
        print(f"[H] {k:6s} same {t['same_accuracy']:.3f} cross {t['cross_accuracy']:.3f} "
              f"lower in {t['n_cross_lower']}/{t['n_classes']}  p {t['p']:.2g} holm {h:.2g}  "
              f"| level removed: same {desc[k]['same_accuracy']:.3f} cross {desc[k]['cross_accuracy']:.3f}")
    metrics = {"preregistration": "PREREGISTRATION.md: Round 3, H (commit d3b3bf9)",
               "columns": cols, "tests": tests, "level_removed_descriptive": desc}
    run = runlog.save_run("force_transfer", {"features": features_path}, metrics, tables, synthetic=False)
    print("saved", run)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_round3")
    ap.add_argument("--features", default="data/round2_features/features.csv")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.features)
    df = df[df.in_band.astype(bool)].reset_index(drop=True)
    print(f"{len(df)} in-band recordings, {df.material_id.nunique()} materials")
    analysis_e(df, a.features)
    analysis_h(df, a.features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
