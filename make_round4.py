"""Round 4, analyses I (rig whitening) and K (test-retest reliability).

    python make_round4.py --features data/round2_features/features.csv

Plan: PREREGISTRATION.md, "Round 4" (commit a86cb06).
Saves runs `rig_whitening` and `reliability`.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import runlog
from haptic import benchmark as B
from haptic import identity as I
from haptic import round4 as R4
from haptic.separability import holm_adjust

CLOTH = range(102, 118)
FEATURES = list(B.FEATURE_SETS["bands"])


def analysis_i(df: pd.DataFrame, path: str) -> None:
    white = R4.whiten(df, FEATURES)
    tests, folds = {}, {}
    for name, ids in (("cloth", CLOTH), ("all118", None)):
        sel = df.material_id.isin(ids) if ids is not None else np.ones(len(df), bool)
        acc = {}
        for kind, table in (("raw", df), ("whitened", white)):
            r = B.linear(table[sel].reset_index(drop=True), FEATURES, "material_id", "direction")
            acc[kind] = r
            folds[f"{name}_{kind}"] = pd.Series({f["fold"]: f["accuracy"] for f in r["folds"]})
        diff = folds[f"{name}_whitened"] - folds[f"{name}_raw"]
        p = float(wilcoxon(diff, alternative="greater").pvalue)
        tests[name] = {"raw_accuracy": acc["raw"]["accuracy"],
                       "whitened_accuracy": acc["whitened"]["accuracy"],
                       "raw_macro_f1": acc["raw"]["macro_f1"],
                       "whitened_macro_f1": acc["whitened"]["macro_f1"],
                       "folds_improved": int((diff > 0).sum()), "n_folds": int(diff.size),
                       "chance": acc["raw"]["chance"], "p": p}
    holm = holm_adjust(np.array([t["p"] for t in tests.values()]))
    for (k, t), h in zip(tests.items(), holm):
        t.update(holm_p=float(h), significant=bool(h < 0.05), predicted="whitened > raw",
                 prediction_held=bool(h < 0.05))
        print(f"[I] {k:6s} raw {t['raw_accuracy']:.3f} whitened {t['whitened_accuracy']:.3f} "
              f"improved {t['folds_improved']}/{t['n_folds']}  p {t['p']:.3g} holm {h:.3g}")
    metrics = {"preregistration": "PREREGISTRATION.md: Round 4, I (commit a86cb06)",
               "features": FEATURES, "n_recordings": len(df), "tests": tests}
    run = runlog.save_run("rig_whitening", {"features": path}, metrics,
                          {"folds": pd.DataFrame(folds).reset_index(names="fold")}, synthetic=False)
    print("saved", run)


def analysis_k(df: pd.DataFrame, path: str) -> None:
    d = df.copy()
    for c in I.RAW4:
        if c != "pattern":
            d[c] = np.log(d[c])
    rows = []
    for c in [*I.RAW4, *FEATURES]:
        rows.append({"feature": c, **R4.reliability(d, c)})
    tab = pd.DataFrame(rows)
    tab["good"] = tab.icc3_1 >= 0.75
    print(tab.round(3).to_string(index=False))
    metrics = {"preregistration": "PREREGISTRATION.md: Round 4, K (commit a86cb06)",
               "threshold_icc": 0.75, "n_good": int(tab.good.sum()), "n_features": len(tab),
               "good_features": tab.loc[tab.good, "feature"].tolist(),
               "icc_median": float(tab.icc3_1.median()),
               "split_half_sb_median": float(tab.split_half_sb.median()),
               "by_feature": tab.set_index("feature").round(4).to_dict(orient="index")}
    run = runlog.save_run("reliability", {"features": path, "log_positive_raw4": True},
                          metrics, {"reliability": tab}, synthetic=False)
    print("saved", run)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_round4")
    ap.add_argument("--features", default="data/round2_features/features.csv")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.features)
    df = df[df.in_band.astype(bool)].reset_index(drop=True)
    analysis_i(df, a.features)
    analysis_k(df, a.features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
