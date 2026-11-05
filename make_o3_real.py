"""Objective 3 on real data: image -> haptic descriptor, and the velocity question.

    python make_o3_real.py

Inputs, never recomputed here:
- descriptors from the latest saved `o1_gate` run (real Cluster cloth);
- per-crop classifier outputs and features from `vision.cluster_scans`
  (data/derived/cluster_crops.npz): 100 crops per material are the views.

Strategies, as built in haptic/association.py:
    class_pair   MINC-2500 class x DTD textile class (fold-1 models)
    cosine       DTD fold-1 ResNet-50 penultimate features

A material's descriptor is the mean of its in-band recordings at the reference
condition (40 mm/s, 1.0 N), as in the synthetic runner. Saves `o3_real`.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import runlog
from haptic import association as A
from haptic import descriptor as D
from haptic import velocity as Y
from haptic.loader import CLOTH_NAMES

N_FOLDS = 4


def flat(t: pd.DataFrame) -> pd.DataFrame:
    if isinstance(t.columns, pd.MultiIndex):
        t = t.copy()
        t.columns = ["_".join(str(c) for c in col if c != "") for col in t.columns]
    return t


def build_views(df: pd.DataFrame, crops) -> tuple[A.Views, list[int], list[int]]:
    at_ref = df[(df["velocity_mm_s"] == D.REF_VELOCITY_MM_S) & (df["force_n"] == 1.0)
                & df["in_band"].astype(bool)]
    table = at_ref.groupby("material_id")[list(A.DIMS)].mean()
    mids = np.array(sorted(set(crops["material_id"])))
    keep = [m for m in mids if m in table.index]
    dropped = [int(m) for m in mids if m not in table.index]
    n_views = int((crops["material_id"] == mids[0]).sum())

    def per_material(a):
        return np.stack([a[crops["material_id"] == m] for m in keep])

    fold1 = list(crops["dtd_folds"]).index(1)
    views = A.Views(tuple(CLOTH_NAMES[m] for m in keep), table.loc[keep].to_numpy(),
                    per_material(crops["dtd_features"]), per_material(crops["minc_probs"]),
                    per_material(crops["dtd_probs"][fold1]), "real")
    assert views.n_views == n_views
    return views, keep, dropped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_o3_real")
    ap.add_argument("--crops", default="data/derived/cluster_crops.npz")
    a = ap.parse_args(argv)
    pd.set_option("display.width", 200)

    gate = runlog.load_run("o1_gate")
    if gate.info["synthetic"]:
        raise SystemExit("latest o1_gate run is synthetic")
    df = gate.table("descriptors")
    df["in_band"] = df["in_band"].astype(bool)
    crops = dict(np.load(a.crops))
    saved = {}

    views, keep, dropped = build_views(df, crops)
    print(A.split_statement(views, N_FOLDS))
    if dropped:
        print(f"dropped, no in-band recording at 40 mm/s and 1.0 N: {dropped}")
    closed = pd.concat([A.closed_set(views, s, N_FOLDS) for s in A.STRATEGIES])
    closed_summary = A.summarise_closed(closed, views.n)
    opened = pd.concat([A.open_set(views, s) for s in A.STRATEGIES])
    open_summary = A.summarise_open(opened)
    print("\n# closed-set retrieval (which material is this crop?)")
    print(flat(closed_summary).round(3).to_string())
    print("\n# open-set: material held out, standardised MAE of the substitute")
    print(open_summary[[("std_mae", "mean"), ("std_mae", "std"),
                        ("std_mae_mean_baseline", "mean"),
                        ("std_mae_random_baseline", "mean")]].round(3).to_string())

    tests = []
    for dim in A.DIMS:
        pick = opened[opened.dimension == dim]
        cos = pick[pick.strategy == "cosine"].sort_values("fold")
        cp = pick[pick.strategy == "class_pair"].sort_values("fold")
        tests.append({"dimension": dim, "comparison": "cosine - class_pair",
                      **A.paired_test(cos.std_mae, cp.std_mae)})
        for s, part in (("cosine", cos), ("class_pair", cp)):
            tests.append({"dimension": dim, "comparison": f"{s} - mean baseline",
                          **A.paired_test(part.std_mae, part.std_mae_mean_baseline)})
    tests = pd.DataFrame(tests)
    from haptic.separability import holm_adjust
    tests["p_holm"] = holm_adjust(tests["p"].to_numpy())
    print("\n# paired Wilcoxon over the held-out materials (negative = first is better)")
    print(tests.round(4).to_string(index=False))

    subs = A.substitutes(views, "cosine")
    sub_ids = {keep[k]: keep[v] for k, v in subs.items()}
    print("\ncosine substitutes:", {CLOTH_NAMES[k]: CLOTH_NAMES[v] for k, v in sub_ids.items()})

    exps = Y.scaling_exponents(df)
    cv = Y.variation_across_velocity(df)
    print("\n# velocity scaling exponents (f = v/lambda predicts frequency 1, duration -1)")
    print(exps.describe().loc[["mean", "50%", "std", "min", "max"]].round(3).to_string())
    print("\n# coefficient of variation across 20-60 mm/s, median over material x force")
    print(cv.drop(columns=["material_id", "force_n"]).median().round(3).to_string())

    verdicts = {}
    for label, kw in (("given", {}), ("unseen_velocity", {"exclude_query_velocity": True}),
                      ("cosine_substitute", {"substitute": sub_ids})):
        with np.errstate(all="ignore"):
            table = Y.compare(Y.run(df, **kw))
        saved[f"velocity_{label}"] = table
        verdicts[label] = table.groupby("verdict").size().to_dict()
        print(f"\n# velocity-conditioned vs agnostic: {label}")
        print(table.round(3).to_string(index=False))

    metrics = {"gate_run": gate.info["run_id"], "n_materials": views.n,
               "n_views": views.n_views, "dropped": dropped,
               "closed": {s: {"top1": float(closed[closed.strategy == s].top1.mean()),
                              "top3": float(closed[closed.strategy == s].top3.mean())}
                          for s in A.STRATEGIES},
               "chance": {"top1": 1 / views.n, "top3": 3 / views.n},
               "velocity_verdicts": verdicts,
               "scaling_exponent_median": exps[list(Y.SCALING_DIMS)].median().to_dict(),
               "cv_median": cv.drop(columns=["material_id", "force_n"]).median().to_dict()}
    path = runlog.save_run(
        "o3_real", {**vars(a), "n_folds": N_FOLDS, "strategies": list(A.STRATEGIES)},
        metrics,
        {"closed_folds": closed, "closed_summary": flat(closed_summary),
         "open_folds": opened, "open_summary": flat(open_summary), "strategy_tests": tests,
         "substitutes": pd.DataFrame({"material_id": list(sub_ids),
                                      "substitute": list(sub_ids.values())}),
         "scaling_exponents": exps, "variation_across_velocity": cv,
         "excluded": Y.excluded(df).reset_index(), **saved},
        synthetic=False)
    print(f"\nsaved run {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
