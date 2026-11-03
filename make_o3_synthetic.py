"""Run every Objective 3 protocol on synthetic stand-ins.

    python make_o3_synthetic.py

SYNTHETIC -- SCAFFOLDING ONLY. Nothing printed here may be quoted as a result. The
images and descriptors are invented, and the velocity relations are built into
the generator. It shows the experiments run, and that they can fail.

Every table is saved as a run under results/o3_synthetic/, marked synthetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import runlog
from haptic import association as A
from haptic import descriptor as D
from haptic import synth_o3 as S
from haptic import velocity as Y

N_FOLDS = 4
COUPLINGS = (0.8, 0.0)


def build_views(df, cfg: S.ViewConfig) -> tuple[A.Views, list[str]]:
    """Material descriptors at the reference condition. A material with no in-band
    recording there is dropped and named."""
    at_ref = df[(df["velocity_mm_s"] == D.REF_VELOCITY_MM_S) & (df["force_n"] == 1.0)
                & df["in_band"]]
    table = at_ref.groupby("material_id")[list(A.DIMS)].mean()
    keep = table.index.to_numpy()
    names = [m.name for m in S.FABRICS]
    dropped = [n for i, n in enumerate(names) if i not in keep]
    mats, texs = S.simulated_class_probs(cfg)
    return A.Views(tuple(names[i] for i in keep), table.to_numpy(),
                   S.simulated_embeddings(cfg)[keep], mats[keep], texs[keep],
                   "synthetic"), dropped


def retrieval(df, saved: dict, dropped_by: dict) -> None:
    for coupling in COUPLINGS:
        views, dropped = build_views(df, S.ViewConfig(coupling=coupling))
        tag = f"coupling{coupling}"
        dropped_by[tag] = dropped
        print(f"\n===== coupling {coupling} =====")
        print(A.split_statement(views, N_FOLDS))
        if dropped:
            print(f"Dropped, no in-band recording at the reference condition: {dropped}")
        closed = [A.closed_set(views, s, N_FOLDS) for s in A.STRATEGIES]
        print("\n# closed-set, per fold")
        for c in closed:
            print(c.round(3).to_string(index=False))
        closed_summary = A.summarise_closed(pd.concat(closed), views.n)
        print("\n# closed-set summary")
        print(closed_summary.round(3).T.to_string())
        opened = pd.concat([A.open_set(views, s) for s in A.STRATEGIES])
        print("\n# open-set, standardised MAE across folds")
        s = A.summarise_open(opened)
        print(s[[("std_mae", "mean"), ("std_mae", "std"), ("std_mae", "min"), ("std_mae", "max"),
                 ("std_mae_mean_baseline", "mean"), ("std_mae_random_baseline", "mean")]]
              .round(3).to_string())
        saved |= {f"{tag}_closed_folds": pd.concat(closed),
                  f"{tag}_closed_summary": closed_summary,
                  f"{tag}_open_folds": opened, f"{tag}_open_summary": s}


def velocity(df, label: str, substitute=None) -> pd.DataFrame:
    print(f"\n===== velocity: {label} =====")
    table = Y.compare(Y.run(df, substitute))
    print(table.drop(columns=["p"]).round(3).to_string(index=False))
    print(table.groupby("verdict").size().to_string())
    return table


def flat(t: pd.DataFrame) -> pd.DataFrame:
    """Join multi-level column names so the saved CSV has one header row."""
    if isinstance(t.columns, pd.MultiIndex):
        t = t.copy()
        t.columns = ["_".join(str(c) for c in col if c != "") for col in t.columns]
    return t


def save(saved: dict, dropped_by: dict):
    metrics = {"dropped": dropped_by}
    for k, t in saved.items():
        if k.startswith("velocity_"):
            metrics[f"{k}_verdicts"] = t.groupby("verdict").size().to_dict()
    config = {"n_folds": N_FOLDS, "couplings": COUPLINGS,
              "view_config": S.ViewConfig(), "strategies": list(A.STRATEGIES)}
    return runlog.save_run("o3_synthetic", config, metrics,
                           {k: flat(t) for k, t in saved.items()}, synthetic=True)


def main() -> int:
    print(A.SYNTHETIC_BANNER)
    df = S.recordings()
    saved, dropped_by = {}, {}
    retrieval(df, saved, dropped_by)
    saved["excluded"] = Y.excluded(df)
    print("\nOut-of-band recordings excluded:\n", saved["excluded"].to_string())
    saved["velocity_generator"] = velocity(
        df, "generator obeys the relations (conditioning wins BY CONSTRUCTION)")
    saved["velocity_free"] = velocity(
        S.recordings(velocity_free=True),
        "velocity_free, relations false (conditioning must lose)")
    views, _ = build_views(df, S.ViewConfig())
    ids = {n: i for i, n in enumerate(m.name for m in S.FABRICS)}
    names = list(views.names)
    sub = {ids[names[k]]: ids[names[v]] for k, v in A.substitutes(views, "cosine").items()}
    saved["velocity_substitute"] = velocity(
        df[df["material_id"].isin(sub)], "cosine substitute, material held out", sub)
    print("\n" + A.SYNTHETIC_BANNER)
    print(f"saved run {save(saved, dropped_by)}")
    return 0


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    raise SystemExit(main())
