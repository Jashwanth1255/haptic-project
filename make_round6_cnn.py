"""Round 6, analysis P: the round-2 CNN on whitened acceleration spectrograms.
Plan: PREREGISTRATION.md, "Round 6" (commit 869b910).

    python make_round6_cnn.py --features <round-2 features dir> --gate-keys <csv>
                              --raw-cloth-folds <csv> --raw-118-folds <csv>

Saves run `whitened_cnn` after each part, cumulatively; fold x seed results are
checkpointed and resume.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import make_round2_cnn as R2
import runlog
from haptic import learned as Lr
from haptic import loader as L
from haptic import round6 as R6
from haptic.separability import holm_adjust

SEEDS = R2.SEEDS
RAW = {"P1": "audio_vs_accel/20261107T110948_431117Z accel_rerun",
       "P2": "benchmark118_cnn/20261107T125519_960150Z material_id|direction"}


def fold_means(res: pd.DataFrame) -> pd.Series:
    return res.groupby(res.fold.astype(str)).accuracy.mean()


def compare(white: pd.DataFrame, raw: pd.DataFrame) -> dict:
    w, r = fold_means(white), fold_means(raw)
    folds = sorted(set(w.index) & set(r.index), key=float)
    if len(folds) != 8:
        raise SystemExit(f"expected 8 matching folds, got {folds}")
    diff = w.loc[folds].to_numpy() - r.loc[folds].to_numpy()
    return {"raw_accuracy": float(r.loc[folds].mean()), "whitened_fold_mean": float(w.loc[folds].mean()),
            "fold_differences": dict(zip(folds, map(float, diff))),
            "folds_improved": int((diff > 0).sum()),
            "p": float(wilcoxon(diff, alternative="greater").pvalue)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_round6_cnn")
    ap.add_argument("--features", required=True)
    ap.add_argument("--gate-keys", required=True)
    ap.add_argument("--raw-cloth-folds", required=True)
    ap.add_argument("--raw-118-folds", required=True)
    ap.add_argument("--checkpoints", default="derived/round6")
    ap.add_argument("--epochs", type=int, default=Lr.Config.epochs, help="smoke tests only")
    ap.add_argument("--limit-materials", type=int, default=None, help="smoke tests only")
    a = ap.parse_args(argv)
    cfg = Lr.Config(epochs=a.epochs)
    ck = Path(a.checkpoints)

    specs, meta = R2.load_specs(Path(a.features) / "accel_specs.npz")
    medians = R6.condition_medians(specs, meta)            # all 118 materials
    white = R6.whiten_specs(specs, meta, medians)
    if a.limit_materials:
        keep = np.flatnonzero(meta.material_id.isin(sorted(meta.material_id.unique())[:a.limit_materials]
                                                     + list(range(102, 118))))
        white, meta = [white[i] for i in keep], meta.iloc[keep].reset_index(drop=True)
    nf = Lr.frames_per_window(L.ACCEL_FS_HZ)
    feats = pd.read_csv(Path(a.features) / "features.csv")
    cats = sorted(feats.category.unique())
    meta["category"] = [cats.index(L.category_of(int(m))) for m in meta.material_id]

    gate = pd.read_csv(a.gate_keys)
    gate = gate[gate["in_band"].astype(bool)]
    raw_cloth = pd.read_csv(a.raw_cloth_folds)
    raw_cloth = raw_cloth[raw_cloth.input == "accel_rerun"]
    raw_118 = pd.read_csv(a.raw_118_folds)

    metrics = {"preregistration": "PREREGISTRATION.md: Round 6, P (commit 869b910)",
               "raw_reference": RAW, "results": {}, "tests": {}}
    tables = {}

    def save():
        return runlog.save_run("whitened_cnn", {**vars(a), "seeds": SEEDS, "config": vars(cfg)},
                               metrics, tables, synthetic=False)

    parts = [("P1", "material_id", "cloth"), ("P2", "material_id", "all118"),
             ("category", "category", "all118")]
    results = {}
    for name, label, subset in parts:
        if subset == "cloth":
            s, m = R2.restrict(white, meta, gate)
        else:
            s, m = white, meta
        print(f"\n[P] {name}: {label}, {len(m)} recordings", flush=True)
        r = Lr.evaluate(s, m, "direction", nf, seeds=SEEDS, cfg=cfg, label=label,
                        checkpoint=ck / f"P_{name}.jsonl")
        results[name] = r
        summ = R2.summarise(r, top="top5")
        counts = m[label].value_counts()
        summ.update(n=int(len(m)), chance=1 / m[label].nunique(),
                    majority_baseline=float(counts.max() / counts.sum()))
        metrics["results"][name] = summ
        tables[f"folds_{name}"] = R2._drop(r)
        tables[f"recall_{name}"] = R2.recall_table(r)
        print(f"[P] {name}: accuracy {summ['accuracy_mean']:.3f}, macro-F1 {summ['macro_f1_mean']:.3f}",
              flush=True)
        if name in ("P1", "P2"):
            metrics["tests"][name] = compare(r, raw_cloth if name == "P1" else raw_118)
            t = metrics["tests"][name]
            print(f"[P] {name}: raw {t['raw_accuracy']:.3f} -> whitened {t['whitened_fold_mean']:.3f}, "
                  f"improved {t['folds_improved']}/8, p {t['p']:.4f}", flush=True)
        if len(metrics["tests"]) == 2:
            names = list(metrics["tests"])
            for n, h in zip(names, holm_adjust(np.array([metrics["tests"][n]["p"] for n in names]))):
                metrics["tests"][n].update(holm_p=float(h), predicted="whitened > raw",
                                           prediction_held=bool(h < 0.05))
                print(f"[P] {n}: Holm p {h:.4f}", flush=True)
        print("saved", save(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
