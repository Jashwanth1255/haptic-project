"""Round 2, GPU stage: CNN experiments B, C and D, in that order.

    python make_round2_cnn.py --features <dir from make_round2_features.py>
                              --gate-keys <csv> --baseline-folds <csv>
                              --k-confirmed <float> [--only B C D]

Plans are in PREREGISTRATION.md, "Benchmark and findings round 2". Each analysis
saves its own run as soon as it finishes, so a timeout loses at most the one
that was running. Fold x seed results are checkpointed and resume.

B  audio_vs_accel     cloth, direction held out: audio 20-8000 Hz, audio 20-1000 Hz,
                      and acceleration rerun with this code. Confirmatory test:
                      audio_full > acceleration run o1_learned_baseline (Wilcoxon
                      one-sided over the 8 folds, seed-averaged).
C  velocity_warp      cloth, velocity held out, log-f spectrogram warped by
                      k in {0, k_confirmed, 1}. Confirmatory: k_confirmed beats 0
                      and beats 1 (exact sign tests, one-sided, Holm).
D  benchmark118_cnn   all 118 materials; material and category; direction held out
                      and repeat 0 -> 1. Descriptive.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import runlog
from haptic import learned as Lr
from haptic import loader as L

KEYS = ["material_id", "velocity_mm_s", "force_n", "direction_deg", "repeat"]
SEEDS = (0, 1, 2)
CFG = Lr.Config()
FS_ACCEL = L.ACCEL_FS_HZ


def load_specs(path: Path):
    z = np.load(path, allow_pickle=False)
    meta = pd.DataFrame({k: z[k] for k in KEYS})
    off = z["offsets"]
    spec = z["spec"]
    return [spec[off[i]:off[i + 1]].astype(np.float32) for i in range(len(meta))], meta


def restrict(specs, meta, keys: pd.DataFrame):
    want = set(map(tuple, keys[KEYS].itertuples(index=False)))
    have = [tuple(r) for r in meta[KEYS].itertuples(index=False)]
    idx = [i for i, k in enumerate(have) if k in want]
    if len(idx) != len(want):
        raise SystemExit(f"{len(idx)} of {len(want)} wanted recordings present")
    return [specs[i] for i in idx], meta.iloc[idx].reset_index(drop=True)


def summarise(res: pd.DataFrame, top: str | None = None) -> dict:
    per_seed = res.groupby("seed").apply(lambda g: np.average(g.accuracy, weights=g.n))
    out = {"accuracy_mean": float(per_seed.mean()),
           "accuracy_sd_over_seeds": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else None,
           "fold_min": float(res.accuracy.min()), "fold_max": float(res.accuracy.max())}
    if res.fold.nunique() > 1:
        out["accuracy_sd_over_folds"] = float(res.groupby("fold").accuracy.mean().std(ddof=1))
    f1 = [Lr.macro_f1(np.concatenate(g.truth.to_list()), np.concatenate(g.predictions.to_list()))
          for _, g in res.groupby("seed")]
    out["macro_f1_mean"] = float(np.mean(f1))
    if top and top in res:
        out[f"{top}_mean"] = float(res.groupby("seed").apply(
            lambda g: np.average(g[top], weights=g.n)).mean())
    return out


def recall_table(res: pd.DataFrame) -> pd.DataFrame:
    long = res.explode(["predictions", "truth"])
    return (long.assign(hit=long.predictions == long.truth)
            .groupby(["split", "truth"]).hit.mean().unstack(0).reset_index()
            .rename(columns={"truth": "class"}))


def majority_correct(res: pd.DataFrame) -> pd.Series:
    """Per recording (row index), whether most seeds were correct."""
    long = res.explode(["rows", "predictions", "truth"])
    long["hit"] = long.predictions == long.truth
    return long.groupby("rows").hit.mean() > 0.5


def _drop(res):
    return res.drop(columns=["predictions", "truth", "rows"], errors="ignore")


# --------------------------------------------------------------------------- B

def analysis_b(a, gate_keys, ck: Path):
    from scipy.stats import wilcoxon
    inputs = {}
    for name in ("audio_full", "audio_band"):
        s, m = load_specs(Path(a.features) / f"cloth_{name}.npz")
        inputs[name] = (*restrict(s, m, gate_keys),
                        Lr.frames_per_window(L.AUDIO_FS_HZ, Lr.AUDIO_N_FFT, Lr.AUDIO_HOP))
    s, m = load_specs(Path(a.features) / "accel_specs.npz")
    inputs["accel_rerun"] = (*restrict(s, m, gate_keys), Lr.frames_per_window(FS_ACCEL))
    results, metrics = {}, {}
    for name, (specs, meta, nf) in inputs.items():
        print(f"\n[B] {name}: {len(meta)} recordings, {specs[0].shape[1]} bands, {nf}-frame windows",
              flush=True)
        r = Lr.evaluate(specs, meta, "direction", nf, seeds=SEEDS, cfg=CFG, checkpoint=ck / f"B_{name}.jsonl")
        results[name] = r
        metrics[name] = summarise(r)
        print(f"[B] {name}: accuracy {metrics[name]['accuracy_mean']:.3f}", flush=True)
    base = pd.read_csv(a.baseline_folds)
    base = base[base.split == "direction"].groupby(base.fold.astype(str)).accuracy.mean()
    audio = results["audio_full"].groupby(results["audio_full"].fold.astype(str)).accuracy.mean()
    folds = sorted(set(base.index) & set(audio.index), key=float)   # all 8 on real data
    diff = audio.loc[folds].to_numpy() - base.loc[folds].to_numpy()
    w = wilcoxon(diff, alternative="greater")
    test = {"comparison": "audio_full vs o1_learned_baseline/20261106T214141_196418Z",
            "n_folds": len(folds),
            "fold_differences": dict(zip(folds, map(float, diff))),
            "wilcoxon_greater_p": float(w.pvalue), "significant": bool(w.pvalue < 0.05),
            "predicted": True}
    test["prediction_held"] = test["significant"]
    print(f"[B] audio_full - acceleration per fold: {np.round(diff, 3)}  p = {w.pvalue:.4f}", flush=True)
    folds_tab = pd.concat([_drop(r).assign(input=n) for n, r in results.items()])
    path = runlog.save_run(
        "audio_vs_accel", {**vars(a), "seeds": SEEDS, "config": vars(CFG)},
        {"preregistration": "PREREGISTRATION.md: Benchmark and findings round 2, B",
         "n": int(len(inputs["audio_full"][1])), "chance": 1 / 16, "inputs": metrics,
         "confirmatory_test": test},
        {"folds": folds_tab,
         **{f"recall_{n}": recall_table(r) for n, r in results.items()}}, synthetic=False)
    print(f"saved run {path}", flush=True)


# --------------------------------------------------------------------------- C

def analysis_c(a, gate_keys, ck: Path):
    from scipy.stats import binomtest
    s, m = load_specs(Path(a.features) / "cloth_logf.npz")
    specs, meta = restrict(s, m, gate_keys)
    nf = Lr.frames_per_window(FS_ACCEL, Lr.LOGF_N_FFT, Lr.LOGF_HOP)
    ks = {"k0": 0.0, "k_confirmed": a.k_confirmed, "k1": 1.0}
    results, metrics, hits = {}, {}, {}
    for name, k in ks.items():
        warped = [Lr.warp(sp, v, k) for sp, v in zip(specs, meta.velocity_mm_s)]
        print(f"\n[C] {name} (k = {k:.3f}): {len(meta)} recordings, {nf}-frame windows", flush=True)
        r = Lr.evaluate(warped, meta, "velocity", nf, seeds=SEEDS, cfg=CFG, checkpoint=ck / f"C_{name}.jsonl")
        results[name] = r
        metrics[name] = {**summarise(r), "k": k,
                         "per_velocity": r.groupby("fold").accuracy.mean().to_dict()}
        hits[name] = majority_correct(r)
        print(f"[C] {name}: accuracy {metrics[name]['accuracy_mean']:.3f}", flush=True)
    tests = {}
    for other in ("k0", "k1"):
        h1, h0 = hits["k_confirmed"], hits[other].reindex(hits["k_confirmed"].index)
        win, lose = int((h1 & ~h0).sum()), int((~h1 & h0).sum())
        p = binomtest(win, win + lose, 0.5, alternative="greater").pvalue if win + lose else 1.0
        tests[f"k_confirmed_vs_{other}"] = {"wins": win, "losses": lose, "p": float(p)}
    ps = np.array([t["p"] for t in tests.values()])
    order = np.argsort(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        t = list(tests.values())[i]
        t["holm_p"] = float(run)
        t["significant"] = bool(run < 0.05)
        t["predicted"] = True
        t["prediction_held"] = t["significant"]
    for n, t in tests.items():
        print(f"[C] {n}: {t['wins']} wins, {t['losses']} losses, Holm p = {t['holm_p']:.4f}", flush=True)
    path = runlog.save_run(
        "velocity_warp", {**vars(a), "seeds": SEEDS, "config": vars(CFG)},
        {"preregistration": "PREREGISTRATION.md: Benchmark and findings round 2, C",
         "n": int(len(meta)), "chance": 1 / 16, "conditions": metrics, "tests": tests},
        {"folds": pd.concat([_drop(r).assign(condition=n) for n, r in results.items()])},
        synthetic=False)
    print(f"saved run {path}", flush=True)


# --------------------------------------------------------------------------- D

def analysis_d(a, ck: Path):
    specs, meta = load_specs(Path(a.features) / "accel_specs.npz")
    feats = pd.read_csv(Path(a.features) / "features.csv")
    cats = sorted(feats.category.unique())
    meta["category"] = [cats.index(L.category_of(int(m))) for m in meta.material_id]
    nf = Lr.frames_per_window(FS_ACCEL)
    metrics, tables = {"preregistration": "PREREGISTRATION.md: Benchmark and findings round 2, D",
                       "n": int(len(meta)), "n_materials": int(meta.material_id.nunique()),
                       "category_codes": dict(enumerate(cats)), "results": {}}, {}
    for label in ("material_id", "category"):
        for split in ("direction", "repeat"):
            print(f"\n[D] {label} / {split}: {len(meta)} recordings", flush=True)
            r = Lr.evaluate(specs, meta, split, nf, seeds=SEEDS, cfg=CFG, label=label,
                            checkpoint=ck / f"D_{label}_{split}.jsonl")
            if r.empty:
                print(f"[D] {label} / {split}: no test data, skipped", flush=True)
                continue
            n_cls = meta[label].nunique()
            s = summarise(r, top="top5")
            counts = meta[label].value_counts()
            s.update(chance=1 / n_cls, majority_baseline=float(counts.max() / counts.sum()))
            metrics["results"][f"{label}|{split}|cnn"] = s
            tables[f"folds_{label}_{split}"] = _drop(r)
            tables[f"recall_{label}_{split}"] = recall_table(r)
            print(f"[D] {label} / {split}: accuracy {s['accuracy_mean']:.3f}, "
                  f"macro-F1 {s['macro_f1_mean']:.3f}", flush=True)
            path = runlog.save_run("benchmark118_cnn", {**vars(a), "seeds": SEEDS,
                                                        "config": vars(CFG)},
                                   metrics, tables, synthetic=False)
            print(f"saved run {path} (cumulative)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_round2_cnn")
    ap.add_argument("--features", required=True)
    ap.add_argument("--gate-keys", required=True)
    ap.add_argument("--baseline-folds", required=True)
    ap.add_argument("--k-confirmed", type=float, required=True)
    ap.add_argument("--checkpoints", default="derived/round2")
    ap.add_argument("--only", nargs="*", default=["B", "C", "D"])
    ap.add_argument("--epochs", type=int, default=Lr.Config.epochs,
                    help="smoke tests only; results use the default")
    a = ap.parse_args(argv)
    ck = Path(a.checkpoints)
    global CFG
    CFG = Lr.Config(epochs=a.epochs)
    gate = pd.read_csv(a.gate_keys)
    gate_keys = gate[gate["in_band"].astype(bool)]
    if "B" in a.only:
        analysis_b(a, gate_keys, ck)
    if "C" in a.only:
        analysis_c(a, gate_keys, ck)
    if "D" in a.only:
        analysis_d(a, ck)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
