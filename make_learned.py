"""Learned baseline on real cloth (haptic/learned.py). EXPLORATORY.

    python make_learned.py --root data/cluster

Uses exactly the in-band recordings of the latest saved `o1_gate` run, so the
accuracy compares directly with the descriptor-based identification in
`o1_real_analysis`. Spectrograms are cached under data/derived. Saves
`o1_learned_baseline`. CPU only.
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


def cache(root: Path, wanted: pd.DataFrame, path: Path):
    if path.exists():
        z = np.load(path, allow_pickle=False)
        meta = pd.DataFrame({k: z[k] for k in KEYS})
        offsets = z["offsets"]
        specs = [z["spec"][offsets[i]:offsets[i + 1]] for i in range(len(meta))]
        return specs, meta, float(z["fs"])
    want = set(map(tuple, wanted[KEYS].itertuples(index=False)))
    specs, rows, fs = [], [], None
    for st in L.discover(root):
        try:
            rec = L.load(st)
        except ValueError:
            continue
        key = (rec["material_id"], rec["velocity_mm_s"], rec["force_n"],
               rec["direction_deg"], rec["repeat"])
        if key not in want:
            continue
        specs.append(Lr.spectrogram(rec["accel"], rec["fs"]))
        rows.append(dict(zip(KEYS, key)))
        fs = rec["fs"]
    meta = pd.DataFrame(rows)
    offsets = np.r_[0, np.cumsum([len(s) for s in specs])]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, spec=np.concatenate(specs), offsets=offsets, fs=fs,
                        **{k: meta[k].to_numpy() for k in KEYS})
    return specs, meta, fs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_learned")
    ap.add_argument("--root", default="data/cluster")
    ap.add_argument("--cache", default="data/derived/spectrograms.npz")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=Lr.Config.epochs)
    a = ap.parse_args(argv)

    gate = runlog.load_run("o1_gate")
    d = gate.table("descriptors")
    wanted = d[d["in_band"].astype(bool)]
    specs, meta, fs = cache(Path(a.root), wanted, Path(a.cache))
    if len(meta) != len(wanted):
        raise SystemExit(f"cache holds {len(meta)} recordings, gate run {len(wanted)}")
    n_frames = Lr.frames_per_window(fs)
    cfg = Lr.Config(epochs=a.epochs)
    print(f"{len(meta)} recordings, {specs[0].shape[1]} frequency bins, "
          f"{n_frames}-frame windows", flush=True)

    results = []
    for split in ("direction", "repeat"):
        r = Lr.evaluate(specs, meta, split, n_frames, seeds=a.seeds, cfg=cfg,
                        checkpoint=Path(a.cache).with_name("learned_folds.jsonl"))
        results.append(r)
        per_seed = r.groupby("seed").apply(lambda g: np.average(g.accuracy, weights=g.n))
        print(f"{split}: accuracy {per_seed.mean():.3f} ± {per_seed.std(ddof=1):.3f} over "
              f"{len(per_seed)} seeds (per-fold range {r.accuracy.min():.3f}-"
              f"{r.accuracy.max():.3f})", flush=True)
    res = pd.concat(results)

    long = res.explode(["predictions", "truth"])
    recall = (long.assign(hit=long.predictions == long.truth)
              .groupby(["split", "truth"]).hit.mean().unstack(0))
    recall.index.name = "material_id"
    recall.insert(0, "name", recall.index.map(L.CLOTH_NAMES))
    print(recall.round(2).to_string())

    metrics = {"gate_run": gate.info["run_id"], "n": len(meta), "chance": 1 / 16,
               "exploratory": True}
    for split, g in res.groupby("split"):
        per_seed = g.groupby("seed").apply(lambda x: np.average(x.accuracy, weights=x.n))
        metrics[split] = {"accuracy_mean": float(per_seed.mean()),
                          "accuracy_sd_over_seeds": float(per_seed.std(ddof=1)),
                          "fold_min": float(g.accuracy.min()), "fold_max": float(g.accuracy.max())}
    path = runlog.save_run("o1_learned_baseline", {**vars(a), "config": vars(cfg)}, metrics,
                           {"folds": res.drop(columns=["predictions", "truth", "rows"], errors="ignore"),
                            "recall": recall.reset_index()}, synthetic=False)
    print(f"saved run {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
