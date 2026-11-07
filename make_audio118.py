"""Round 4, analysis M: contact audio at 118-material scale.

    python make_audio118.py extract --root <cluster> --out <dir>    # Kaggle CPU
    python make_audio118.py analyse --audio <dir>/audio118.csv      # here

extract: for every audio file, the steady window from its position file (same
loader rules as O1), then 16 log band shares over 20-1000 Hz and over 20-8000 Hz,
each with log RMS. analyse: 118-way logistic regression with direction held out,
against D's acceleration `bands`. Plan: PREREGISTRATION.md, "Round 4", M (a86cb06).
Saves run `audio118`.
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from haptic import descriptor as D
from haptic import loader as L

LINEAR_RUN = "20261107T103411_132525Z"
N_BANDS = 16
RANGES = {"a1k": (20.0, 1000.0), "a8k": (20.0, 8000.0)}


def columns(prefix: str) -> list[str]:
    return [f"{prefix}_b{k:02d}" for k in range(N_BANDS)] + [f"{prefix}_log_rms"]


def band_shares(x: np.ndarray, fs: float, low: float, high: float, prefix: str) -> dict:
    """Log share of [low, high) power in 16 log bands, and log RMS in the range."""
    from scipy.signal import butter, sosfiltfilt, welch
    sos = butter(4, [low / (fs / 2), high / (fs / 2)], btype="band", output="sos")
    xb = sosfiltfilt(sos, np.asarray(x, float) - np.mean(x))
    f, p = welch(xb, fs, nperseg=min(16384, len(xb)))   # ~2.7 Hz, as fine as the accel bands
    edges = np.geomspace(low, high, N_BANDS + 1)
    total = p[(f >= low) & (f < high)].sum()
    out = {c: float(np.log(p[(f >= lo) & (f < hi)].sum() / total + 1e-12))
           for c, lo, hi in zip(columns(prefix), edges[:-1], edges[1:])}
    out[f"{prefix}_log_rms"] = float(np.log(np.sqrt(np.mean(xb ** 2)) + 1e-12))
    return out


def work(path: str):
    import soundfile as sf
    p = Path(path)
    info = L.parse_name(p.with_suffix(".parquet"))   # same naming, parquet suffix expected
    if info is None:
        return {"skip": p.name, "reason": "unparsed name"}
    pos = p.parents[2] / "position" / p.parent.name / (p.stem + ".parquet")
    try:
        window = L.steady_window(L.read_table(pos), info["velocity_mm_s"])
        if window is None or window[1] - window[0] < L.MIN_STEADY_S:
            raise ValueError(f"under {L.MIN_STEADY_S} s of steady sliding")
        x, fs = sf.read(p, dtype="float64")
        if x.ndim > 1:
            raise ValueError(f"expected mono, got {x.shape}")
        x = x[int(window[0] * fs):int(window[1] * fs)]
        row = {"material_id": info["material_id"], "category": L.category_of(info["material_id"]),
               "velocity_mm_s": info["velocity_mm_s"], "force_n": info["force_n"],
               "direction_deg": info["direction_deg"],
               "repeat": L.repeat_index(info["direction_deg"], info["repeat"])}
        for prefix, (lo, hi) in RANGES.items():
            row.update(band_shares(x, fs, lo, hi, prefix))
        return {"row": row}
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        return {"skip": p.name, "reason": str(e)[:200]}


def extract(root: Path, out: Path, workers: int) -> None:
    files = sorted(str(p) for p in (root / "sensor_data" / "audio").rglob("*.flac"))
    print(f"{len(files)} audio files", flush=True)
    with Pool(workers) as pool:
        res = pool.map(work, files, chunksize=32)
    rows = [r["row"] for r in res if "row" in r]
    skips = [{"file": r["skip"], "reason": r["reason"]} for r in res if "skip" in r]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "audio118.csv", index=False)
    pd.DataFrame(skips, columns=["file", "reason"]).to_csv(out / "audio118_skipped.csv", index=False)
    print(f"{len(rows)} rows, {len(skips)} skipped", flush=True)


def analyse(audio_csv: str, features_csv: str) -> None:
    from scipy.stats import wilcoxon

    import runlog
    from haptic import benchmark as B
    audio = pd.read_csv(audio_csv)
    accel = pd.read_csv(features_csv)
    accel = accel[accel.in_band.astype(bool)]
    keys = ["material_id", "velocity_mm_s", "force_n", "direction_deg", "repeat"]
    # same recordings for both: the in-band acceleration set, where audio exists
    df = accel.merge(audio.drop(columns="category"), on=keys, how="inner").reset_index(drop=True)
    print(f"{len(audio)} audio rows; {len(df)} matched to in-band acceleration", flush=True)
    sets = {"accel_bands": list(B.FEATURE_SETS["bands"]),
            "audio_1k": columns("a1k"), "audio_8k": columns("a8k")}
    results, folds = {}, {}
    for label in ("material_id", "category"):
        for split in ("direction", "repeat"):
            for name, cols in sets.items():
                r = B.linear(df, cols, label, split)
                results[f"{label}|{split}|{name}"] = {k: v for k, v in r.items() if k != "folds"}
                if label == "material_id" and split == "direction":
                    folds[name] = pd.Series({f["fold"]: f["accuracy"] for f in r["folds"]})
                print(f"{label:11s} {split:9s} {name:12s} acc {r['accuracy']:.3f}  "
                      f"macro-F1 {r['macro_f1']:.3f}", flush=True)
    folds = pd.DataFrame(folds)
    p = float(wilcoxon(folds.audio_8k - folds.accel_bands, alternative="greater").pvalue)
    # the registered comparator is D's saved run; check the matched rerun agrees
    d = runlog.load_run("benchmark118_linear")
    if d.info["run_id"] != LINEAR_RUN:
        raise SystemExit(f"pre-registration names benchmark118_linear/{LINEAR_RUN}")
    d_acc = d.metrics["results"]["material_id|direction|bands"]["accuracy"]
    d_folds = pd.Series({f["fold"]: f["accuracy"]
                         for f in d.metrics["results"]["material_id|direction|bands"]["folds"]})
    p_d = float(wilcoxon(folds.audio_8k - d_folds.reindex(folds.index), alternative="greater").pvalue)
    metrics = {"preregistration": "PREREGISTRATION.md: Round 4, M (commit a86cb06)",
               "n_audio": len(audio), "n_matched": len(df), "results": results,
               "test": {"comparison": f"audio_8k vs benchmark118_linear/{LINEAR_RUN} bands",
                        "registered_accuracy": d_acc, "p_vs_registered_run": p_d,
                        "p_vs_matched_rerun": p, "significant": bool(p_d < 0.05),
                        "predicted": "audio_8k > accel bands", "prediction_held": bool(p_d < 0.05)},
               "fold_accuracy": folds.round(4).to_dict(orient="index")}
    print(f"[M] audio_8k vs registered accel bands ({d_acc:.3f}): p {p_d:.4f}; vs matched rerun p {p:.4f}")
    run = runlog.save_run("audio118", {"audio": audio_csv, "features": features_csv}, metrics,
                          {"folds": folds.reset_index(names="fold")}, synthetic=False)
    print("saved", run)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_audio118")
    ap.add_argument("step", choices=("extract", "analyse"))
    ap.add_argument("--root")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--audio", default="data/round2_features/audio118.csv")
    ap.add_argument("--features", default="data/round2_features/features.csv")
    a = ap.parse_args(argv)
    if a.step == "extract":
        extract(Path(a.root), Path(a.out), a.workers)
    else:
        analyse(a.audio, a.features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
