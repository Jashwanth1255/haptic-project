"""Round 2, CPU stage: features for analyses B, C and D, then the linear benchmark.

    python make_round2_features.py --root <cluster> --out <dir> [--workers 4]

For every recording of all 118 materials (8 directions, 2 repeats, 5 velocities,
2 forces) this extracts the O1 descriptors, the 16 log bands + RMS, and the
acceleration spectrogram that the CNN reads. For the 16 cloth materials it also
extracts the log-frequency spectrogram (analysis C) and the contact-audio
spectrograms over the same steady-sliding window (analysis B). Loader rules are
those of O1. It writes:

    features.csv          one row per loaded recording, with in_band
    skipped.csv
    accel_specs.npz       acceleration spectrograms, float16, in-band rows only
    cloth_logf.npz        log-frequency spectrograms, cloth
    cloth_audio_full.npz  audio 20-8000 Hz, cloth
    cloth_audio_band.npz  audio 20-1000 Hz, cloth

Then it runs the pre-registered linear benchmark (analysis D) and saves run
`benchmark118_linear`. The plan is in PREREGISTRATION.md, round 2, D.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import runlog
from haptic import benchmark as B
from haptic import descriptor as D
from haptic import identity as I
from haptic import learned as Lr
from haptic import loader as L

KEYS = ["material_id", "velocity_mm_s", "force_n", "direction_deg", "repeat"]
AUDIO_FULL = (20.0, 8000.0)
AUDIO_BAND = (D.BAND_LOW_HZ, D.BAND_HIGH_HZ)


def _audio_path(stub) -> Path:
    p = Path(stub["path"])
    root = p.parents[2]                    # <root>/sensor_data/accel/<mid>/<file>
    return root / "audio" / p.parent.name / (p.stem + ".flac")


def work(stub):
    """Everything for one recording, or a skip reason."""
    try:
        rec = L.load(stub)
        d = D.extract(rec["accel"], rec["fs"], material_id=stub["material_id"],
                      velocity_mm_s=stub["velocity_mm_s"], force_n=stub["force_n"])
    except ValueError as e:
        return {"skip": {"file": Path(stub["path"]).name, "reason": str(e).split(":")[-1].strip()}}
    row = {**d.as_dict(), **I.band_features(rec["accel"], rec["fs"]),
           "category": stub["category"], "direction_deg": stub["direction_deg"],
           "repeat": rec["repeat"]}
    out = {"row": row, "accel": Lr.spectrogram(rec["accel"], rec["fs"]).astype(np.float16)}
    if stub["material_id"] in L.CLOTH_IDS:
        import soundfile as sf
        out["logf"] = Lr.logf_spectrogram(rec["accel"], rec["fs"]).astype(np.float16)
        window = L.steady_window(L.read_table(L._sibling(Path(stub["path"]), "position")),
                                 stub["velocity_mm_s"])
        x, fs = sf.read(_audio_path(stub), dtype="float64")
        if x.ndim > 1:
            raise RuntimeError(f"{_audio_path(stub)}: expected mono, got {x.shape}")
        x = x[int(window[0] * fs):int(window[1] * fs)]
        out["audio_full"] = Lr.audio_spectrogram(x, fs, *AUDIO_FULL).astype(np.float16)
        out["audio_band"] = Lr.audio_spectrogram(x, fs, *AUDIO_BAND).astype(np.float16)
    return out


def save_specs(path: Path, specs, meta: pd.DataFrame):
    offsets = np.r_[0, np.cumsum([len(s) for s in specs])]
    np.savez_compressed(path, spec=np.concatenate(specs), offsets=offsets,
                        **{k: meta[k].to_numpy() for k in KEYS})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_round2_features")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    stubs = L.discover(a.root, material_ids=tuple(range(118)))
    print(f"{len(stubs)} recordings found", flush=True)
    from multiprocessing import Pool
    results = []
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap(work, stubs, chunksize=16), 1):
            results.append(r)
            if k % 1000 == 0:
                print(f"  {k}/{len(stubs)}", flush=True)

    skipped = pd.DataFrame([r["skip"] for r in results if "skip" in r], columns=["file", "reason"])
    ok = [r for r in results if "row" in r]
    df = D.normalise_frame(pd.DataFrame([r["row"] for r in ok]))
    df.to_csv(out / "features.csv", index=False)
    skipped.to_csv(out / "skipped.csv", index=False)
    inb = df["in_band"].to_numpy(bool)
    print(f"{len(df)} loaded, {len(skipped)} skipped, {int((~inb).sum())} out of band", flush=True)

    save_specs(out / "accel_specs.npz", [r["accel"] for r, keep in zip(ok, inb) if keep], df[inb])
    cloth = df.material_id.isin(L.CLOTH_IDS).to_numpy()
    for name in ("logf", "audio_full", "audio_band"):
        save_specs(out / f"cloth_{name}.npz", [r[name] for r, c in zip(ok, cloth) if c], df[cloth])

    # ---- analysis D, linear models (descriptive, plan fixed in advance) --------
    bench = df[inb].reset_index(drop=True)
    metrics, tables = {"preregistration": "PREREGISTRATION.md: Benchmark and findings round 2, D",
                       "n_recordings": int(len(bench)), "n_skipped": int(len(skipped)),
                       "n_out_of_band": int((~inb).sum()), "n_materials": int(bench.material_id.nunique()),
                       "results": {}}, {}
    for label in ("material_id", "category"):
        for split in ("direction", "repeat"):
            for fs_name, cols in B.FEATURE_SETS.items():
                r = B.linear(bench, cols, label, split)
                if r is None:
                    print(f"{label}|{split}: no test data, skipped", flush=True)
                    continue
                key = f"{label}|{split}|{fs_name}"
                tables[key.replace("|", "_").replace("+", "_")] = pd.DataFrame(
                    [{"class": k, "recall": v} for k, v in r.pop("recall").items()])
                metrics["results"][key] = r
                print(f"{key:40s} acc {r['accuracy']:.3f}  macro-F1 {r['macro_f1']:.3f}  "
                      f"chance {r['chance']:.4f}  majority {r['majority_baseline']:.3f}", flush=True)
    path = runlog.save_run("benchmark118_linear", vars(a), metrics,
                           {"skipped": skipped, **tables}, synthetic=False)
    (out / "linear_run.json").write_text(json.dumps({"path": str(path)}))
    print(f"saved run {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
