"""Round 3, analysis F: what survives the on/off vibration channel.

    python make_onoff.py --root data/cluster

For each of the 2,493 cloth gate recordings and each source (acceleration, contact
audio; both band-passed to 20-1000 Hz), the first 500 ms of the amplitude envelope
is rendered with scheme B (10 ms tick) through the assumed actuator lag. Fabric
identity is then measured from the intended and from the rendered envelope:
16-way logistic regression, direction held out.

Plan and predictions: PREREGISTRATION.md, "Round 3", F (commit d3b3bf9).
Saves run `onoff_channel`.
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import runlog
from haptic import descriptor as D
from haptic import loader as L
from haptic import round3 as R3
from haptic.separability import holm_adjust

GATE_RUN = "20261104T153936_758197Z"
KEYS = ["material_id", "velocity_mm_s", "force_n", "direction_deg", "repeat"]
SOURCES = ("accel", "audio")
FEAT = [f"m{k:02d}" for k in range(len(R3.MOD_BANDS) - 1)] + ["log_level"]


def _audio_path(stub) -> Path:
    p = Path(stub["path"])
    return p.parents[2] / "audio" / p.parent.name / (p.stem + ".flac")


def signals(stub):
    """Steady-window acceleration and audio for one recording, or a skip reason."""
    import soundfile as sf
    try:
        rec = L.load(stub)
        window = L.steady_window(L.read_table(L._sibling(Path(stub["path"]), "position")),
                                 stub["velocity_mm_s"])
        x, fs = sf.read(_audio_path(stub), dtype="float64")
        if x.ndim > 1:
            raise RuntimeError(f"{_audio_path(stub)}: expected mono, got {x.shape}")
        audio = x[int(window[0] * fs):int(window[1] * fs)]
        out = {}
        for name, sig, rate in (("accel", D.to_magnitude(rec["accel"]), rec["fs"]), ("audio", audio, fs)):
            xb = D.bandpass(np.asarray(sig, float) - np.mean(sig), rate)
            out[name] = {"x": sig, "fs": rate, "intensity": float(np.sqrt(np.mean(xb ** 2)))}
        return {"key": {k: stub[k] if k != "repeat" else rec["repeat"] for k in KEYS}, "sig": out}
    except (ValueError, FileNotFoundError) as e:
        return {"skip": f"{Path(stub['path']).name}: {e}"}


def intensities(stub):
    """Pass 1: only the key and each source's in-band RMS, so the large signals
    are not shipped between processes."""
    r = signals(stub)
    if "skip" in r:
        return r
    return {"key": r["key"], "stub": stub,
            "intensity": {s: r["sig"][s]["intensity"] for s in SOURCES}}


def render(args):
    stub, ref = args
    loaded = signals(stub)
    if "skip" in loaded:
        return loaded
    sig = loaded["sig"]
    out = {}
    for name in SOURCES:
        s = sig[name]
        try:
            intended, rendered = R3.render_pair(s["x"], s["fs"], s["intensity"], ref[name])
        except ValueError as e:
            return {"skip": str(e)}
        out[name] = (R3.envelope_features(intended), R3.envelope_features(rendered))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_onoff")
    ap.add_argument("--root", default="data/cluster")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)

    run = runlog.load_run("o1_gate")
    if run.info["run_id"] != GATE_RUN:
        raise SystemExit(f"pre-registration names o1_gate/{GATE_RUN}, latest is {run.info['run_id']}")
    gate = run.table("descriptors")
    gate = gate[gate.in_band.astype(bool)]
    keep = set(map(tuple, gate[KEYS].to_numpy().tolist()))
    stubs = L.discover(a.root)
    print(f"{len(stubs)} cloth recordings found; gate set has {len(keep)}", flush=True)

    with Pool(a.workers) as pool:
        loaded = pool.map(intensities, stubs, chunksize=8)
    skipped = [r["skip"] for r in loaded if "skip" in r]
    loaded = [r for r in loaded if "key" in r
              and tuple(r["key"][k] for k in KEYS) in keep]
    print(f"{len(loaded)} in the gate set, {len(skipped)} failed to load", flush=True)
    ref = {s: float(np.percentile([r["intensity"][s] for r in loaded], 95)) for s in SOURCES}

    with Pool(a.workers) as pool:
        feats = pool.map(render, [(r["stub"], ref) for r in loaded], chunksize=8)
    rows = []
    for r, f in zip(loaded, feats):
        if "skip" in f:
            skipped.append(f"{r['key']}: {f['skip']}")
            continue
        for s in SOURCES:
            for stage, v in zip(("intended", "rendered"), f[s]):
                rows.append({**r["key"], "source": s, "stage": stage, **dict(zip(FEAT, v))})
    table = pd.DataFrame(rows)
    n_rec = len(table) // 4
    print(f"{n_rec} recordings rendered, {len(skipped)} skipped in all", flush=True)

    folds, results = {}, {}
    for (s, stage), part in table.groupby(["source", "stage"]):
        acc, r = R3.fold_accuracies(part.reset_index(drop=True), FEAT)
        folds[f"{s}_{stage}"] = acc
        results[f"{s}_{stage}"] = {k: r[k] for k in ("accuracy", "macro_f1", "fold_std", "n", "chance")}
        print(f"[F] {s:5s} {stage:8s} accuracy {r['accuracy']:.3f}  macro-F1 {r['macro_f1']:.3f}")
    folds = pd.DataFrame(folds)
    chance = 1 / 16
    p = [float(wilcoxon(folds.accel_rendered - chance, alternative="greater").pvalue),
         float(wilcoxon(folds.accel_intended - folds.accel_rendered, alternative="greater").pvalue),
         float(wilcoxon(folds.audio_rendered - folds.accel_rendered, alternative="two-sided").pvalue)]
    holm = holm_adjust(np.array(p))
    names = [("rendered_accel_above_chance", "rendered accel > 1/16"),
             ("rendering_loses_identity", "intended accel > rendered accel"),
             ("audio_vs_accel_rendered", "none (two-sided)")]
    tests = {}
    for (n, pred), pi, h in zip(names, p, holm):
        tests[n] = {"p": pi, "holm_p": float(h), "significant": bool(h < 0.05), "predicted": pred}
        if pred != "none (two-sided)":
            tests[n]["prediction_held"] = bool(h < 0.05)
        print(f"[F] {n:30s} p {pi:.3g} holm {h:.3g}")
    metrics = {"preregistration": "PREREGISTRATION.md: Round 3, F (commit d3b3bf9)",
               "n_recordings": n_rec, "n_skipped": len(skipped), "intensity_ref": ref,
               "env_ms": R3.ENV_MS, "tick_ms": R3.TICK_MS, "chance": chance,
               "results": results, "tests": tests,
               "fold_accuracy": folds.round(4).to_dict(orient="index")}
    run = runlog.save_run("onoff_channel", {"root": a.root, "gate_run": GATE_RUN,
                                            "features": FEAT},
                          metrics, {"features": table, "folds": folds.reset_index(names="fold"),
                                    "skipped": pd.DataFrame({"reason": skipped})},
                          synthetic=False)
    print("saved", run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
