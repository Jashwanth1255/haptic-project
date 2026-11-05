"""Objective 1 on real Cluster cloth: everything after the gate.

    python make_o1_real.py --root data/cluster

Reads the descriptors from the latest saved `o1_gate` run (it never recomputes
them) and saves one run, `o1_real_analysis`, holding:

- the **pre-registered group test** (haptic/groups.py), and its sensitivity to the
  one reading the registration left open;
- where each descriptor's variance comes from: material, sliding direction, their
  interaction, velocity, force;
- **exploratory**, not registered: material identification with the sliding
  direction held out, from the 4-D descriptors and from a 16-band spectrum
  (haptic/identity.py). Band features are computed here and cached.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import runlog
from haptic import groups as G
from haptic import identity as I
from haptic import loader as L


def band_table(root: Path, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    stubs = L.discover(root)
    for k, st in enumerate(stubs, 1):
        if k % 250 == 0:
            print(f"  bands {k}/{len(stubs)}", flush=True)
        try:
            rec = L.load(st)
        except ValueError:
            continue
        rows.append({"material_id": rec["material_id"], "velocity_mm_s": rec["velocity_mm_s"],
                     "force_n": rec["force_n"], "direction_deg": rec["direction_deg"],
                     "repeat": rec["repeat"], **I.band_features(rec["accel"], rec["fs"])})
    out = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_o1_real")
    ap.add_argument("--root", default="data/cluster")
    ap.add_argument("--bands-cache", default="data/derived/band_features.csv")
    a = ap.parse_args(argv)

    gate = runlog.load_run("o1_gate")
    if gate.info["synthetic"]:
        raise SystemExit("latest o1_gate run is synthetic")
    df = gate.table("descriptors")
    work = df[df["in_band"].astype(bool)].copy()
    print(f"gate run {gate.info['run_id']}: {len(df)} descriptors, {len(work)} in band")

    groups = G.run(work)
    verdict = G.verdict(groups)
    sens = G.sensitivity(work)
    raw_groups = G.run(work, dims=I.RAW4)
    print("\n# pre-registered group test (velocity-normalised)")
    print(groups.drop(columns=["materials"]).round(4).to_string(index=False))
    print(verdict)
    print("\n# sensitivity to the reading of 'other assigned materials'")
    print(sens.round(4).to_string(index=False))
    print("\n# same test on raw descriptors (not registered; for comparison)")
    print(raw_groups.drop(columns=["materials"]).round(4).to_string(index=False))

    shares = I.variance_shares(work, (*I.RAW4, *I.NORM4[:2], I.NORM4[3]))
    print("\n# variance shares")
    print(shares.round(3).to_string(index=False))

    bands = band_table(Path(a.root), Path(a.bands_cache))
    keys = ["material_id", "velocity_mm_s", "force_n", "direction_deg", "repeat"]
    merged = work.merge(bands, on=keys, how="inner", validate="one_to_one")
    band_cols = (*I.BAND_COLUMNS, "log_rms")
    sets = {"raw4": I.RAW4, "norm4": I.NORM4, "bands": band_cols,
            "bands+raw4": (*band_cols, *I.RAW4)}
    ident, recall = {}, {}
    print("\n# EXPLORATORY: material identification, direction held out (chance 1/16)")
    for name, cols in sets.items():
        r = I.identify(merged, cols)
        ident[name] = {k: r[k] for k in ("accuracy", "fold_mean", "fold_std", "n_folds",
                                          "chance", "n")}
        recall[name] = pd.Series(r["recall"])
        print(f"  {name:<11} accuracy {r['accuracy']:.3f}  (fold mean {r['fold_mean']:.3f} "
              f"± {r['fold_std']:.3f}, n={r['n']})")
    recall = pd.DataFrame(recall)
    recall.index.name = "material_id"
    recall.insert(0, "name", recall.index.map(L.CLOTH_NAMES))
    print(recall.round(2).to_string())

    metrics = {"gate_run": gate.info["run_id"],
               "gate": {"raw": gate.metrics["raw"], "normalised": gate.metrics["normalised"]},
               "group_test": verdict,
               "groups": {r.group: {"n": int(r.n), "ratio": r.ratio, "p": r.p,
                                    "p_holm": r.p_holm, "tested": bool(r.tested)}
                          for r in groups.itertuples()},
               "identification_exploratory": ident}
    path = runlog.save_run(
        "o1_real_analysis", vars(a), metrics,
        {"group_test": groups.assign(materials=groups.materials.astype(str)),
         "group_sensitivity": sens,
         "group_test_raw": raw_groups.assign(materials=raw_groups.materials.astype(str)),
         "variance_shares": shares, "identification_recall": recall.reset_index()},
        synthetic=False)
    print(f"\nsaved run {path}")
    return 0


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    raise SystemExit(main())
