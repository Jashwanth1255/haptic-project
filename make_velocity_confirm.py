"""Confirmatory test: does cloth's spectrum shift with velocity on held-out data?

    python make_velocity_confirm.py --root data/cluster

Implements PREREGISTRATION.md, "Confirmation: cloth spectra shift with velocity
after whitening", exactly as written there:

    held-out   cloth 102-117, directions 45/135/180/225/270/315 both repeats, and
               the second repeat at 0 and 90
    reference  median log spectrum over all 118 materials per (velocity, force),
               from directions 0/90, first repeat only
    tests      cloth > 0 (Wilcoxon, one-sided); cloth held-out > other 102 from the
               original run (Mann-Whitney, one-sided); Holm across the two

Saves `velocity_confirm_cloth`.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

import runlog
from haptic import loader as L
from haptic import velocity as Y
from haptic.separability import holm_adjust

ORIGINAL_DIRECTIONS = (0, 90)
META = ["material_id", "velocity_mm_s", "force_n", "direction_deg", "repeat"]


def spectra(root, stubs):
    rows, specs, skipped = [], [], 0
    for k, st in enumerate(stubs, 1):
        if k % 500 == 0:
            print(f"  {k}/{len(stubs)}", flush=True)
        try:
            rec = L.load(st)
        except ValueError:
            skipped += 1
            continue
        specs.append(Y.log_spectrum(rec["accel"], rec["fs"]))
        rows.append({k_: st[k_] for k_ in META})
    return np.array(specs), pd.DataFrame(rows), skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_velocity_confirm")
    ap.add_argument("--root", default="data/cluster")
    a = ap.parse_args(argv)

    original = runlog.load_run("velocity_all_materials")
    if "spectral_shift_exponent" not in original.metrics:
        raise SystemExit("latest velocity_all_materials run has no shift exponents")
    others = original.table("shift_exponents_per_material")
    others = others[others.category != "cloth"].whitened.to_numpy()

    stubs = L.discover(a.root, material_ids=tuple(range(118)))
    ref_stubs = [s for s in stubs if s["direction_deg"] in ORIGINAL_DIRECTIONS and s["repeat"] == 0]
    held_stubs = [s for s in stubs if s["material_id"] in L.CLOTH_IDS
                  and not (s["direction_deg"] in ORIGINAL_DIRECTIONS and s["repeat"] == 0)]
    print(f"reference {len(ref_stubs)} recordings, held out {len(held_stubs)}", flush=True)

    ref_spec, ref_meta, ref_skipped = spectra(a.root, ref_stubs)
    reference = Y.whitening_reference(ref_spec, ref_meta)
    held_spec, held_meta, held_skipped = spectra(a.root, held_stubs)
    cells = Y.shift_exponents(held_spec, held_meta, whiten=True, reference=reference,
                              keys=("material_id", "force_n", "direction_deg", "repeat"))
    per = cells.groupby("material_id").shift_exponent.mean()

    w = stats.wilcoxon(per, alternative="greater")
    mw = stats.mannwhitneyu(per, others, alternative="greater")
    holm = holm_adjust([w.pvalue, mw.pvalue])
    passed = bool(holm[0] < 0.05 and holm[1] < 0.05)
    print(f"\n{len(cells)} held-out cells over {per.size} cloth materials "
          f"({held_skipped} held-out files skipped)")
    print(f"cloth held-out median exponent {per.median():.3f} "
          f"[{per.quantile(.25):.3f}, {per.quantile(.75):.3f}]; others (original) "
          f"{np.median(others):.3f}")
    print(f"  1. cloth > 0: Wilcoxon p={w.pvalue:.3g}, Holm {holm[0]:.3g}")
    print(f"  2. cloth > others: Mann-Whitney p={mw.pvalue:.3g}, Holm {holm[1]:.3g}")
    print(f"  confirmed: {passed}")
    table = per.rename("held_out").to_frame()
    table.insert(0, "name", table.index.map(L.CLOTH_NAMES))
    print(table.round(3).to_string())

    metrics = {"original_run": original.info["run_id"], "n_cells": len(cells),
               "n_materials": int(per.size), "held_out_skipped": held_skipped,
               "reference_skipped": ref_skipped, "cloth_median": float(per.median()),
               "others_median": float(np.median(others)),
               "wilcoxon_gt0_p": float(w.pvalue), "mannwhitney_gt_others_p": float(mw.pvalue),
               "holm": [float(h) for h in holm], "confirmed": passed,
               "preregistration": "PREREGISTRATION.md: Confirmation: cloth spectra shift "
                                  "with velocity after whitening"}
    path = runlog.save_run("velocity_confirm_cloth", vars(a), metrics,
                           {"cells": cells, "per_material": table.reset_index()},
                           synthetic=False)
    print(f"saved run {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
