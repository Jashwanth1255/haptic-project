"""Is the cloth's velocity-invariant frequency a property of cloth, or of the rig?

    python make_velocity_all.py --root data/cluster

Added 2026-11-05, after O3 found the spectral centroid of real cloth nearly
independent of sliding speed (median exponent -0.03, where f = v / lambda predicts
1). Two explanations predict different things across the other 102 materials:

    rig      a resonance of the probe tip and accelerometer sets the frequency, so
             every material, rough wood and stone included, shows an exponent near 0
             and a centroid in the same place
    cloth    soft, compliant fabric damps the spatial texture; harder materials with
             a real surface period show exponents nearer 1

Subset, identical for every material: directions 0 and 90 degrees, first repeat,
all five velocities, both forces. Saves `velocity_all_materials`.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

import runlog
from haptic import descriptor as D
from haptic import loader as L
from haptic import velocity as Y

DIRECTIONS = (0, 90)


def descriptors(root) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rows, skipped, spectra = [], [], []
    stubs = [s for s in L.discover(root, material_ids=tuple(range(118)))
             if s["direction_deg"] in DIRECTIONS and s["repeat"] == 0]
    for k, st in enumerate(stubs, 1):
        if k % 500 == 0:
            print(f"  {k}/{len(stubs)}", flush=True)
        try:
            rec = L.load(st)
            d = D.extract(rec["accel"], rec["fs"], material_id=st["material_id"],
                          velocity_mm_s=st["velocity_mm_s"], force_n=st["force_n"])
        except ValueError as e:
            skipped.append({"file": st["path"].name, "reason": str(e).split(":")[-1].strip()})
            continue
        spectra.append(Y.log_spectrum(rec["accel"], rec["fs"]))
        rows.append({**d.as_dict(), "category": st["category"],
                     "direction_deg": st["direction_deg"], "repeat": rec["repeat"]})
    df = D.normalise_frame(pd.DataFrame(rows))
    return df, pd.DataFrame(skipped, columns=["file", "reason"]), np.array(spectra)


def shift_summary(spectra, df, cat) -> tuple[pd.DataFrame, dict]:
    """Spectral-shift exponents, raw and whitened (EXPLORATORY, added 2026-11-06)."""
    meta = df[["material_id", "velocity_mm_s", "force_n", "direction_deg"]]
    per = []
    for whiten in (False, True):
        e = Y.shift_exponents(spectra, meta, whiten=whiten)
        per.append(e.groupby("material_id").shift_exponent.mean()
                   .rename("whitened" if whiten else "raw"))
    per = pd.concat(per, axis=1).reset_index()
    per["category"] = per.material_id.map(cat)
    out = {}
    for col in ("raw", "whitened"):
        x = per[col]
        out[col] = {"median": float(x.median()), "q25": float(x.quantile(.25)),
                    "q75": float(x.quantile(.75)),
                    "share_within_025_of_1": float((np.abs(x - 1) < 0.25).mean()),
                    "share_within_025_of_0": float((np.abs(x) < 0.25).mean()),
                    "vs_0_p": float(stats.wilcoxon(x).pvalue),
                    "vs_1_p": float(stats.wilcoxon(x - 1).pvalue),
                    "cloth_median": float(x[per.category == "cloth"].median()),
                    "other_median": float(x[per.category != "cloth"].median()),
                    "cloth_vs_rest_p": float(stats.mannwhitneyu(
                        x[per.category == "cloth"], x[per.category != "cloth"],
                        alternative="two-sided").pvalue),
                    "cloth_vs_0_p": float(stats.wilcoxon(x[per.category == "cloth"]).pvalue)}
    return per, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_velocity_all")
    ap.add_argument("--root", default="data/cluster")
    a = ap.parse_args(argv)
    pd.set_option("display.width", 200)

    df, skipped, spectra = descriptors(a.root)
    print(f"{len(df)} descriptors from {df.material_id.nunique()} materials; "
          f"{len(skipped)} skipped, {int((~df.in_band).sum())} out of band")
    cat = df.groupby("material_id").category.first()
    exps = Y.scaling_exponents(df)
    exps["category"] = exps.material_id.map(cat)
    per_material = exps.groupby(["material_id", "category"], as_index=False)[
        list(Y.SCALING_DIMS)].mean()

    by_cat = per_material.groupby("category")[list(Y.SCALING_DIMS)].agg(["median", "count"])
    print("\n# exponent of each dimension on velocity, per category (median over materials)")
    print(by_cat.round(3).to_string())

    f = per_material.frequency_hz
    cloth = per_material.category == "cloth"
    kw = stats.kruskal(*[g.frequency_hz for _, g in per_material.groupby("category")])
    mw = stats.mannwhitneyu(f[cloth], f[~cloth], alternative="two-sided")
    one = stats.wilcoxon(f[~cloth] - 1.0)
    zero = stats.wilcoxon(f[~cloth])
    print(f"\nfrequency exponent: cloth median {f[cloth].median():.3f} (n={cloth.sum()}), "
          f"others {f[~cloth].median():.3f} (n={(~cloth).sum()})")
    print(f"  Kruskal-Wallis across categories p={kw.pvalue:.3g}; cloth vs rest "
          f"Mann-Whitney p={mw.pvalue:.3g}")
    print(f"  non-cloth vs 1 (f = v/lambda): Wilcoxon p={one.pvalue:.3g}; vs 0: p={zero.pvalue:.3g}")
    share_near_one = float((np.abs(f - 1) < 0.25).mean())
    share_near_zero = float((np.abs(f) < 0.25).mean())
    print(f"  materials with exponent within 0.25 of 1: {share_near_one:.2f}; of 0: {share_near_zero:.2f}")

    at40 = df[(df.velocity_mm_s == 40) & df.in_band].groupby("material_id").frequency_hz.median()
    cent = at40.groupby(at40.index.map(cat)).agg(["median", "min", "max"])
    print("\n# spectral centroid at 40 mm/s (Hz), per category")
    print(cent.round(0).to_string())

    shifts, shift_metrics = shift_summary(spectra, df, cat)
    print()
    print("# spectral-shift exponent per material (EXPLORATORY; 1 = spectrum moves as v/lambda)")
    print(shifts.groupby("category")[["raw", "whitened"]].median().round(3).to_string())
    for k, v in shift_metrics.items():
        print(f"  {k}: median {v['median']:.3f} [{v['q25']:.3f}, {v['q75']:.3f}], within 0.25 of 1: "
              f"{v['share_within_025_of_1']:.2f}, of 0: {v['share_within_025_of_0']:.2f}, "
              f"vs 0 p={v['vs_0_p']:.3g}, vs 1 p={v['vs_1_p']:.3g}; cloth {v['cloth_median']:.3f} "
              f"vs rest {v['other_median']:.3f} p={v['cloth_vs_rest_p']:.3g}, cloth vs 0 p={v['cloth_vs_0_p']:.3g}")

    metrics = {"n_descriptors": len(df), "n_materials": int(df.material_id.nunique()),
               "n_skipped": len(skipped), "n_out_of_band": int((~df.in_band).sum()),
               "frequency_exponent": {
                   "cloth_median": float(f[cloth].median()),
                   "other_median": float(f[~cloth].median()),
                   "kruskal_p": float(kw.pvalue), "cloth_vs_rest_p": float(mw.pvalue),
                   "other_vs_1_p": float(one.pvalue), "other_vs_0_p": float(zero.pvalue),
                   "share_within_025_of_1": share_near_one,
                   "share_within_025_of_0": share_near_zero},
               "by_category_median": per_material.groupby("category")[
                   list(Y.SCALING_DIMS)].median().to_dict(orient="index"),
               "spectral_shift_exponent": shift_metrics}
    path = runlog.save_run("velocity_all_materials",
                           {**vars(a), "directions": DIRECTIONS, "repeat": 0}, metrics,
                           {"descriptors": df, "skipped": skipped,
                            "exponents": exps, "exponents_per_material": per_material,
                            "shift_exponents_per_material": shifts,
                "centroid_40mm_s": at40.rename("frequency_hz").reset_index()},
                           synthetic=False)
    print(f"\nsaved run {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
