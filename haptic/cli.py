"""Objective 1 command line.

    python -m haptic.cli inspect --root <dataset>     what is actually on disk
    python -m haptic.cli gate    --root <dataset>     run the O1 gate on real data
    python -m haptic.cli gate    --synthetic          run it on synthetic signals
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import runlog

from . import descriptor as D
from . import loader as L
from . import separability as S
from . import signals as G


def _build_synthetic(seconds: float, repeats: int) -> pd.DataFrame:
    recs = G.synth_dataset(repeats=repeats, seconds=seconds)
    ds = D.extract_many(recs)
    df = pd.DataFrame([d.as_dict() for d in ds])
    df["material_name"] = [r["material_name"] for r in recs]
    return D.normalise_frame(df)


def _build_real(root: Path, seconds: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(descriptors, skipped recordings with the reason for each)."""
    stubs = L.discover(root)
    names = L.load_names(root)
    print(f"found {len(stubs)} cloth recordings", file=sys.stderr)
    rows, skipped = [], []
    for i, st in enumerate(stubs, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(stubs)}", file=sys.stderr)
        try:
            rec = L.load(st)
            if seconds:
                rec["accel"] = rec["accel"][:int(seconds * rec["fs"])]
            d = D.extract(rec["accel"], rec["fs"],
                          material_id=rec["material_id"],
                          velocity_mm_s=rec["velocity_mm_s"],
                          force_n=rec["force_n"])
        except ValueError as e:
            skipped.append({"file": st["path"].name, "material_id": st["material_id"],
                            "velocity_mm_s": st["velocity_mm_s"], "reason": str(e)})
            continue
        row = d.as_dict()
        row["material_name"] = names.get(st["material_id"], rec["material_name"])
        for k in ("direction_deg", "repeat", "measured_fs_hz", "rate_off", "steady_s"):
            row[k] = rec[k]
        rows.append(row)
    skipped = pd.DataFrame(skipped, columns=["file", "material_id", "velocity_mm_s", "reason"])
    if len(skipped):
        print(f"skipped {len(skipped)} recordings; reasons:\n"
              f"{skipped['reason'].str.split(':').str[-1].value_counts().to_string()}",
              file=sys.stderr)
    if not rows:
        raise RuntimeError("no recordings produced a descriptor")
    return D.normalise_frame(pd.DataFrame(rows)), skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="haptic.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="describe the dataset directory")
    pi.add_argument("--root", required=True)

    pg = sub.add_parser("gate", help="run the Objective 1 separability gate")
    pg.add_argument("--root", help="dataset root (omit with --synthetic)")
    pg.add_argument("--synthetic", action="store_true")
    pg.add_argument("--seconds", type=float,
                    help="truncate each recording (synthetic default 1.5; real: whole steady window)")
    pg.add_argument("--repeats", type=int, default=3)
    pg.add_argument("--out", default="descriptors.csv")
    pg.add_argument("--keep-out-of-band", action="store_true",
                    help="do not drop recordings whose centroid is at a band edge")
    pg.add_argument("--no-save", action="store_true",
                    help="do not save the run under results/")

    a = ap.parse_args(argv)

    if a.cmd == "inspect":
        print(L.inspect(a.root))
        return 0

    if a.synthetic:
        print("SYNTHETIC SIGNALS — scaffolding only, not a result.\n",
              file=sys.stderr)
        df, skipped = _build_synthetic(a.seconds or 1.5, a.repeats), None
    elif a.root:
        df, skipped = _build_real(Path(a.root), a.seconds)
    else:
        ap.error("gate needs --root or --synthetic")

    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(df)} descriptors)\n", file=sys.stderr)

    n_oob = int((~df["in_band"]).sum())
    if n_oob:
        print(f"{n_oob} of {len(df)} recordings are out of band "
              f"(centroid pinned at a band edge).")
        bad = df[~df["in_band"]].groupby("material_name", observed=True).size()
        for name, n in bad.items():
            print(f"    {name}: {n}")
        print()
    work = df if a.keep_out_of_band else df[df["in_band"]]

    metrics = {"n_recordings": len(df), "n_out_of_band": n_oob,
               "out_of_band_by_material": (df[~df["in_band"]]
                                           .groupby("material_name", observed=True)
                                           .size().to_dict())}
    artifacts = {"descriptors": df}
    if skipped is not None:
        metrics["n_skipped"] = len(skipped)
        artifacts["skipped"] = skipped
    for key, label, dims in (("raw", "RAW", D.Descriptor.CORE),
                             ("normalised", "VELOCITY-NORMALISED", D.NORMALISED_CORE)):
        gate = S.run_gate(work, dims)
        per_v = S.separability_by_velocity(work, dims)
        print(f"\n### {label}")
        print(gate.summary())
        print("\nper-velocity silhouette:")
        print(per_v.to_string(index=False))
        metrics[key] = gate.as_dict()
        artifacts |= {f"{key}_anova": gate.anova, f"{key}_per_velocity": per_v,
                      f"{key}_summary": gate.summary()}

    if not a.no_save:
        path = runlog.save_run("o1_gate_synthetic" if a.synthetic else "o1_gate",
                               vars(a), metrics, artifacts, synthetic=a.synthetic)
        print(f"\nsaved run {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
