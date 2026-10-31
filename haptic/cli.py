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


def _build_real(root: Path, seconds: float | None) -> pd.DataFrame:
    stubs = L.discover(root)
    names = L.load_names(root)
    print(f"found {len(stubs)} cloth recordings", file=sys.stderr)
    rows = []
    for i, st in enumerate(stubs, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(stubs)}", file=sys.stderr)
        rec = L.load(st)
        if seconds:
            n = int(seconds * rec["fs"])
            rec["accel"] = rec["accel"][:n]
        try:
            d = D.extract(rec["accel"], rec["fs"],
                          material_id=rec["material_id"],
                          velocity_mm_s=rec["velocity_mm_s"],
                          force_n=rec["force_n"])
        except ValueError as e:
            print(f"  skipped {st['path'].name}: {e}", file=sys.stderr)
            continue
        row = d.as_dict()
        row["material_name"] = names.get(st["material_id"], rec["material_name"])
        rows.append(row)
    if not rows:
        raise RuntimeError("no recordings produced a descriptor")
    return D.normalise_frame(pd.DataFrame(rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="haptic.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="describe the dataset directory")
    pi.add_argument("--root", required=True)

    pg = sub.add_parser("gate", help="run the Objective 1 separability gate")
    pg.add_argument("--root", help="dataset root (omit with --synthetic)")
    pg.add_argument("--synthetic", action="store_true")
    pg.add_argument("--seconds", type=float, default=1.5)
    pg.add_argument("--repeats", type=int, default=3)
    pg.add_argument("--out", default="descriptors.csv")
    pg.add_argument("--keep-out-of-band", action="store_true",
                    help="do not drop recordings whose centroid is at a band edge")

    a = ap.parse_args(argv)

    if a.cmd == "inspect":
        print(L.inspect(a.root))
        return 0

    if a.synthetic:
        print("SYNTHETIC SIGNALS — scaffolding only, not a result.\n",
              file=sys.stderr)
        df = _build_synthetic(a.seconds, a.repeats)
    elif a.root:
        df = _build_real(Path(a.root), a.seconds)
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

    for label, dims in (("RAW", D.Descriptor.CORE),
                        ("VELOCITY-NORMALISED", D.NORMALISED_CORE)):
        print(f"\n### {label}")
        print(S.run_gate(work, dims).summary())
        print("\nper-velocity silhouette:")
        print(S.separability_by_velocity(work, dims).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
