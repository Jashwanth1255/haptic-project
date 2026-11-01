"""Summarise the restricted 12-way DTD scores across folds.

    python make_o2_folds.py

Reads results/o2/dtd_fold*_seed0/textile12/textile_metrics.json, keeping fold 1
apart from folds 2-10: the class list was chosen after seeing fold 1, so only the
others can be held out from that choice.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vision.textile import summarise_folds

HERE = Path(__file__).parent


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.glob("dtd_fold*_seed0/textile12/textile_metrics.json")):
        m = json.loads(p.read_text())
        u = m.get("restricted_unseen")
        rows.append({"fold": m.get("fold", 1), "n": m["restricted"]["n"],
                     "accuracy": m["restricted"]["top1"],
                     "macro_f1": m["restricted"]["macro_f1"],
                     "n_unseen": u["n"] if u else 0,
                     "accuracy_unseen": u["top1"] if u else None,
                     "macro_f1_unseen": u["macro_f1"] if u else None})
    if not rows:
        raise FileNotFoundError(f"no textile metrics under {root}")
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def main() -> int:
    df = collect(HERE / "results" / "o2")
    print(df.round(4).to_string(index=False))
    held = df[df["fold"] > 1]
    if len(held) != 9:
        print(f"\nonly {len(held)} of 9 held-out folds present: no summary")
        return 1
    print("\nfolds 2-10, mean +/- std:")
    for col in ("accuracy", "macro_f1", "accuracy_unseen", "macro_f1_unseen"):
        s = summarise_folds(held[col])
        print(f"  {col:<16} {s['mean']:.4f} +/- {s['std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
