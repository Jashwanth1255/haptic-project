"""Does field of view explain why the MINC classifier rejects Cluster's scans? EXPLORATORY.

    python -m vision.cluster_scan_scale --root data/cluster --models models

The 224 px crops cover about 19 mm of fabric, much less than a MINC-2500 photo.
This classifies, for each of the 16 cloth materials, the whole 1181 px scan
(100 mm field) and its four quadrants (50 mm field), each with the standard eval
preprocessing (resize to 256, centre crop 224), and reports the fraction called
"fabric" at each field of view.

The first run of this analysis (`o2_cluster_scan_scale/20261104T161606_361379Z`)
came from an unsaved stdin script. This file reproduces it so the paper can cite
a run made from committed code.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import runlog
from haptic.groups import ASSIGNMENT

from .cluster_scans import _run, softmax

QUADRANTS = ("q0", "q1", "q2", "q3")


def views(scan: Path, tmp: Path) -> list[tuple[str, str, Path]]:
    """(view, scale, path): the whole scan, then its four quadrants in reading order."""
    from PIL import Image
    out = [("whole", "100 mm", scan)]
    with Image.open(scan) as im:
        im = im.convert("RGB")
        w, h = im.size
        boxes = [(0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2),
                 (0, h // 2, w // 2, h), (w // 2, h // 2, w, h)]
        for name, box in zip(QUADRANTS, boxes):
            p = tmp / f"{scan.stem}_{name}.png"           # lossless, no second JPEG pass
            im.crop(box).save(p)
            out.append((name, "50 mm", p))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cluster_scan_scale")
    ap.add_argument("--root", default="data/cluster")
    ap.add_argument("--models", default="models")
    a = ap.parse_args(argv)
    scans = Path(a.root, "images", "images_scan_area")
    with tempfile.TemporaryDirectory() as t:
        rows = []
        for mid in sorted(ASSIGNMENT):
            for view, scale, path in views(scans / f"img{mid}.jpg", Path(t)):
                rows.append({"mid": mid, "view": view, "scale": scale, "path": path})
        items = pd.DataFrame(rows)
        classes, logits, _ = _run(Path(a.models, "minc2500_fold1_seed0", "best.pt"),
                                  list(items.path))
    pred = np.array(classes)[softmax(logits).argmax(1)]
    table = items.drop(columns="path").assign(pred=pred)[["mid", "view", "pred", "scale"]]
    frac = table.assign(fabric=table.pred == "fabric").groupby("scale").fabric.mean()
    per_class = pd.read_csv(Path("results", "o2", "minc2500_fold1_seed0", "per_class.csv"))
    minc_recall = float(per_class.set_index("class").loc["fabric", "recall"])
    metrics = {"exploratory": True,
               "fabric_fraction_by_scale": {k: float(v) for k, v in frac.items()},
               "n_images": int(len(table)),
               "minc_test_fabric_recall": round(minc_recall, 3),
               "minc_test_fabric_recall_source": "results/o2/minc2500_fold1_seed0/per_class.csv"}
    config = {"views": "whole 1181px scan (100 mm) and 4 quadrants (50 mm), resized 256",
              "model": f"{a.models}/minc2500_fold1_seed0"}
    for k, v in metrics["fabric_fraction_by_scale"].items():
        print(f"{k}: {v:.4f} called fabric")
    print("saved", runlog.save_run("o2_cluster_scan_scale", config, metrics,
                                   {"predictions": table}, synthetic=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
