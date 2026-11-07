"""Round 3, analysis G: does a fabric classifier trained on Cluster's flatbed
scans transfer to product photographs?

    python make_photo_transfer.py select   --labels <labels_front.csv> --out data/deepfashion_mm
    python make_photo_transfer.py fetch    --out data/deepfashion_mm
    python make_photo_transfer.py run      --cluster data/cluster --photos data/deepfashion_mm

Plan and prediction: PREREGISTRATION.md, "Round 3", G (commit 2b1b3c7).
Photos: DeepFashion-MultiModal, Kaggle mirror silverstone1903/deep-fashion-multimodal
(CC BY-NC-SA 4.0). Images stay under data/, which is not committed.
Saves run `photo_transfer`.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

MIRROR = "silverstone1903/deep-fashion-multimodal"
CLASSES = ("cotton", "chiffon", "leather")
CLUSTER = {"cotton": (104, 105, 106), "chiffon": (107,), "leather": tuple(range(87, 102))}
CAPTION_FABRICS = ("cotton", "denim", "chiffon", "knitting", "leather", "furry", "other", "mixed")
PER_CLASS = 100
TRAIN_CROPS = range(1, 71)
TEST_CROPS = range(71, 101)
N_BOOT = 10_000

_PAT = re.compile(r"\b(" + "|".join(CAPTION_FABRICS) + r")\b(?= fabric)|fabric is (\w+)")


def caption_fabrics(caption: str) -> set[str]:
    """Every fabric a caption names: '<x> fabric' or 'its fabric is <x>'."""
    out = set()
    for a, b in _PAT.findall(caption.lower()):
        word = a or b
        if word in CAPTION_FABRICS:
            out.add(word)
    return out


def caption_label(caption: str) -> str | None:
    """The single fabric of a photo, or None when the caption names none or several."""
    f = caption_fabrics(caption)
    return next(iter(f)) if len(f) == 1 else None


def select(labels: pd.DataFrame, per_class: int = PER_CLASS, seed: int = 0) -> pd.DataFrame:
    lab = labels.assign(fabric=labels.caption.map(caption_label))
    lab = lab[lab.fabric.isin(CLASSES)]
    rng = np.random.default_rng(seed)
    parts = []
    for c in CLASSES:
        part = lab[lab.fabric == c].sort_values("image_id")
        k = min(per_class, len(part))
        parts.append(part.iloc[np.sort(rng.choice(len(part), k, replace=False))])
    return pd.concat(parts)[["image_id", "path", "fabric", "product_type", "gender"]]


def balanced_accuracy(y, pred) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def bootstrap(y, hit, n=N_BOOT, seed=0) -> np.ndarray:
    """Balanced accuracy over class-stratified bootstrap resamples."""
    rng = np.random.default_rng(seed)
    y, hit = np.asarray(y), np.asarray(hit, float)
    idx = [np.flatnonzero(y == c) for c in np.unique(y)]
    out = np.empty(n)
    for b in range(n):
        out[b] = np.mean([hit[rng.choice(i, i.size)].mean() for i in idx])
    return out


def _features(paths, batch=32):
    import torch
    from PIL import Image
    from torchvision.models import ResNet50_Weights, resnet50
    w = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=w)
    model.fc = torch.nn.Identity()
    model.eval()
    tf = w.transforms()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])
            out.append(model(x).numpy())
            print(f"  features {min(i + batch, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(out)


def _fit(x, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, class_weight="balanced")).fit(x, y)


def run(cluster: Path, photos: Path) -> None:
    import runlog
    from sklearn.model_selection import StratifiedKFold
    sel = pd.read_csv(photos / "selected.csv")
    sel["file"] = [photos / "images" / p for p in sel.path]
    missing = [str(f) for f in sel.file if not f.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} photos missing, e.g. {missing[:2]}; run fetch")
    scans = pd.DataFrame([{"fabric": c, "material_id": m, "crop": k,
                           "file": cluster / "images" / "crop_images" / f"img{m}" / f"img{m}_{k}.jpg"}
                          for c, ids in CLUSTER.items() for m in ids for k in range(1, 101)])
    print(f"{len(scans)} scan crops, {len(sel)} photos {sel.fabric.value_counts().to_dict()}", flush=True)
    xs, xp = _features([str(f) for f in scans.file]), _features([str(f) for f in sel.file])
    train = scans.crop.isin(TRAIN_CROPS).to_numpy()
    test = scans.crop.isin(TEST_CROPS).to_numpy()
    model = _fit(xs[train], scans.fabric[train])
    scan_pred = model.predict(xs[test])
    photo_pred = model.predict(xp)
    ys, yp = scans.fabric[test].to_numpy(), sel.fabric.to_numpy()
    ba_scan, ba_photo = balanced_accuracy(ys, scan_pred), balanced_accuracy(yp, photo_pred)
    b_scan = bootstrap(ys, scan_pred == ys, seed=1)
    b_photo = bootstrap(yp, photo_pred == yp, seed=2)
    p = float(np.mean(b_photo >= b_scan))
    # descriptive: reverse direction and within-photo reference
    rev = _fit(xp, yp).predict(xs)
    ba_rev = balanced_accuracy(scans.fabric, rev)
    cv = np.empty(len(yp), dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(xp, yp):
        cv[te] = _fit(xp[tr], yp[tr]).predict(xp[te])
    ba_cv = balanced_accuracy(yp, cv)
    conf = lambda y, pr: pd.crosstab(pd.Series(y, name="true"), pd.Series(pr, name="pred"))
    metrics = {
        "preregistration": "PREREGISTRATION.md: Round 3, G (commit 2b1b3c7)",
        "classes": list(CLASSES), "chance": 1 / 3,
        "n_scan_test": int(test.sum()), "n_photos": len(sel),
        "photos_per_class": sel.fabric.value_counts().to_dict(),
        "scan_balanced_accuracy": ba_scan,
        "scan_ci95": np.percentile(b_scan, [2.5, 97.5]).tolist(),
        "photo_balanced_accuracy": ba_photo,
        "photo_ci95": np.percentile(b_photo, [2.5, 97.5]).tolist(),
        "gap": ba_scan - ba_photo,
        "test": {"p_bootstrap": p, "significant": bool(p < 0.05),
                 "predicted": "photo < scan", "prediction_held": bool(p < 0.05)},
        "photo_recall": {c: float((photo_pred[yp == c] == c).mean()) for c in CLASSES},
        "scan_recall": {c: float((scan_pred[ys == c] == c).mean()) for c in CLASSES},
        "descriptive": {"photo_to_scan_balanced_accuracy": ba_rev,
                        "photo_cv_balanced_accuracy": ba_cv},
    }
    print(f"[G] scans {ba_scan:.3f}  photos {ba_photo:.3f}  gap {ba_scan - ba_photo:+.3f}  "
          f"p {p:.4f} | reverse {ba_rev:.3f}  photo CV {ba_cv:.3f}")
    print("photo confusion\n", conf(yp, photo_pred))
    saved = runlog.save_run(
        "photo_transfer",
        {"mirror": MIRROR, "weights": "IMAGENET1K_V2", "per_class": PER_CLASS, "seed": 0,
         "cluster_ids": {c: list(v) for c, v in CLUSTER.items()}},
        metrics,
        {"photos": sel.drop(columns="file").assign(pred=photo_pred),
         "confusion_photo": conf(yp, photo_pred).reset_index(),
         "confusion_scan": conf(ys, scan_pred).reset_index()},
        synthetic=False)
    print("saved", saved)


def fetch(out: Path) -> None:
    sel = pd.read_csv(out / "selected.csv")
    (out / "images").mkdir(parents=True, exist_ok=True)
    todo = [p for p in sel.path if not (out / "images" / p).exists()]
    print(f"{len(todo)} of {len(sel)} photos to fetch", flush=True)
    for k, p in enumerate(todo, 1):
        subprocess.run(["kaggle", "datasets", "download", MIRROR, "-f", f"selected_images/{p}",
                        "-p", str(out / "images"), "-q"], check=True, capture_output=True)
        z = out / "images" / (p + ".zip")
        if z.exists():
            import zipfile
            with zipfile.ZipFile(z) as f:
                f.extractall(out / "images")
            z.unlink()
        if k % 25 == 0:
            print(f"  {k}/{len(todo)}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_photo_transfer")
    ap.add_argument("step", choices=("select", "fetch", "run"))
    ap.add_argument("--labels")
    ap.add_argument("--out", default="data/deepfashion_mm")
    ap.add_argument("--cluster", default="data/cluster")
    ap.add_argument("--photos", default="data/deepfashion_mm")
    a = ap.parse_args(argv)
    if a.step == "select":
        sel = select(pd.read_csv(a.labels))
        Path(a.out).mkdir(parents=True, exist_ok=True)
        sel.to_csv(Path(a.out) / "selected.csv", index=False)
        print(sel.fabric.value_counts().to_dict())
    elif a.step == "fetch":
        fetch(Path(a.out))
    else:
        run(Path(a.cluster), Path(a.photos))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
