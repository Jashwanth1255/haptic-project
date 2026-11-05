"""The trained classifiers on Cluster's 300 dpi cloth scans: the scan side of O2.

    python -m vision.cluster_scans --root data/cluster --models models \
        --out data/derived/cluster_crops.npz

Each of the 16 cloth materials has 100 crops of 224x224 px cut from its scan
(`images/crop_images/img<id>/`). Two questions, both answerable without labelling
anything by hand:

- **Material (MINC-2500).** Every crop is cloth, so the right MINC class is
  "fabric" for all 1,600. Reported as the fraction called fabric, overall and per
  material, with what the rest were called.
- **Texture (DTD, 10 folds).** Each crop is classified 12-way over the registered
  textile subset, and the class is mapped to the registered group (pile, weave,
  drape, or neither for frilly and lacelike). Group accuracy is scored against the
  assignment in PREREGISTRATION.md, which was recorded before this ran. The DTD
  test accuracy cannot be compared directly: that is 12-way class accuracy, this
  is 4-way group accuracy against a label made by eye from the name and scan.

Per-crop softmax probabilities and the penultimate (2048-d) ResNet-50 features
are saved for Objective 3, which needs one view per crop.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from haptic.groups import ASSIGNMENT

from .textile import TEXTILE_CLASSES

GROUP_OF_CLASS = {
    "fibrous": "pile", "matted": "pile",
    "braided": "weave", "woven": "weave", "interlaced": "weave", "meshed": "weave",
    "knitted": "weave",
    "gauzy": "drape", "pleated": "drape", "wrinkled": "drape",
    "frilly": "neither", "lacelike": "neither",
}
CROP = re.compile(r"img(\d+)_(\d+)\.(?:jpg|png)$", re.I)


def crop_items(root: str | Path, material_ids=tuple(ASSIGNMENT)) -> pd.DataFrame:
    """One row per crop: path, material_id, crop number. Sorted, so views line up."""
    rows = []
    for p in Path(root, "images", "crop_images").rglob("*"):
        m = CROP.search(p.name)
        if m and int(m.group(1)) in material_ids:
            rows.append({"path": p, "material_id": int(m.group(1)), "crop": int(m.group(2))})
    if not rows:
        raise FileNotFoundError(f"no crop images under {root}/images/crop_images")
    return pd.DataFrame(rows).sort_values(["material_id", "crop"]).reset_index(drop=True)


def _run(checkpoint: Path, paths, batch_size: int = 64):
    """(class list, logits, penultimate features) for every image."""
    import torch

    from . import data as V
    from . import train as T
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    classes = list(ck["classes"])
    model = T.build_model(len(classes), pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()
    fc, model.fc = model.fc, torch.nn.Identity()
    ds = V.ImageList([(p, 0) for p in paths], V.transforms_for(train=False))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    feats = []
    with torch.no_grad():
        for x, _ in loader:
            feats.append(model(x))
    feats = torch.cat(feats)
    with torch.no_grad():
        logits = fc(feats)
    return classes, logits.numpy(), feats.numpy()


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def textile_probs(logits: np.ndarray, classes) -> np.ndarray:
    """Softmax over the 12 registered textile classes only, in TEXTILE_CLASSES order."""
    keep = [list(classes).index(c) for c in TEXTILE_CLASSES]
    return softmax(logits[:, keep])


def minc_summary(items: pd.DataFrame, probs: np.ndarray, classes) -> tuple[dict, pd.DataFrame]:
    pred = np.array(classes)[probs.argmax(1)]
    df = items.assign(pred=pred)
    per = df.groupby("material_id").agg(
        fabric=("pred", lambda s: float((s == "fabric").mean())),
        top_other=("pred", lambda s: s[s != "fabric"].value_counts().index[0]
                   if (s != "fabric").any() else ""))
    overall = {"fabric_fraction": float((pred == "fabric").mean()),
               "n_crops": len(pred),
               "called": df.pred.value_counts().to_dict(),
               "materials_majority_fabric": int((per.fabric > 0.5).sum())}
    return overall, per.reset_index()


def dtd_summary(items: pd.DataFrame, probs_by_fold: dict[int, np.ndarray]):
    """Per-fold group accuracy (crops, and per-material majority), and the
    per-material distribution of predicted classes pooled over folds."""
    truth = items.material_id.map(ASSIGNMENT).to_numpy()
    assigned = truth != "neither"
    folds, dist = [], []
    for fold, p in sorted(probs_by_fold.items()):
        cls = np.array(TEXTILE_CLASSES)[p.argmax(1)]
        grp = np.array([GROUP_OF_CLASS[c] for c in cls])
        maj = (pd.Series(grp).groupby(items.material_id.to_numpy())
               .agg(lambda s: s.value_counts().index[0]))
        mtruth = maj.index.map(ASSIGNMENT).to_numpy()
        folds.append({"fold": fold,
                      "crop_group_acc_all16": float((grp == truth).mean()),
                      "crop_group_acc_assigned13": float((grp[assigned] == truth[assigned]).mean()),
                      "material_majority_acc_all16": float((maj.to_numpy() == mtruth).mean())})
        dist.append(pd.DataFrame({"material_id": items.material_id, "fold": fold,
                                  "class": cls, "group": grp}))
    folds = pd.DataFrame(folds)
    dist = pd.concat(dist)
    by_class = (dist.groupby("material_id")["class"].value_counts(normalize=True)
                .unstack(fill_value=0.0).reindex(columns=list(TEXTILE_CLASSES), fill_value=0.0))
    by_group = (dist.groupby("material_id")["group"].value_counts(normalize=True)
                .unstack(fill_value=0.0))
    by_group.insert(0, "assigned", by_group.index.map(ASSIGNMENT))
    # The chance a label-blind guesser gets, always saying the commonest group
    counts = pd.Series(truth).value_counts(normalize=True)
    return folds, by_class.reset_index(), by_group.reset_index(), float(counts.iloc[0])


def main(argv=None) -> int:
    import runlog
    ap = argparse.ArgumentParser(prog="vision.cluster_scans")
    ap.add_argument("--root", required=True)
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="data/derived/cluster_crops.npz")
    a = ap.parse_args(argv)

    items = crop_items(a.root)
    print(f"{len(items)} crops from {items.material_id.nunique()} materials", flush=True)
    paths = items.path.tolist()
    minc_ck = Path(a.models, "minc2500_fold1_seed0", "best.pt")
    minc_classes, minc_logits, minc_feat = _run(minc_ck, paths)
    print("MINC done", flush=True)
    dtd, dtd_feat = {}, None
    for ck in sorted(Path(a.models).glob("dtd_fold*_seed0/best.pt")):
        fold = int(re.search(r"fold(\d+)", ck.parent.name).group(1))
        classes, logits, feat = _run(ck, paths)
        dtd[fold] = textile_probs(logits, classes)
        if fold == 1:
            dtd_feat = feat
        print(f"DTD fold {fold} done", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, material_id=items.material_id.to_numpy(),
                        crop=items.crop.to_numpy(),
                        minc_classes=np.array(minc_classes), minc_probs=softmax(minc_logits),
                        minc_features=minc_feat.astype(np.float32),
                        textile_classes=np.array(TEXTILE_CLASSES),
                        dtd_folds=np.array(sorted(dtd)),
                        dtd_probs=np.stack([dtd[f] for f in sorted(dtd)]),
                        dtd_features=dtd_feat.astype(np.float32))

    minc_overall, minc_per = minc_summary(items, softmax(minc_logits), minc_classes)
    folds, by_class, by_group, majority = dtd_summary(items, dtd)
    metrics = {"minc": minc_overall,
               "dtd_group": {c: {"mean": float(folds[c].mean()), "std": float(folds[c].std(ddof=1)),
                                 "n_folds": len(folds)}
                             for c in folds.columns if c != "fold"},
               "majority_group_baseline_all16": majority}
    path = runlog.save_run("o2_cluster_scans", vars(a), metrics,
                           {"minc_per_material": minc_per, "dtd_group_per_fold": folds,
                            "dtd_class_by_material": by_class,
                            "dtd_group_by_material": by_group},
                           synthetic=False)
    pd.set_option("display.width", 200)
    print(f"\nMINC: {minc_overall['fabric_fraction']:.3f} of crops called fabric; "
          f"{minc_overall['materials_majority_fabric']}/16 materials by majority")
    print(minc_per.to_string(index=False))
    print("\nDTD group accuracy over folds:")
    print(folds.describe().loc[["mean", "std"]].round(3).to_string())
    print(f"always-weave baseline: {majority:.3f}")
    print(by_group.round(2).to_string(index=False))
    print(f"\nsaved {out} and run {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
