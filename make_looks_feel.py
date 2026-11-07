"""Pre-registered analysis A: does looks predict feel? (118 materials)

    python make_looks_feel.py --root data/cluster

PREREGISTRATION.md, "Benchmark and findings round 2", section A. Visual distance:
cosine between per-material mean ImageNet ResNet-50 features of the dataset's 100
crops. Haptic distance (primary): Euclidean between per-material mean log spectra
of the in-band recordings in the velocity_all_materials subset (0 and 90 degrees,
first repeat). Three one-sided Spearman Mantel tests, Holm across them:
all 118 free; all 118 within category; the 16 cloth materials. Saves `looks_feel`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import runlog
from haptic import identity as I
from haptic import loader as L
from haptic import rsa

N_PERM = 9999
PREDICTED = {"all_free": True, "all_within_category": False, "cloth_only": False}


def visual_features(root: Path, ids, batch: int = 50) -> np.ndarray:
    import torch
    from PIL import Image
    from torchvision.models import resnet50

    from vision.data import transforms_for
    from vision.train import WEIGHTS
    model = resnet50(weights=WEIGHTS)
    model.fc = torch.nn.Identity()
    model.eval()
    tf = transforms_for(train=False)
    out = []
    for mid in ids:
        paths = sorted(Path(root, "images", "crop_images", f"img{mid}").glob("*.jpg"))
        if len(paths) != 100:
            raise SystemExit(f"material {mid}: {len(paths)} crops, expected 100")
        feats = []
        with torch.no_grad():
            for b in range(0, len(paths), batch):
                x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[b:b + batch]])
                feats.append(model(x).numpy())
        out.append(np.concatenate(feats).mean(0))
        if mid % 20 == 0:
            print(f"  visual features up to material {mid}", flush=True)
    return np.array(out)


def haptic_tables(root: Path):
    import make_velocity_all as VA
    df, skipped, spectra = VA.descriptors(root)
    keep = df["in_band"].to_numpy(bool)
    df, spectra = df[keep].reset_index(drop=True), spectra[keep]
    ids = np.sort(df.material_id.unique())
    spec = np.array([spectra[(df.material_id == m).to_numpy()].mean(0) for m in ids])
    x = I._matrix(df, I.RAW4)                       # logs positive dimensions, as identify does
    x = (x - x.mean(0)) / x.std(0)
    desc = np.array([x[(df.material_id == m).to_numpy()].mean(0) for m in ids])
    return ids, spec, desc, len(df), len(skipped)


def holm(ps):
    order = np.argsort(ps)
    out, run = np.empty(len(ps)), 0.0
    for rank, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[k]))
        out[k] = run
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_looks_feel")
    ap.add_argument("--root", default="data/cluster")
    ap.add_argument("--perm", type=int, default=N_PERM)
    a = ap.parse_args(argv)
    root = Path(a.root)

    ids, spec, desc, n_rec, n_skip = haptic_tables(root)
    if len(ids) != 118:
        raise SystemExit(f"haptic side has {len(ids)} materials, expected 118")
    print(f"haptic: {n_rec} in-band recordings, {n_skip} skipped", flush=True)
    vis = visual_features(root, ids)
    cat = np.array([L.category_of(int(m)) for m in ids])
    cloth = np.isin(ids, L.CLOTH_IDS)

    dv = rsa.cosine_distance(vis)
    dh = rsa.euclidean_distance(spec)
    dd = rsa.euclidean_distance(desc)
    tests = {
        "all_free": rsa.mantel(dv, dh, a.perm),
        "all_within_category": rsa.mantel(dv, dh, a.perm, strata=cat),
        "cloth_only": rsa.mantel(dv[np.ix_(cloth, cloth)], dh[np.ix_(cloth, cloth)], a.perm),
    }
    adj = holm(np.array([t["p"] for t in tests.values()]))
    for (name, t), p in zip(tests.items(), adj):
        t["holm_p"] = float(p)
        t["significant"] = bool(p < 0.05)
        t["predicted_significant"] = PREDICTED[name]
        t["prediction_held"] = t["significant"] == PREDICTED[name]
        print(f"{name:22s} r = {t['r']:+.3f}  p = {t['p']:.4f}  Holm {p:.4f}  "
              f"predicted {'sig' if PREDICTED[name] else 'n.s.'} -> "
              f"{'held' if t['prediction_held'] else 'FAILED'}", flush=True)

    secondary = {  # descriptive, not tested against a prediction
        "descriptor_all_free": rsa.mantel(dv, dd, a.perm),
        "descriptor_cloth_only": rsa.mantel(dv[np.ix_(cloth, cloth)], dd[np.ix_(cloth, cloth)], a.perm),
        "spectrum_vs_descriptor_all": rsa.mantel(dh, dd, a.perm),
    }
    for name, t in secondary.items():
        print(f"(secondary) {name:28s} r = {t['r']:+.3f}  p = {t['p']:.4f}")

    i, j = np.triu_indices(len(ids), 1)
    pairs = pd.DataFrame({"a": ids[i], "b": ids[j], "category_a": cat[i], "category_b": cat[j],
                          "visual_cosine": dv[i, j], "haptic_spectrum": dh[i, j],
                          "haptic_descriptor": dd[i, j]})
    metrics = {"preregistration": "PREREGISTRATION.md: Benchmark and findings round 2, A",
               "n_materials": int(len(ids)), "n_recordings": n_rec, "n_skipped": n_skip,
               "tests": tests, "secondary": secondary,
               "all_predictions_held": all(t["prediction_held"] for t in tests.values())}
    path = runlog.save_run("looks_feel", {**vars(a), "visual": "resnet50 IMAGENET1K_V2, 100 crops"},
                           metrics, {"pairs": pairs}, synthetic=False)
    print(f"saved run {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
