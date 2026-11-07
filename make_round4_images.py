"""Round 4, analyses J (stricter scan -> photo transfer, CLIP zero-shot) and L
(image -> category -> descriptor at 118-material scale).

    python make_round4_images.py features          # ResNet-50 features of all crops (cached)
    python make_round4_images.py j
    python make_round4_images.py l

Plan: PREREGISTRATION.md, "Round 4" (commit a86cb06). Saves runs
`photo_transfer_strict` and `image_to_descriptor`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import make_photo_transfer as G
import runlog
from haptic import loader as L
from haptic.separability import holm_adjust

CLUSTER = Path("data/cluster")
PHOTOS = Path("data/deepfashion_mm")
CACHE = Path("data/derived/round4")
CLIP_MODEL = "openai/clip-vit-base-patch32"
G_RUN = "20261107T144009_695846Z"


def crop_table() -> pd.DataFrame:
    return pd.DataFrame([{"material_id": m, "category": L.category_of(m), "crop": k,
                          "file": str(CLUSTER / "images" / "crop_images" / f"img{m}" / f"img{m}_{k}.jpg")}
                         for m in range(118) for k in range(1, 101)])


def resnet_crops() -> tuple[pd.DataFrame, np.ndarray]:
    tab = crop_table()
    f = CACHE / "crops_resnet50_v2.npy"
    if not f.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(f, G._features(tab.file.tolist()).astype(np.float32))
    return tab, np.load(f)


def resnet_photos(sel: pd.DataFrame) -> np.ndarray:
    f = CACHE / "photos_resnet50_v2.npy"
    if not f.exists():
        np.save(f, G._features([str(PHOTOS / "images" / p) for p in sel.path]).astype(np.float32))
    return np.load(f)


def clip_scores(files, prompts) -> np.ndarray:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_MODEL).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    out, emb = [], []
    with torch.no_grad():
        t = proc(text=prompts, return_tensors="pt", padding=True)
        text = model.get_text_features(**t)
        text = text / text.norm(dim=-1, keepdim=True)
        for i in range(0, len(files), 32):
            im = proc(images=[Image.open(p).convert("RGB") for p in files[i:i + 32]], return_tensors="pt")
            e = model.get_image_features(**im)
            e = e / e.norm(dim=-1, keepdim=True)
            emb.append(e.numpy())
            out.append((e @ text.T).numpy())
            print(f"  clip {min(i + 32, len(files))}/{len(files)}", flush=True)
    return np.concatenate(out), np.concatenate(emb)


def _boot_less(y_a, hit_a, y_b, hit_b, seed):
    """p for 'balanced accuracy of a < b': share of resamples with a >= b."""
    a = G.bootstrap(y_a, hit_a, seed=seed)
    b = G.bootstrap(y_b, hit_b, seed=seed + 1)
    return float(np.mean(a >= b)), np.percentile(a, [2.5, 97.5]).tolist(), np.percentile(b, [2.5, 97.5]).tolist()


def analysis_j() -> None:
    g = runlog.load_run("photo_transfer")
    if g.info["run_id"] != G_RUN:
        raise SystemExit(f"pre-registration names photo_transfer/{G_RUN}")
    sel = pd.read_csv(PHOTOS / "selected.csv")
    tab, xs = resnet_crops()
    xp = resnet_photos(sel)
    fab = {m: c for c, ids in G.CLUSTER.items() for m in ids}
    tab["fabric"] = tab.material_id.map(fab)

    # J1: cotton vs leather, leave one material out
    two = tab.fabric.isin(["cotton", "leather"]).to_numpy()
    t2, x2 = tab[two].reset_index(drop=True), xs[two]
    lomo = np.empty(len(t2), dtype=object)
    for m in t2.material_id.unique():
        test = (t2.material_id == m).to_numpy()
        lomo[test] = G._fit(x2[~test], t2.fabric[~test]).predict(x2[test])
    # one vote per material, so 100 crops of one swatch do not count 100 times
    per_mat = pd.DataFrame({"m": t2.material_id, "y": t2.fabric, "hit": lomo == t2.fabric})
    mat = per_mat.groupby("m").agg(y=("y", "first"), hit=("hit", "mean"))
    ba_scan_crop = G.balanced_accuracy(t2.fabric, lomo)
    psel = sel.fabric.isin(["cotton", "leather"]).to_numpy()
    ph_pred = G._fit(x2, t2.fabric).predict(xp[psel])
    yp = sel.fabric[psel].to_numpy()
    ba_photo = G.balanced_accuracy(yp, ph_pred)
    p1, ci_photo, ci_scan = _boot_less(yp, ph_pred == yp, t2.fabric.to_numpy(), lomo == t2.fabric, seed=10)

    # J2: CLIP zero-shot on all scans of the three fabrics and on the photos
    three = tab.fabric.notna().to_numpy()
    t3 = tab[three].reset_index(drop=True)
    prompts = [f"a photo of {c} fabric" for c in G.CLASSES]
    s_scan, e_scan = clip_scores(t3.file.tolist(), prompts)
    s_photo, e_photo = clip_scores([str(PHOTOS / "images" / p) for p in sel.path], prompts)
    cls = np.array(G.CLASSES)
    zs_scan, zs_photo = cls[s_scan.argmax(1)], cls[s_photo.argmax(1)]
    y3, y_ph = t3.fabric.to_numpy(), sel.fabric.to_numpy()
    ba_zs_scan, ba_zs_photo = G.balanced_accuracy(y3, zs_scan), G.balanced_accuracy(y_ph, zs_photo)
    # prediction: photo > scan, i.e. scan < photo
    p2, ci_zs_scan, ci_zs_photo = _boot_less(y3, zs_scan == y3, y_ph, zs_photo == y_ph, seed=20)
    holm = holm_adjust(np.array([p1, p2]))
    # descriptive: CLIP linear probe, scans (crops 1-70) -> photos
    train = t3.crop.isin(G.TRAIN_CROPS).to_numpy()
    probe = G._fit(e_scan[train], y3[train])
    ba_probe_photo = G.balanced_accuracy(y_ph, probe.predict(e_photo))
    ba_probe_scan = G.balanced_accuracy(y3[~train], probe.predict(e_scan[~train]))

    metrics = {
        "preregistration": "PREREGISTRATION.md: Round 4, J (commit a86cb06)",
        "j1_lomo": {"classes": ["cotton", "leather"], "scan_balanced_accuracy": ba_scan_crop,
                    "scan_ci95": ci_scan, "scan_per_material_hit": mat.hit.round(3).to_dict(),
                    "photo_balanced_accuracy": ba_photo, "photo_ci95": ci_photo,
                    "n_photos": int(psel.sum()), "p": p1, "holm_p": float(holm[0]),
                    "predicted": "photo < LOMO scan", "prediction_held": bool(holm[0] < 0.05)},
        "j2_clip_zero_shot": {"model": CLIP_MODEL, "prompts": prompts,
                              "scan_balanced_accuracy": ba_zs_scan, "scan_ci95": ci_zs_scan,
                              "photo_balanced_accuracy": ba_zs_photo, "photo_ci95": ci_zs_photo,
                              "scan_recall": {c: float((zs_scan[y3 == c] == c).mean()) for c in cls},
                              "photo_recall": {c: float((zs_photo[y_ph == c] == c).mean()) for c in cls},
                              "p": p2, "holm_p": float(holm[1]),
                              "predicted": "photo > scan", "prediction_held": bool(holm[1] < 0.05)},
        "descriptive_clip_probe": {"scan_test_balanced_accuracy": ba_probe_scan,
                                   "photo_balanced_accuracy": ba_probe_photo},
    }
    print(f"[J1] LOMO scans {ba_scan_crop:.3f}  photos {ba_photo:.3f}  p {p1:.4f} holm {holm[0]:.4f}")
    print(f"[J2] CLIP zero-shot scans {ba_zs_scan:.3f}  photos {ba_zs_photo:.3f}  p {p2:.4f} holm {holm[1]:.4f}")
    print(f"     CLIP probe scans->scans {ba_probe_scan:.3f}  scans->photos {ba_probe_photo:.3f}")
    run = runlog.save_run("photo_transfer_strict", {"g_run": G_RUN, "clip": CLIP_MODEL}, metrics,
                          {"clip_photos": sel.assign(zero_shot=zs_photo),
                           "lomo_per_material": mat.reset_index()}, synthetic=False)
    print("saved", run)


DIMS = {"log_frequency": ("frequency_hz", True), "log_intensity": ("intensity", True),
        "pattern": ("pattern", False), "log_duration": ("duration_ms", True)}


def analysis_l(features: str) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(features)
    df = df[df.in_band.astype(bool)]
    for k, (c, lg) in DIMS.items():
        df[k] = np.log(df[c]) if lg else df[c]
    desc = df.groupby("material_id")[list(DIMS)].mean()
    cat = pd.Series({m: L.category_of(m) for m in desc.index})
    tab, xs = resnet_crops()
    rows = []
    for m in desc.index:
        test = (tab.material_id == m).to_numpy()
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
        clf.fit(xs[~test], tab.category[~test])
        votes = pd.Series(clf.predict(xs[test])).value_counts()
        pred_cat = votes.index[0]
        others = desc.drop(index=m)
        oc = cat.drop(index=m)
        row = {"material_id": m, "category": cat[m], "pred_category": pred_cat,
               "vote_share": float(votes.iloc[0] / votes.sum())}
        for k in DIMS:
            truth = desc.loc[m, k]
            row[f"{k}_err_image"] = abs(others.loc[oc == pred_cat, k].mean() - truth)
            row[f"{k}_err_global"] = abs(others[k].mean() - truth)
            same = others.loc[oc == cat[m], k]
            row[f"{k}_err_oracle"] = abs(same.mean() - truth) if len(same) else np.nan
        rows.append(row)
        print(f"  material {m}: {cat[m]} -> {pred_cat}", flush=True)
    res = pd.DataFrame(rows)
    p = [float(wilcoxon(res[f"{k}_err_global"] - res[f"{k}_err_image"], alternative="greater").pvalue)
         for k in DIMS]
    holm = holm_adjust(np.array(p))
    tests = {}
    for (k, pi), h in zip(zip(DIMS, p), holm):
        tests[k] = {"mae_image": float(res[f"{k}_err_image"].mean()),
                    "mae_global": float(res[f"{k}_err_global"].mean()),
                    "mae_oracle": float(res[f"{k}_err_oracle"].mean()),
                    "n_image_better": int((res[f"{k}_err_image"] < res[f"{k}_err_global"]).sum()),
                    "p": pi, "holm_p": float(h), "predicted": "image < global",
                    "prediction_held": bool(h < 0.05)}
        t = tests[k]
        print(f"[L] {k:14s} MAE image {t['mae_image']:.3f} global {t['mae_global']:.3f} "
              f"oracle {t['mae_oracle']:.3f}  better {t['n_image_better']}/118  p {pi:.3g} holm {h:.3g}")
    cat_acc = float((res.category == res.pred_category).mean())
    per_cat = res.groupby("category").apply(lambda d: float((d.category == d.pred_category).mean())).to_dict()
    print(f"[L] category accuracy from images, leave-one-material-out: {cat_acc:.3f}")
    metrics = {"preregistration": "PREREGISTRATION.md: Round 4, L (commit a86cb06)",
               "n_materials": len(res), "category_accuracy": cat_acc,
               "category_accuracy_by_category": per_cat,
               "majority_category_share": float(cat.value_counts().iloc[0] / len(cat)),
               "tests": tests}
    run = runlog.save_run("image_to_descriptor", {"features": features, "weights": "IMAGENET1K_V2"},
                          metrics, {"per_material": res}, synthetic=False)
    print("saved", run)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_round4_images")
    ap.add_argument("step", choices=("features", "j", "l"))
    ap.add_argument("--features", default="data/round2_features/features.csv")
    a = ap.parse_args(argv)
    if a.step == "features":
        resnet_crops()
        resnet_photos(pd.read_csv(PHOTOS / "selected.csv"))
    elif a.step == "j":
        analysis_j()
    else:
        analysis_l(a.features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
