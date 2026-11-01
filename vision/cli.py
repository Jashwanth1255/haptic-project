"""Objective 2 command line.

    python -m vision.cli check --dataset dtd --data <path>      verify the splits
    python -m vision.cli train --dataset dtd --data <path>      fine-tune and report
    python -m vision.cli textile --data <path> --checkpoint <path>   12-way DTD scores

Re-running `train` with the same --out resumes from the last finished epoch.
A run given --limit-per-class is a smoke run and its numbers are not results.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import data as V

# Few images per class on DTD means many more epochs than MINC for the same
# number of updates.
DEFAULT_EPOCHS = {"dtd": 30, "minc2500": 15}


def _limit(items: list, n: int) -> list:
    if not n:
        return items
    kept, seen = [], {}
    for path, y in items:
        seen[y] = seen.get(y, 0) + 1
        if seen[y] <= n:
            kept.append((path, y))
    return kept


def _textile(a, test_items, classes, seen_items) -> int:
    import hashlib
    import json

    import pandas as pd
    import torch

    from . import textile as X
    from . import train as T

    path = Path(a.checkpoint)
    if path.is_dir():
        found = sorted(path.rglob("best.pt"))
        if len(found) != 1:
            raise RuntimeError(f"expected one best.pt under {path}, found {found}")
        path = found[0]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["classes"] != list(classes):
        raise RuntimeError(f"{path} was trained on a different class list")
    if ck["config"]["limit_per_class"]:
        raise RuntimeError(f"{path} is a smoke run")
    if ck["config"]["fold"] != a.fold:
        # Folds resplit one image pool: another fold's model trained on this
        # fold's test images, and would score them as if unseen.
        raise RuntimeError(f"{path} was trained on fold {ck['config']['fold']}; "
                           f"scoring it on fold {a.fold} would leak test images")

    model = T.build_model(len(classes), pretrained=False)
    model.load_state_dict(ck["model"])
    keep = set(X.subset_indices(classes))
    items = [it for it in test_items if it[1] in keep]
    ds = V.ImageList(items, V.transforms_for(train=False))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logits, y = X.logits_of(model, ds, device, num_workers=a.workers)
    s = X.score_subset(logits, y, classes)
    r, f = s["restricted"], s["filtered"]
    unseen = X.unseen_mask(items, seen_items)
    u = X.score_subset(logits[torch.from_numpy(unseen)], y[unseen],
                       classes)["restricted"] if unseen.any() else None

    print(f"checkpoint {path}\n  sha256 {digest}\n")
    print(f"unrestricted (47-way)  n={s['n']}  accuracy "
          f"{s['unrestricted_accuracy']:.4f}")
    print(f"filtered               n={f.n}  accuracy {f.top1:.4f}  "
          f"macro F1 {f.macro_f1:.4f}")
    print(f"restricted (12-way)    n={r.n}  accuracy {r.top1:.4f}  "
          f"macro F1 {r.macro_f1:.4f}")
    if u is not None:
        print(f"restricted, unseen     n={u.n}  accuracy {u.top1:.4f}  "
              f"macro F1 {u.macro_f1:.4f}  (outside fold 1's test split)")
    print()
    print(r.per_class.round(3).to_string(index=False))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "textile_metrics.json").write_text(json.dumps({
        "checkpoint_sha256": digest, "best_epoch": ck["epoch"], "n": s["n"],
        "n_predicted_outside": s["n_predicted_outside"],
        "unrestricted_accuracy": s["unrestricted_accuracy"],
        "fold": a.fold, "filtered": f.as_dict(), "restricted": r.as_dict(),
        "restricted_unseen": u.as_dict() if u is not None else None,
    }, indent=2))
    r.per_class.to_csv(out / "restricted_per_class.csv", index=False)
    pd.DataFrame(r.confusion, index=r.classes,
                 columns=r.classes).to_csv(out / "restricted_confusion.csv")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vision.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("check", help="verify the published splits on disk")
    pt = sub.add_parser("train", help="fine-tune and report on the test split")
    px = sub.add_parser("textile", help="score DTD's textile subset, masked to 12 ways")
    for p in (pc, pt, px):
        p.add_argument("--dataset", default="dtd" if p is px else None,
                       required=p is not px, choices=sorted(V.SPECS))
        p.add_argument("--data", required=True, help="dataset root or any parent")
        p.add_argument("--fold", type=int, default=1)

    px.add_argument("--checkpoint", required=True,
                    help="best.pt, or a directory holding exactly one")
    px.add_argument("--out", default="runs/textile")
    px.add_argument("--workers", type=int, default=2)

    pt.add_argument("--out", default="runs")
    pt.add_argument("--epochs", type=int)
    pt.add_argument("--freeze-epochs", type=int, default=3)
    pt.add_argument("--batch-size", type=int, default=64)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--workers", type=int, default=2)
    pt.add_argument("--limit-per-class", type=int, default=0)
    pt.add_argument("--no-pretrained", action="store_true")
    pt.add_argument("--require-gpu", action="store_true",
                    help="fail at once rather than fall back to the CPU")
    pt.add_argument("--resume-from",
                    help="directory holding last.pt from an earlier session")

    a = ap.parse_args(argv)
    spec = V.SPECS[a.dataset]
    root = V.find_root(a.data, spec)
    splits = V.load_splits(root, spec, a.fold)
    classes = V.class_names(root)
    sizes = {k: len(v) for k, v in splits.items()}
    print(f"{spec.name} fold {a.fold}: {len(classes)} classes, {sizes}  ({root})")
    if a.cmd == "check":
        return 0
    if a.cmd == "textile":
        return _textile(a, splits["test"], classes,
                        V.read_split(root, spec, "test", fold=1))

    import torch

    from . import train as T

    if a.require_gpu and not torch.cuda.is_available():
        raise RuntimeError("--require-gpu given but no GPU is visible to torch")

    cfg = T.TrainConfig(dataset=spec.name, fold=a.fold,
                        epochs=a.epochs or DEFAULT_EPOCHS[spec.name],
                        freeze_epochs=a.freeze_epochs, batch_size=a.batch_size,
                        seed=a.seed, pretrained=not a.no_pretrained,
                        limit_per_class=a.limit_per_class)
    if cfg.is_smoke:
        print("SMOKE RUN -- subset of the data, not a result.\n", file=sys.stderr)

    tag = f"{spec.name}_fold{a.fold}_seed{a.seed}" + ("_smoke" if cfg.is_smoke else "")
    out = Path(a.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    if a.resume_from and not (out / "last.pt").exists():
        for name in ("last.pt", "best.pt"):
            shutil.copy(Path(a.resume_from) / name, out / name)

    train_ds, val_ds, test_ds = (
        V.ImageList(_limit(splits[k], a.limit_per_class),
                    V.transforms_for(train=k == "train"))
        for k in ("train", "val", "test"))

    model = T.build_model(len(classes), cfg.pretrained)
    fit = T.fit(model, train_ds, val_ds, classes, cfg, out, num_workers=a.workers)
    report = T.test_report(model, test_ds, classes, cfg, out, num_workers=a.workers)

    print(f"\n### {spec.name} fold {a.fold} -- test split"
          f"{' (SMOKE, not a result)' if cfg.is_smoke else ''}")
    print(f"  best epoch : {fit['best_epoch'] + 1} of {cfg.epochs}, "
          f"by validation macro F1 ({fit['best_val_macro_f1']:.4f})")
    print(report.summary())
    print("\nmost confused:")
    print(report.most_confused().to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
