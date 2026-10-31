"""Objective 2 command line.

    python -m vision.cli check --dataset dtd --data <path>      verify the splits
    python -m vision.cli train --dataset dtd --data <path>      fine-tune and report

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vision.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("check", help="verify the published splits on disk")
    pt = sub.add_parser("train", help="fine-tune and report on the test split")
    for p in (pc, pt):
        p.add_argument("--dataset", required=True, choices=sorted(V.SPECS))
        p.add_argument("--data", required=True, help="dataset root or any parent")
        p.add_argument("--fold", type=int, default=1)

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
