"""Fine-tune an ImageNet-pretrained ResNet-50, resumably.

Two rules hold this file together.

**The test set is not reachable from `fit`.** The best epoch is chosen on
validation macro F1, and `test_report` is a separate call made once, afterwards,
on the saved best weights. Choosing the epoch on test scores would inflate the
number the paper reports.

**An interrupted run must equal an uninterrupted one.** Kaggle sessions end
without warning. Everything random is derived from (seed, epoch) and the learning
rate is a pure function of the epoch, so a resumed run retraces the same path
rather than merely continuing from a nearby point.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import metrics as M

WEIGHTS = "IMAGENET1K_V2"


@dataclass(frozen=True)
class TrainConfig:
    dataset: str
    fold: int = 1
    epochs: int = 30
    freeze_epochs: int = 3      # head only, so a random head cannot wreck the backbone
    batch_size: int = 64
    lr_head: float = 0.01
    lr_backbone: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4
    seed: int = 0
    pretrained: bool = True
    limit_per_class: int = 0    # non-zero marks a smoke run, never a result

    def __post_init__(self):
        if not 0 <= self.freeze_epochs < self.epochs:
            raise ValueError("freeze_epochs must be below epochs")

    @property
    def is_smoke(self) -> bool:
        return self.limit_per_class > 0


def build_model(n_classes: int, pretrained: bool = True) -> nn.Module:
    from torchvision.models import resnet50
    model = resnet50(weights=WEIGHTS if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def lr_at(cfg: TrainConfig, epoch: int) -> tuple[float, float]:
    """(backbone, head) learning rates. Cosine decay once unfrozen."""
    if epoch < cfg.freeze_epochs:
        return 0.0, cfg.lr_head
    t = (epoch - cfg.freeze_epochs) / (cfg.epochs - cfg.freeze_epochs)
    c = 0.5 * (1 + math.cos(math.pi * t))
    return cfg.lr_backbone * c, cfg.lr_head * c


def _split_params(model: nn.Module):
    head = list(model.fc.parameters())
    ids = {id(p) for p in head}
    return [p for p in model.parameters() if id(p) not in ids], head


def _set_phase(model: nn.Module, frozen: bool) -> None:
    """Frozen means no gradients and fixed batch-norm statistics.

    requires_grad alone is not enough: batch norm updates its running statistics
    in train mode regardless, so a "frozen" backbone would still drift.
    """
    backbone, _ = _split_params(model)
    for p in backbone:
        p.requires_grad_(not frozen)
    model.train(not frozen)
    model.fc.train()


def _save(obj: dict, path: Path) -> None:
    # Written then renamed, so a session killed mid-write leaves the old file.
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _load(path: Path, device) -> dict:
    return torch.load(path, map_location=device, weights_only=True)


@torch.no_grad()
def predict(model: nn.Module, dataset, device, batch_size: int = 64,
            num_workers: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Integer labels and predictions over a dataset, in dataset order."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    trues, preds = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        trues.append(y)
    return torch.cat(trues).numpy(), torch.cat(preds).numpy()


def _train_epoch(model, optimizer, scaler, train_ds, cfg, epoch, device,
                 num_workers) -> float:
    # Seeded per epoch: shuffle order and augmentation depend on the epoch
    # number alone, which is what makes resuming exact.
    torch.manual_seed(cfg.seed + epoch)
    gen = torch.Generator().manual_seed(cfg.seed + epoch)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        generator=gen, num_workers=num_workers,
                        pin_memory=device.type == "cuda",
                        drop_last=len(train_ds) > cfg.batch_size)
    frozen = epoch < cfg.freeze_epochs
    _set_phase(model, frozen)
    backbone_lr, head_lr = lr_at(cfg, epoch)
    optimizer.param_groups[0]["lr"] = backbone_lr
    optimizer.param_groups[1]["lr"] = head_lr

    loss_fn = nn.CrossEntropyLoss()
    total, seen = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            loss = loss_fn(model(x), y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss went non-finite at epoch {epoch}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(y)
        seen += len(y)
    return total / max(seen, 1)


def fit(model: nn.Module, train_ds, val_ds, classes, cfg: TrainConfig,
        out: str | Path, device=None, num_workers: int = 0,
        stop_after: int | None = None) -> dict:
    """Train, or resume from `out/last.pt` if it exists.

    Checkpoints after every epoch. `stop_after` ends the session early, leaving
    the same state an interrupted session would.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    backbone, head = _split_params(model)
    optimizer = torch.optim.SGD(
        [{"params": backbone}, {"params": head}], lr=cfg.lr_head,
        momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    start, best_score, best_epoch, history = 0, -1.0, -1, []
    last = out / "last.pt"
    if last.exists():
        ck = _load(last, device)
        if ck["config"] != asdict(cfg) or ck["classes"] != list(classes):
            raise RuntimeError(f"{last} belongs to a different run; refusing to "
                               f"resume. Use a new output directory.")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start, best_score = ck["epoch"], ck["best_score"]
        best_epoch, history = ck["best_epoch"], ck["history"]
        print(f"resumed at epoch {start}/{cfg.epochs}", flush=True)

    for epoch in range(start, cfg.epochs):
        if stop_after is not None and epoch >= stop_after:
            break
        began = time.monotonic()
        loss = _train_epoch(model, optimizer, scaler, train_ds, cfg, epoch,
                            device, num_workers)
        y, p = predict(model, val_ds, device, cfg.batch_size, num_workers)
        val = M.evaluate(y, p, classes)
        seconds = time.monotonic() - began
        history.append({"epoch": epoch, "train_loss": loss,
                        "val_top1": val.top1, "val_macro_f1": val.macro_f1,
                        "seconds": seconds, "device": device.type})
        print(f"epoch {epoch + 1}/{cfg.epochs}  loss {loss:.4f}  "
              f"val top-1 {val.top1:.4f}  val macro F1 {val.macro_f1:.4f}  "
              f"{seconds:.0f}s on {device.type}", flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2))

        if val.macro_f1 > best_score:
            best_score, best_epoch = val.macro_f1, epoch
            _save({"model": model.state_dict(), "epoch": epoch,
                   "config": asdict(cfg), "classes": list(classes)},
                  out / "best.pt")
        _save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scaler": scaler.state_dict(), "epoch": epoch + 1,
               "best_score": best_score, "best_epoch": best_epoch,
               "history": history, "config": asdict(cfg),
               "classes": list(classes)}, last)

    (out / "history.json").write_text(json.dumps(history, indent=2))
    return {"best_epoch": best_epoch, "best_val_macro_f1": best_score,
            "history": history}


def test_report(model: nn.Module, test_ds, classes, cfg: TrainConfig,
                out: str | Path, device=None,
                num_workers: int = 0) -> M.ClassificationReport:
    """Score the best validation epoch on the test split and write the results."""
    out = Path(out)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    best = _load(out / "best.pt", device)
    if best["classes"] != list(classes):
        raise RuntimeError("best.pt was trained on a different class list")
    model.to(device)
    model.load_state_dict(best["model"])

    y, p = predict(model, test_ds, device, cfg.batch_size, num_workers)
    report = M.evaluate(y, p, classes)

    payload = report.as_dict()
    payload.update(config=asdict(cfg), best_epoch=best["epoch"],
                   smoke=cfg.is_smoke, torch=torch.__version__,
                   weights=WEIGHTS if cfg.pretrained else None)
    (out / "test_metrics.json").write_text(json.dumps(payload, indent=2))
    report.per_class.to_csv(out / "per_class.csv", index=False)
    import pandas as pd
    pd.DataFrame(report.confusion, index=report.classes,
                 columns=report.classes).to_csv(out / "confusion.csv")
    return report


test_report.__test__ = False    # pytest: the name is not a test
