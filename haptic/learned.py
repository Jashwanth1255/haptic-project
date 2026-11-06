"""Learned baseline: can a CNN on spectrograms identify the cloth? EXPLORATORY.

Added 2026-11-05, after the O1 gate failed, to answer the objection a reviewer
will raise: that the four-number descriptor, not the signal, is the problem. If a
learned model with direction held out also stays near chance, the negative result
is about the signal; if it does well, the descriptor discards identity.

Input: log power spectrogram of the band-passed acceleration magnitude, 20-1000 Hz,
over the steady-sliding stretch (as `loader.load` returns it). Velocity and force
are given to the classifier head, since both are known when a query is made.
Training draws random WINDOW_S windows; a test recording is scored by averaging
the logits of all its non-overlapping windows.

Splits:
    direction   8 folds, one sliding direction held out each: the honest test
    repeat      train repeat 0, test repeat 1, every direction seen: upper bound
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import descriptor as D

N_FFT = 256
HOP = 64
WINDOW_S = 0.5


def spectrogram(accel: np.ndarray, fs: float) -> np.ndarray:
    """(frames, bins) log power over 20-1000 Hz."""
    from scipy.signal import stft
    x = D.bandpass(D.to_magnitude(accel), fs)
    f, _, z = stft(x, fs, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None, padded=False)
    keep = (f >= D.BAND_LOW_HZ) & (f <= D.BAND_HIGH_HZ)
    return np.log(np.abs(z[keep]) ** 2 + 1e-10).T.astype(np.float32)


def frames_per_window(fs: float) -> int:
    return int((WINDOW_S * fs - N_FFT) // HOP) + 1


@dataclass
class Config:
    epochs: int = 25
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-3
    windows_per_recording: int = 4      # random windows drawn per recording per epoch
    channels: int = 32
    dropout: float = 0.3
    device: str = "auto"                # auto = cuda when available, else cpu


def _net(n_bins: int, n_classes: int, cfg: Config):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            c = cfg.channels
            self.body = nn.Sequential(
                nn.Conv2d(1, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(c, 2 * c, 3, padding=1), nn.BatchNorm2d(2 * c), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 4)))      # pool time away, keep coarse frequency
            self.head = nn.Sequential(nn.Dropout(cfg.dropout),
                                      nn.Linear(2 * c * 4 + 2, 64), nn.ReLU(),
                                      nn.Dropout(cfg.dropout), nn.Linear(64, n_classes))

        def forward(self, spec, cond):
            h = self.body(spec.unsqueeze(1)).flatten(1)
            return self.head(torch.cat([h, cond], 1))

    return Net()


def _windows(spec: np.ndarray, n: int, rng=None, count: int = 1) -> np.ndarray:
    """`count` random windows of `n` frames (or all non-overlapping ones if rng is None)."""
    if len(spec) < n:
        spec = np.pad(spec, ((0, n - len(spec)), (0, 0)), mode="edge")
    if rng is None:
        starts = np.arange(0, len(spec) - n + 1, n)
    else:
        starts = rng.integers(0, len(spec) - n + 1, size=count)
    return np.stack([spec[s:s + n] for s in starts])


def fit_predict(specs, meta: pd.DataFrame, train: np.ndarray, test: np.ndarray,
                n_frames: int, cfg: Config = Config(), seed: int = 0) -> np.ndarray:
    """Train on `train` rows, return predicted class labels for `test` rows."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    classes = np.sort(meta.material_id.unique())
    y = np.searchsorted(classes, meta.material_id.to_numpy())
    cond = np.column_stack([np.log(meta.velocity_mm_s / 40.0), meta.force_n - 0.75]).astype(np.float32)
    # standardise with training statistics only
    stack = np.concatenate([specs[i] for i in train])
    mu, sd = stack.mean(0), stack.std(0) + 1e-6
    norm = [(s - mu) / sd for s in specs]

    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                       if cfg.device == "auto" else cfg.device)
    model = _net(norm[0].shape[1], len(classes), cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, epochs=cfg.epochs,
        steps_per_epoch=int(np.ceil(len(train) * cfg.windows_per_recording / cfg.batch_size)))
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(cfg.epochs):
        model.train()
        idx = np.repeat(train, cfg.windows_per_recording)
        rng.shuffle(idx)
        for b in range(0, len(idx), cfg.batch_size):
            rows = idx[b:b + cfg.batch_size]
            x = torch.from_numpy(np.stack([_windows(norm[i], n_frames, rng)[0] for i in rows])).to(dev)
            out = model(x, torch.from_numpy(cond[rows]).to(dev))
            loss = lossf(out, torch.from_numpy(y[rows]).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    pred = []
    with torch.no_grad():
        for i in test:
            w = torch.from_numpy(_windows(norm[i], n_frames)).to(dev)
            c = torch.from_numpy(np.repeat(cond[i:i + 1], len(w), 0)).to(dev)
            pred.append(int(model(w, c).mean(0).argmax()))
    return classes[np.array(pred)]


def evaluate(specs, meta: pd.DataFrame, split: str, n_frames: int,
             seeds=(0, 1, 2), cfg: Config = Config(), checkpoint=None) -> pd.DataFrame:
    """One row per fold x seed: accuracy and n. `split` is direction or repeat.
    `checkpoint`, a path, keeps finished fold x seed rows as JSON lines so an
    interrupted run resumes where it stopped."""
    import json
    from pathlib import Path
    done = {}
    if checkpoint is not None and Path(checkpoint).exists():
        for line in Path(checkpoint).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["split"], str(r["fold"]), r["seed"])] = r
    meta = meta.reset_index(drop=True)
    if split == "direction":
        folds = [(np.flatnonzero(meta.direction_deg != d), np.flatnonzero(meta.direction_deg == d), d)
                 for d in sorted(meta.direction_deg.unique())]
    elif split == "repeat":
        r = meta["repeat"] % 2          # loader's repeat index: direction-major, parity = repeat
        folds = [(np.flatnonzero(r == 0), np.flatnonzero(r == 1), "repeat0->1")]
    else:
        raise ValueError(split)
    rows = []
    for train, test, label in folds:
        for seed in seeds:
            key = (split, str(label), seed)
            if key in done:
                rows.append(done[key])
                continue
            pred = fit_predict(specs, meta, train, test, n_frames, cfg, seed)
            truth = meta.material_id.to_numpy()[test]
            row = {"split": split, "fold": str(label), "seed": seed, "n": len(test),
                   "accuracy": float((pred == truth).mean()),
                   "predictions": [int(p) for p in pred], "truth": [int(t) for t in truth]}
            rows.append(row)
            if checkpoint is not None:
                Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({**row, "fold": str(label)}) + "\n")
                print(f"  {split} fold {label} seed {seed}: accuracy {row['accuracy']:.3f}",
                      flush=True)
    return pd.DataFrame(rows)
