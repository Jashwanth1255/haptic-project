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
    velocity    5 folds, one velocity held out each (round 2, analysis C)

Round 2 (2026-11-07) added other inputs: a log-frequency spectrogram that can be
warped along log f by k*ln(v/40) (analysis C), and contact-audio spectrograms
(analysis B). It also replaced the per-sample Python window loop with batched
indexing on one padded tensor. The method is unchanged, but the random stream
is not, so `o1_learned_baseline/20261106T214141_196418Z` (the old loop) is not
bit-reproducible with this code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import descriptor as D

N_FFT = 256
HOP = 64
WINDOW_S = 0.5

LOGF_N_FFT = 1024
LOGF_HOP = 128
LOGF_BANDS = 48

AUDIO_N_FFT = 2048
AUDIO_HOP = 512
AUDIO_BANDS = 64


def spectrogram(accel: np.ndarray, fs: float) -> np.ndarray:
    """(frames, bins) log power over 20-1000 Hz."""
    from scipy.signal import stft
    x = D.bandpass(D.to_magnitude(accel), fs)
    f, _, z = stft(x, fs, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None, padded=False)
    keep = (f >= D.BAND_LOW_HZ) & (f <= D.BAND_HIGH_HZ)
    return np.log(np.abs(z[keep]) ** 2 + 1e-10).T.astype(np.float32)


def band_spectrogram(x: np.ndarray, fs: float, n_fft: int, hop: int, n_bands: int,
                     low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    """(frames, n_bands) log power pooled into log-spaced bands; also the band
    centres (Hz). Bands holding no STFT bin take the value interpolated on log f."""
    from scipy.signal import stft
    f, _, z = stft(x, fs, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    p = np.abs(z) ** 2
    edges = np.geomspace(low, high, n_bands + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])
    out = np.full((p.shape[1], n_bands), np.nan)
    for k in range(n_bands):
        m = (f >= edges[k]) & (f < edges[k + 1])
        if m.any():
            out[:, k] = np.log(p[m].mean(0) + 1e-10)
    have = ~np.isnan(out[0])
    if not have.all():
        lc = np.log(centres)
        for t in range(out.shape[0]):
            out[t, ~have] = np.interp(lc[~have], lc[have], out[t, have])
    return out.astype(np.float32), centres


def logf_spectrogram(accel: np.ndarray, fs: float) -> np.ndarray:
    """Acceleration, log-frequency bands over 20-1000 Hz (analysis C)."""
    x = D.bandpass(D.to_magnitude(accel), fs)
    return band_spectrogram(x, fs, LOGF_N_FFT, LOGF_HOP, LOGF_BANDS,
                            D.BAND_LOW_HZ, D.BAND_HIGH_HZ)[0]


def audio_spectrogram(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    """Contact audio, log-frequency bands (analysis B)."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [low / (fs / 2), min(high * 1.05, fs / 2 * 0.99) / (fs / 2)],
                 btype="band", output="sos")
    return band_spectrogram(sosfiltfilt(sos, x - x.mean()), fs, AUDIO_N_FFT, AUDIO_HOP,
                            AUDIO_BANDS, low, high)[0]


def warp(spec: np.ndarray, velocity_mm_s: float, k: float,
         low: float = D.BAND_LOW_HZ, high: float = D.BAND_HIGH_HZ) -> np.ndarray:
    """Resample a log-frequency spectrogram so band i reads the power at
    f_i * (v/40)^k. With k = 1, a spectrum obeying f = v/lambda lands on the same
    bands at every speed; with k = 0 nothing changes. Outside the band the edge
    value is held."""
    if k == 0:
        return spec
    n = spec.shape[1]
    lc = np.log(np.sqrt(np.geomspace(low, high, n + 1)[:-1] * np.geomspace(low, high, n + 1)[1:]))
    target = lc + k * np.log(velocity_mm_s / 40.0)
    out = np.empty_like(spec)
    for t in range(spec.shape[0]):
        out[t] = np.interp(target, lc, spec[t])
    return out


def frames_per_window(fs: float, n_fft: int = N_FFT, hop: int = HOP) -> int:
    return int((WINDOW_S * fs - n_fft) // hop) + 1


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


def _padded(norm, n: int):
    """One (N, T_max, bins) array, each recording edge-padded to at least n frames
    and zero-padded after its end; plus each recording's usable length."""
    lens = np.array([max(len(s), n) for s in norm])
    out = np.zeros((len(norm), lens.max(), norm[0].shape[1]), np.float32)
    for i, s in enumerate(norm):
        if len(s) < n:
            s = np.pad(s, ((0, n - len(s)), (0, 0)), mode="edge")
        out[i, :len(s)] = s
    return out, lens


def fit_predict(specs, meta: pd.DataFrame, train: np.ndarray, test: np.ndarray,
                n_frames: int, cfg: Config = Config(), seed: int = 0,
                label: str = "material_id", return_scores: bool = False):
    """Train on `train` rows, return predicted labels for `test` rows (and, with
    `return_scores`, the (n_test, n_classes) mean logits and the class list)."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    labels = meta[label].to_numpy()
    classes = np.unique(labels)
    y = np.searchsorted(classes, labels)
    cond = np.column_stack([np.log(meta.velocity_mm_s.to_numpy(float) / 40.0),
                            meta.force_n.to_numpy(float) - 0.75]).astype(np.float32)
    # standardise with training statistics only
    stack = np.concatenate([specs[i] for i in train]).astype(np.float64)
    mu, sd = stack.mean(0), stack.std(0) + 1e-6
    norm = [((np.asarray(s, np.float64) - mu) / sd).astype(np.float32) for s in specs]

    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                       if cfg.device == "auto" else cfg.device)
    padded, lens = _padded(norm, n_frames)
    P = torch.from_numpy(padded).to(dev)
    span = torch.from_numpy(lens - n_frames + 1).to(dev)
    C = torch.from_numpy(cond).to(dev)
    Y = torch.from_numpy(y).to(dev)
    offs = torch.arange(n_frames, device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed)

    model = _net(padded.shape[2], len(classes), cfg).to(dev)
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
            rows = torch.from_numpy(idx[b:b + cfg.batch_size]).to(dev)
            start = (torch.rand(len(rows), generator=gen, device=dev) * span[rows]).long()
            x = P[rows[:, None], start[:, None] + offs]
            loss = lossf(model(x, C[rows]), Y[rows])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    # every non-overlapping window of every test recording, scored in batches
    rec, starts = [], []
    for i in test:
        s = np.arange(0, lens[i] - n_frames + 1, n_frames)
        rec.append(np.full(len(s), i))
        starts.append(s)
    rec, starts = np.concatenate(rec), np.concatenate(starts)
    pos = np.searchsorted(test, rec) if np.all(np.diff(test) > 0) else \
        np.array([np.flatnonzero(test == r)[0] for r in rec])
    scores = torch.zeros(len(test), len(classes), device=dev)
    counts = torch.zeros(len(test), device=dev)
    with torch.no_grad():
        for b in range(0, len(rec), 1024):
            r = torch.from_numpy(rec[b:b + 1024]).to(dev)
            s = torch.from_numpy(starts[b:b + 1024]).to(dev)
            out = model(P[r[:, None], s[:, None] + offs], C[r])
            p = torch.from_numpy(pos[b:b + 1024]).to(dev)
            scores.index_add_(0, p, out)
            counts.index_add_(0, p, torch.ones(len(p), device=dev))
    scores = (scores / counts[:, None]).cpu().numpy()
    pred = classes[scores.argmax(1)]
    return (pred, scores, classes) if return_scores else pred


def folds_for(meta: pd.DataFrame, split: str):
    meta = meta.reset_index(drop=True)
    if split == "direction":
        return [(np.flatnonzero(meta.direction_deg != d), np.flatnonzero(meta.direction_deg == d), d)
                for d in sorted(meta.direction_deg.unique())]
    if split == "velocity":
        return [(np.flatnonzero(meta.velocity_mm_s != v), np.flatnonzero(meta.velocity_mm_s == v), v)
                for v in sorted(meta.velocity_mm_s.unique())]
    if split == "repeat":
        r = meta["repeat"] % 2          # loader's repeat index: direction-major, parity = repeat
        return [(np.flatnonzero(r == 0), np.flatnonzero(r == 1), "repeat0->1")]
    raise ValueError(split)


def evaluate(specs, meta: pd.DataFrame, split: str, n_frames: int,
             seeds=(0, 1, 2), cfg: Config = Config(), checkpoint=None,
             label: str = "material_id", top_k: int = 5) -> pd.DataFrame:
    """One row per fold x seed: accuracy, top-k accuracy and n. `split` is
    direction, repeat or velocity. `checkpoint`, a path, keeps finished fold x seed
    rows as JSON lines so an interrupted run resumes where it stopped."""
    import json
    from pathlib import Path
    done = {}
    if checkpoint is not None and Path(checkpoint).exists():
        for line in Path(checkpoint).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["split"], str(r["fold"]), r["seed"])] = r
    meta = meta.reset_index(drop=True)
    rows = []
    for train, test, label_ in folds_for(meta, split):
        if len(test) == 0 or len(train) == 0:
            continue
        for seed in seeds:
            key = (split, str(label_), seed)
            if key in done:
                rows.append(done[key])
                continue
            pred, scores, classes = fit_predict(specs, meta, train, test, n_frames, cfg, seed,
                                                label=label, return_scores=True)
            truth = meta[label].to_numpy()[test]
            k = min(top_k, len(classes))
            topk = classes[np.argsort(-scores, 1)[:, :k]]
            row = {"split": split, "fold": str(label_), "seed": seed, "n": len(test),
                   "accuracy": float((pred == truth).mean()),
                   f"top{k}": float((topk == truth[:, None]).any(1).mean()),
                   "rows": [int(t) for t in test],
                   "predictions": [int(p) for p in pred], "truth": [int(t) for t in truth]}
            rows.append(row)
            if checkpoint is not None:
                Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"  {split} fold {label_} seed {seed}: accuracy {row['accuracy']:.3f}",
                      flush=True)
    return pd.DataFrame(rows)


def macro_f1(truth, pred) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(truth, pred, average="macro"))
