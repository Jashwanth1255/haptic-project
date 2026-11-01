"""Image to vibration pattern, behind the web interface.

    image -> 12-way textile class -> descriptor -> four patterns + fidelity

**Retrieval here is a placeholder.** `CLASS_TO_FABRIC` assigns each textile class a
fictional fabric from `signals.py`, by hand and with no evidence. Objective 3
replaces it. Until then every descriptor, waveform and spectrum the interface
shows is synthetic, and the response says so in `synthetic`.

The classifier is the DTD fold 1 model. It has not been tested on product
photographs: that is Objective 2's open transfer question, so a prediction on a
product photo is unvalidated.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import signal as sps

from haptic import descriptor as D
from haptic import render as R
from haptic import signals as G
from vision.textile import TEXTILE_CLASSES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "dtd_fold1_seed0" / "best.pt"
REFERENCE_VELOCITY_MM_S = 40.0
REFERENCE_FORCE_N = 1.0
SYNTHESIS_SEED = 0
MAX_POINTS = 600

CLASS_TO_FABRIC = {     # PLACEHOLDER: arbitrary until Objective 3 exists
    "braided": "twill-diagonal",
    "fibrous": "pile-fleece",
    "frilly": "satin-dense",
    "gauzy": "satin-slick",
    "interlaced": "plain-weave-coarse",
    "knitted": "pile-velvet",
    "lacelike": "denim-heavy",
    "matted": "felt-matte",
    "meshed": "mesh-open",
    "pleated": "corduroy-wide",
    "woven": "plain-weave-fine",
    "wrinkled": "corduroy-narrow",
}

Classifier = Callable[[bytes], list[tuple[str, float]]]


def load_classifier(checkpoint: str | Path = DEFAULT_CHECKPOINT) -> Classifier:
    """12-way textile classifier: non-textile logits masked, as in evaluation."""
    import torch
    from PIL import Image

    from vision import data as V
    from vision import textile as X
    from vision.train import build_model

    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    classes = ck["classes"]
    keep = X.subset_indices(classes)
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()
    transform = V.transforms_for(train=False)

    @torch.no_grad()
    def classify(image: bytes) -> list[tuple[str, float]]:
        with Image.open(io.BytesIO(image)) as im:
            x = transform(im.convert("RGB")).unsqueeze(0)
        probs = model(x).squeeze(0)[keep].softmax(0)
        order = probs.argsort(descending=True).tolist()
        return [(TEXTILE_CLASSES[i], float(probs[i])) for i in order]

    return classify


@dataclass(frozen=True)
class Retrieved:
    fabric: str
    signal: np.ndarray
    envelope: np.ndarray
    descriptor: D.Descriptor


def build_library() -> tuple[dict[str, Retrieved], R.RenderConfig]:
    """One synthetic recording per class, at a fixed exploration condition."""
    by_name = {m.name: m for m in G.FABRIC_BANK}
    library = {}
    for cls, fabric in CLASS_TO_FABRIC.items():
        x = G.synthesize(by_name[fabric], REFERENCE_VELOCITY_MM_S, REFERENCE_FORCE_N,
                         seed=SYNTHESIS_SEED)
        library[cls] = Retrieved(fabric, x, R.signal_envelope(x, G.FS_HZ),
                                 D.extract(x, G.FS_HZ))
    ref = R.intensity_reference(r.descriptor for r in library.values())
    return library, R.RenderConfig(intensity_ref=ref)


def _thin(y: np.ndarray, n: int = MAX_POINTS) -> list[float]:
    if y.size <= n:
        return [round(float(v), 5) for v in y]
    return [round(float(v), 5) for v in sps.resample_poly(y, n, y.size)]


def _waveform(r: Retrieved) -> dict:
    """The first 200 ms of signal: a whole recording would be a solid block."""
    n = int(0.2 * G.FS_HZ)
    return {"duration_ms": 200.0, "y": [round(float(v), 4) for v in r.signal[:n]]}


def _spectrum(r: Retrieved) -> dict:
    """Display only. Fidelity never uses this: see `haptic.render`."""
    x = r.signal - r.signal.mean()
    f, p = sps.welch(x, fs=G.FS_HZ, nperseg=1024)
    keep = (f >= 5) & (f <= 1500)
    db = 10 * np.log10(np.maximum(p[keep], 1e-12))
    return {"f_hz": [round(float(v), 2) for v in f[keep]],
            "db": [round(float(v), 2) for v in db]}


def _patterns(r: Retrieved, cfg: R.RenderConfig) -> tuple[dict, dict]:
    level = min(r.descriptor.intensity / cfg.intensity_ref, 1.0)
    patterns, scores = {}, {}
    for name, p in R.encode_all(r.descriptor, cfg, r.envelope).items():
        patterns[name] = {"durations": list(p.durations), "duty": round(p.duty, 4),
                          "total_ms": p.total_ms,
                          "motor": _thin(R.achievable_envelope(p), 400)}
        # a solid on or off pattern has no modulation, so no spectrum to compare
        scores[name] = (R.fidelity(r.envelope, p, level).as_dict()
                        if np.ptp(p.binary()) else None)
    return patterns, scores


def analyse(image: bytes, classify: Classifier, library: dict[str, Retrieved],
            cfg: R.RenderConfig) -> dict:
    """Everything the page shows for one image, with server-side timings."""
    t0 = time.perf_counter()
    ranked = classify(image)
    t1 = time.perf_counter()
    top, prob = ranked[0]
    r = library[top]
    t2 = time.perf_counter()
    patterns, scores = _patterns(r, cfg)
    t3 = time.perf_counter()
    d = r.descriptor
    return {
        "synthetic": True,
        "predicted": {"class": top, "probability": round(prob, 4),
                      "top3": [{"class": c, "probability": round(p, 4)}
                               for c, p in ranked[:3]]},
        "fabric": {"name": r.fabric, "velocity_mm_s": REFERENCE_VELOCITY_MM_S,
                   "force_n": REFERENCE_FORCE_N},
        "descriptor": {k: round(float(getattr(d, k)), 4) for k in D.Descriptor.CORE},
        "waveform": _waveform(r),
        "spectrum": _spectrum(r),
        "envelope": {"duration_ms": cfg.duration_ms,
                     "y": _thin(r.envelope[:cfg.duration_ms], 400)},
        "patterns": patterns,
        "fidelity": scores,
        "limits": {"switching_hz": R.switching_limit_hz(cfg.tick_ms),
                   "tick_ms": cfg.tick_ms, "band_hz": [D.BAND_LOW_HZ, D.BAND_HIGH_HZ]},
        "timings_ms": {"classify": round((t1 - t0) * 1000, 2),
                       "retrieve": round((t2 - t1) * 1000, 2),
                       "encode": round((t3 - t2) * 1000, 2)},
    }
