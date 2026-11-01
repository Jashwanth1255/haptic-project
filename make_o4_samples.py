"""Generate the interface's sample images.

    python make_o4_samples.py

These are procedural test patterns, not product photographs. They exist so the
page can be exercised without uploading anything, and the classifier's output on
them means nothing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

OUT = Path(__file__).parent / "webapp" / "static" / "samples"
SIZE = 320


def _save(gray: np.ndarray, tint, name: str) -> None:
    g = (gray - gray.min()) / np.ptp(gray)
    rgb = np.stack([g * c for c in tint], axis=-1)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(OUT / f"{name}.png", optimize=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    grain = rng.normal(0, 0.08, (SIZE, SIZE))

    over = ((x // 12 + y // 12) % 2).astype(float)
    threads = np.where(over > 0, np.sin(y * np.pi / 6), np.sin(x * np.pi / 6)) ** 2
    _save(threads + over * 0.4 + grain, (0.62, 0.70, 0.88), "weave")

    loops = np.sin(x * np.pi / 10 + 2 * np.sin(y * np.pi / 14)) ** 2
    _save(loops + 0.5 * np.sin(y * np.pi / 7) ** 2 + grain, (0.90, 0.62, 0.55), "knit")

    _save(np.sin((x + 0.3 * y) * np.pi / 40) ** 2 + grain, (0.82, 0.80, 0.66), "pleat")

    fibres = gaussian_filter(rng.normal(size=(SIZE, SIZE)), 1.6)
    _save(fibres + 0.5 * gaussian_filter(rng.normal(size=(SIZE, SIZE)), 9) * 8,
          (0.60, 0.74, 0.60), "felt")
    print(f"wrote 4 samples to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
