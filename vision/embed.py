"""ResNet-50 embeddings from the Objective 2 models, for Objective 3.

Retrieval needs a query and a database that share no pixels, or it can only find
the same image again. Cluster has one scan per material, so `tiles` cuts a scan
into a grid: some tiles build the database and the rest are queries.

Only `MINC` and `DTD` weights trained in Objective 2 are meant to be used here. A
query embedding is the two models' L2-normalised 2048-d vectors joined together.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from . import data as V
from .train import build_model


def load_embedder(checkpoint: str | Path, device="cpu") -> nn.Module:
    """The model from `checkpoint`, ending at the 2048-d average pool."""
    ck = torch.load(checkpoint, map_location=device, weights_only=True)
    model = build_model(len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["model"])
    model.fc = nn.Identity()
    return model.to(device).eval()


def tiles(image: Image.Image, rows: int, cols: int) -> list[Image.Image]:
    """Non-overlapping grid cells. Any remainder at the edges is dropped."""
    w, h = image.size[0] // cols, image.size[1] // rows
    if w < 1 or h < 1:
        raise ValueError(f"{image.size} is too small for a {rows}x{cols} grid")
    return [image.crop((c * w, r * h, (c + 1) * w, (r + 1) * h))
            for r in range(rows) for c in range(cols)]


@torch.no_grad()
def embed(models, images, device="cpu") -> np.ndarray:
    """(images, 2048 x len(models)), each model's part L2-normalised."""
    tf = V.transforms_for(train=False)
    x = torch.stack([tf(im.convert("RGB")) for im in images]).to(device)
    parts = [nn.functional.normalize(m(x), dim=1) for m in models]
    return torch.cat(parts, dim=1).cpu().numpy()
