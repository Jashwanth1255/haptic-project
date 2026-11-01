"""Embedding extraction, on random weights and flat images: mechanics only."""
import numpy as np
import pytest
import torch
from PIL import Image

from vision import embed as E
from vision.train import build_model


def _ckpt(tmp_path, k=4):
    p = tmp_path / "best.pt"
    torch.save({"model": build_model(k, pretrained=False).state_dict(),
                "classes": [f"c{i}" for i in range(k)]}, p)
    return p


def test_tiles_share_no_pixels():
    im = Image.fromarray(np.arange(120 * 120 * 3, dtype=np.uint32).reshape(120, 120, 3)
                         .astype(np.uint8))
    parts = E.tiles(im, 2, 3)
    assert len(parts) == 6 and parts[0].size == (40, 60)
    boxes = [(c * 40, r * 60) for r in range(2) for c in range(3)]
    assert len(set(boxes)) == 6


def test_tile_contents_come_from_their_own_cell():
    a = np.zeros((40, 40, 3), np.uint8)
    a[:, 20:] = 255
    left, right = E.tiles(Image.fromarray(a), 1, 2)
    assert np.asarray(left).max() == 0 and np.asarray(right).min() == 255


def test_too_small_for_the_grid():
    with pytest.raises(ValueError, match="too small"):
        E.tiles(Image.new("RGB", (2, 2)), 4, 4)


def test_embedding_has_the_pooled_size(tmp_path):
    m = E.load_embedder(_ckpt(tmp_path))
    out = E.embed([m], [Image.new("RGB", (64, 64), (200, 10, 10))])
    assert out.shape == (1, 2048)
    assert np.linalg.norm(out[0]) == pytest.approx(1.0, abs=1e-5)


def test_two_models_are_joined(tmp_path):
    m = E.load_embedder(_ckpt(tmp_path))
    out = E.embed([m, m], [Image.new("RGB", (64, 64), (1, 2, 3))] * 2)
    assert out.shape == (2, 4096)
