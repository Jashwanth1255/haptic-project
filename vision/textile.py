"""The 12-class DTD textile subset, scored three ways from the same logits.

    unrestricted  47-way argmax, scored on textile images
    filtered      as above, but dropping images predicted outside the subset
    restricted    logits of non-textile classes masked before argmax

Masking can only help: an image already correct stays correct, because removing
competitors cannot unseat the winner. Filtering looks better still, but only by
discarding hard images, so it answers a narrower question than the other two.

The class list is fixed in PREREGISTRATION.md. Change neither without the other.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import metrics as M

TEXTILE_CLASSES = ("braided", "fibrous", "frilly", "gauzy", "interlaced",
                   "knitted", "lacelike", "matted", "meshed", "pleated",
                   "woven", "wrinkled")


def subset_indices(classes, subset=TEXTILE_CLASSES) -> list[int]:
    classes = list(classes)
    missing = [c for c in subset if c not in classes]
    if missing:
        raise ValueError(f"not in this class list: {missing}")
    return [classes.index(c) for c in subset]


def restrict(logits: torch.Tensor, keep: list[int]) -> torch.Tensor:
    """Copy of `logits` with every class outside `keep` set to -inf."""
    masked = torch.full_like(logits, -torch.inf)
    masked[:, keep] = logits[:, keep]
    return masked


@torch.no_grad()
def logits_of(model, dataset, device, batch_size: int = 64,
              num_workers: int = 0) -> tuple[torch.Tensor, np.ndarray]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    logits, trues = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        trues.append(y)
    return torch.cat(logits), torch.cat(trues).numpy()


def score_subset(logits: torch.Tensor, y, classes,
                 subset=TEXTILE_CLASSES) -> dict:
    """All three scores for images whose true class is in `subset`."""
    y = np.asarray(y)
    keep = subset_indices(classes, subset)
    if not np.isin(y, keep).all():
        raise ValueError("an image's true class is outside the subset")
    position = {c: i for i, c in enumerate(keep)}
    to_subset = np.vectorize(position.__getitem__)
    y_sub = to_subset(y)

    open_pred = logits.argmax(1).numpy()
    inside = np.isin(open_pred, keep)
    filtered = M.evaluate(y_sub[inside], to_subset(open_pred[inside]), subset)
    restricted = M.evaluate(y_sub, to_subset(restrict(logits, keep).argmax(1).numpy()),
                            subset)
    return {
        "n": int(y.size),
        "n_predicted_outside": int((~inside).sum()),
        "unrestricted_accuracy": float((open_pred == y).mean()),
        "filtered": filtered,
        "restricted": restricted,
    }
