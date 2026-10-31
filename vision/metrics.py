"""Objective 2 classification metrics -- top-1, macro F1, confusion matrix.

Everything reported for O2 comes through `evaluate`, so that training-time
validation and final test numbers cannot drift apart. Takes plain integer
arrays; nothing here depends on torch.

The class list is passed in explicitly rather than inferred from the labels. A
class the model never predicts and that is absent from a subset would otherwise
vanish from the matrix, and its zero F1 would quietly leave the macro average.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


@dataclass
class ClassificationReport:
    classes: tuple[str, ...]
    n: int
    top1: float
    macro_f1: float
    confusion: np.ndarray           # rows = true class, columns = predicted
    per_class: pd.DataFrame

    def as_dict(self) -> dict:
        return {
            "classes": list(self.classes),
            "n": self.n,
            "top1": self.top1,
            "macro_f1": self.macro_f1,
            "confusion": self.confusion.tolist(),
            "per_class": self.per_class.to_dict(orient="records"),
        }

    def most_confused(self, k: int = 10) -> pd.DataFrame:
        """Largest off-diagonal cells: true class, predicted class, count."""
        off = self.confusion.copy()
        np.fill_diagonal(off, 0)
        rows = [{"true": self.classes[i], "predicted": self.classes[j],
                 "count": int(off[i, j]),
                 "rate": float(off[i, j] / max(1, self.confusion[i].sum()))}
                for i, j in zip(*np.nonzero(off))]
        if not rows:
            return pd.DataFrame(columns=["true", "predicted", "count", "rate"])
        return (pd.DataFrame(rows).sort_values("count", ascending=False)
                .head(k).reset_index(drop=True))

    def summary(self) -> str:
        lines = [f"  images   : {self.n}",
                 f"  classes  : {len(self.classes)}",
                 f"  top-1    : {self.top1:.4f}",
                 f"  macro F1 : {self.macro_f1:.4f}"]
        return "\n".join(lines)


def evaluate(y_true, y_pred, classes) -> ClassificationReport:
    """Score integer predictions against integer labels.

    Labels index into `classes`. Anything out of range raises: it means a
    checkpoint is being scored against a different class list, which would
    otherwise produce a plausible-looking and entirely wrong number.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = tuple(classes)
    k = len(classes)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("no predictions to score")
    if len(set(classes)) != k:
        raise ValueError("duplicate class names")
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"{name} must be integer class indices")
        if arr.min() < 0 or arr.max() >= k:
            raise ValueError(f"{name} holds an index outside 0..{k - 1}")

    labels = np.arange(k)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    per_class = pd.DataFrame({
        "class": classes,
        "support": support,
        "precision": tp / np.maximum(predicted, 1),
        "recall": tp / np.maximum(support, 1),
        "f1": f1,
    })
    return ClassificationReport(
        classes=classes, n=int(y_true.size),
        top1=float(tp.sum() / y_true.size),
        macro_f1=float(f1.mean()),
        confusion=cm, per_class=per_class,
    )
