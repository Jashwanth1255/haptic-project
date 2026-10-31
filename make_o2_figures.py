"""Confusion matrix figures for Objective 2, from results/o2/*/test_metrics.json.

    python make_o2_figures.py

Cells show recall: each row sums to 1, so a dark off-diagonal cell reads as "this
share of the true class was called that". Smoke runs are refused.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent


def recall_matrix(confusion) -> np.ndarray:
    c = np.asarray(confusion, dtype=float)
    return c / np.maximum(c.sum(axis=1, keepdims=True), 1)


def plot(metrics: dict, title: str, path: Path) -> None:
    if metrics["smoke"]:
        raise ValueError(f"{path.name}: refusing to plot a smoke run")
    m = recall_matrix(metrics["confusion"])
    classes = metrics["classes"]
    k = len(classes)
    annotate = k <= 25      # a number in every cell is unreadable at 47 classes
    side = max(7.0, 0.27 * k)

    fig, ax = plt.subplots(figsize=(side + 1.2, side), constrained_layout=True)
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k), classes, rotation=90, fontsize=8)
    ax.set_yticks(range(k), classes, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{title}\ntop-1 {metrics['top1']:.3f}, "
                 f"macro F1 {metrics['macro_f1']:.3f}, n={metrics['n']}",
                 fontsize=11, loc="left")
    ax.set_xticks(np.arange(k + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(k + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if annotate:
        for i, j in zip(*np.nonzero(m >= 0.05)):
            ax.text(j, i, f"{m[i, j]:.2f}"[1:] if m[i, j] < 1 else "1", ha="center",
                    va="center", fontsize=6.5,
                    color="white" if m[i, j] > 0.55 else "#1a1a1a")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Share of true class (recall)")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    for run, title in (("dtd_fold1_seed0", "DTD texture, fold 1, test split"),
                       ("minc2500_fold1_seed0", "MINC-2500 material, fold 1, test split")):
        d = HERE / "results" / "o2" / run
        plot(json.loads((d / "test_metrics.json").read_text()), title,
             d / "confusion.png")
        print(f"wrote {d / 'confusion.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
