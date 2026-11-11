"""Figures made only for the Phase I report.

    python report/make_report_figures.py

fig_dtd_textile_confusion.png: 12 textile classes, DTD texture model, confusion
summed over the 10 folds' test splits and normalised per true class (recall).
Source: results/o2/dtd_fold<k>_seed0/textile12/restricted_confusion.csv.

fig_identification_summary.png: material identification with sliding direction
held out, for every input and model tried in Phase I, read from the saved runs
(o1_real_analysis, rig_whitening, audio_vs_accel, whitened_cnn,
benchmark118_cnn, audio118).
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
OUT = ROOT / "report" / "figures"


def textile_confusion() -> Path:
    files = sorted(glob.glob(str(ROOT / "results/o2/dtd_fold*_seed0/textile12/restricted_confusion.csv")))
    if len(files) != 10:
        raise SystemExit(f"expected 10 folds, found {len(files)}")
    tabs = [pd.read_csv(f, index_col=0) for f in files]
    total = sum(t.to_numpy(float) for t in tabs)
    classes = list(tabs[0].index)
    recall = total / total.sum(1, keepdims=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(recall, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = recall[i, j]
            if round(v * 100) >= 1:
                ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 else "#1f2937")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("DTD texture model on the 12 textile classes\n"
                 "(10 folds combined; % of each true class)", fontsize=10, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Recall")
    cb.outline.set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig_dtd_textile_confusion.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    diag = np.diag(recall)
    print(path, "mean recall", round(float(diag.mean()), 3),
          "lowest", sorted(zip(diag.round(3), classes))[:3])
    return path


ACCEL, SOUND = "#2a6fdb", "#e8743b"   # validated categorical pair (blue, orange)


def identification_summary() -> Path:
    import runlog
    o1 = runlog.load_run("o1_real_analysis").metrics["identification_exploratory"]
    rw = runlog.load_run("rig_whitening").metrics["tests"]
    av = runlog.load_run("audio_vs_accel").metrics["inputs"]
    wc = runlog.load_run("whitened_cnn").metrics["results"]
    b118 = runlog.load_run("benchmark118_cnn").metrics["results"]
    a118 = runlog.load_run("audio118").metrics["results"]
    cloth = [("4 descriptors, linear", o1["raw4"]["accuracy"], ACCEL),
             ("16-band spectrum, linear", o1["bands"]["accuracy"], ACCEL),
             ("Spectrum, whitened, linear", rw["cloth"]["whitened_accuracy"], ACCEL),
             ("Spectrogram CNN", av["accel_rerun"]["accuracy_mean"], ACCEL),
             ("Spectrogram CNN, whitened", wc["P1"]["accuracy_mean"], ACCEL),
             ("Contact sound CNN", av["audio_full"]["accuracy_mean"], SOUND)]
    allm = [("16-band spectrum, linear", rw["all118"]["raw_accuracy"], ACCEL),
            ("Spectrum, whitened, linear", rw["all118"]["whitened_accuracy"], ACCEL),
            ("Spectrogram CNN", b118["material_id|direction|cnn"]["accuracy_mean"], ACCEL),
            ("Spectrogram CNN, whitened", wc["P2"]["accuracy_mean"], ACCEL),
            ("Contact sound, linear", a118["material_id|direction|audio_8k"]["accuracy"], SOUND)]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.4), gridspec_kw={"height_ratios": [6, 5]})
    for ax, rows, title, chance in ((axes[0], cloth, "16 cloth materials", 1 / 16),
                                    (axes[1], allm, "All 118 materials", 1 / 118)):
        labels = [r[0] for r in rows][::-1]; vals = [r[1] for r in rows][::-1]; cols = [r[2] for r in rows][::-1]
        ax.barh(labels, vals, color=cols, height=0.62, edgecolor="white", linewidth=2)
        for y, v in enumerate(vals):
            ax.text(v + 0.01, y, f"{v * 100:.1f}%", va="center", fontsize=8.5, color="#1f2937")
        ax.axvline(chance, color="#6b7280", ls="--", lw=1)
        ax.text(chance + 0.005, len(vals) - 0.45, f"chance {chance * 100:.2f}%", fontsize=7.5, color="#4b5563")
        ax.set_xlim(0, 0.8); ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
        ax.grid(axis="x", color="#e5e7eb", lw=0.8); ax.set_axisbelow(True)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[1].set_xlabel("Identification accuracy, sliding direction held out")
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=ACCEL, label="Acceleration (vibration)"), Patch(color=SOUND, label="Contact sound")],
                   loc="center right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    path = OUT / "fig_identification_summary.png"
    fig.savefig(path, dpi=300); plt.close(fig)
    print(path, [round(v, 3) for _, v, _ in cloth], [round(v, 3) for _, v, _ in allm])
    return path


if __name__ == "__main__":
    textile_confusion()
    identification_summary()
