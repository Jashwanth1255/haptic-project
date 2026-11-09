"""Paper figures, drawn only from saved runs (never recomputed here).

    python make_paper_figures.py            # writes figures/paper/*.png and *.pdf

Each figure names the run it was drawn from in the PDF metadata and in
figures/paper/SOURCES.json, so every plotted number traces back to a run id.

    fig1_variance_shares     o1_real_analysis      what drives each descriptor
    fig2_scan_transfer       o2_cluster_scans, o2_cluster_scan_scale
    fig3_velocity_exponents  velocity_all_materials (all 118 materials)
    fig4_identification      o1_real_analysis, o1_learned_baseline (if saved)

Colours: the validated categorical slots below (dataviz validator, light mode,
all checks pass with a contrast warning, hence direct labels on every series).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import runlog  # noqa: E402

OUT = Path("figures/paper")
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
NEUTRAL = "#c3c6cc"
INK, INK2, GRID = "#1f2328", "#59636e", "#e4e6ea"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})


def _save(fig, name: str, runs: dict, sources: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"Subject": "; ".join(f"{k} {v}" for k, v in runs.items())}
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf", metadata=meta)
    plt.close(fig)
    sources[name] = runs
    print(f"  {name}: {meta['Subject']}")


def _bar_labels(ax, bars, values, fmt="{:.2f}", pad=0.01):
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + pad, fmt.format(v),
                ha="center", va="bottom", fontsize=8, color=INK)


# --- figure 1 -------------------------------------------------------------------

COMPONENTS = [("material_id", "Material"), ("direction_deg", "Direction"),
              ("material_id:direction_deg", "Material × direction"),
              ("velocity_mm_s", "Velocity"), ("force_n", "Force"), ("Residual", "Residual")]
DIM_LABEL = {"frequency_hz": "Frequency\n(centroid)", "intensity": "Intensity\n(RMS)",
             "pattern": "Pattern\n(entropy)", "duration_ms": "Duration\n(envelope)"}


def fig_variance(sources):
    run = runlog.load_run("o1_real_analysis")
    v = run.table("variance_shares").set_index("dimension").loc[list(DIM_LABEL)]
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    y = np.arange(len(v))[::-1]
    left = np.zeros(len(v))
    colours = SLOTS + [NEUTRAL]
    for (col, label), c in zip(COMPONENTS, colours):
        w = v[col].to_numpy()
        ax.barh(y, w, left=left, color=c, height=0.62, edgecolor="white", linewidth=2,
                label=label)
        for yi, l, wi in zip(y, left, w):
            if wi >= 0.07:
                ax.text(l + wi / 2, yi, f"{wi:.2f}", ha="center", va="center", fontsize=7.5,
                        color=INK)
        left += w
    ax.set_yticks(y, [DIM_LABEL[d] for d in v.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of variance (type II sums of squares)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=6, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7.5,
              handlelength=1, columnspacing=1)
    ax.set_title("Material identity explains 8–15% of each descriptor; direction and residual dominate",
                 pad=22, fontsize=9)
    _save(fig, "fig1_variance_shares", {"o1_real_analysis": run.info["run_id"]}, sources)


# --- figure 2 -------------------------------------------------------------------

def fig_transfer(sources):
    scans = runlog.load_run("o2_cluster_scans")
    scale = runlog.load_run("o2_cluster_scan_scale")
    labels = ["MINC-2500\ntest photos", "Cluster scan\ncrops", "Scan rescaled\n50 mm field",
              "Scan rescaled\n100 mm field"]
    values = [scale.metrics["minc_test_fabric_recall"], scans.metrics["minc"]["fabric_fraction"],
              scale.metrics["fabric_fraction_by_scale"]["50 mm"],
              scale.metrics["fabric_fraction_by_scale"]["100 mm"]]
    per_scale = scale.table("predictions").groupby("scale").size()
    ns = ["", f"n={scans.metrics['minc']['n_crops']}", f"n={per_scale['50 mm']}",
          f"n={per_scale['100 mm']}"]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    bars = ax.bar(range(4), values, color=[SLOTS[0]] + [SLOTS[1]] * 3, width=0.62,
                  edgecolor="white", linewidth=2)
    _bar_labels(ax, bars, values, "{:.3f}")
    ax.set_xticks(range(4), labels, fontsize=8)
    for i, n in enumerate(ns):
        if n:
            ax.text(i, -0.2, n, ha="center", va="top", fontsize=7, color=INK2,
                    transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share called “fabric” by MINC ResNet-50")
    ax.set_title("A fabric classifier does not recognise flatbed scans of fabric")
    _save(fig, "fig2_scan_transfer", {"o2_cluster_scans": scans.info["run_id"],
                                      "o2_cluster_scan_scale": scale.info["run_id"]}, sources)


# --- figure 3 -------------------------------------------------------------------

CATEGORY_ORDER = ["wood", "stone", "polymer", "metal", "glass", "composite", "ceramic",
                  "leather", "cloth"]
CATEGORY_SHORT = {"composite": "comp.", "ceramic": "cer.", "polymer": "poly.",
                  "leather": "leath."}


def fig_velocity(sources):
    run = runlog.load_run("velocity_all_materials")
    cen = run.table("exponents_per_material")
    shift = run.table("shift_exponents_per_material")
    try:
        confirm = runlog.load_run("velocity_confirm_cloth")
    except FileNotFoundError:
        confirm = None
    if confirm is not None and confirm.metrics["original_run"] != run.info["run_id"]:
        raise SystemExit("velocity_confirm_cloth was not run against the latest velocity run")
    panels = [(cen, "frequency_hz", "Spectral-centroid exponent"),
              (shift, "whitened", "Spectral-shift exponent (rig-whitened)")]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), sharey=True, width_ratios=(9, 10.2),
                             gridspec_kw={"wspace": 0.08})
    rng = np.random.default_rng(0)
    for ax, (df, col, title) in zip(axes, panels):
        for i, cat in enumerate(CATEGORY_ORDER):
            y = df.loc[df.category == cat, col].to_numpy()
            c = SLOTS[1] if cat == "cloth" else SLOTS[0]
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(y)), y, s=14, color=c,
                       edgecolor="white", linewidth=0.6, zorder=3)
            ax.plot([i - 0.28, i + 0.28], [np.median(y)] * 2, color=INK, lw=1.4, zorder=4)
        for ref in (0, 1):
            ax.axhline(ref, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
        ticks = [f"{CATEGORY_SHORT.get(c, c)}\n({(df.category == c).sum()})"
                 for c in CATEGORY_ORDER]
        if col == "whitened" and confirm is not None:
            y = confirm.table("per_material").held_out.to_numpy()
            i = len(CATEGORY_ORDER)
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(y)), y, s=14, facecolor="white",
                       edgecolor=SLOTS[1], linewidth=1.2, zorder=3)
            ax.plot([i - 0.28, i + 0.28], [np.median(y)] * 2, color=INK, lw=1.4, zorder=4)
            ticks.append(f"held-\nout\n({len(y)})")
        ax.set_xticks(range(len(ticks)), ticks, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Exponent of frequency on sliding velocity")
    axes[0].set_ylim(-0.8, 1.2)
    right_edge = axes[1].get_xlim()[1]
    for ref, lab in ((1, "f = v/λ"), (0, "fixed\nfrequency")):
        axes[1].text(right_edge + 0.1, ref, lab, ha="left", va="center",
                     fontsize=7.5, color=INK2, clip_on=False)
    m = run.metrics
    fig.suptitle(f"No material follows f = v/λ; once the shared rig spectrum is removed, only cloth "
                 f"shifts with speed\n(all {m['n_materials']} materials; bar = median; orange = "
                 f"cloth; hollow = held-out replication, cloth)",
                 x=0.06, y=1.07, ha="left", fontsize=9, fontweight="bold")
    runs = {"velocity_all_materials": run.info["run_id"]}
    if confirm is not None:
        runs["velocity_confirm_cloth"] = confirm.info["run_id"]
    _save(fig, "fig3_velocity_exponents", runs, sources)


# --- figure 4 -------------------------------------------------------------------

def fig_identification(sources):
    run = runlog.load_run("o1_real_analysis")
    ident = run.metrics["identification_exploratory"]
    rows = [("4 descriptors\n(raw)", ident["raw4"]["accuracy"], ident["raw4"]["fold_std"]),
            ("4 descriptors\n(v-normalised)", ident["norm4"]["accuracy"], ident["norm4"]["fold_std"]),
            ("16 log bands", ident["bands"]["accuracy"], ident["bands"]["fold_std"]),
            ("Bands + raw\ndescriptors", ident["bands+raw4"]["accuracy"],
             ident["bands+raw4"]["fold_std"])]
    runs = {"o1_real_analysis": run.info["run_id"]}
    try:
        cnn = runlog.load_run("o1_learned_baseline")
    except (FileNotFoundError, LookupError):
        cnn = None
    if cnn is not None:
        d = cnn.metrics["direction"]
        # same error bar as the descriptor bars: SD over the 8 direction folds
        # (each fold's accuracy averaged over seeds), not SD over seeds
        folds = cnn.table("folds")
        per_fold = folds[folds["split"] == "direction"].groupby("fold")["accuracy"].mean()
        rows.append(("CNN on\nspectrogram", d["accuracy_mean"], float(np.std(per_fold, ddof=1))))
        runs["o1_learned_baseline"] = cnn.info["run_id"]
    labels, acc, sd = zip(*rows)
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    colours = [SLOTS[0]] * 2 + [SLOTS[2]] * 2 + [SLOTS[3]] * (len(rows) - 4)
    bars = ax.bar(range(len(rows)), acc, yerr=sd, color=colours, width=0.62, edgecolor="white",
                  linewidth=2, error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
    for b, a, e in zip(bars, acc, sd):
        ax.text(b.get_x() + b.get_width() / 2, a + e + 0.01, f"{a:.3f}", ha="center",
                va="bottom", fontsize=8, color=INK)
    chance = ident["raw4"]["chance"]
    ax.axhline(chance, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(len(rows) - 0.45, chance, "chance\n1/16", ha="left", va="center",
            fontsize=7.5, color=INK2, clip_on=False)
    ax.set_xticks(range(len(rows)), labels, fontsize=7.5)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("Accuracy, sliding direction held out")
    note = "" if cnn is not None else " (CNN run not saved)"
    ax.set_title(f"16-way cloth identification from acceleration{note}", fontsize=9)
    ax.grid(axis="x", visible=False)
    _save(fig, "fig4_identification", runs, sources)


def main() -> int:
    sources: dict = {}
    for f in (fig_variance, fig_transfer, fig_velocity, fig_identification):
        f(sources)
    (OUT / "SOURCES.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    print(f"wrote {len(sources)} figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
