"""The three-stage representability profile, assembled from saved runs only.

    python -m study.representability --out representability.png

Nothing is recomputed. Each stage's number per attribute is read from its run:

    Stage 1   stage1_perception        Spearman as a fraction of sqrt(ICC(2,k))
    Stage 2   stage2_representability  leave-one-material-out R², normalised descriptors
    Stage 3   study_discrimination_*   Spearman between pair d' and rating gap,
                                       mean over the non-control schemes

The three are on different scales, so the figure gives each stage its own panel
rather than one shared axis. Warmth is marked as the control, roughness as the
reference.

`ordering_verdict` applies the rule in PREREGISTRATION.md ("Prediction:
representability ordering") to Stage 2, and reports each part either way.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import runlog
from vision.encoders import ATTRIBUTES

ALPHA = 0.05
STAGES = ("stage1", "stage2", "stage3")
LABELS = {"stage1": "Stage 1: images to perception\n(Spearman / ceiling)",
          "stage2": "Stage 2: haptics to perception\n(out-of-sample R²)",
          "stage3": "Stage 3: rendered discrimination\n(Spearman of d' with rating gap)"}
CONTROL, REFERENCE = "warmth", "roughness"


def _column(table: pd.DataFrame, key: str, value: str) -> pd.Series:
    return table.set_index(key)[value].reindex(list(ATTRIBUTES))


def profile(stage1: runlog.Run, stage2: runlog.Run, stage3: runlog.Run) -> pd.DataFrame:
    s1 = _column(stage1.table("scores"), "target", "fraction_of_ceiling")
    s2t = stage2.table("scores")
    s2t = s2t[s2t["descriptors"] == "normalised"]
    s2 = _column(s2t, "target", "r2")
    p2 = _column(s2t, "target", "p_permutation")
    track = stage3.table("attribute_tracking")
    s3 = track.groupby("attribute")["spearman"].mean().reindex(list(ATTRIBUTES))
    out = pd.DataFrame({"stage1": s1, "stage2": s2, "stage2_p": p2, "stage3": s3})
    out.index.name = "attribute"
    out.attrs["runs"] = {k: str(r.path) for k, r in
                         (("stage1", stage1), ("stage2", stage2), ("stage3", stage3))}
    return out


def ordering_verdict(r2: pd.Series, p: pd.Series, dropped=()) -> dict:
    """The pre-registered rule: min(roughness, bumpiness) > slipperiness >
    softness > warmth in the point estimates, and warmth not beating the mean
    predictor (permutation p >= 0.05). Attributes dropped for low ICC are skipped,
    and any comparison that loses an attribute is reported as untestable."""
    r2 = r2.drop(list(dropped), errors="ignore")
    have = lambda *a: all(x in r2.index and np.isfinite(r2[x]) for x in a)
    top = (min(r2[a] for a in ("roughness", "bumpiness") if a in r2.index)
           if any(a in r2.index for a in ("roughness", "bumpiness")) else np.nan)
    parts = {}
    for name, ok, test in (
            ("top_above_slipperiness", np.isfinite(top) and have("slipperiness"),
             lambda: top > r2["slipperiness"]),
            ("slipperiness_above_softness", have("slipperiness", "softness"),
             lambda: r2["slipperiness"] > r2["softness"]),
            ("softness_above_warmth", have("softness", "warmth"),
             lambda: r2["softness"] > r2["warmth"])):
        parts[name] = bool(test()) if ok else None
    control = (bool(p[CONTROL] >= ALPHA) if CONTROL in p.index and CONTROL not in dropped
               and np.isfinite(p[CONTROL]) else None)
    tested = [v for v in parts.values() if v is not None]
    return {"parts": parts, "ordering_held": all(tested) if tested else None,
            "n_comparisons_tested": len(tested), "control_held": control,
            "dropped": list(dropped)}


def figure(table: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    blue, grey, ink, muted = "#2a78d6", "#9a9a92", "#1d1d1b", "#66665f"
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    for ax, stage in zip(axes, STAGES):
        vals = table[stage]
        x = np.arange(len(table))
        colours = [grey if a == CONTROL else blue for a in table.index]
        bars = ax.bar(x, vals.fillna(0), width=0.6, color=colours, edgecolor="white",
                      linewidth=2)
        for b, a in zip(bars, table.index):
            if a == CONTROL:
                b.set_hatch("//")
            if a == REFERENCE:
                b.set_edgecolor(ink)
                b.set_linewidth(1.2)
        for xi, v in zip(x, vals):
            ax.annotate("n/a" if pd.isna(v) else f"{v:.2f}", (xi, 0 if pd.isna(v) else v),
                        xytext=(0, 3 if (pd.isna(v) or v >= 0) else -12),
                        textcoords="offset points", ha="center", fontsize=8, color=muted)
        ax.axhline(0, color=muted, linewidth=0.8)
        ax.margins(y=0.15)                  # room for labels on negative bars
        ax.set_xticks(x, table.index, rotation=30, ha="right", fontsize=9, color=ink)
        ax.set_title(LABELS[stage], fontsize=9, color=ink, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(muted)
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(axis="y", color="#e6e6e0", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=blue, label="attribute"),
                        Patch(facecolor=blue, edgecolor=ink, label=f"{REFERENCE} (reference)"),
                        Patch(facecolor=grey, hatch="//", label=f"{CONTROL} (control)")],
               loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200)
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="study.representability")
    ap.add_argument("--stage1", default="stage1_perception")
    ap.add_argument("--stage2", default="stage2_representability")
    ap.add_argument("--stage3", default="study_discrimination_main")
    ap.add_argument("--reliability", default="study_reliability_main",
                    help="its unreliable attributes are dropped from the ordering")
    ap.add_argument("--out", default="representability.png")
    a = ap.parse_args(argv)

    runs = {k: runlog.load_run(getattr(a, k)) for k in ("stage1", "stage2", "stage3")}
    dropped = runlog.load_run(a.reliability).metrics.get("unreliable", [])
    table = profile(runs["stage1"], runs["stage2"], runs["stage3"])
    verdict = ordering_verdict(table["stage2"], table["stage2_p"], dropped)
    fig = figure(table, a.out)
    print(table.round(3).to_string())
    print("\npre-registered ordering (Stage 2):")
    for k, v in verdict["parts"].items():
        print(f"  {k:<30} {'held' if v else 'untestable' if v is None else 'FAILED'}")
    print(f"  {'warmth control':<30} "
          f"{'held' if verdict['control_held'] else 'untestable' if verdict['control_held'] is None else 'FAILED'}")
    saved = runlog.save_run(
        "representability_profile",
        {**vars(a), "source_runs": {k: str(r.path) for k, r in runs.items()},
         "source_commits": {k: r.info["git_commit"] for k, r in runs.items()}},
        {"ordering": verdict, "profile": table.to_dict(orient="index")},
        {"profile": table, "figure": fig}, synthetic=any(r.info["synthetic"] for r in runs.values()))
    print(f"\nwrote {a.out}\nsaved run {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
