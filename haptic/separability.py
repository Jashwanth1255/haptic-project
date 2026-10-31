"""Objective 1 go/no-go gate — are the materials distinguishable at all?

Three questions, in order of how badly a "no" hurts:

1. **ANOVA** — does any descriptor dimension vary significantly across materials?
   If none does, the descriptors carry no class information and Objectives 3 and
   5 cannot work.
2. **Silhouette** — in the 4-D descriptor space, do recordings of the same
   material cluster together? This is the headline number for the gate.
3. **Pairwise collapse** — which specific material pairs are indistinguishable?
   This is Draft B's diversity/similarity module, re-aimed at our own descriptors
   instead of at TexSenseGAN's outputs.

A failing gate is not a failed project. It means the class list must be coarsened
-- group the collapsed pairs and report the grouping -- which is a legitimate and
reportable finding about how much fabric identity survives a 4-D descriptor.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .descriptor import Descriptor

# Gate thresholds. These are conventions, not laws -- state them in the paper
# and keep them fixed once data collection starts, so the gate cannot be moved
# after seeing the result.
ALPHA = 0.05            # ANOVA significance level
SILHOUETTE_MIN = 0.30   # below this, clusters are not meaningfully separated
COLLAPSE_RATIO = 1.0    # centroid distance / pooled spread; below => overlapping
MIN_SEPARABLE = 8       # project needs at least this many distinct classes


@dataclass
class GateResult:
    passed: bool
    silhouette: float
    n_materials: int
    n_recordings: int
    anova: pd.DataFrame
    collapsed_pairs: list[tuple[str, str, float]]
    effective_classes: int
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 68,
            "OBJECTIVE 1 GATE",
            "=" * 68,
            f"  materials          : {self.n_materials}",
            f"  recordings         : {self.n_recordings}",
            f"  silhouette (4-D)   : {self.silhouette:.3f}   "
            f"(threshold {SILHOUETTE_MIN})",
            f"  effective classes  : {self.effective_classes}   "
            f"(need {MIN_SEPARABLE})",
            "",
            "  ANOVA across materials, per descriptor dimension:",
        ]
        for _, r in self.anova.iterrows():
            if r.get("note"):
                lines.append(f"    {r['dimension']:<22} -- {r['note']}")
                continue
            flag = ("significant" if r["p_holm"] < ALPHA
                    else "NOT significant (Holm-corrected)")
            lines.append(
                f"    {r['dimension']:<22} F={r['F']:9.2f}  "
                f"p={r['p_value']:.2e}  p_holm={r['p_holm']:.2e}  "
                f"omega2={r['omega_squared']:+.3f}  {flag}"
            )
        if self.collapsed_pairs:
            lines += ["", f"  collapsed pairs ({len(self.collapsed_pairs)}):"]
            for a, b, d in self.collapsed_pairs:
                lines.append(f"    {a:<22} ~ {b:<22} separation={d:.2f}")
        else:
            lines += ["", "  collapsed pairs    : none"]
        if self.notes:
            lines += ["", "  notes:"]
            lines += [f"    - {n}" for n in self.notes]
        lines += ["", f"  RESULT: {'PASS' if self.passed else 'FAIL'}", "=" * 68]
        return "\n".join(lines)


def _labels(df: pd.DataFrame) -> pd.Series:
    return df["material_name"] if "material_name" in df else df["material_id"]


def anova_by_dimension(df: pd.DataFrame,
                       dims: tuple[str, ...] = Descriptor.CORE) -> pd.DataFrame:
    """One-way ANOVA across materials, for each descriptor dimension.

    Effect size is reported alongside p, because with many recordings per
    material even a trivial difference reaches p < 0.05.

    Both eta-squared and omega-squared are given, and **omega-squared is the one
    to use**. Eta-squared is biased upward by small samples: with k groups and n
    observations its expected value under a true null is about (k-1)/(n-1), so
    three identical materials with four recordings each produce eta2 ~ 0.18
    purely from noise, which reads as a moderate effect and is not one.
    Omega-squared subtracts that bias and can go negative, which is the honest
    signal for "no effect here".
    """
    lab = _labels(df)
    rows = []
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].unique().tolist()
        raise ValueError(
            f"duplicate columns {dupes}: df[col] returns a DataFrame rather "
            f"than a Series and every per-dimension statistic below is wrong")
    for dim in dims:
        groups = [np.ravel(g[dim].dropna().values)
                  for _, g in df.groupby(lab, observed=True)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            rows.append({"dimension": dim, "F": np.nan, "p_value": np.nan,
                         "eta_squared": np.nan, "omega_squared": np.nan,
                         "note": "fewer than 2 usable groups"})
            continue
        # A dimension with no within-group variance makes F undefined. On
        # synthetic signals this happens because some generator parameters are
        # deterministic; on real recordings it means a constant column, which
        # is worth seeing rather than silently turning into NaN.
        if all(np.ptp(g) == 0 for g in groups):
            rows.append({"dimension": dim, "F": np.nan, "p_value": np.nan,
                         "eta_squared": np.nan, "omega_squared": np.nan,
                         "note": "zero within-group variance"})
            continue
        F, p = stats.f_oneway(*groups)
        allv = np.concatenate(groups)
        grand = allv.mean()
        k = len(groups)
        n = len(allv)
        ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_t = ((allv - grand) ** 2).sum()
        ss_w = ss_t - ss_b
        ms_w = ss_w / (n - k) if n > k else np.nan

        eta2 = float(ss_b / ss_t) if ss_t > 0 else np.nan
        omega2 = (float((ss_b - (k - 1) * ms_w) / (ss_t + ms_w))
                  if np.isfinite(ms_w) and (ss_t + ms_w) > 0 else np.nan)

        rows.append({"dimension": dim, "F": float(F), "p_value": float(p),
                     "eta_squared": eta2, "omega_squared": omega2, "note": ""})

    out = pd.DataFrame(rows)

    # Holm-Bonferroni across the dimensions tested together. Four simultaneous
    # tests at alpha=0.05 give roughly a 1-in-5 chance of at least one false
    # positive; the corrected column is the one to quote.
    out["p_holm"] = np.nan
    valid = out["p_value"].notna()
    if valid.any():
        idx = out.index[valid]
        order = out.loc[idx, "p_value"].sort_values().index
        m = len(order)
        running = 0.0
        for rank, i in enumerate(order):
            adj = min(1.0, out.at[i, "p_value"] * (m - rank))
            running = max(running, adj)     # Holm is monotone non-decreasing
            out.at[i, "p_holm"] = running
    return out


def descriptor_matrix(df: pd.DataFrame,
                      dims: tuple[str, ...] = Descriptor.CORE):
    """Standardised descriptor matrix and labels, NaN rows dropped."""
    sub = df[list(dims)].copy()
    ok = sub.notna().all(axis=1)
    X = StandardScaler().fit_transform(sub[ok].values)
    return X, _labels(df)[ok].values, ok


def cluster_quality(df: pd.DataFrame,
                    dims: tuple[str, ...] = Descriptor.CORE) -> float:
    X, y, _ = descriptor_matrix(df, dims)
    if len(np.unique(y)) < 2 or len(X) < 3:
        return float("nan")
    return float(silhouette_score(X, y))


def pairwise_separation(df: pd.DataFrame,
                        dims: tuple[str, ...] = Descriptor.CORE) -> pd.DataFrame:
    """Separation ratio between every pair of materials.

    ratio = ||centroid_i - centroid_j|| / (spread_i + spread_j)

    where spread is the mean distance of a material's recordings from its own
    centroid. A ratio below 1 means the two clouds overlap: no classifier will
    reliably tell them apart from these descriptors alone.
    """
    X, y, _ = descriptor_matrix(df, dims)
    names = sorted(set(y))
    cents, spreads = {}, {}
    for n in names:
        pts = X[y == n]
        c = pts.mean(axis=0)
        cents[n] = c
        spreads[n] = float(np.mean(np.linalg.norm(pts - c, axis=1))) if len(pts) > 1 else 0.0

    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = float(np.linalg.norm(cents[a] - cents[b]))
            denom = spreads[a] + spreads[b]
            rows.append({
                "material_a": a, "material_b": b,
                "centroid_distance": d,
                "pooled_spread": denom,
                "separation_ratio": d / denom if denom > 0 else np.inf,
            })
    return pd.DataFrame(rows).sort_values("separation_ratio").reset_index(drop=True)


def separability_by_velocity(df: pd.DataFrame,
                             dims: tuple[str, ...] = Descriptor.CORE) -> pd.DataFrame:
    """Silhouette computed within each sliding velocity.

    Feeds Objective 3: if separability holds at every velocity individually but
    collapses when velocities are pooled, the descriptors are velocity-dependent
    and retrieval must be velocity-conditioned.
    """
    rows = []
    for v, g in df.groupby("velocity_mm_s", observed=True):
        rows.append({"velocity_mm_s": v, "n": len(g),
                     "silhouette": cluster_quality(g, dims)})
    rows.append({"velocity_mm_s": "ALL POOLED", "n": len(df),
                 "silhouette": cluster_quality(df, dims)})
    return pd.DataFrame(rows)


def run_gate(descriptors: list[Descriptor] | pd.DataFrame,
             dims: tuple[str, ...] = Descriptor.CORE) -> GateResult:
    """Apply the full Objective 1 gate."""
    if isinstance(descriptors, list):
        df = pd.DataFrame([d.as_dict() for d in descriptors])
    else:
        df = descriptors.copy()

    lab = _labels(df)
    n_mat = int(lab.nunique())

    aov = anova_by_dimension(df, dims)
    sil = cluster_quality(df, dims)
    pairs = pairwise_separation(df, dims)

    collapsed = [
        (r.material_a, r.material_b, r.separation_ratio)
        for r in pairs.itertuples()
        if r.separation_ratio < COLLAPSE_RATIO
    ]

    # effective classes: merge collapsed pairs transitively
    parent = {n: n for n in lab.unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in collapsed:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    effective = len({find(n) for n in parent})

    notes = []
    sig = aov[aov["p_holm"] < ALPHA]["dimension"].tolist()
    if not sig:
        notes.append("no descriptor dimension separates materials significantly")
    else:
        notes.append(f"significant dimensions: {', '.join(sig)}")
    weak = aov[(aov["p_holm"] < ALPHA)
               & (aov["omega_squared"] < 0.06)]["dimension"].tolist()
    if weak:
        notes.append(
            f"significant but weak effect (omega2<0.06), driven by sample size "
            f"rather than a real difference: {', '.join(weak)}")
    if collapsed:
        notes.append(
            f"{len(collapsed)} pair(s) overlap; {n_mat} materials reduce to "
            f"{effective} effective classes")

    passed = bool(
        np.isfinite(sil) and sil >= SILHOUETTE_MIN
        and effective >= MIN_SEPARABLE
        and len(sig) > 0
    )

    return GateResult(
        passed=passed, silhouette=sil, n_materials=n_mat,
        n_recordings=len(df), anova=aov, collapsed_pairs=collapsed,
        effective_classes=effective, notes=notes,
    )
