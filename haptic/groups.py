"""The pre-registered group test: do pile, weave and drape cluster in haptic space?

As registered in PREREGISTRATION.md ("Prediction: haptic clusters mirror visual
clusters" and "Prediction: the drape group does not transfer"), with the group
assignment recorded there on 2026-11-04 before any descriptor existed:

- velocity-normalised descriptors, standardised over recordings, one mean centroid
  per material, Euclidean distance;
- for each group, mean within-group distance against mean distance to the other
  assigned materials (those in another named group; "neither" takes no part);
- a one-sided permutation test on the group labels. With 13 assigned materials
  every relabelling is enumerated, so p is exact, not sampled;
- pile and weave are Holm-corrected together at alpha 0.05 (the confirmatory
  family); drape is tested alone at alpha 0.05 (exploratory);
- a group with fewer than MIN_GROUP materials is reported as untested.

One reading had to be chosen: "the other assigned materials" is taken to include
drape, since the drape entry extends the assignment to four labels. `sensitivity`
reports the other readings beside it, so the choice is visible.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .descriptor import NORMALISED_CORE
from .separability import holm_adjust

ALPHA = 0.05
MIN_GROUP = 3
CONFIRMATORY = ("pile", "weave")
EXPLORATORY = ("drape",)

# PREREGISTRATION.md, "Cluster cloth group assignment", 2026-11-04. Never edit.
ASSIGNMENT = {
    102: "pile", 103: "pile",
    104: "weave", 105: "weave", 111: "weave", 112: "weave", 114: "weave",
    116: "weave", 117: "weave",
    106: "drape", 107: "drape", 109: "drape", 110: "drape",
    108: "neither", 113: "neither", 115: "neither",
}


def centroids(df: pd.DataFrame, dims=NORMALISED_CORE) -> pd.DataFrame:
    """Standardise each dimension over recordings, then average per material."""
    x = df[list(dims)].astype(float)
    z = (x - x.mean()) / x.std(ddof=0)
    return z.groupby(df["material_id"].to_numpy()).mean()


def distances(cent: pd.DataFrame) -> pd.DataFrame:
    v = cent.to_numpy()
    d = np.sqrt(((v[:, None, :] - v[None, :, :]) ** 2).sum(-1))
    return pd.DataFrame(d, index=cent.index, columns=cent.index)


def _within_between(d: np.ndarray, members: np.ndarray, others: np.ndarray) -> tuple[float, float]:
    w = d[np.ix_(members, members)][np.triu_indices(len(members), 1)].mean()
    b = d[np.ix_(members, others)].mean()
    return float(w), float(b)


def group_test(dist: pd.DataFrame, labels: dict[int, str], group: str,
               pool: tuple[str, ...] | None = None) -> dict:
    """Within/between ratio for `group` and its exact one-sided permutation p.

    `pool` is the set of labels whose materials take part (the group itself and its
    comparison set); None means every label except "neither"."""
    ids = [m for m in dist.index if m in labels]
    pool = pool or tuple(sorted({labels[m] for m in ids} - {"neither"}))
    part = [m for m in ids if labels[m] in pool]
    members = [m for m in part if labels[m] == group]
    out = {"group": group, "n": len(members), "n_pool": len(part),
           "materials": members, "within": np.nan, "between": np.nan,
           "ratio": np.nan, "p": np.nan, "n_permutations": 0,
           "tested": len(members) >= MIN_GROUP and len(part) > len(members)}
    if not out["tested"]:
        return out
    d = dist.loc[part, part].to_numpy()
    idx = np.arange(len(part))
    k = len(members)
    obs_w, obs_b = _within_between(d, np.array([part.index(m) for m in members]),
                                   np.array([i for i, m in enumerate(part) if m not in members]))
    obs = obs_w / obs_b
    ratios = []
    for combo in combinations(idx, k):
        mem = np.array(combo)
        w, b = _within_between(d, mem, np.setdiff1d(idx, mem))
        ratios.append(w / b)
    ratios = np.array(ratios)
    out.update(within=obs_w, between=obs_b, ratio=obs,
               p=float(np.mean(ratios <= obs + 1e-12)), n_permutations=len(ratios))
    return out


def run(df: pd.DataFrame, labels: dict[int, str] = ASSIGNMENT,
        dims=NORMALISED_CORE) -> pd.DataFrame:
    """The registered analysis: one row per group, with its verdict."""
    dist = distances(centroids(df, dims))
    rows = [group_test(dist, labels, g) for g in (*CONFIRMATORY, *EXPLORATORY)]
    res = pd.DataFrame(rows)
    res["family"] = ["confirmatory" if g in CONFIRMATORY else "exploratory" for g in res.group]
    conf = res[(res.family == "confirmatory")]
    # Holm over the registered family of two, even when one cannot be tested:
    # an untested group is entered as p = 1, which never helps the other.
    res["p_holm"] = np.nan
    res.loc[conf.index, "p_holm"] = holm_adjust(conf["p"].fillna(1.0).to_numpy())
    p_used = res["p_holm"].where(res.family == "confirmatory", res["p"])
    res["clusters"] = np.where(res.tested, p_used < ALPHA, None)
    return res


def verdict(res: pd.DataFrame) -> dict:
    conf = res[res.family == "confirmatory"]
    if not conf.tested.all():
        confirmatory = "untested: " + ", ".join(
            f"{g} has {n} materials (minimum {MIN_GROUP})"
            for g, n, t in zip(conf.group, conf.n, conf.tested) if not t)
        failed = conf[conf.tested & (conf.clusters == False)]  # noqa: E712
        if len(failed):
            confirmatory = "failed (" + ", ".join(failed.group) + " does not cluster); " + confirmatory
    else:
        confirmatory = "supported" if conf.clusters.all() else "failed"
    d = res[res.group == "drape"].iloc[0]
    if not d.tested:
        drape = "untested"
    elif d.clusters:
        drape = "drape clusters: the exploratory prediction failed"
    else:
        drape = ("drape does not cluster, as predicted (weak evidence: non-significance "
                 f"with {d.n} materials)")
    return {"confirmatory": confirmatory, "drape": drape}


def sensitivity(df: pd.DataFrame, labels: dict[int, str] = ASSIGNMENT,
                dims=NORMALISED_CORE) -> pd.DataFrame:
    """The same test under the other readings of "the other assigned materials":
    pile+weave only (the two-label wording), and all 16 materials."""
    dist = distances(centroids(df, dims))
    everyone = tuple(sorted(set(labels.values())))
    rows = []
    for reading, pool in (("pile+weave+drape (registered reading)", None),
                          ("pile+weave only", ("pile", "weave")),
                          ("all 16, neither included", everyone)):
        for g in (*CONFIRMATORY, *EXPLORATORY):
            if pool is not None and g not in pool:
                continue
            r = group_test(dist, labels, g, pool)
            rows.append({"reading": reading, **{k: r[k] for k in
                         ("group", "n", "n_pool", "ratio", "p", "tested")}})
    return pd.DataFrame(rows)
