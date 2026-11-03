"""Pilot and main-study reliability from the rating instrument's exported CSVs.

    python -m study.reliability exports/*.csv --name pilot

Each CSV is one participant's session from `study/rating_instrument.html`, rows of

    participant, device_model, block, image_id, attribute, rating, repeat, rt_ms

Only rows with block "rating" are used here; discrimination rows are analysed by
`study.discrimination`.

- **ICC(2,k) and ICC(2,1)** per attribute: two-way random effects, absolute
  agreement (Shrout and Fleiss 1979; McGraw and Wong's A,k and A,1), on first
  presentations only. ICC(2,k) is the reliability of the mean rating, which is
  what every model is trained on. Below 0.50 an attribute is flagged: raters do
  not agree on it, and no model should be expected to predict it.
- **Intra-rater Spearman**: each participant's first against second rating of the
  repeated images.
- **Timing and click-through checks.** The thresholds below are instrument quality
  checks, fixed here before the pilot. A flag is a prompt to look, not an
  automatic exclusion; any exclusion must be reported with its reason.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ICC_MIN = 0.50              # below this, the attribute is flagged as unreliable
FAST_MEDIAN_RT_MS = 1500    # median time per rating below this looks like clicking through
STRAIGHT_LINE = 0.80        # share of one participant's ratings that are a single value
INTRA_RATER_MIN = 0.0       # at or below this, a participant disagrees with themself

COLUMNS = ["participant", "device_model", "block", "image_id", "attribute",
           "rating", "repeat", "rt_ms"]


def read_sessions(paths, block: str = "rating") -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        missing = {"participant", "device_model", "block"} - set(df.columns)
        if missing:
            raise ValueError(f"{p}: missing columns {sorted(missing)}")
        frames.append(df.assign(source=Path(p).name))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["block"] == block].copy()
    df["participant"] = df["participant"].astype(str)
    if block == "rating":
        missing = set(COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"missing rating columns {sorted(missing)}")
        dup = df.duplicated(["participant", "image_id", "attribute", "repeat"])
        if dup.any():
            first = df[dup].iloc[0]
            raise ValueError(f"{int(dup.sum())} duplicated ratings, e.g. participant "
                             f"{first['participant']} {first['image_id']} {first['attribute']}")
    return df


def icc2(matrix) -> dict:
    """ICC(2,1) and ICC(2,k) for a complete targets x raters matrix."""
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError("need a complete matrix with at least 2 targets and 2 raters")
    n, k = x.shape
    grand = x.mean()
    ss_rows = k * np.sum((x.mean(1) - grand) ** 2)
    ss_cols = n * np.sum((x.mean(0) - grand) ** 2)
    ss_err = np.sum((x - grand) ** 2) - ss_rows - ss_cols
    msr, msc = ss_rows / (n - 1), ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    single = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    average = (msr - mse) / (msr + (msc - mse) / n)
    return {"icc2_1": float(single), "icc2_k": float(average), "n_targets": n,
            "n_raters": k}


def icc_by_attribute(df: pd.DataFrame) -> pd.DataFrame:
    """First presentations only. Images not rated by every participant are dropped
    and counted, because ICC(2,k) needs a complete matrix."""
    first = df[df["repeat"] == 0]
    rows = []
    for attr, g in first.groupby("attribute", sort=False):
        wide = g.pivot(index="image_id", columns="participant", values="rating")
        complete = wide.dropna()
        r = icc2(complete.to_numpy())
        rows.append({"attribute": attr, **r,
                     "n_images_dropped": len(wide) - len(complete),
                     "unreliable": r["icc2_k"] < ICC_MIN})
    return pd.DataFrame(rows)


def intra_rater(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman between first and second ratings of the repeated images."""
    cols = ["participant", "attribute", "n_repeated", "spearman"]
    wide = df.pivot_table(index=["participant", "attribute", "image_id"],
                          columns="repeat", values="rating")
    if 0 not in wide.columns or 1 not in wide.columns:
        return pd.DataFrame(columns=cols)
    wide = wide.dropna(subset=[0, 1])
    rows = []
    for (p, attr), g in wide.groupby(level=[0, 1]):
        ok = len(g) >= 3 and g[0].nunique() > 1 and g[1].nunique() > 1
        rows.append({"participant": p, "attribute": attr, "n_repeated": len(g),
                     "spearman": float(spearmanr(g[0], g[1]).statistic) if ok else np.nan})
    return pd.DataFrame(rows, columns=cols)


def timing(df: pd.DataFrame) -> pd.DataFrame:
    """Median time per rating for each participant, and the share of their
    ratings that are one single value."""
    g = df.groupby("participant")
    return pd.DataFrame({
        "device_model": g["device_model"].first(),
        "n_ratings": g.size(),
        "median_rt_ms": g["rt_ms"].median(),
        "modal_share": g["rating"].agg(lambda r: r.value_counts(normalize=True).iloc[0]),
    }).reset_index()


def flags(df: pd.DataFrame) -> pd.DataFrame:
    """Participants worth a second look, with every reason that applies."""
    t = timing(df)
    ir = intra_rater(df).groupby("participant")["spearman"].median()
    t["median_intra_rater"] = t["participant"].map(ir)
    reasons = []
    for r in t.itertuples():
        why = []
        if r.median_rt_ms < FAST_MEDIAN_RT_MS:
            why.append(f"median time {r.median_rt_ms:.0f} ms < {FAST_MEDIAN_RT_MS}")
        if r.modal_share >= STRAIGHT_LINE:
            why.append(f"{r.modal_share:.0%} of ratings one value")
        if np.isfinite(r.median_intra_rater) and r.median_intra_rater <= INTRA_RATER_MIN:
            why.append(f"intra-rater rho {r.median_intra_rater:.2f}")
        reasons.append("; ".join(why))
    t["flags"] = reasons
    return t


def per_image_time(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("image_id")["rt_ms"].median().rename("median_rt_ms")
            .sort_values().reset_index())


def mean_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Image x attribute mean of first presentations: the models' targets."""
    return (df[df["repeat"] == 0]
            .pivot_table(index="image_id", columns="attribute", values="rating"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="study.reliability")
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--name", default="pilot", help="run name suffix")
    a = ap.parse_args(argv)

    import runlog

    df = read_sessions(a.csv)
    icc = icc_by_attribute(df)
    ir = intra_rater(df)
    fl = flags(df)
    print(f"{df['participant'].nunique()} participants, {len(df)} ratings\n")
    print("ICC per attribute (first presentations):")
    print(icc.round(3).to_string(index=False))
    print("\nintra-rater Spearman, median across participants:")
    print(ir.groupby("attribute")["spearman"].median().round(3).to_string())
    print("\nparticipants:")
    print(fl.round(3).to_string(index=False))
    bad = icc.loc[icc["unreliable"], "attribute"].tolist()
    if bad:
        print(f"\nUNRELIABLE (ICC(2,k) < {ICC_MIN}): {bad}")
    saved = runlog.save_run(
        f"study_reliability_{a.name}",
        {"csv": a.csv, "icc_min": ICC_MIN, "fast_median_rt_ms": FAST_MEDIAN_RT_MS,
         "straight_line": STRAIGHT_LINE, "intra_rater_min": INTRA_RATER_MIN},
        {"n_participants": df["participant"].nunique(), "n_ratings": len(df),
         "icc2k": dict(zip(icc["attribute"], icc["icc2_k"])),
         "unreliable": bad,
         "flagged_participants": fl.loc[fl["flags"] != "", "participant"].tolist()},
        {"icc": icc, "intra_rater": ir, "participants": fl,
         "per_image_time": per_image_time(df), "mean_ratings": mean_ratings(df)},
        synthetic=False)
    print(f"\nsaved run {saved}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
