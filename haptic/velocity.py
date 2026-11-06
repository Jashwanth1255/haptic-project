"""Objective 3's velocity experiment: does conditioning on velocity help?

**NO RESULT FROM THIS MODULE IS VALID UNTIL REAL CLUSTER DESCRIPTORS EXIST.** On
`signals.py` output conditioning wins *by construction*, because the generator was
written with f = v / lambda in it. Recovering that proves the arithmetic. Only real
fabric can say whether the relations hold. The `velocity_free` recordings in
`synth_o3` make them false, and there conditioning must lose: that is the evidence
this experiment can give a negative answer.

    agnostic      the material's descriptor averaged over velocities, whatever the
                  query velocity is
    conditioned   the material's normalised descriptor, with the query velocity
                  re-applied through `descriptor.denormalise`

Both use the same reference recordings at the query's force, and are scored on
query recordings that were not used to build them: lower repeat numbers build the
reference and the rest are queries. On Cluster the repeats are the eight sliding
directions. Out-of-band recordings are excluded on both sides, and counted.

The material is the unit of analysis: errors are averaged within a material before
testing, so repeats do not inflate n.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import descriptor as D
from .association import DIMS, paired_test
from .separability import holm_adjust

ALPHA = 0.05
MIN_PAIRS = 6       # below this a Wilcoxon test cannot reach p < 0.05 two-sided


def smallest_holm_p(n_materials: int, n_tests: int) -> float:
    """The best Holm p a table of this size could possibly report. Above ALPHA, a
    "no clear difference" verdict is forced by the sample and says nothing."""
    return min(1.0, n_tests * 2 / 2 ** n_materials)


def split_repeats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeats = np.sort(df["repeat"].unique())
    if len(repeats) < 2:
        raise ValueError("need at least 2 repeats to separate reference from query")
    cut = repeats[len(repeats) // 2]
    return df[df["repeat"] < cut], df[df["repeat"] >= cut]


def _across_velocities(ref: pd.DataFrame, columns) -> pd.Series:
    """Median within a velocity, then mean across velocities.

    The median because `duration_ms` is heavy-tailed: a rare recording locks onto a
    spurious low-frequency envelope peak, ten times too long, and one such value
    would otherwise set a material's reference. The per-velocity step means a
    velocity with fewer in-band recordings keeps its equal weight.
    """
    return ref.groupby("velocity_mm_s")[list(columns)].median().mean()


def _agnostic(ref: pd.DataFrame) -> dict:
    return _across_velocities(ref, DIMS).to_dict()


def _conditioned(ref: pd.DataFrame, velocity: float, force: float) -> dict:
    m = _across_velocities(ref, D.NORMALISED_CORE)
    return D.denormalise(m["wavelength_mm"], m["roughness_index"], m["pattern"],
                         m["burst_spacing_mm"], velocity, force)


def run(df: pd.DataFrame, substitute: dict | None = None,
        exclude_query_velocity: bool = False) -> pd.DataFrame:
    """One row per material x velocity x force x dimension: mean absolute errors.

    `substitute` maps a material id to the id retrieved in its place. Without it
    the right material is given, which isolates the velocity question from any
    retrieval error. `exclude_query_velocity` also removes the query's velocity
    from the reference, a harder test of generalising to an unseen speed.
    """
    usable = df[df["in_band"]]
    usable = usable[usable[list(DIMS) + list(D.NORMALISED_CORE)].notna().all(axis=1)]
    ref_all, query = split_repeats(usable)
    rows = []
    for (mid, v, fo), q in query.groupby(["material_id", "velocity_mm_s", "force_n"]):
        src = substitute[mid] if substitute else mid
        ref = ref_all[(ref_all["material_id"] == src) & (ref_all["force_n"] == fo)]
        if exclude_query_velocity:
            ref = ref[ref["velocity_mm_s"] != v]
        if ref.empty:
            continue
        ag, co = _agnostic(ref), _conditioned(ref, v, fo)
        for dim in DIMS:
            rows.append({"material_id": mid, "velocity_mm_s": v, "force_n": fo,
                         "dimension": dim,
                         "err_agnostic": float((q[dim] - ag[dim]).abs().mean()),
                         "err_conditioned": float((q[dim] - co[dim]).abs().mean())})
    return pd.DataFrame(rows)


def compare(errors: pd.DataFrame) -> pd.DataFrame:
    """Per velocity and dimension: does conditioning help, by how much, is it
    significant? Holm-corrected across every test in the table.

    `improvement_pct` is the reduction in MAE: negative means conditioning hurt.
    """
    per_material = errors.groupby(["material_id", "velocity_mm_s", "dimension"],
                                  as_index=False)[["err_agnostic", "err_conditioned"]].mean()
    rows = []
    for (v, dim), g in per_material.groupby(["velocity_mm_s", "dimension"], sort=True):
        ag, co = g["err_agnostic"].mean(), g["err_conditioned"].mean()
        test = (paired_test(g["err_agnostic"], g["err_conditioned"])
                if len(g) >= MIN_PAIRS else {"p": np.nan})
        rows.append({"velocity_mm_s": v, "dimension": dim, "n_materials": len(g),
                     "mae_agnostic": ag, "mae_conditioned": co,
                     "improvement_pct": 100 * (ag - co) / ag if ag > 0 else 0.0,
                     "p": test["p"]})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p"].values)
    out["verdict"] = [_verdict(r) for r in out.itertuples()]
    n = int(out["n_materials"].min())
    if smallest_holm_p(n, len(out)) >= ALPHA:
        warnings.warn(f"only {n} materials: {len(out)} Holm-corrected tests cannot "
                      f"reach p < {ALPHA}, so no verdict here is informative")
    return out


def _verdict(r) -> str:
    if not np.isfinite(r.p_holm):
        return "too few materials"
    if r.p_holm >= ALPHA:
        return "no clear difference"
    return "conditioning helps" if r.mae_conditioned < r.mae_agnostic else "conditioning hurts"


def excluded(df: pd.DataFrame) -> pd.DataFrame:
    """Out-of-band recordings per velocity. These must be reported, never hidden."""
    return (df.assign(out_of_band=~df["in_band"]).groupby("velocity_mm_s")["out_of_band"]
            .agg(["sum", "size"]).rename(columns={"sum": "excluded", "size": "total"}))


SCALING_DIMS = ("frequency_hz", "intensity", "duration_ms")


def scaling_exponents(df: pd.DataFrame, dims=SCALING_DIMS) -> pd.DataFrame:
    """Per material and force: the slope b of log(dimension) = a + b log(velocity),
    fitted to the per-velocity medians of in-band recordings.

    f = v / lambda, the relation velocity normalisation assumes, predicts b = 1 for
    frequency and b = -1 for duration. A frequency fixed by the probe or the
    fabric's own resonance predicts b = 0.
    """
    usable = df[df["in_band"].astype(bool)]
    rows = []
    for (mid, fo), g in usable.groupby(["material_id", "force_n"]):
        med = g.groupby("velocity_mm_s")[list(dims)].median()
        if len(med) < 3:
            continue
        lv = np.log(med.index.to_numpy(dtype=float))
        row = {"material_id": mid, "force_n": fo, "n_velocities": len(med)}
        for dim in dims:
            row[dim] = float(np.polyfit(lv, np.log(med[dim].to_numpy(dtype=float)), 1)[0])
        rows.append(row)
    return pd.DataFrame(rows)


def variation_across_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Coefficient of variation across the five velocities of each dimension's
    per-velocity median, per material and force, raw beside normalised. Smaller is
    more velocity-invariant. Pattern has no normalised form and appears once."""
    usable = df[df["in_band"].astype(bool)]
    cols = [*DIMS, *[c for c in D.NORMALISED_CORE if c not in DIMS]]
    rows = []
    for (mid, fo), g in usable.groupby(["material_id", "force_n"]):
        med = g.groupby("velocity_mm_s")[cols].median()
        cv = (med.std(ddof=1) / med.mean().abs()).to_dict()
        rows.append({"material_id": mid, "force_n": fo, **cv})
    return pd.DataFrame(rows)


# --- spectral shift: does any part of the spectrum move with velocity? ---------
# Added 2026-11-06, EXPLORATORY. The centroid is one number and a fixed resonance
# can dominate it while a texture peak underneath still scales as v / lambda. This
# aligns whole log-frequency spectra instead: if the spectrum at velocity v is the
# one at 40 mm/s shifted by log(v / 40), the shift exponent is 1; a spectrum fixed
# in frequency gives 0. "Whitened" divides out the median spectrum of all
# materials at the same velocity and force, the part common to every surface and
# so most plausibly the rig's transfer function.

SHIFT_GRID = np.geomspace(20.0, 1000.0, 241)     # 0.0163 log-units per step
MAX_SHIFT_STEPS = 60                             # +-0.98 in ln f, beyond 60/20 mm/s


def log_spectrum(accel: np.ndarray, fs: float) -> np.ndarray:
    """Log Welch power of the band-passed magnitude on SHIFT_GRID."""
    from scipy.signal import welch
    from . import descriptor as D
    x = D.bandpass(D.to_magnitude(accel), fs)
    f, p = welch(x, fs, nperseg=2048)
    return np.interp(np.log(SHIFT_GRID), np.log(f[1:]), np.log(p[1:] + 1e-20))


def best_shift(ref: np.ndarray, other: np.ndarray, max_steps: int = MAX_SHIFT_STEPS) -> float:
    """ln-frequency shift that best maps `ref` onto `other` (positive = `other` higher)."""
    step = np.log(SHIFT_GRID[1] / SHIFT_GRID[0])
    best, arg = -np.inf, 0
    for k in range(-max_steps, max_steps + 1):
        a, b = (ref[:len(ref) - k], other[k:]) if k >= 0 else (ref[-k:], other[:len(other) + k])
        if len(a) < len(ref) // 2:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if r > best:
            best, arg = r, k
    return arg * step


def whitening_reference(spectra: np.ndarray, meta: pd.DataFrame) -> dict:
    """Median log spectrum per (velocity, force): the part shared by every surface."""
    meta = meta.reset_index(drop=True)
    return {k: np.median(spectra[g.index], axis=0)
            for k, g in meta.groupby(["velocity_mm_s", "force_n"])}


def shift_exponents(spectra: np.ndarray, meta: pd.DataFrame, ref_velocity: float = 40.0,
                    whiten: bool = False, reference: dict | None = None,
                    keys=("material_id", "force_n", "direction_deg")) -> pd.DataFrame:
    """Per cell of `keys`: slope of best ln-shift on ln(v / ref) through the
    origin. `spectra` rows align with `meta` (material_id, velocity_mm_s,
    force_n, direction_deg, and any extra key). `whiten` subtracts the median
    spectrum per (velocity, force) of these same rows, or of `reference` (from
    `whitening_reference`) when given, so held-out rows can be whitened by a
    reference they did not contribute to."""
    meta = meta.reset_index(drop=True)
    s = spectra.copy()
    if whiten:
        ref_spec = reference if reference is not None else whitening_reference(spectra, meta)
        for k, g in meta.groupby(["velocity_mm_s", "force_n"]):
            s[g.index] -= ref_spec[k]
    keys = list(keys)
    rows = []
    for key, g in meta.groupby(keys):
        ref = g[g.velocity_mm_s == ref_velocity]
        others = g[g.velocity_mm_s != ref_velocity]
        if len(ref) != 1 or len(others) < 2:
            continue
        x = np.log(others.velocity_mm_s.to_numpy() / ref_velocity)
        y = np.array([best_shift(s[ref.index[0]], s[i]) for i in others.index])
        rows.append(dict(zip(keys, key),
                         shift_exponent=float(x @ y / (x @ x))))
    return pd.DataFrame(rows)
