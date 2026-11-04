"""Loader for the Cluster Haptic Texture Dataset.

Reference
---------
Cluster Haptic Texture Dataset, Scientific Data 13, 756 (2026).
    DOI 10.1038/s41597-026-06760-z
    data: DOI 10.6084/m9.figshare.29438288 (zip, CSV and WAV)
    mirror: huggingface.co/datasets/tamago117/cluster-haptic-texture-dataset (Parquet)
    code: github.com/cluster-lab/Cluster-Haptic-Texture-Dataset

Verified on 2026-11-03 from the repository's documents/dataset_details.md and
documents/texture_list.pdf, and the figshare and Hugging Face file listings:

- Licence: Apache 2.0 on both the figshare record and the Hugging Face mirror.
- Sensor files are named `<id>_<direction>_<velocity>_<force>_<repeat>`:
  direction 0-315 degrees in steps of 45, velocity 20-60 mm/s, force in **mN**
  (500 or 1000), repeat 0 or 1. Figshare holds `sensor_data/accel/<name>.csv`;
  the mirror holds `sensor_data/accel/<id>/<name>.parquet`.
- Acceleration files have columns time, X, Y, Z (g): `time` in seconds in the
  figshare CSVs, `time_ns` in nanoseconds in the mirror Parquet. Documented as
  6 kHz; the mirror files measure about 8.6 kHz (median step 111 us) and are
  unevenly spaced, so every recording is resampled to a uniform 6 kHz grid
  through an anti-aliasing filter, and its measured rate is reported.
- Position files have columns time / time_ns, X, Y (mm) at 100 Hz.
- Cloth is IDs 102-117; their names are in `CLOTH_NAMES`.

Checked on the real files 2026-11-04: accel and position share a time origin
(both start within a millisecond of zero and span the same length).
`steady_window` relies on it, and `load` still refuses a window that falls
outside the acceleration recording.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CLOTH_IDS = tuple(range(102, 118))          # 16 cloth materials
VELOCITIES_MM_S = (20.0, 30.0, 40.0, 50.0, 60.0)
FORCES_N = (0.5, 1.0)
DIRECTIONS_DEG = (0, 45, 90, 135, 180, 225, 270, 315)
REPEATS = (0, 1)
ACCEL_FS_HZ = 6000.0
AUDIO_FS_HZ = 44100.0

# documents/texture_list.pdf, read 2026-11-03
CLOTH_NAMES = {
    102: "Velvet", 103: "Corduroy", 104: "Cotton", 105: "Cotton broadcloth",
    106: "Cotton gauze", 107: "Chiffon", 108: "Lace", 109: "Georgette",
    110: "Organdy", 111: "Mesh", 112: "Wool", 113: "Nylon", 114: "Polyester",
    115: "Silk", 116: "Linen", 117: "Cotton linen",
}

CATEGORIES = {
    "wood": range(0, 38), "stone": range(38, 49), "polymer": range(49, 65),
    "metal": range(65, 74), "glass": range(74, 79), "composite": range(79, 82),
    "ceramic": range(82, 87), "leather": range(87, 102), "cloth": range(102, 118),
}

SENSOR_NAME = re.compile(
    r"^(?P<mid>\d+)_(?P<dir>\d+)_(?P<vel>\d+)_(?P<force>\d+)_(?P<rep>\d+)\.(?:csv|parquet)$")
SIGNAL_SUFFIXES = (".csv", ".parquet")

STEADY_TOLERANCE = 0.25     # speed within 25% of nominal counts as steady sliding
STEADY_SMOOTH = 5           # position samples averaged before differentiating (50 ms)
MIN_STEADY_S = 0.5          # shorter than this, the recording is not used
RATE_TOLERANCE = 0.2        # measured accel rate this far from MEASURED_FS_HZ is reported
MEASURED_FS_HZ = 8600.0     # what the mirror files actually hold (2026-11-04), not the documented 6 kHz
OVERSAMPLE = 3              # interpolation grid, as a multiple of the analysis rate


def category_of(material_id: int) -> str:
    for name, rng in CATEGORIES.items():
        if material_id in rng:
            return name
    return "unknown"


def parse_name(path: str | Path) -> dict | None:
    """The acquisition condition encoded in a sensor file name, or None if the
    name does not follow the documented convention or holds an impossible value.
    Force is converted from mN to N."""
    m = SENSOR_NAME.match(Path(path).name)
    if not m:
        return None
    mid, direction, vel, force, rep = (int(m.group(k)) for k in
                                       ("mid", "dir", "vel", "force", "rep"))
    if (direction not in DIRECTIONS_DEG or float(vel) not in VELOCITIES_MM_S
            or force / 1000 not in FORCES_N or rep not in REPEATS):
        return None
    return {"material_id": mid, "direction_deg": direction,
            "velocity_mm_s": float(vel), "force_n": force / 1000, "repeat": rep}


def repeat_index(direction_deg: int, repeat: int) -> int:
    """0-15: direction-major, so the lower half of the indices is the first four
    directions. `velocity.split_repeats` uses the lower half as reference and the
    upper half as queries, which puts queries in directions the reference never saw."""
    return DIRECTIONS_DEG.index(direction_deg) * len(REPEATS) + repeat


def _sibling(path: Path, signal: str) -> Path | None:
    """The same recording's file for another signal: swap the `accel` folder."""
    parts = list(path.parts)
    if "accel" not in parts:
        return None
    parts[len(parts) - 1 - parts[::-1].index("accel")] = signal
    p = Path(*parts)
    return p if p.exists() else None


def discover(root: str | Path, material_ids=CLOTH_IDS) -> list[dict]:
    """Every acceleration recording under `root`, as stubs (no data read yet)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"dataset root not found: {root}\n"
            f"Fetch the cloth subset with `python fetch_cluster.py --out <dir>`.")
    wanted = set(material_ids) if material_ids else None
    found, unparsed = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SIGNAL_SUFFIXES or "accel" not in p.parts:
            continue
        info = parse_name(p)
        if info is None:
            unparsed.append(p.name)
            continue
        if wanted is not None and info["material_id"] not in wanted:
            continue
        found.append({"path": p, "category": category_of(info["material_id"]), **info})
    if not found:
        raise RuntimeError(
            f"no acceleration recordings matched under {root} "
            f"({len(unparsed)} accel files with other names, e.g. {unparsed[:3]}).\n"
            f"Run `python -m haptic.cli inspect --root {root}` to see what is there.")
    return found


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def with_seconds(table: pd.DataFrame) -> pd.DataFrame:
    """`time` in seconds, from `time` (figshare CSV) or `time_ns` (mirror Parquet)."""
    if "time" in table:
        return table
    if "time_ns" in table:
        return table.assign(time=table["time_ns"].to_numpy(dtype=float) / 1e9)
    raise ValueError(f"no time column; columns are {list(table.columns)}")


def resample(table: pd.DataFrame, columns=("X", "Y", "Z"),
             fs: float = ACCEL_FS_HZ) -> tuple[np.ndarray, np.ndarray, float]:
    """(uniform time grid, values on it, measured mean rate). Duplicate or
    out-of-order timestamps are sorted and dropped before interpolating.

    The real files arrive at about 8.6 kHz although documented as 6 kHz, so the
    samples are interpolated onto a grid at OVERSAMPLE x `fs` and then brought down
    to `fs` through an anti-aliasing filter; interpolating straight onto `fs` would
    fold 3-4.5 kHz content into the analysis band."""
    from scipy.signal import resample_poly
    t = with_seconds(table)["time"].to_numpy(dtype=float)
    order = np.argsort(t, kind="stable")
    t = t[order]
    keep = np.r_[True, np.diff(t) > 0]
    t = t[keep]
    if len(t) < 2:
        raise ValueError("fewer than two distinct timestamps")
    values = table[list(columns)].to_numpy(dtype=float)[order][keep]
    measured = (len(t) - 1) / (t[-1] - t[0])
    fine = np.arange(t[0], t[-1], 1 / (OVERSAMPLE * fs))
    up = np.column_stack([np.interp(fine, t, values[:, k]) for k in range(values.shape[1])])
    out = resample_poly(up - up.mean(axis=0), 1, OVERSAMPLE, axis=0) + up.mean(axis=0)
    grid = t[0] + np.arange(len(out)) / fs
    return grid, out, float(measured)


def steady_window(position: pd.DataFrame, velocity_mm_s: float,
                  tolerance: float = STEADY_TOLERANCE,
                  smooth: int = STEADY_SMOOTH) -> tuple[float, float] | None:
    """The longest stretch where the probe moves at the nominal speed, so the
    approach, acceleration and stop are not described as texture."""
    p = with_seconds(position).sort_values("time").drop_duplicates("time")
    t, x, y = (p[c].to_numpy(dtype=float) for c in ("time", "X", "Y"))
    if len(t) < smooth + 2:
        return None
    kernel = np.ones(smooth) / smooth
    xs, ys = np.convolve(x, kernel, "valid"), np.convolve(y, kernel, "valid")
    ts = np.convolve(t, kernel, "valid")
    speed = np.hypot(np.diff(xs), np.diff(ys)) / np.diff(ts)
    ok = np.abs(speed - velocity_mm_s) <= tolerance * velocity_mm_s
    best, start = (0, 0), None
    for i, flag in enumerate(np.r_[ok, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best[1] == best[0]:
        return None
    return float(ts[best[0]]), float(ts[best[1]])


def load(stub: dict, fs: float = ACCEL_FS_HZ, steady: bool = True) -> dict:
    """One recording, resampled to `fs`, as `descriptor.extract` consumes it.

    With `steady`, only the stretch at the nominal speed is kept. A recording
    without a position file, or with less than MIN_STEADY_S of steady sliding,
    raises ValueError, so the caller can count it rather than describe it.
    """
    grid, accel, measured = resample(read_table(stub["path"]), fs=fs)
    window = None
    if steady:
        pos_path = _sibling(stub["path"], "position")
        if pos_path is None:
            raise ValueError(f"no position file for {stub['path'].name}")
        window = steady_window(read_table(pos_path), stub["velocity_mm_s"])
        if window is None or window[1] - window[0] < MIN_STEADY_S:
            raise ValueError(f"{stub['path'].name}: under {MIN_STEADY_S} s of steady sliding")
        if window[0] < grid[0] or window[1] > grid[-1]:
            raise ValueError(f"{stub['path'].name}: steady window {window} lies outside "
                             f"the acceleration recording {grid[0]:.2f}-{grid[-1]:.2f} s")
        inside = (grid >= window[0]) & (grid <= window[1])
        accel = accel[inside]
    mid = stub["material_id"]
    return {"accel": accel, "fs": fs, "material_id": mid,
            "material_name": CLOTH_NAMES.get(mid, f"id{mid:03d}"),
            "velocity_mm_s": stub["velocity_mm_s"], "force_n": stub["force_n"],
            "direction_deg": stub["direction_deg"],
            "repeat": repeat_index(stub["direction_deg"], stub["repeat"]),
            "measured_fs_hz": measured,
            "rate_off": abs(measured - MEASURED_FS_HZ) > RATE_TOLERANCE * MEASURED_FS_HZ,
            "steady_s": None if window is None else window[1] - window[0]}


def load_names(root: str | Path) -> dict[int, str]:
    """texture_list.xlsx or .csv at the root if present, else `CLOTH_NAMES`."""
    root = Path(root)
    for cand in ("texture_list.xlsx", "texture_list.csv"):
        p = root / cand
        if not p.exists():
            continue
        tbl = pd.read_excel(p) if p.suffix == ".xlsx" else pd.read_csv(p)
        cols = {c.lower().strip(): c for c in tbl.columns}
        idc = next((cols[k] for k in cols if "id" in k), None)
        namec = next((cols[k] for k in cols if "name" in k and "ja" not in k), None)
        if idc and namec:
            return {int(r[idc]): str(r[namec]) for _, r in tbl.iterrows()
                    if str(r[idc]).strip().isdigit()}
    return dict(CLOTH_NAMES)


def inspect(root: str | Path, limit: int = 20) -> str:
    """Describe what is actually on disk. Run this first, before anything else."""
    root = Path(root)
    if not root.exists():
        return f"not found: {root}"
    files = [p for p in root.rglob("*") if p.is_file()]
    lines = [f"root: {root}", f"total files: {len(files)}", "", "files per folder:"]
    folders: dict[str, int] = {}
    for p in files:
        key = "/".join(p.relative_to(root).parts[:2])
        folders[key] = folders.get(key, 0) + 1
    lines += [f"  {k:<40} {v}" for k, v in sorted(folders.items())]

    accel = [p for p in files if "accel" in p.parts and p.suffix.lower() in SIGNAL_SUFFIXES]
    parsed = [(p, parse_name(p)) for p in accel]
    bad = [p for p, i in parsed if i is None]
    lines += ["", f"acceleration files: {len(accel)}, parsed {len(accel) - len(bad)}, "
                  f"unparsed {len(bad)}"]
    lines += [f"  UNPARSED {p.relative_to(root)}" for p in bad[:limit]]
    for p, i in [x for x in parsed if x[1]][:5]:
        lines.append(f"  {p.relative_to(root)} -> {i}")
        lines.append(f"    position file: {'found' if _sibling(p, 'position') else 'MISSING'}")
    ok = [i for _, i in parsed if i]
    if ok:
        df = pd.DataFrame(ok)
        per = df.groupby("material_id").size()
        expected = len(DIRECTIONS_DEG) * len(VELOCITIES_MM_S) * len(FORCES_N) * len(REPEATS)
        lines += ["", f"recordings per material (expected {expected}):"]
        names = load_names(root)
        lines += [f"  {mid:>3} {names.get(mid, '?'):<18} {n}" for mid, n in per.items()]
        if accel:
            t = read_table(accel[0])
            lines += ["", f"first file columns: {list(t.columns)}, rows {len(t)}"]
            if "time" in t or "time_ns" in t:
                dt = np.diff(with_seconds(t)["time"].to_numpy(dtype=float))
                lines.append(f"  time step: median {np.median(dt) * 1e3:.3f} ms, "
                             f"min {dt.min() * 1e3:.3f}, max {dt.max() * 1e3:.3f}")
    return "\n".join(lines)
