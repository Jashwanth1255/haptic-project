"""Loader for the Cluster Haptic Texture Dataset.

Reference
---------
Cluster Haptic Texture Dataset, Scientific Data 13, 756 (2026).
    DOI 10.1038/s41597-026-06760-z
    data: DOI 10.6084/m9.figshare.29438288
    code: github.com/cluster-lab/Cluster-Haptic-Texture-Dataset

Documented contents, from the paper:
    118 materials in 9 categories --
        0-37 Wood, 38-48 Stone, 49-64 Polymer, 65-73 Metal, 74-78 Glass,
        79-81 Composite, 82-86 Ceramic, 87-101 Biological Leather,
        102-117 Cloth.
    Signals: contact audio (44.1 kHz, two microphones), rubber-tip acceleration
    (3-axis MEMS, ~6 kHz), normal force, in-plane (x, y) tip position (100 Hz).
    Sliding velocities 20, 30, 40, 50, 60 mm/s; eight intercardinal directions;
    normal forces 0.5 and 1.0 N. Surfaces also scanned at 300 dpi.

WHAT IS NOT YET VERIFIED
------------------------
The on-disk file layout and the per-file naming convention were not confirmed
when this was written -- the figshare record could not be opened. `CLOTH_IDS`
and the acquisition grid come from the paper and are reliable; the directory
structure in `discover()` is a best guess and is deliberately written to fail
loudly with a readable message rather than silently return nothing.

When you download the dataset, run

    python -m haptic.cli inspect --root /path/to/texture_dataset

which prints what it actually finds. If it does not match, adjust
`FILENAME_PATTERNS` -- that is the only part that should need changing.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

CLOTH_IDS = tuple(range(102, 118))          # 16 cloth materials
VELOCITIES_MM_S = (20.0, 30.0, 40.0, 50.0, 60.0)
FORCES_N = (0.5, 1.0)
ACCEL_FS_HZ = 6000.0
AUDIO_FS_HZ = 44100.0

CATEGORIES = {
    "wood": range(0, 38), "stone": range(38, 49), "polymer": range(49, 65),
    "metal": range(65, 74), "glass": range(74, 79), "composite": range(79, 82),
    "ceramic": range(82, 87), "leather": range(87, 102), "cloth": range(102, 118),
}

# Candidate patterns for pulling (material, velocity, force) out of a path.
# Extend this list rather than rewriting the loader.
FILENAME_PATTERNS = (
    re.compile(r"(?P<mid>\d+)[/_-].*?v(?P<vel>\d+).*?f(?P<force>[\d.]+)", re.I),
    re.compile(r"id(?P<mid>\d+).*?(?P<vel>\d+)mm.*?(?P<force>[\d.]+)N", re.I),
    re.compile(r"(?P<mid>\d+)_(?P<vel>\d+)_(?P<force>[\d.]+)", re.I),
)


def category_of(material_id: int) -> str:
    for name, rng in CATEGORIES.items():
        if material_id in rng:
            return name
    return "unknown"


def _parse(path: Path):
    s = str(path)
    for pat in FILENAME_PATTERNS:
        m = pat.search(s)
        if m:
            try:
                return (int(m.group("mid")), float(m.group("vel")),
                        float(m.group("force")))
            except (ValueError, IndexError):
                continue
    return None


def _read_signal(path: Path) -> np.ndarray:
    suf = path.suffix.lower()
    if suf == ".npy":
        return np.load(path)
    if suf in (".csv", ".txt", ".tsv"):
        delim = "\t" if suf == ".tsv" else ","
        return np.genfromtxt(path, delimiter=delim, skip_header=1)
    if suf == ".wav":
        from scipy.io import wavfile
        _, data = wavfile.read(path)
        return data.astype(float)
    raise ValueError(f"unsupported signal file type: {path.name}")


def discover(root: str | Path, material_ids=CLOTH_IDS,
             pattern: str = "**/*") -> list[dict]:
    """Walk the dataset directory and return recording stubs (no data read yet).

    Each stub: {path, material_id, category, velocity_mm_s, force_n}.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"dataset root not found: {root}\n"
            f"Download from https://doi.org/10.6084/m9.figshare.29438288 "
            f"and unzip into this path.")

    wanted = set(material_ids) if material_ids else None
    found, skipped = [], 0
    for p in root.glob(pattern):
        if not p.is_file() or p.suffix.lower() not in (".npy", ".csv", ".txt", ".tsv", ".wav"):
            continue
        parsed = _parse(p)
        if parsed is None:
            skipped += 1
            continue
        mid, vel, force = parsed
        if wanted is not None and mid not in wanted:
            continue
        found.append({"path": p, "material_id": mid,
                      "category": category_of(mid),
                      "velocity_mm_s": vel, "force_n": force})

    if not found:
        raise RuntimeError(
            f"no recordings matched under {root} "
            f"({skipped} files present but unparseable).\n"
            f"The on-disk layout differs from what FILENAME_PATTERNS expects. "
            f"Run `python -m haptic.cli inspect --root {root}` to see the real "
            f"structure, then add a pattern.")
    return found


def load(stub: dict, fs: float = ACCEL_FS_HZ) -> dict:
    """Read one recording into the shape `descriptor.extract_many` consumes."""
    return {
        "accel": _read_signal(stub["path"]),
        "fs": fs,
        "material_id": stub["material_id"],
        "material_name": f"id{stub['material_id']:03d}",
        "velocity_mm_s": stub["velocity_mm_s"],
        "force_n": stub["force_n"],
    }


def load_names(root: str | Path) -> dict[int, str]:
    """Read texture_list.xlsx / .csv if present, to map ids to material names."""
    root = Path(root)
    for cand in ("texture_list.xlsx", "texture_list.csv", "texture_list.tsv"):
        p = root / cand
        if not p.exists():
            continue
        import pandas as pd
        tbl = pd.read_excel(p) if p.suffix == ".xlsx" else pd.read_csv(p)
        cols = {c.lower().strip(): c for c in tbl.columns}
        idc = next((cols[k] for k in cols if "id" in k), None)
        namec = next((cols[k] for k in cols if "name" in k or "material" in k), None)
        if idc and namec:
            return {int(r[idc]): str(r[namec]) for _, r in tbl.iterrows()
                    if str(r[idc]).strip().isdigit()}
    return {}


def inspect(root: str | Path, limit: int = 40) -> str:
    """Describe what is actually on disk. Run this first, before anything else."""
    root = Path(root)
    if not root.exists():
        return f"not found: {root}"
    lines = [f"root: {root}", ""]
    subs = sorted(p for p in root.iterdir() if p.is_dir())
    lines.append(f"top-level directories ({len(subs)}):")
    lines += [f"  {p.name}" for p in subs[:limit]]
    if len(subs) > limit:
        lines.append(f"  ... and {len(subs) - limit} more")

    files = [p for p in root.glob("**/*") if p.is_file()]
    lines += ["", f"total files: {len(files)}", "", "extensions:"]
    exts: dict[str, int] = {}
    for p in files:
        exts[p.suffix.lower()] = exts.get(p.suffix.lower(), 0) + 1
    for e, n in sorted(exts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {e or '(none)':<10} {n}")

    lines += ["", "sample paths (relative):"]
    for p in files[:limit]:
        rel = p.relative_to(root)
        parsed = _parse(p)
        tag = (f"-> id={parsed[0]} v={parsed[1]} f={parsed[2]}"
               if parsed else "-> UNPARSED")
        lines.append(f"  {rel}  {tag}")

    names = load_names(root)
    if names:
        lines += ["", f"material names found ({len(names)}):"]
        for mid in CLOTH_IDS:
            if mid in names:
                lines.append(f"  {mid}: {names[mid]}")
    else:
        lines += ["", "texture_list not found at root -- cloth names unknown."]
    return "\n".join(lines)
