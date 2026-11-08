"""Saved runs: every number quoted in the paper must trace back to one.

    save_run("o3_synthetic", config, metrics, {"closed": df}, synthetic=True)

writes `results/<name>/<run_id>/` holding

    run.json        name, time, git commit, dirty-tree flag, synthetic flag, argv
    config.json     what the run was asked to do
    metrics.json    the numbers it produced
    <artifact>      tables as .csv, text as .txt, JSON-able objects as .json,
                    figures as .png, existing files copied

A run directory appears only once it is complete. `synthetic` is required, so no
run can be saved without saying whether its numbers are scaffolding.

Set RUNLOG_ROOT to save somewhere other than the repository's `results/`. Where
there is no git checkout, as on Kaggle, RUNLOG_COMMIT supplies the commit.
"""
from __future__ import annotations

import dataclasses
import json
import math
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = "results"
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
RUN_FILES = ("run.json", "config.json", "metrics.json")


def results_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    return Path(os.environ.get("RUNLOG_ROOT") or REPO / RESULTS_DIR)


def git_state(repo: Path = REPO) -> tuple[str | None, bool | None]:
    """(commit, dirty). Saved results under `results/` do not make a tree dirty,
    or every run would mark the next one."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("RUNLOG_COMMIT") or None, None
    return head, bool(_code_changes(status))


def _code_changes(status: str) -> list[str]:
    """Paths from `git status --porcelain` that can change what code runs: any
    tracked change outside results/, and untracked .py files. Untracked documents
    (a plan, a draft) do not make a run dirty."""
    out = []
    for line in status.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip('"')
        if path.startswith(RESULTS_DIR + "/"):
            continue
        if line.startswith("??") and not (path.endswith(".py") or path.endswith("/")):
            continue
        out.append(path)
    return out


SOURCE_SKIP = {".git", RESULTS_DIR, "data", "derived", "__pycache__", ".pytest_cache", "paper"}


def source_digest(repo: Path = REPO) -> str:
    """sha256 over every .py file under the repository (path and bytes, sorted),
    skipping data, results and caches. Two runs with the same digest ran the same
    code, committed or not."""
    h = hashlib.sha256()
    files = []
    for d, sub, names in os.walk(repo):
        sub[:] = sorted(s for s in sub if s not in SOURCE_SKIP and not s.startswith("."))
        files += [Path(d) / n for n in names if n.endswith(".py")]
    files.sort(key=lambda f: f.relative_to(repo).as_posix())
    for f in files:
        h.update(f.relative_to(repo).as_posix().encode() + bytes([0]))
        h.update(f.read_bytes().replace(bytes([13, 10]), bytes([10])) + bytes([0]))
    return h.hexdigest()


def dirty_paths(repo: Path = REPO) -> list[str] | None:
    try:
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return _code_changes(status)


def plain(obj):
    """JSON-ready copy. NaN and infinity become null; tables are refused, because
    they belong in artifacts, where they keep their columns."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        raise TypeError("save tables as artifacts, not inside config or metrics")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return plain(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, np.generic):
        return plain(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot save {type(obj).__name__} as JSON")


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(plain(obj), indent=2, allow_nan=False), encoding="utf-8")


def _write_artifact(folder: Path, name: str, value) -> None:
    if not NAME.match(name) or name.split(".")[0] in {"run", "config", "metrics"}:
        raise ValueError(f"bad artifact name {name!r}")
    if isinstance(value, pd.DataFrame):
        value.to_csv(folder / f"{name}.csv",
                     index=not isinstance(value.index, pd.RangeIndex))
    elif isinstance(value, pd.Series):
        value.to_frame().to_csv(folder / f"{name}.csv")
    elif isinstance(value, str):
        (folder / f"{name}.txt").write_text(value, encoding="utf-8")
    elif isinstance(value, Path):
        shutil.copy2(value, folder / f"{name}{value.suffix}")
    elif hasattr(value, "savefig"):
        value.savefig(folder / f"{name}.png", dpi=150, bbox_inches="tight")
    else:
        _dump(folder / f"{name}.json", value)


def save_run(name: str, config: dict, metrics: dict, artifacts: dict | None = None,
             *, synthetic: bool, root: str | Path | None = None) -> Path:
    """Write one run and return its directory."""
    if not NAME.match(name):
        raise ValueError(f"bad run name {name!r}")
    if not isinstance(synthetic, bool):
        raise TypeError("synthetic must be True or False")
    commit, dirty = git_state()
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S_%fZ")
    base = results_root(root) / name
    base.mkdir(parents=True, exist_ok=True)
    tmp = base / f".partial-{uuid.uuid4().hex}"
    tmp.mkdir()
    try:
        _dump(tmp / "run.json", {
            "name": name, "run_id": run_id, "created_utc": now.isoformat(),
            "synthetic": synthetic, "git_commit": commit, "git_dirty": dirty,
            "dirty_paths": dirty_paths() if dirty else [],
            "source_sha256": source_digest(),
            "argv": sys.argv, "python": sys.version.split()[0],
            "platform": platform.platform()})
        _dump(tmp / "config.json", config)
        _dump(tmp / "metrics.json", metrics)
        for key, value in (artifacts or {}).items():
            _write_artifact(tmp, key, value)
        final = base / run_id
        while final.exists():
            final = base / f"{run_id}_{uuid.uuid4().hex[:6]}"
        tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if dirty:
        print(f"warning: run {name} saved from a tree with uncommitted changes",
              file=sys.stderr)
    return final


@dataclass(frozen=True)
class Run:
    path: Path
    info: dict
    config: dict
    metrics: dict

    @property
    def artifacts(self) -> dict[str, Path]:
        return {p.stem: p for p in sorted(self.path.iterdir())
                if p.is_file() and p.name not in RUN_FILES}

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.path / f"{name}.csv")


def _runs(base: Path) -> list[Path]:
    return sorted(p for p in base.iterdir()
                  if p.is_dir() and not p.name.startswith(".")
                  and (p / "run.json").exists())


def load_run(where: str | Path, root: str | Path | None = None) -> Run:
    """A run directory, or a run name, which loads that name's latest run."""
    path = Path(where)
    if not (path / "run.json").exists():
        base = results_root(root) / str(where)
        found = _runs(base) if base.is_dir() else []
        if not found:
            raise FileNotFoundError(f"no saved run for {where!r}")
        path = found[-1]
    read = lambda f: json.loads((path / f).read_text(encoding="utf-8"))
    return Run(path, read("run.json"), read("config.json"), read("metrics.json"))


def list_runs(name: str | None = None, root: str | Path | None = None) -> pd.DataFrame:
    """One row per saved run, oldest first."""
    top = results_root(root)
    bases = [top / name] if name else sorted(p for p in top.glob("*") if p.is_dir())
    rows = []
    for base in bases:
        if not base.is_dir():
            continue
        for p in _runs(base):
            info = json.loads((p / "run.json").read_text(encoding="utf-8"))
            rows.append({k: info.get(k) for k in
                         ("name", "run_id", "created_utc", "synthetic",
                          "git_commit", "git_dirty")} | {"path": str(p)})
    cols = ["name", "run_id", "created_utc", "synthetic", "git_commit", "git_dirty", "path"]
    return pd.DataFrame(rows, columns=cols).sort_values("created_utc", ignore_index=True)
