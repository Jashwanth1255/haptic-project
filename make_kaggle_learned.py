"""Pack the CNN baseline (make_learned.py) into one Kaggle kernel script.

    python make_kaggle_learned.py --out <scratch>/kaggle_learned.py

The kernel downloads the cloth accel and position files from the same pinned
Hugging Face revision as fetch_cluster.py, rebuilds the spectrograms with the same
code, and runs make_learned.py on the GPU. It carries a slim copy of the latest
o1_gate run (run.json, config.json and the descriptor keys plus in_band), so it
trains on exactly the recordings the local analyses used. The repository stays the
source of truth: the kernel holds a generated copy and its results come back as a
saved run under results/.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import runlog

HERE = Path(__file__).parent
SOURCES = [*sorted((HERE / "haptic").glob("*.py")), HERE / "runlog" / "__init__.py",
           HERE / "fetch_cluster.py", HERE / "make_learned.py"]
KEYS = ["material_id", "velocity_mm_s", "force_n", "direction_deg", "repeat", "in_band"]

TEMPLATE = '''\
import os, pathlib, subprocess, sys, shutil

CODE = {code!r}
SOURCES = {sources!r}
GATE = {gate!r}

src = pathlib.Path("/kaggle/working/src")
for name, text in SOURCES.items():
    p = src / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
gate_dir = pathlib.Path("/kaggle/working/results/o1_gate") / GATE["run_id"]
gate_dir.mkdir(parents=True, exist_ok=True)
for name, text in GATE["files"].items():
    (gate_dir / name).write_text(text)

env = dict(os.environ, RUNLOG_COMMIT=CODE, RUNLOG_ROOT="/kaggle/working/results",
           PYTHONUTF8="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
print("code", CODE, flush=True)
import torch
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

def run(*args):
    print("\\n$", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-W", "ignore", *args], cwd=src, env=env, check=True)

run("fetch_cluster.py", "--out", "/tmp/cluster", "--no-images", "--retries", "40")
run("make_learned.py", "--root", "/tmp/cluster",
    "--cache", "/kaggle/working/derived/spectrograms.npz")
os.remove("/kaggle/working/derived/spectrograms.npz")
shutil.rmtree(gate_dir.parent)          # the gate stub is an input, not a result
print("done", flush=True)
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_kaggle_learned")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    sources = {str(p.relative_to(HERE)).replace("\\", "/"): p.read_text(encoding="utf-8")
               for p in SOURCES}
    digest = hashlib.sha256("".join(sources[k] for k in sorted(sources)).encode()).hexdigest()[:12]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", *sources], cwd=HERE,
                                capture_output=True, text=True).stdout.strip())
    code = f"{commit}{'+uncommitted' if dirty else ''} sources-sha256:{digest}"

    gate = runlog.load_run("o1_gate")
    d = gate.table("descriptors")[KEYS]
    files = {"run.json": (gate.path / "run.json").read_text(encoding="utf-8"),
             "config.json": (gate.path / "config.json").read_text(encoding="utf-8"),
             "metrics.json": "{}",
             "descriptors.csv": d.to_csv(index=False)}
    text = TEMPLATE.format(code=code, sources=sources,
                           gate={"run_id": gate.info["run_id"], "files": files})
    Path(a.out).write_text(text, encoding="utf-8")
    print(f"{a.out}: {len(text) / 1024:.0f} KB, code {code}, gate run {gate.info['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
