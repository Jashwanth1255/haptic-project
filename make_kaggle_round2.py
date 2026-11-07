"""Pack round 2 into two Kaggle kernel scripts.

    python make_kaggle_round2.py --out-dir <dir>

features kernel (CPU): downloads accel + position for all 118 materials and the
    cloth contact audio from the pinned Hugging Face revision, runs
    make_round2_features.py (features + linear benchmark, analysis D). Download
    time is spent on CPU, not GPU quota.
cnn kernel (GPU): reads the features kernel's output and runs make_round2_cnn.py
    (analyses B, C, D). It carries the cloth gate keys, the o1_learned_baseline
    folds and the confirmed velocity exponent.

The repository stays the source of truth: kernels hold generated copies, and
results come back as saved runs under results/.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import runlog

HERE = Path(__file__).parent
SOURCES = [*sorted((HERE / "haptic").glob("*.py")), HERE / "runlog" / "__init__.py",
           HERE / "fetch_cluster.py", HERE / "make_round2_features.py", HERE / "make_round2_cnn.py"]
GATE_KEYS = ["material_id", "velocity_mm_s", "force_n", "direction_deg", "repeat", "in_band"]

PRELUDE = '''\
import os, pathlib, subprocess, sys, shutil, glob

CODE = {code!r}
SOURCES = {sources!r}
EXTRA = {extra!r}

src = pathlib.Path("/kaggle/working/src")
for name, text in {{**SOURCES, **EXTRA}}.items():
    p = src / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
env = dict(os.environ, RUNLOG_COMMIT=CODE, RUNLOG_ROOT="/kaggle/working/results",
           PYTHONUTF8="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
print("code", CODE, flush=True)

def run(*args):
    print("\\n$", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-W", "ignore", *args], cwd=src, env=env, check=True)
'''

FEATURES_BODY = '''
try:
    import soundfile  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "soundfile"], check=True)
run("fetch_cluster.py", "--out", "/tmp/cluster", "--all-materials", "--no-images", "--retries", "300")
run("fetch_cluster.py", "--out", "/tmp/cluster", "--no-images", "--with", "audio", "--retries", "300")
run("make_round2_features.py", "--root", "/tmp/cluster", "--out", "/kaggle/working/features",
    "--workers", str(os.cpu_count() or 4))
print("done", flush=True)
'''

CNN_BODY = '''
import torch
print("cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
hits = glob.glob("/kaggle/input/**/features/features.csv", recursive=True)
if not hits:
    raise SystemExit("features kernel output not attached: " + str(os.listdir("/kaggle/input")))
features = str(pathlib.Path(hits[0]).parent)
print("features from", features, flush=True)
run("make_round2_cnn.py", "--features", features,
    "--gate-keys", "inputs/gate_keys.csv", "--baseline-folds", "inputs/baseline_folds.csv",
    "--k-confirmed", K_CONFIRMED, "--checkpoints", "/kaggle/working/derived/round2")
print("done", flush=True)
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_kaggle_round2")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sources = {str(p.relative_to(HERE)).replace("\\", "/"): p.read_text(encoding="utf-8")
               for p in SOURCES}
    digest = hashlib.sha256("".join(sources[k] for k in sorted(sources)).encode()).hexdigest()[:12]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", *sources], cwd=HERE,
                                capture_output=True, text=True).stdout.strip())
    code = f"{commit}{'+uncommitted' if dirty else ''} sources-sha256:{digest}"

    feat = PRELUDE.format(code=code, sources=sources, extra={}) + FEATURES_BODY
    (out / "features.py").write_text(feat, encoding="utf-8")

    gate = runlog.load_run("o1_gate")
    base = runlog.load_run("o1_learned_baseline")
    conf = runlog.load_run("velocity_confirm_cloth")
    if base.info["run_id"] != "20261106T214141_196418Z":
        raise SystemExit("pre-registration names o1_learned_baseline/20261106T214141_196418Z")
    if conf.info["run_id"] != "20261105T203252_596852Z":
        raise SystemExit("pre-registration names velocity_confirm_cloth/20261105T203252_596852Z")
    k = f"{conf.metrics['cloth_median']:.3f}"
    if k != "0.338":
        raise SystemExit(f"confirmed exponent {k}, pre-registration says 0.338")
    extra = {"inputs/gate_keys.csv": gate.table("descriptors")[GATE_KEYS].to_csv(index=False),
             "inputs/baseline_folds.csv": base.table("folds").to_csv(index=False)}
    cnn = (PRELUDE.format(code=code, sources=sources, extra=extra)
           + CNN_BODY.replace("K_CONFIRMED", repr(k)))
    (out / "cnn.py").write_text(cnn, encoding="utf-8")
    for f in ("features.py", "cnn.py"):
        print(f"{out / f}: {(out / f).stat().st_size / 1024:.0f} KB")
    print(f"code {code}; gate {gate.info['run_id']}; k {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
