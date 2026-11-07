"""Pack round 6 (analysis P, whitened-spectrogram CNN) into a Kaggle GPU kernel.

    python make_kaggle_round6.py --out-dir <dir>

The kernel reads the round-2 features kernel's output (`accel_specs.npz`, the
same spectrograms the raw CNN runs used) and carries the gate keys and the two
registered raw-CNN fold tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import make_kaggle_round2 as K2
import runlog

HERE = Path(__file__).parent
SOURCES = [*K2.SOURCES, HERE / "make_round6_cnn.py"]
RAW_CLOTH = ("audio_vs_accel", "20261107T110948_431117Z")
RAW_118 = ("benchmark118_cnn", "20261107T125519_960150Z")

BODY = '''
import torch
print("cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
hits = glob.glob("/kaggle/input/**/features/accel_specs.npz", recursive=True)
if not hits:
    raise SystemExit("features kernel output not attached: " + str(os.listdir("/kaggle/input")))
features = str(pathlib.Path(hits[0]).parent)
print("features from", features, flush=True)
run("make_round6_cnn.py", "--features", features, "--gate-keys", "inputs/gate_keys.csv",
    "--raw-cloth-folds", "inputs/raw_cloth_folds.csv", "--raw-118-folds", "inputs/raw_118_folds.csv",
    "--checkpoints", "/kaggle/working/derived/round6")
print("done", flush=True)
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_kaggle_round6")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import hashlib
    import subprocess
    sources = {str(p.relative_to(HERE)).replace("\\", "/"): p.read_text(encoding="utf-8")
               for p in SOURCES}
    digest = hashlib.sha256("".join(sources[k] for k in sorted(sources)).encode()).hexdigest()[:12]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", *sources], cwd=HERE,
                                capture_output=True, text=True).stdout.strip())
    code = f"{commit}{'+uncommitted' if dirty else ''} sources-sha256:{digest}"

    gate = runlog.load_run("o1_gate")
    cloth = runlog.load_run(RAW_CLOTH[0])
    r118 = runlog.load_run(RAW_118[0])
    for run, (_, rid) in ((cloth, RAW_CLOTH), (r118, RAW_118)):
        if run.info["run_id"] != rid:
            raise SystemExit(f"pre-registration names {rid}, latest is {run.info['run_id']}")
    extra = {"inputs/gate_keys.csv": gate.table("descriptors")[K2.GATE_KEYS].to_csv(index=False),
             "inputs/raw_cloth_folds.csv": cloth.table("folds").to_csv(index=False),
             "inputs/raw_118_folds.csv": r118.table("folds_material_id_direction").to_csv(index=False)}
    (out / "cnn6.py").write_text(K2.PRELUDE.format(code=code, sources=sources, extra=extra) + BODY,
                                 encoding="utf-8")
    print(f"{out / 'cnn6.py'}: {(out / 'cnn6.py').stat().st_size / 1024:.0f} KB; code {code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
