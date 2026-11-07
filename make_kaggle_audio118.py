"""Pack analysis M's extraction into a Kaggle CPU kernel script.

    python make_kaggle_audio118.py --out <file.py>

Downloads audio + position for all 118 materials (pinned mirror revision) and
writes audio118.csv to /kaggle/working/audio118/. Analysis runs locally.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import make_kaggle_round2 as K

BODY = '''
try:
    import soundfile  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "soundfile"], check=True)
run("fetch_cluster.py", "--out", "/tmp/cluster", "--all-materials", "--no-images",
    "--signals", "audio", "position", "--retries", "300", "--threads", "4")
run("make_audio118.py", "extract", "--root", "/tmp/cluster", "--out", "/kaggle/working/audio118",
    "--workers", str(os.cpu_count() or 4))
print("done", flush=True)
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_kaggle_audio118")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    here = K.HERE
    files = [*sorted((here / "haptic").glob("*.py")), here / "runlog" / "__init__.py",
             here / "fetch_cluster.py", here / "make_audio118.py"]
    sources = {str(p.relative_to(here)).replace("\\\\", "/"): p.read_text(encoding="utf-8")
               for p in files}
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                            capture_output=True, text=True).stdout.strip()
    text = K.PRELUDE.format(code=commit, sources=sources, extra={}) + BODY
    Path(a.out).write_text(text, encoding="utf-8")
    print(f"{a.out}: {Path(a.out).stat().st_size / 1024:.0f} KB, code {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
