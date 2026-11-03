"""Pack the `vision` package into one script for a Kaggle kernel.

The repository stays the source of truth: the kernel holds a generated copy,
never edits of its own. Everything after `--` goes to `vision.cli`, once per
`--then`-separated command.

    python make_kaggle_script.py --out k.py -- train --dataset dtd --data /kaggle/input
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

TEMPLATE = '''\
import os
import pathlib
import subprocess
import sys

COMMIT = {commit!r}
SOURCES = {sources!r}
COMMANDS = {commands!r}

src = pathlib.Path("/kaggle/working/src")
for name, text in SOURCES.items():
    path = src / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)

# No git checkout here: saved runs take the commit from the environment.
env = dict(os.environ, RUNLOG_COMMIT=COMMIT,
           RUNLOG_ROOT="/kaggle/working/results")
print("code at commit", COMMIT, flush=True)
for args in COMMANDS:
    print("\\n$ vision.cli", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", "vision.cli", *args],
                   cwd=src, env=env, check=True)
'''

PACKAGES = ("vision", "runlog")


def split_commands(args: list[str]) -> list[list[str]]:
    commands, current = [], []
    for a in args:
        if a == "--then":
            commands.append(current)
            current = []
        else:
            current.append(a)
    commands.append(current)
    if not all(commands):
        raise ValueError("empty command")
    return commands


def build(commands: list[list[str]], commit: str) -> str:
    sources = {f"{pkg}/{p.name}": p.read_text(encoding="utf-8")
               for pkg in PACKAGES for p in sorted((HERE / pkg).glob("*.py"))}
    return TEMPLATE.format(commit=commit, sources=sources, commands=commands)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        raise SystemExit("give the vision.cli command after --")
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv[:cut])
    res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                         capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", *PACKAGES], cwd=HERE,
                           capture_output=True, text=True).stdout.strip()
    commit = res.stdout.strip() + ("+uncommitted" if dirty else "")
    Path(a.out).write_text(build(split_commands(argv[cut + 1:]), commit),
                           encoding="utf-8")
    print(f"wrote {a.out} (code at {commit})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
