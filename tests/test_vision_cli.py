"""End to end through the CLI, on a tiny fake dataset. Scaffolding only."""
import json
import subprocess
import sys
from dataclasses import replace

import pytest
from PIL import Image

import make_kaggle_script as K
from vision import cli as C
from vision import data as V


@pytest.fixture
def fake(tmp_path, monkeypatch):
    spec = replace(V.DTD, n_classes=2, per_class={"train": 3, "val": 1, "test": 1})
    monkeypatch.setitem(V.SPECS, "dtd", spec)
    root = tmp_path / "dtd"
    for c, colour in (("banded", (220, 30, 30)), ("woven", (30, 30, 220))):
        (root / "images" / c).mkdir(parents=True)
        for i in range(5):
            Image.new("RGB", (64, 64), colour).save(root / "images" / c / f"{c}_{i}.jpg")
    (root / "labels").mkdir()
    for fold in range(1, spec.n_folds + 1):
        for split, ids in (("train", (0, 1, 2)), ("val", (3,)), ("test", (4,))):
            (root / "labels" / f"{split}{fold}.txt").write_text("\n".join(
                f"{c}/{c}_{i}.jpg" for c in ("banded", "woven") for i in ids))
    return root


def test_check_passes_on_a_good_dataset(fake, capsys):
    assert C.main(["check", "--dataset", "dtd", "--data", str(fake)]) == 0
    assert "2 classes" in capsys.readouterr().out


def test_train_end_to_end_is_labelled_smoke(fake, tmp_path, monkeypatch, capsys):
    from vision import train as T
    import torchvision.models as tvm
    monkeypatch.setattr(T, "build_model", lambda k, p: _small(tvm, k))

    assert C.main(["train", "--dataset", "dtd", "--data", str(fake),
                   "--out", str(tmp_path / "runs"), "--epochs", "2",
                   "--freeze-epochs", "1", "--limit-per-class", "2",
                   "--batch-size", "2", "--workers", "0",
                   "--no-pretrained"]) == 0
    out = capsys.readouterr().out
    assert "SMOKE, not a result" in out
    saved = json.loads((tmp_path / "runs" / "dtd_fold1_seed0_smoke" /
                        "test_metrics.json").read_text())
    assert saved["smoke"] is True and saved["n"] == 2


def test_require_gpu_fails_before_any_training(fake, tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no GPU"):
        C.main(["train", "--dataset", "dtd", "--data", str(fake),
                "--out", str(tmp_path), "--require-gpu"])
    assert not list(tmp_path.glob("dtd_*"))


def _small(tvm, k):
    from torch import nn
    m = tvm.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, k)
    return m


def test_limit_keeps_every_class():
    items = [("p", 0)] * 5 + [("q", 1)] * 5
    kept = C._limit(items, 2)
    assert [y for _, y in kept] == [0, 0, 1, 1]
    assert C._limit(items, 0) == items


def test_kaggle_script_carries_the_whole_package_and_is_valid_python(tmp_path):
    text = K.build([["check", "--dataset", "dtd", "--data", "/kaggle/input"]], "abc")
    compile(text, "k.py", "exec")
    ns = {}
    head = text.split("src = pathlib.Path")[0]
    exec(head, ns)
    assert {"train.py", "data.py", "metrics.py", "cli.py"} <= set(ns["SOURCES"])
    assert ns["COMMIT"] == "abc"


def test_split_commands():
    assert K.split_commands(["a", "--then", "b", "c"]) == [["a"], ["b", "c"]]
    with pytest.raises(ValueError):
        K.split_commands(["a", "--then"])
