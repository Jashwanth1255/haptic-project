"""Product-photograph transfer: labeller agreement, and the paired gap per fold."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

import runlog
from vision import metrics as M
from vision import transfer as P
from vision.textile import TEXTILE_CLASSES


def _csv(tmp_path, name, rows):
    path = tmp_path / name
    pd.DataFrame(rows, columns=["image", "label"]).to_csv(path, index=False)
    return path


def test_kappa_matches_sklearn():
    rng = np.random.default_rng(0)
    a = rng.choice(TEXTILE_CLASSES, 80)
    b = np.where(rng.random(80) < 0.7, a, rng.choice(TEXTILE_CLASSES, 80))
    assert P.cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))


def test_kappa_is_zero_at_chance_and_one_when_identical():
    assert P.cohen_kappa(["woven"] * 2 + ["gauzy"] * 2,
                         ["woven", "gauzy"] * 2) == pytest.approx(0.0)
    assert P.cohen_kappa(["woven", "gauzy"], ["woven", "gauzy"]) == 1.0


def test_labels_outside_the_registered_classes_are_refused(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        P.read_labels(_csv(tmp_path, "a.csv", [("x.jpg", "denim")]))


def test_duplicate_images_are_refused(tmp_path):
    with pytest.raises(ValueError, match="duplicated"):
        P.read_labels(_csv(tmp_path, "a.csv", [("x.jpg", "woven"), ("x.jpg", "gauzy")]))


def test_only_agreed_photographs_are_scored_and_the_rest_counted():
    a = pd.Series({"1": "woven", "2": "gauzy", "3": "knitted", "4": "woven"})
    b = pd.Series({"1": "woven", "2": "pleated", "3": "knitted", "5": "woven"})
    assert list(P.agreed(a, b).index) == ["1", "3"]
    g = P.agreement(a, b)
    assert (g["n_both"], g["n_agreed"], g["only_a"], g["only_b"]) == (3, 2, ["4"], ["5"])


def _report(correct_per_class: dict, n=4):
    y, p = [], []
    for i, c in enumerate(TEXTILE_CLASSES):
        k = correct_per_class.get(c, n)
        y += [i] * n
        p += [i] * k + [(i + 1) % len(TEXTILE_CLASSES)] * (n - k)
    return M.evaluate(np.array(y), np.array(p), TEXTILE_CLASSES)


def _reference(recall=1.0, top1=0.8):
    return top1, pd.DataFrame({"class": TEXTILE_CLASSES, "recall": recall})


def test_gap_is_photo_minus_dtd_per_fold_and_per_class():
    photo = {1: _report({"woven": 1}), 2: _report({"woven": 3})}
    folds, per_class = P.gap_tables(photo, {1: _reference(), 2: _reference()})
    assert folds["gap_top1"].tolist() == pytest.approx(
        [r.top1 - 0.8 for r in photo.values()])
    woven = per_class.set_index("class").loc["woven"]
    assert woven["gap_mean"] == pytest.approx(-0.5)
    assert per_class.iloc[0]["class"] == "woven"        # worst first
    assert per_class["few_photos"].all()             # 4 photos each, below 5


def test_dtd_reference_reads_the_saved_fold(tmp_path):
    folder = tmp_path / "dtd_fold3_seed0" / "textile12"
    folder.mkdir(parents=True)
    (folder / "textile_metrics.json").write_text(json.dumps(
        {"fold": 3, "restricted": {"top1": 0.81}}))
    pd.DataFrame({"class": ["woven"], "recall": [0.5]}).to_csv(
        folder / "restricted_per_class.csv", index=False)
    top1, table = P.dtd_reference(tmp_path, 3)
    assert top1 == 0.81 and table["recall"].iloc[0] == 0.5


def test_committed_dtd_references_exist_for_every_fold():
    from pathlib import Path
    for fold in range(1, 11):
        top1, table = P.dtd_reference(Path(__file__).parent.parent / "results/o2", fold)
        assert 0 < top1 < 1 and set(table["class"]) == set(TEXTILE_CLASSES)


def test_transfer_cli_end_to_end_on_fake_photographs(tmp_path, monkeypatch, capsys):
    import torch
    import torchvision.models as tvm
    from PIL import Image
    from torch import nn

    from vision import cli as C
    from vision import train as T

    torch.manual_seed(0)
    classes = ["banded", *TEXTILE_CLASSES]

    def small(k, pretrained):
        m = tvm.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, k)
        return m
    monkeypatch.setattr(T, "build_model", small)

    photos = tmp_path / "photos"
    photos.mkdir()
    rows = []
    for c in TEXTILE_CLASSES:
        Image.new("RGB", (40, 40), (len(c) * 9, 80, 120)).save(photos / f"{c}.jpg")
        rows.append((f"{c}.jpg", c))
    a = _csv(tmp_path, "a.csv", rows)
    b = _csv(tmp_path, "b.csv", rows[:-1] + [(rows[-1][0], "woven")])

    for fold in (1, 2):
        d = tmp_path / "models" / f"dtd_fold{fold}_seed0"
        d.mkdir(parents=True)
        torch.save({"model": small(len(classes), False).state_dict(), "classes": classes,
                    "config": {"limit_per_class": 0, "fold": fold}}, d / "best.pt")
        ref = tmp_path / "ref" / f"dtd_fold{fold}_seed0" / "textile12"
        ref.mkdir(parents=True)
        (ref / "textile_metrics.json").write_text(json.dumps(
            {"fold": fold, "restricted": {"top1": 0.8}}))
        pd.DataFrame({"class": TEXTILE_CLASSES, "recall": 0.8}).to_csv(
            ref / "restricted_per_class.csv", index=False)

    assert C.main(["transfer", "--photos", str(photos), "--labels-a", str(a),
                   "--labels-b", str(b), "--models", str(tmp_path / "models"),
                   "--reference", str(tmp_path / "ref")]) == 0
    run = runlog.load_run("o2_transfer_dtd_photos")
    assert run.info["synthetic"] is False
    assert run.metrics["n_scored"] == 11 and run.metrics["meets_minimum"] is False
    assert run.config["folds"] == [1, 2]
    assert len(run.table("per_fold")) == 2
    assert "fewer than the 50" in capsys.readouterr().err
