"""Split loading for Objective 2, on tiny fake datasets written to tmp_path.

The fakes copy the published layout. What they prove is that a damaged or
mismatched dataset is refused -- not that the real copies are intact, which is
what running against them on Kaggle checks.
"""
from dataclasses import replace

import pytest
import torch
from PIL import Image

from vision import data as V

SPEC = replace(V.DTD, name="fake", n_classes=2, n_folds=2,
               per_class={"train": 2, "val": 1, "test": 1})
MINCISH = replace(V.MINC2500, name="fakeminc", n_classes=2, n_folds=1,
                  per_class={"train": 2, "validate": 1, "test": 1})


def _build(root, spec, listing_prefix=""):
    for cls in ("banded", "woven"):
        d = root / "images" / cls
        d.mkdir(parents=True)
        for i in range(4):
            Image.new("RGB", (300, 240), (i * 40, 90, 200)).save(d / f"{cls}_{i}.jpg")
    labels = root / "labels"
    labels.mkdir()
    for fold in range(1, spec.n_folds + 1):
        for split, ids in (("train", (0, 1)), (spec.val_split, (2,)), ("test", (3,))):
            rows = [f"{listing_prefix}{c}/{c}_{i}.jpg"
                    for c in ("banded", "woven") for i in ids]
            (labels / f"{split}{fold}.txt").write_text("\n".join(rows) + "\n")
    return root


@pytest.fixture
def root(tmp_path):
    return _build(tmp_path / "dtd", SPEC)


def test_reads_labels_from_the_directory_name(root):
    items = V.read_split(root, SPEC, "train")
    assert len(items) == 4
    assert {p.parent.name: y for p, y in items} == {"banded": 0, "woven": 1}


def test_handles_the_minc_listing_format(tmp_path):
    """MINC lines start with images/, DTD lines do not."""
    root = _build(tmp_path / "minc", MINCISH, listing_prefix="images/")
    assert len(V.load_splits(root, MINCISH)["train"]) == 4


def test_find_root_descends_into_kaggle_nesting(tmp_path):
    _build(tmp_path / "input" / "x" / "dtd", SPEC)
    assert V.find_root(tmp_path / "input", SPEC).name == "dtd"


def test_find_root_picks_the_right_dataset_when_both_are_attached(tmp_path):
    _build(tmp_path / "a_other", SPEC)
    (tmp_path / "a_other" / "images" / "third").mkdir()
    _build(tmp_path / "b_wanted", SPEC)
    assert V.find_root(tmp_path, SPEC).name == "b_wanted"


def test_find_root_rejects_a_copy_without_splits(tmp_path):
    (tmp_path / "images" / "banded").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="published splits"):
        V.find_root(tmp_path, SPEC)


def test_overlapping_splits_are_refused(root):
    """Test-set leakage must stop the run."""
    (root / "labels" / "test1.txt").write_text(
        "banded/banded_0.jpg\nwoven/woven_3.jpg\n")
    with pytest.raises(RuntimeError, match="overlap"):
        V.load_splits(root, SPEC)


def test_wrong_per_class_count_is_refused(root):
    (root / "labels" / "train1.txt").write_text(
        "banded/banded_0.jpg\nwoven/woven_0.jpg\nwoven/woven_1.jpg\n")
    with pytest.raises(RuntimeError, match="per class"):
        V.read_split(root, SPEC, "train")


def test_missing_image_is_refused(root):
    (root / "images" / "banded" / "banded_0.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="missing image"):
        V.read_split(root, SPEC, "train")


def test_wrong_class_count_is_refused(root):
    (root / "images" / "third").mkdir()
    with pytest.raises(RuntimeError, match="expected 2 classes"):
        V.read_split(root, SPEC, "train")


@pytest.mark.parametrize("kw, err", [({"split": "dev"}, "no split"),
                                     ({"split": "train", "fold": 3}, "folds")])
def test_bad_split_or_fold_raises(root, kw, err):
    with pytest.raises(ValueError, match=err):
        V.read_split(root, SPEC, **kw)


def test_images_come_out_normalised_and_sized(root):
    ds = V.ImageList(V.read_split(root, SPEC, "train"), V.transforms_for(train=False))
    x, y = ds[0]
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32
    assert x.min() < 0, "ImageNet normalisation was not applied"
    assert y == 0


def test_eval_transform_is_deterministic_and_train_is_not(tmp_path):
    p = tmp_path / "noise.jpg"
    Image.effect_noise((300, 240), 80).convert("RGB").save(p)
    items = [(p, 0)]
    ev = V.ImageList(items, V.transforms_for(train=False))
    assert torch.equal(ev[0][0], ev[0][0])
    tr = V.ImageList(items, V.transforms_for(train=True))
    assert not torch.equal(tr[0][0], tr[0][0])
