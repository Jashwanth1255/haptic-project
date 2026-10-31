"""Textile subset scoring, on logits small enough to work out by hand."""
from pathlib import Path

import numpy as np
import pytest
import torch

from vision import textile as X

CLASSES = ("a", "b", "c", "d")
SUBSET = ("a", "c")


def _case():
    """Four images of a or c; d is outside the subset.

      image 0  true a, top a            correct everywhere
      image 1  true c, top a            wrong everywhere
      image 2  true a, top d, then a    outside; masking rescues it
      image 3  true c, top d, then a    outside; masking gets it wrong
    """
    logits = torch.tensor([[5., 1., 2., 0.],
                           [5., 0., 2., 1.],
                           [3., 0., 1., 9.],
                           [3., 0., 1., 9.]])
    return logits, np.array([0, 2, 0, 2])


def test_restrict_removes_only_the_other_classes():
    m = X.restrict(torch.tensor([[1., 9., 2., 8.]]), [0, 2])
    assert m[0, 0] == 1 and m[0, 2] == 2
    assert torch.isinf(m[0, 1]) and torch.isinf(m[0, 3])


def test_three_scores_by_hand():
    s = X.score_subset(*_case(), CLASSES, SUBSET)
    assert s["n"] == 4 and s["n_predicted_outside"] == 2
    assert s["unrestricted_accuracy"] == pytest.approx(1 / 4)
    assert s["filtered"].n == 2
    assert s["filtered"].top1 == pytest.approx(1 / 2)
    assert s["restricted"].n == 4
    assert s["restricted"].top1 == pytest.approx(2 / 4)
    assert s["restricted"].confusion.tolist() == [[2, 0], [2, 0]]


def test_restricted_scores_every_image():
    s = X.score_subset(*_case(), CLASSES, SUBSET)
    assert s["restricted"].confusion.sum() == s["n"]


def test_masking_never_loses_a_correct_prediction():
    rng = np.random.default_rng(0)
    logits = torch.tensor(rng.normal(size=(500, 47)), dtype=torch.float32)
    names = [f"c{i}" for i in range(47)]
    subset = tuple(names[:12])
    y = rng.integers(0, 12, 500)
    s = X.score_subset(logits, y, names, subset)
    assert s["restricted"].top1 >= s["unrestricted_accuracy"]


def test_filtered_can_exceed_restricted():
    """Dropping hard images flatters the score. This is the reason the filtered
    figure must not be quoted as a classifier's accuracy."""
    logits = torch.tensor([[5., 0., 0., 0.]] * 3 + [[0., 0., 1., 9.]] * 3)
    y = np.array([0, 0, 0, 0, 0, 0])
    s = X.score_subset(logits, y, CLASSES, SUBSET)
    assert s["filtered"].top1 == 1.0
    assert s["restricted"].top1 == pytest.approx(0.5)


def test_image_from_outside_the_subset_is_refused():
    with pytest.raises(ValueError, match="outside the subset"):
        X.score_subset(torch.zeros(1, 4), np.array([3]), CLASSES, SUBSET)


def test_missing_class_is_refused():
    with pytest.raises(ValueError, match="not in this class list"):
        X.subset_indices(("a", "b"), ("a", "z"))


def test_list_matches_the_preregistration():
    text = (Path(__file__).parents[1] / "PREREGISTRATION.md").read_text()
    listed = [l.split(". ", 1)[1].strip() for l in text.splitlines()
              if l[:1].isdigit() and ". " in l]
    assert tuple(listed) == X.TEXTILE_CLASSES
    assert len(X.TEXTILE_CLASSES) == 12
