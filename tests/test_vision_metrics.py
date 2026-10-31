"""Objective 2 metrics, checked against values worked out by hand."""
import numpy as np
import pytest

from vision import metrics as M

CLASSES = ("a", "b", "c")


def test_perfect_predictions():
    y = [0, 1, 2, 0]
    r = M.evaluate(y, y, CLASSES)
    assert r.top1 == 1.0
    assert r.macro_f1 == 1.0
    assert r.confusion.trace() == 4


def test_hand_worked_example():
    """true a a a b, predicted a a b b.

    a: precision 1,   recall 2/3, F1 0.8
    b: precision 1/2, recall 1,   F1 2/3
    c: absent from both, F1 0
    """
    r = M.evaluate([0, 0, 0, 1], [0, 0, 1, 1], CLASSES)
    assert r.top1 == pytest.approx(0.75)
    assert r.macro_f1 == pytest.approx((0.8 + 2 / 3 + 0) / 3)
    assert r.confusion.tolist() == [[2, 1, 0], [0, 1, 0], [0, 0, 0]]


def test_confusion_rows_are_true_class():
    r = M.evaluate([0, 0], [1, 1], CLASSES)
    assert r.confusion[0, 1] == 2 and r.confusion[1, 0] == 0


def test_macro_f1_exposes_a_collapsed_classifier():
    """Always predicting the majority class gets high top-1 and must not get a
    high macro F1 -- this is the reason macro F1 is reported at all."""
    y_true = np.array([0] * 90 + [1] * 5 + [2] * 5)
    r = M.evaluate(y_true, np.zeros(100, dtype=int), CLASSES)
    assert r.top1 == pytest.approx(0.90)
    assert r.macro_f1 < 0.35


def test_absent_class_stays_in_the_matrix():
    r = M.evaluate([0, 1], [0, 1], CLASSES)
    assert r.confusion.shape == (3, 3)
    assert list(r.per_class["class"]) == list(CLASSES)
    assert r.per_class.loc[2, "support"] == 0


def test_chance_level_on_random_predictions():
    """A gate that cannot read low is no gate."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 23, 5000)
    p = rng.integers(0, 23, 5000)
    r = M.evaluate(y, p, tuple(f"c{i}" for i in range(23)))
    assert r.top1 == pytest.approx(1 / 23, abs=0.015)


def test_most_confused_ranks_by_count():
    r = M.evaluate([0, 0, 0, 1, 2], [1, 1, 2, 1, 2], CLASSES)
    top = r.most_confused(1).iloc[0]
    assert (top["true"], top["predicted"], top["count"]) == ("a", "b", 2)


def test_most_confused_empty_when_perfect():
    assert M.evaluate([0, 1], [0, 1], CLASSES).most_confused().empty


@pytest.mark.parametrize("y_true, y_pred, match", [
    ([0, 3], [0, 1], "outside"),
    ([0, 1], [0, -1], "outside"),
    ([0, 1], [0], "shape"),
    ([], [], "no predictions"),
    ([0.0, 1.0], [0.0, 1.0], "integer"),
])
def test_bad_input_raises(y_true, y_pred, match):
    y_true = np.asarray(y_true, dtype=float if match == "integer" else int)
    y_pred = np.asarray(y_pred, dtype=float if match == "integer" else int)
    with pytest.raises(ValueError, match=match):
        M.evaluate(y_true, y_pred, CLASSES)


def test_duplicate_class_names_raise():
    with pytest.raises(ValueError, match="duplicate"):
        M.evaluate([0], [0], ("a", "a"))


def test_as_dict_is_json_serialisable():
    import json
    json.dumps(M.evaluate([0, 1], [0, 1], CLASSES).as_dict())
