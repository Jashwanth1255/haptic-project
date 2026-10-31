"""Objective 2 figures: the normalisation, and the refusal of smoke runs."""
import numpy as np
import pytest

import make_o2_figures as F


def test_rows_sum_to_one_and_empty_rows_stay_zero():
    m = F.recall_matrix([[3, 1], [0, 0]])
    assert m[0].tolist() == [0.75, 0.25]
    assert m[1].tolist() == [0.0, 0.0]


def test_smoke_runs_cannot_become_figures(tmp_path):
    metrics = {"smoke": True, "confusion": np.eye(2).tolist(), "classes": ["a", "b"],
               "top1": 1.0, "macro_f1": 1.0, "n": 2}
    with pytest.raises(ValueError, match="smoke"):
        F.plot(metrics, "x", tmp_path / "x.png")
    assert not (tmp_path / "x.png").exists()
