"""Scoring of the classifiers on Cluster crops, on invented outputs."""
import numpy as np
import pandas as pd

from haptic.groups import ASSIGNMENT
from vision import cluster_scans as C
from vision.textile import TEXTILE_CLASSES


def _items(n=4):
    return pd.DataFrame([{"material_id": m, "crop": k} for m in ASSIGNMENT for k in range(n)])


def _probs_for(classes_by_material, items):
    p = np.full((len(items), len(TEXTILE_CLASSES)), 0.01)
    for i, m in enumerate(items.material_id):
        p[i, TEXTILE_CLASSES.index(classes_by_material[m])] = 1.0
    return p / p.sum(1, keepdims=True)


def test_every_textile_class_has_a_registered_group():
    assert set(C.GROUP_OF_CLASS) == set(TEXTILE_CLASSES)
    assert set(C.GROUP_OF_CLASS.values()) == {"pile", "weave", "drape", "neither"}


def test_a_perfect_classifier_scores_one_and_a_constant_one_scores_its_group_share():
    items = _items()
    one_class = {"pile": "fibrous", "weave": "woven", "drape": "gauzy", "neither": "lacelike"}
    perfect = _probs_for({m: one_class[g] for m, g in ASSIGNMENT.items()}, items)
    constant = _probs_for({m: "woven" for m in ASSIGNMENT}, items)
    folds, _, by_group, majority = C.dtd_summary(items, {1: perfect, 2: constant})
    f = folds.set_index("fold")
    assert f.loc[1, "crop_group_acc_all16"] == 1.0
    assert f.loc[2, "crop_group_acc_all16"] == 7 / 16 == majority
    assert f.loc[2, "crop_group_acc_assigned13"] == 7 / 13
    assert set(by_group.assigned) == {"pile", "weave", "drape", "neither"}


def test_minc_counts_fabric():
    items = _items(2)
    classes = ["carpet", "fabric", "wood"]
    p = np.tile([0.1, 0.8, 0.1], (len(items), 1))
    p[items.material_id.to_numpy() == 102] = [0.8, 0.1, 0.1]
    overall, per = C.minc_summary(items, p, classes)
    assert overall["fabric_fraction"] == 15 / 16
    assert per.set_index("material_id").loc[102, "top_other"] == "carpet"
    assert overall["materials_majority_fabric"] == 15


def test_textile_softmax_uses_only_the_twelve_classes():
    classes = ["banded", *TEXTILE_CLASSES]
    logits = np.zeros((1, len(classes)))
    logits[0, 0] = 50.0
    p = C.textile_probs(logits, classes)
    assert p.shape == (1, 12) and np.allclose(p, 1 / 12)
