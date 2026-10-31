"""Training loop for Objective 2, on a tiny network and synthetic tensors.

SYNTHETIC -- scaffolding only. These numbers say the loop works, nothing about
any material or texture.
"""
import json

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from vision import train as T

CLASSES = ("a", "b", "c")


class Tiny(nn.Module):
    """Has batch norm and a `.fc`, the two things the loop treats specially."""

    def __init__(self, k=3):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1),
                                  nn.BatchNorm2d(8), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(8, k)

    def forward(self, x):
        return self.fc(self.body(x))


def _data(n=60, seed=0, learnable=True):
    """Each class has its own mean colour, unless `learnable` is off."""
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % 3
    x = 0.3 * torch.randn(n, 3, 8, 8, generator=g)
    if learnable:
        for c in range(3):
            x[y == c, c] += 2.0
    else:
        y = y[torch.randperm(n, generator=g)]
    return TensorDataset(x, y)


def _cfg(**kw):
    base = dict(dataset="fake", epochs=4, freeze_epochs=1, batch_size=8,
                lr_head=0.05, lr_backbone=0.05, pretrained=False)
    return T.TrainConfig(**{**base, **kw})


def _model(seed=0):
    torch.manual_seed(seed)
    return Tiny()


def test_learns_a_separable_problem(tmp_path):
    r = T.fit(_model(), _data(), _data(seed=1), CLASSES, _cfg(epochs=8), tmp_path)
    assert r["best_val_macro_f1"] > 0.9


def test_stays_at_chance_on_shuffled_labels(tmp_path):
    """Nothing to learn: validation on fresh shuffled labels must stay near 1/3.
    A loop leaking labels or scoring the training data would fail here."""
    r = T.fit(_model(), _data(learnable=False), _data(seed=1, learnable=False),
              CLASSES, _cfg(epochs=8), tmp_path)
    assert r["history"][-1]["val_top1"] < 0.6


def test_resumed_run_equals_uninterrupted_run(tmp_path):
    train, val = _data(), _data(seed=1)
    cfg = _cfg(epochs=5, freeze_epochs=1)

    whole = _model()
    T.fit(whole, train, val, CLASSES, cfg, tmp_path / "whole")

    first = _model()
    T.fit(first, train, val, CLASSES, cfg, tmp_path / "split", stop_after=3)
    # a new process: the model starts from nothing and must recover it all
    second = _model(seed=99)
    r = T.fit(second, train, val, CLASSES, cfg, tmp_path / "split")

    assert len(r["history"]) == 5
    for (name, a), b in zip(whole.state_dict().items(),
                            second.state_dict().values()):
        assert torch.equal(a, b), f"{name} differs after resume"


def test_finished_run_does_no_more_work(tmp_path):
    train, val = _data(), _data(seed=1)
    cfg = _cfg(epochs=2)
    T.fit(_model(), train, val, CLASSES, cfg, tmp_path)
    assert len(T.fit(_model(), train, val, CLASSES, cfg, tmp_path)["history"]) == 2


def test_resume_refuses_a_different_config(tmp_path):
    train, val = _data(), _data(seed=1)
    T.fit(_model(), train, val, CLASSES, _cfg(epochs=2), tmp_path)
    with pytest.raises(RuntimeError, match="different run"):
        T.fit(_model(), train, val, CLASSES, _cfg(epochs=2, seed=1), tmp_path)


def test_frozen_phase_leaves_the_backbone_untouched(tmp_path):
    """Including batch-norm running statistics, which ignore requires_grad."""
    model = _model()
    before = {k: v.clone() for k, v in model.body.state_dict().items()}
    fc_before = model.fc.weight.clone()
    T.fit(model, _data(), _data(seed=1), CLASSES,
          _cfg(epochs=2, freeze_epochs=1), tmp_path, stop_after=1)
    for k, v in model.body.state_dict().items():
        assert torch.equal(before[k], v), f"{k} moved while frozen"
    assert not torch.equal(fc_before, model.fc.weight)


def test_backbone_trains_after_unfreezing(tmp_path):
    model = _model()
    before = model.body[0].weight.clone()
    T.fit(model, _data(), _data(seed=1), CLASSES,
          _cfg(epochs=2, freeze_epochs=1), tmp_path)
    assert not torch.equal(before, model.body[0].weight)


def test_best_epoch_is_chosen_on_validation(tmp_path):
    r = T.fit(_model(), _data(), _data(seed=1), CLASSES, _cfg(), tmp_path)
    best = max(h["val_macro_f1"] for h in r["history"])
    assert r["best_val_macro_f1"] == best
    assert r["history"][r["best_epoch"]]["val_macro_f1"] == best


def test_report_scores_the_best_weights_not_the_last(tmp_path):
    train, val = _data(), _data(seed=1)
    cfg = _cfg(epochs=3)
    model = _model()
    T.fit(model, train, val, CLASSES, cfg, tmp_path)
    expected = T._load(tmp_path / "best.pt", "cpu")["model"]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)         # ruin the in-memory weights
    T.test_report(model, val, CLASSES, cfg, tmp_path)
    for k, v in model.state_dict().items():
        assert torch.equal(v, expected[k])


def test_report_writes_the_deliverables(tmp_path):
    train, val = _data(), _data(seed=1)
    cfg = _cfg(limit_per_class=2)
    model = _model()
    T.fit(model, train, val, CLASSES, cfg, tmp_path)
    r = T.test_report(model, val, CLASSES, cfg, tmp_path)
    saved = json.loads((tmp_path / "test_metrics.json").read_text())
    assert saved["top1"] == r.top1 and saved["macro_f1"] == r.macro_f1
    assert saved["smoke"] is True, "smoke runs must stay labelled as smoke"
    assert (tmp_path / "confusion.csv").exists()
    assert (tmp_path / "per_class.csv").exists()


def test_cosine_schedule():
    cfg = T.TrainConfig(dataset="x", epochs=10, freeze_epochs=2,
                        lr_head=0.01, lr_backbone=0.001)
    assert T.lr_at(cfg, 0) == (0.0, 0.01)
    assert T.lr_at(cfg, 2) == pytest.approx((0.001, 0.01))
    assert T.lr_at(cfg, 9)[0] < T.lr_at(cfg, 5)[0] < T.lr_at(cfg, 2)[0]
    assert T.lr_at(cfg, 9)[0] > 0


def test_freeze_must_leave_epochs_to_train():
    with pytest.raises(ValueError):
        T.TrainConfig(dataset="x", epochs=3, freeze_epochs=3)


def test_real_resnet50_has_the_right_head():
    m = T.build_model(47, pretrained=False)
    assert m.fc.out_features == 47
    assert m(torch.zeros(2, 3, 224, 224)).shape == (2, 47)
