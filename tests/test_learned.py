"""The learned baseline must learn identity that is there and not invent it."""
import numpy as np
import pandas as pd

from haptic import learned as Lr

FS = 6000.0


def _data(signal: bool, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 1.0, 1 / FS)
    specs, rows = [], []
    for k, mid in enumerate((1, 2, 3)):
        tone = 150 + 200 * k
        for d in (0, 90, 180, 270):
            for rep in (0, 1):
                z = -1 + 0.02 * rng.normal(size=t.size)
                if signal:
                    z = z + 0.05 * np.sin(2 * np.pi * tone * t + rng.uniform(0, 6))
                acc = np.column_stack([0 * t, 0 * t, z])
                specs.append(Lr.spectrogram(acc, FS))
                rows.append({"material_id": mid, "direction_deg": d, "repeat": 2 * (d // 90) + rep,
                             "velocity_mm_s": 40.0, "force_n": 1.0})
    return specs, pd.DataFrame(rows)


CFG = Lr.Config(epochs=15, windows_per_recording=8, channels=8)


def test_spectrogram_covers_the_band():
    specs, _ = _data(True)
    n_bins = specs[0].shape[1]
    assert 30 <= n_bins <= 50 and specs[0].dtype == np.float32


def test_learns_identity_that_is_there():
    specs, meta = _data(True)
    res = Lr.evaluate(specs, meta, "direction", Lr.frames_per_window(FS), seeds=(0,), cfg=CFG)
    assert res.accuracy.mean() > 0.8 and len(res) == 4


def test_noise_stays_near_chance():
    accs = []
    for s in range(3):
        specs, meta = _data(False, seed=s)
        res = Lr.evaluate(specs, meta, "direction", Lr.frames_per_window(FS), seeds=(s,), cfg=CFG)
        accs.append(res.accuracy.mean())
    assert np.mean(accs) < 0.6


def test_repeat_split_trains_on_even_and_tests_on_odd():
    specs, meta = _data(True)
    res = Lr.evaluate(specs, meta, "repeat", Lr.frames_per_window(FS), seeds=(0,), cfg=CFG)
    assert res.n.iloc[0] == 12


def test_checkpoint_resumes_without_refitting(tmp_path, monkeypatch):
    specs, meta = _data(True)
    ck = tmp_path / "folds.jsonl"
    first = Lr.evaluate(specs, meta, "repeat", Lr.frames_per_window(FS), seeds=(0,), cfg=CFG,
                        checkpoint=ck)
    monkeypatch.setattr(Lr, "fit_predict", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    again = Lr.evaluate(specs, meta, "repeat", Lr.frames_per_window(FS), seeds=(0,), cfg=CFG,
                        checkpoint=ck)
    assert again.accuracy.tolist() == first.accuracy.tolist()
