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


# --- round 2 additions -----------------------------------------------------------

def _tone_logf(freq, seconds=1.0):
    t = np.arange(0, seconds, 1 / FS)
    z = -1 + 0.05 * np.sin(2 * np.pi * freq * t)
    return Lr.logf_spectrogram(np.column_stack([0 * t, 0 * t, z]), FS)


def _peak_band(spec):
    return int(np.argmax(spec.mean(0)))


def test_logf_spectrogram_shape_and_peak_moves_with_tone():
    lo, hi = _tone_logf(100.0), _tone_logf(400.0)
    assert lo.shape[1] == Lr.LOGF_BANDS and not np.isnan(lo).any()
    assert _peak_band(hi) > _peak_band(lo)


def test_warp_with_k1_aligns_a_spectrum_that_scales_with_velocity():
    # a texture of fixed wavelength: 200 Hz at 40 mm/s, 300 Hz at 60 mm/s
    s40, s60 = _tone_logf(200.0), _tone_logf(300.0)
    assert _peak_band(s60) != _peak_band(s40)
    assert abs(_peak_band(Lr.warp(s60, 60.0, 1.0)) - _peak_band(s40)) <= 1
    # k = 0 leaves it where it was
    assert _peak_band(Lr.warp(s60, 60.0, 0.0)) == _peak_band(s60)


def test_warp_with_k1_misaligns_a_fixed_resonance():
    s40, s60 = _tone_logf(250.0), _tone_logf(250.0)
    assert abs(_peak_band(Lr.warp(s60, 60.0, 1.0)) - _peak_band(s40)) >= 3


def test_audio_spectrogram_bands():
    fs = 44100.0
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 3000 * t) + 0.01 * np.random.default_rng(0).normal(size=t.size)
    s = Lr.audio_spectrogram(x, fs, 20.0, 8000.0)
    assert s.shape[1] == Lr.AUDIO_BANDS and not np.isnan(s).any()
    centres = np.sqrt(np.geomspace(20, 8000, 65)[:-1] * np.geomspace(20, 8000, 65)[1:])
    assert abs(np.log(centres[_peak_band(s)] / 3000)) < 0.15


def test_velocity_split_holds_out_each_velocity():
    meta = pd.DataFrame({"velocity_mm_s": [20, 40, 60, 20, 40, 60],
                         "direction_deg": 0, "repeat": 0})
    folds = Lr.folds_for(meta, "velocity")
    assert [f[2] for f in folds] == [20, 40, 60]
    assert all(set(meta.velocity_mm_s.iloc[te]) == {v} for _, te, v in folds)


def test_category_label_and_topk_reported():
    specs, meta = _data(True)
    meta["category"] = np.where(meta.material_id == 1, 0, 1)
    res = Lr.evaluate(specs, meta, "direction", Lr.frames_per_window(FS), seeds=(0,),
                      cfg=CFG, label="category")
    assert set(res.truth.explode().unique()) <= {0, 1}
    assert "top2" in res and (res.top2 >= res.accuracy).all()
