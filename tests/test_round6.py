import numpy as np
import pandas as pd

from haptic import round6 as R6


def _data():
    rng = np.random.default_rng(0)
    rows, specs = [], []
    for m in range(5):
        for v in (20, 40):
            for d in (0, 90):
                rows.append({"material_id": m, "velocity_mm_s": v, "force_n": 0.5, "direction_deg": d})
                rig = np.linspace(0, 3, 8) * (v / 20) + d / 90       # shared by every material
                specs.append(rig + m * 0.1 + rng.normal(0, 0.01, (30, 8)))
    return specs, pd.DataFrame(rows)


def test_whitening_removes_shared_condition_response():
    specs, meta = _data()
    w = R6.whiten_specs(specs, meta)
    # after whitening, the same material looks the same at every condition
    for m in range(5):
        means = np.stack([w[i].mean(0) for i in np.flatnonzero(meta.material_id == m)])
        assert np.abs(means - means.mean(0)).max() < 0.05
    # before, it did not
    raw = np.stack([specs[i].mean(0) for i in np.flatnonzero(meta.material_id == 0)])
    assert np.abs(raw - raw.mean(0)).max() > 1.0


def test_whitening_keeps_material_differences_and_shape():
    specs, meta = _data()
    w = R6.whiten_specs(specs, meta)
    assert all(a.shape == b.shape and a.dtype == np.float32 for a, b in zip(w, specs))
    gap = w[np.flatnonzero(meta.material_id == 4)[0]].mean() - w[np.flatnonzero(meta.material_id == 0)[0]].mean()
    assert abs(gap - 0.4) < 0.02


def test_external_medians_are_used():
    specs, meta = _data()
    med = R6.condition_medians(specs, meta)
    sub = meta.material_id == 0
    w = R6.whiten_specs([s for s, k in zip(specs, sub) if k], meta[sub], med)
    assert abs(np.mean([x.mean() for x in w]) + 0.2) < 0.02     # material 0 sits 0.2 below the median material
