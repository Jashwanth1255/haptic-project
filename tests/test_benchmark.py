"""Linear benchmark: finds identity that is there, stays near chance on noise,
and reports what the plan says. Scaffolding data only."""
import numpy as np
import pandas as pd

from haptic import benchmark as B
from haptic import identity as I


def _frame(signal: float, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for mid in range(6):
        for d in (0, 45, 90, 135):
            for rep in range(10):
                r = {"material_id": mid, "category": "a" if mid < 3 else "b",
                     "direction_deg": d, "repeat": 2 * (d // 45) + rep % 2}
                for c in (*I.BAND_COLUMNS, "log_rms"):
                    r[c] = rng.normal()
                r[f"band{mid:02d}"] += signal
                r.update(frequency_hz=200 + rng.normal(), intensity=1 + 0.1 * rng.random(),
                         pattern=0.5, duration_ms=50 + rng.random())
                rows.append(r)
    return pd.DataFrame(rows)


def test_finds_identity():
    r = B.linear(_frame(5.0), B.FEATURE_SETS["bands"], "material_id", "direction")
    assert r["accuracy"] > 0.9 and r["top5"] >= r["accuracy"] and len(r["folds"]) == 4


def test_noise_near_chance():
    r = B.linear(_frame(0.0, seed=1), B.FEATURE_SETS["bands"], "material_id", "direction")
    assert r["accuracy"] < 0.45 and abs(r["chance"] - 1 / 6) < 1e-9


def test_category_task_and_repeat_split():
    r = B.linear(_frame(5.0), B.FEATURE_SETS["bands"], "category", "repeat")
    assert r["n_classes"] == 2 and r["n"] == 120 and r["majority_baseline"] == 0.5
