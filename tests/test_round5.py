"""Round 5 helpers on invented data (scaffolding, not results)."""
import numpy as np
import pandas as pd

import make_round5 as R5


def test_sign_flip_detects_shift_and_not_null():
    rng = np.random.default_rng(0)
    assert R5.sign_flip_p(rng.normal(0.3, 1, 400)) < 0.01
    assert R5.sign_flip_p(rng.normal(0.0, 1, 400)) > 0.05


def test_velocity_sd_zero_when_flat():
    rows = [{"material_id": m, "velocity_mm_s": v, "f": m * 1.0}
            for m in range(3) for v in (20, 40, 60) for _ in range(2)]
    sd = R5.velocity_sd(pd.DataFrame(rows), ["f"])
    assert np.allclose(sd.f, 0)
