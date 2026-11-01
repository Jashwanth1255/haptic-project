"""Fidelity of the four schemes over the synthetic fabric bank.

    python make_o4_table.py

SYNTHETIC SIGNALS -- scaffolding only, not a result. This shows how the metric
behaves and which envelopes fall outside the playable band. It says nothing about
what a person feels, and the motor lag behind every number is an assumption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from haptic import descriptor as D
from haptic import render as R
from haptic import signals as G

VELOCITIES = (20.0, 40.0, 60.0)
TAUS_MS = (10.0, R.ACTUATOR_TAU_MS, 50.0)
SEEDS = range(3)


def rows() -> pd.DataFrame:
    recs = []
    for m in G.FABRIC_BANK:
        for v in VELOCITIES:
            for s in SEEDS:
                x = G.synthesize(m, v, seed=s)
                recs.append((m, v, s, x, D.extract(x, G.FS_HZ)))
    cfg = R.RenderConfig(intensity_ref=R.intensity_reference([r[4] for r in recs]))

    out = []
    for m, v, s, x, d in recs:
        env = R.signal_envelope(x, G.FS_HZ)
        level = min(d.intensity / cfg.intensity_ref, 1.0)
        for scheme, p in R.encode_all(d, cfg, env, seed=s).items():
            if np.ptp(p.binary()) == 0:     # solid on or off: no modulation to score
                continue
            for tau in TAUS_MS:
                f = R.fidelity(env, p, level, tau_ms=tau)
                out.append({"material": m.name, "velocity": v, "scheme": scheme,
                            "tau_ms": tau, "lsd_db": f.log_spectral_db,
                            "wasserstein_hz": f.wasserstein_hz,
                            "level_error": f.level_error,
                            "renderable": f.renderable})
    return pd.DataFrame(out)


def main() -> int:
    print("SYNTHETIC SIGNALS -- scaffolding only, not a result.\n")
    df = rows()
    base = df[df["tau_ms"] == R.ACTUATOR_TAU_MS]
    n = base[base["scheme"] == "B"]
    print(f"renderable envelopes: {int(n['renderable'].sum())} of {len(n)} "
          f"recordings (limit {R.switching_limit_hz():.0f} Hz)\n")
    for label, part in (("renderable", base[base["renderable"]]),
                        ("not renderable", base[~base["renderable"]])):
        print(f"### {label}: median per scheme")
        print(part.groupby("scheme")[["lsd_db", "wasserstein_hz", "level_error"]]
              .median().round(3).to_string(), "\n")
    print("### renderable: median Wasserstein (Hz) by assumed motor lag")
    print(df[df["renderable"]].pivot_table(index="scheme", columns="tau_ms",
          values="wasserstein_hz", aggfunc="median").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
