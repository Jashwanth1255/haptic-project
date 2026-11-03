"""Fidelity of the four schemes over the synthetic fabric bank.

    python make_o4_table.py

SYNTHETIC SIGNALS -- scaffolding only, not a result. This shows how the metric
behaves and which envelopes fall outside the playable band. It says nothing about
what a person feels, and the motor lag behind every number is an assumption.

The table is saved as a run under results/o4_fidelity_synthetic/, marked synthetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import runlog
from haptic import descriptor as D
from haptic import render as R
from haptic import signals as G

VELOCITIES = (20.0, 40.0, 60.0)
TAUS_MS = (10.0, R.ACTUATOR_TAU_MS, 50.0)
SEEDS = range(3)
SCORES = ["lsd_db", "wasserstein_hz", "level_error"]


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
    medians = {}
    for label, part in (("renderable", base[base["renderable"]]),
                        ("not renderable", base[~base["renderable"]])):
        medians[label] = part.groupby("scheme")[SCORES].median()
        print(f"### {label}: median per scheme")
        print(medians[label].round(3).to_string(), "\n")
    by_tau = df[df["renderable"]].pivot_table(index="scheme", columns="tau_ms",
                                              values="wasserstein_hz", aggfunc="median")
    print("### renderable: median Wasserstein (Hz) by assumed motor lag")
    print(by_tau.round(3).to_string())

    metrics = {"renderable_recordings": int(n["renderable"].sum()),
               "recordings": len(n), "switching_limit_hz": R.switching_limit_hz(),
               "median_renderable": medians["renderable"].to_dict(orient="index"),
               "median_not_renderable": medians["not renderable"].to_dict(orient="index")}
    config = {"velocities": VELOCITIES, "taus_ms": TAUS_MS, "seeds": list(SEEDS),
              "assumed_tau_ms": R.ACTUATOR_TAU_MS,
              "materials": [m.name for m in G.FABRIC_BANK]}
    by_tau.columns = [f"tau_{c:g}ms" for c in by_tau.columns]
    path = runlog.save_run("o4_fidelity_synthetic", config, metrics,
                           {"fidelity": df, "wasserstein_by_tau": by_tau},
                           synthetic=True)
    print(f"\nsaved run {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
