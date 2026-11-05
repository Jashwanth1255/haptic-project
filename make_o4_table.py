"""Fidelity of the four schemes over the synthetic fabric bank, or real cloth.

    python make_o4_table.py                       synthetic bank
    python make_o4_table.py --real data/cluster   real Cluster cloth recordings

With --real, the recordings are Cluster cloth at 20, 40 and 60 mm/s, 1.0 N,
directions 0 and 90 degrees, first repeat: 96 at most. The signals are real; the
motor lag and the 10 ms tick are still assumptions, and nothing here says what a
person feels. Saved as `o4_fidelity_real`, not synthetic.

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
REAL_DURATION_MS = 800    # real steady windows run 0.5-4.5 s; 1600 ms would drop the fast ones
SCORES = ["lsd_db", "wasserstein_hz", "level_error"]


def _synthetic_recordings():
    for m in G.FABRIC_BANK:
        for v in VELOCITIES:
            for s in SEEDS:
                x = G.synthesize(m, v, seed=s)
                yield m.name, v, s, x, G.FS_HZ


def _real_recordings(root):
    from haptic import loader as L
    for st in L.discover(root):
        if (st["velocity_mm_s"] in VELOCITIES and st["force_n"] == 1.0
                and st["direction_deg"] in (0, 90) and st["repeat"] == 0):
            try:
                rec = L.load(st)
            except ValueError:
                continue
            yield rec["material_name"], rec["velocity_mm_s"], st["direction_deg"],                 D.to_magnitude(rec["accel"]), rec["fs"]


def rows(real_root=None) -> pd.DataFrame:
    recs = []
    source = _real_recordings(real_root) if real_root else _synthetic_recordings()
    for name, v, s, x, fs in source:
        d = D.extract(x, fs)
        if real_root and not (2 * D.BAND_LOW_HZ < d.frequency_hz < 0.5 * D.BAND_HIGH_HZ):
            continue
        recs.append((name, v, s, x, d, fs))
    duration = REAL_DURATION_MS if real_root else 1600
    need = duration * R.ENV_FS_HZ / 1000
    recs = [r for r in recs if len(r[3]) / r[5] * R.ENV_FS_HZ >= need]
    cfg = R.RenderConfig(intensity_ref=R.intensity_reference([r[4] for r in recs]),
                         duration_ms=duration)

    out = []
    for name, v, s, x, d, fs in recs:
        env = R.signal_envelope(x, fs)
        level = min(d.intensity / cfg.intensity_ref, 1.0)
        for scheme, p in R.encode_all(d, cfg, env, seed=s).items():
            if np.ptp(p.binary()) == 0:     # solid on or off: no modulation to score
                continue
            for tau in TAUS_MS:
                f = R.fidelity(env, p, level, tau_ms=tau)
                out.append({"material": name, "velocity": v, "scheme": scheme,
                            "tau_ms": tau, "lsd_db": f.log_spectral_db,
                            "wasserstein_hz": f.wasserstein_hz,
                            "level_error": f.level_error,
                            "renderable": f.renderable})
    return pd.DataFrame(out)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="make_o4_table")
    ap.add_argument("--real", metavar="ROOT", help="Cluster dataset root")
    a = ap.parse_args(argv)
    if a.real:
        print("REAL Cluster cloth signals. Motor lag and tick are assumptions.\n")
    else:
        print("SYNTHETIC SIGNALS -- scaffolding only, not a result.\n")
    df = rows(a.real)
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
              "materials": sorted(df["material"].unique()), "real_root": a.real,
              "duration_ms": REAL_DURATION_MS if a.real else 1600}
    by_tau.columns = [f"tau_{c:g}ms" for c in by_tau.columns]
    path = runlog.save_run("o4_fidelity_real" if a.real else "o4_fidelity_synthetic",
                           config, metrics,
                           {"fidelity": df, "wasserstein_by_tau": by_tau},
                           synthetic=not a.real)
    print(f"\nsaved run {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
