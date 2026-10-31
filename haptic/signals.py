"""Synthetic texture signals — a stand-in for Cluster Haptic until it is downloaded.

Two jobs:

1. **Verification.** Every signal is generated from known parameters, so the
   extractor in `descriptor.py` can be checked against ground truth. If it cannot
   recover a centroid it was handed, it will not recover one from real data either.

2. **Scaffolding.** The whole O1 -> O3 chain can be built and debugged today, and
   switching to the real dataset is a change of loader, not a change of pipeline.

**This is test scaffolding, not data.** Nothing produced here may appear in the
paper as a result. The physics below is a plausible caricature, not a model of
fabric mechanics.

Physical mapping used
---------------------
A tool sliding at velocity `v` over a surface with spatial wavelength `lambda`
excites a temporal frequency `f = v / lambda`. That is the one real constraint
worth encoding, because it is what makes velocity invariance testable: the
velocity-invariant property of a material is `lambda`, and the measured centroid
is not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps


@dataclass(frozen=True)
class SyntheticMaterial:
    """A fictional fabric with known spatial properties.

    name                 : label only
    wavelength_mm        : dominant spatial period of the surface texture
    periodicity          : 0 = fully stochastic, 1 = strongly periodic.
                           Maps inversely onto measured spectral flatness.
    roughness            : amplitude scale (arbitrary units, as for a raw accelerometer)
    burst_per_mm         : rate of macro roughness bursts per mm of travel;
                           sets the envelope modulation and so `duration_ms`
    rel_bandwidth        : spectral width of the carrier relative to its centre
    """

    name: str
    wavelength_mm: float
    periodicity: float
    roughness: float
    burst_per_mm: float
    rel_bandwidth: float = 0.45

    def carrier_hz(self, velocity_mm_s: float) -> float:
        return velocity_mm_s / self.wavelength_mm

    def envelope_hz(self, velocity_mm_s: float) -> float:
        return velocity_mm_s * self.burst_per_mm


# Twelve fictional fabrics, spread so that some pairs are deliberately close.
# The close pairs exist to make the separability gate in `separability.py` do
# real work rather than trivially passing.
FABRIC_BANK: tuple[SyntheticMaterial, ...] = (
    SyntheticMaterial("plain-weave-fine",   0.22, 0.55, 0.9, 0.55),
    SyntheticMaterial("plain-weave-coarse", 0.48, 0.50, 1.4, 0.45),
    SyntheticMaterial("satin-slick",        0.14, 0.30, 0.5, 0.30),
    SyntheticMaterial("satin-dense",        0.16, 0.28, 0.55, 0.32),   # close to above
    SyntheticMaterial("twill-diagonal",     0.62, 0.72, 1.6, 0.85),
    SyntheticMaterial("denim-heavy",        0.70, 0.68, 2.1, 0.80),
    SyntheticMaterial("corduroy-wide",      2.40, 0.90, 2.8, 0.42),
    SyntheticMaterial("corduroy-narrow",    1.30, 0.88, 2.4, 0.78),
    SyntheticMaterial("pile-velvet",        0.30, 0.12, 1.1, 1.60),
    SyntheticMaterial("pile-fleece",        0.34, 0.10, 1.3, 1.75),    # close to above
    SyntheticMaterial("mesh-open",          1.10, 0.62, 1.7, 1.05),
    SyntheticMaterial("felt-matte",         0.26, 0.08, 0.8, 1.90),
)

# Cluster's own acquisition grid.
VELOCITIES_MM_S = (20.0, 30.0, 40.0, 50.0, 60.0)
FORCES_N = (0.5, 1.0)
FS_HZ = 6000.0


def _band_noise(n: int, fs: float, centre: float, rel_bw: float,
                rng: np.random.Generator) -> np.ndarray:
    """Noise shaped into a band around `centre`."""
    x = rng.standard_normal(n)
    nyq = fs / 2.0
    half = max(centre * rel_bw, 5.0)
    lo = max(centre - half, 1.0)
    hi = min(centre + half, nyq * 0.98)
    if lo >= hi:
        lo, hi = max(hi * 0.5, 1.0), hi
    sos = sps.butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    y = sps.sosfiltfilt(sos, x)
    s = np.std(y)
    return y / s if s > 0 else y


def synthesize(material: SyntheticMaterial,
               velocity_mm_s: float,
               force_n: float = 1.0,
               seconds: float = 2.0,
               fs: float = FS_HZ,
               snr_db: float = 30.0,
               seed: int | None = None) -> np.ndarray:
    """Generate one acceleration recording.

    Returns a (N,) array. Amplitude rises with normal force and with the square
    root of velocity, which is the usual coarse approximation for sliding
    contact; it keeps the intensity dimension from being velocity-invariant, as
    it is not in real measurements either.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs

    f_c = material.carrier_hz(velocity_mm_s)
    f_env = material.envelope_hz(velocity_mm_s)

    # carrier: a periodic component plus a stochastic one, mixed by `periodicity`
    tone = np.sin(2 * np.pi * f_c * t + rng.uniform(0, 2 * np.pi))
    noise = _band_noise(n, fs, f_c, material.rel_bandwidth, rng)
    w = float(np.clip(material.periodicity, 0.0, 1.0))
    carrier = w * tone + (1.0 - w) * noise

    # envelope: macro roughness bursts, never fully silent
    env = 1.0 + 0.8 * np.sin(2 * np.pi * f_env * t + rng.uniform(0, 2 * np.pi))
    env = np.clip(env, 0.05, None)

    amp = material.roughness * force_n * np.sqrt(velocity_mm_s / 40.0)
    x = amp * carrier * env

    # measurement noise
    sig_p = float(np.mean(x ** 2))
    if sig_p > 0:
        noise_p = sig_p / (10 ** (snr_db / 10.0))
        x = x + rng.standard_normal(n) * np.sqrt(noise_p)
    return x


def synth_dataset(materials=FABRIC_BANK,
                  velocities=VELOCITIES_MM_S,
                  forces=FORCES_N,
                  repeats: int = 3,
                  seconds: float = 2.0,
                  fs: float = FS_HZ,
                  seed: int = 0) -> list[dict]:
    """Full synthetic acquisition grid, in the shape `extract_many` consumes.

    Mirrors Cluster's structure: every material at every velocity and force.
    `repeats` stands in for Cluster's eight sliding directions.
    """
    out = []
    counter = 0
    for mid, mat in enumerate(materials):
        for v in velocities:
            for fo in forces:
                for r in range(repeats):
                    counter += 1
                    out.append({
                        "accel": synthesize(mat, v, fo, seconds, fs,
                                            seed=seed * 100000 + counter),
                        "fs": fs,
                        "material_id": mid,
                        "material_name": mat.name,
                        "velocity_mm_s": v,
                        "force_n": fo,
                        "repeat": r,
                    })
    return out
