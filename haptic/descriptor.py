"""Objective 1 — haptic descriptor extraction.

Band-pass -> FFT -> RMS pipeline producing the four descriptor dimensions the
project is committed to (frequency, intensity, pattern, duration), plus a set of
auxiliary features used only for the separability analysis in `separability.py`.

Designed against the Cluster Haptic Texture Dataset (Scientific Data 13, 756, 2026):
3-axis MEMS acceleration at ~6 kHz, sliding velocities 20-60 mm/s, normal forces
0.5 and 1.0 N.

Every definition here is deliberately a standard, citable quantity. Nothing in this
module is novel, and it should not be presented as such -- the contribution of the
project lives in Objectives 2, 3 and 5, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Iterable

import numpy as np
from scipy import signal as sps

# Texture-relevant vibrotactile band.
#   Lower bound 20 Hz  -- below this is gross hand motion, not texture.
#   Upper bound 1000 Hz -- above this the skin's mechanoreceptors do not respond;
#                          the Pacinian channel peaks around 200-300 Hz.
BAND_LOW_HZ = 20.0
BAND_HIGH_HZ = 1000.0

# Envelope modulation band: the rate at which roughness bursts arrive while
# sliding. This is what "pattern" and "duration" are computed from.
ENV_LOW_HZ = 2.0
ENV_HIGH_HZ = 200.0


@dataclass(frozen=True)
class Descriptor:
    """The four committed descriptor dimensions, plus provenance.

    frequency_hz : spectral centroid of the band-passed acceleration.
                   Perceptually, "how fine" the texture is.
    intensity    : RMS acceleration in the band, in the signal's own units.
                   Perceptually, "how strong".
    pattern      : normalised spectral entropy of the band-passed signal,
                   in [0, 1]. Low => periodic (corduroy, ribbed weave);
                   high => stochastic (fleece, felt).
    duration_ms  : dominant envelope modulation period -- the temporal scale of
                   the roughness bursts. This is the quantity the renderer needs
                   to set pulse timing.
    """

    material_id: int
    velocity_mm_s: float
    force_n: float
    frequency_hz: float
    intensity: float
    pattern: float
    duration_ms: float
    # auxiliary -- separability analysis only, not part of the committed four
    spectral_rolloff_hz: float = float("nan")
    spectral_bandwidth_hz: float = float("nan")
    zero_crossing_rate: float = float("nan")
    env_depth: float = float("nan")
    flatness_db: float = float("nan")
    crest: float = float("nan")

    CORE = ("frequency_hz", "intensity", "pattern", "duration_ms")
    AUX = (
        "spectral_rolloff_hz",
        "spectral_bandwidth_hz",
        "zero_crossing_rate",
        "env_depth",
        "flatness_db",
        "crest",
    )

    def core_vector(self) -> np.ndarray:
        return np.array([getattr(self, f) for f in self.CORE], dtype=float)

    def full_vector(self) -> np.ndarray:
        return np.array(
            [getattr(self, f) for f in (*self.CORE, *self.AUX)], dtype=float
        )

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# signal conditioning
# --------------------------------------------------------------------------- #

def to_magnitude(accel: np.ndarray) -> np.ndarray:
    """Collapse 3-axis acceleration to a scalar magnitude signal.

    Cluster records 3-axis acceleration from a rubber tip. The tool's
    orientation relative to the sliding direction is not something we control,
    and the project renders to a single-channel actuator, so the Euclidean
    magnitude is the honest reduction. Accepts (N,) or (N, 3).
    """
    accel = np.asarray(accel, dtype=float)
    if accel.ndim == 1:
        return accel
    if accel.ndim == 2 and accel.shape[1] == 3:
        return np.linalg.norm(accel, axis=1)
    raise ValueError(f"expected (N,) or (N,3) acceleration, got {accel.shape}")


def bandpass(x: np.ndarray, fs: float,
             low: float = BAND_LOW_HZ, high: float = BAND_HIGH_HZ,
             order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass.

    filtfilt is used rather than lfilter so no group delay is introduced -- the
    envelope timing in `duration_ms` would otherwise be biased.
    """
    nyq = fs / 2.0
    high = min(high, nyq * 0.99)
    if low >= high:
        raise ValueError(f"band {low}-{high} Hz invalid for fs={fs}")
    sos = sps.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sps.sosfiltfilt(sos, x)


def _welch(x: np.ndarray, fs: float):
    """Power spectral density, with a window short enough for brief strokes."""
    nper = int(min(len(x), max(256, 2 ** int(np.log2(max(len(x) // 8, 256))))))
    f, p = sps.welch(x, fs=fs, nperseg=nper, noverlap=nper // 2)
    return f, p


# --------------------------------------------------------------------------- #
# the four committed dimensions
# --------------------------------------------------------------------------- #

def spectral_centroid(f: np.ndarray, p: np.ndarray) -> float:
    """Power-weighted mean frequency -> descriptor dimension `frequency`."""
    total = p.sum()
    return float(np.sum(f * p) / total) if total > 0 else float("nan")


def rms(x: np.ndarray) -> float:
    """Root-mean-square amplitude -> descriptor dimension `intensity`."""
    return float(np.sqrt(np.mean(np.square(x))))


def spectral_entropy(f: np.ndarray, p: np.ndarray,
                     low: float = BAND_LOW_HZ, high: float = BAND_HIGH_HZ,
                     eps: float = 1e-20) -> float:
    """Normalised spectral entropy -> descriptor dimension `pattern`.

    Shannon entropy of the in-band power distribution, divided by log(N) so it
    is bounded [0, 1]. Low => energy concentrated at a few frequencies, i.e. a
    periodic structure such as a rib or a regular weave. High => energy spread
    broadly, i.e. a stochastic surface such as a pile or felt.

    Chosen over Wiener-entropy spectral flatness, which is the more common
    measure but is unusable here: these signals are narrowband, so the
    out-of-band bins drive the geometric mean to ~0 and flatness collapses to
    the same near-zero value for every material regardless of its structure.
    Flatness is still reported in dB as an auxiliary feature.
    """
    p = np.asarray(p, dtype=float)
    band = (f >= low) & (f <= high)
    p = p[band]
    if p.size < 2 or p.sum() <= 0:
        return float("nan")
    q = p / p.sum()
    q = q[q > 0]
    h = -np.sum(q * np.log(q))
    return float(h / np.log(len(p)))


def spectral_flatness_db(f: np.ndarray, p: np.ndarray,
                         low: float = BAND_LOW_HZ, high: float = BAND_HIGH_HZ,
                         eps: float = 1e-20) -> float:
    """Wiener entropy in dB — auxiliary feature, kept for comparability."""
    p = np.asarray(p, dtype=float)
    band = (f >= low) & (f <= high)
    p = p[band]
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    gm = np.exp(np.mean(np.log(p + eps)))
    am = np.mean(p)
    if am <= 0 or gm <= 0:
        return float("nan")
    return float(10.0 * np.log10(gm / am))


def spectral_crest(f: np.ndarray, p: np.ndarray,
                   low: float = BAND_LOW_HZ, high: float = BAND_HIGH_HZ) -> float:
    """Peak-to-mean power ratio in band — auxiliary tonality measure."""
    band = (f >= low) & (f <= high)
    pb = p[band]
    if pb.size == 0 or pb.mean() <= 0:
        return float("nan")
    return float(pb.max() / pb.mean())


def envelope(x: np.ndarray, fs: float) -> np.ndarray:
    """Amplitude envelope via the analytic signal, low-passed to the modulation band."""
    env = np.abs(sps.hilbert(x))
    nyq = fs / 2.0
    cutoff = min(ENV_HIGH_HZ, nyq * 0.99)
    sos = sps.butter(4, cutoff / nyq, btype="low", output="sos")
    return sps.sosfiltfilt(sos, env)


ENV_FS_HZ = 500.0   # envelope is low-passed to ENV_HIGH_HZ, so this is ample


def envelope_period_ms(env: np.ndarray, fs: float) -> float:
    """Dominant envelope modulation period -> descriptor dimension `duration`.

    Computed from the envelope's own spectrum rather than by peak-picking in the
    time domain, which is far less stable on stochastic textures.

    The envelope is decimated to ENV_FS_HZ before the spectrum is taken. At the
    raw 6 kHz rate a Welch window sized for the carrier gives roughly 6 Hz of
    frequency resolution, which is useless here -- a true 16 Hz modulation can
    land three bins away and report a period three times too long. Decimating
    first buys ~1 Hz resolution in the band that matters.
    """
    env = np.asarray(env, dtype=float)
    env = env - np.mean(env)
    if np.allclose(env, 0):
        return float("nan")

    if fs > ENV_FS_HZ * 2:
        q = int(fs // ENV_FS_HZ)
        env = sps.decimate(env, q, ftype="fir", zero_phase=True)
        fs_e = fs / q
    else:
        fs_e = fs

    nper = int(min(len(env), 512))
    if nper < 64:
        return float("nan")
    f, p = sps.welch(env, fs=fs_e, nperseg=nper, noverlap=nper // 2)

    band = (f >= ENV_LOW_HZ) & (f <= min(ENV_HIGH_HZ, fs_e / 2 * 0.99))
    if not band.any() or p[band].sum() <= 0:
        return float("nan")
    f_dom = f[band][int(np.argmax(p[band]))]
    return float(1000.0 / f_dom) if f_dom > 0 else float("nan")


# --------------------------------------------------------------------------- #
# auxiliary features (separability analysis only)
# --------------------------------------------------------------------------- #

def spectral_rolloff(f: np.ndarray, p: np.ndarray, frac: float = 0.85) -> float:
    c = np.cumsum(p)
    if c[-1] <= 0:
        return float("nan")
    return float(f[int(np.searchsorted(c, frac * c[-1]))])


def spectral_bandwidth(f: np.ndarray, p: np.ndarray, centroid: float) -> float:
    total = p.sum()
    if total <= 0 or not np.isfinite(centroid):
        return float("nan")
    return float(np.sqrt(np.sum(((f - centroid) ** 2) * p) / total))


def zero_crossing_rate(x: np.ndarray, fs: float) -> float:
    return float(np.mean(np.abs(np.diff(np.signbit(x).astype(int)))) * fs / 2.0)


def envelope_depth(env: np.ndarray) -> float:
    """Modulation depth of the envelope: (max-min)/(max+min). Bounded [0, 1]."""
    mx, mn = float(np.max(env)), float(np.min(env))
    return (mx - mn) / (mx + mn) if (mx + mn) > 0 else float("nan")


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #

def extract(accel: np.ndarray, fs: float, *,
            material_id: int = -1,
            velocity_mm_s: float = float("nan"),
            force_n: float = float("nan")) -> Descriptor:
    """Run the full pipeline on one recording.

    Parameters
    ----------
    accel : (N,) or (N, 3) acceleration samples
    fs    : sampling rate in Hz (Cluster acceleration is ~6000)
    """
    x = to_magnitude(accel)
    if len(x) < 64:
        raise ValueError(f"recording too short: {len(x)} samples")

    x = x - np.mean(x)                 # remove DC / gravity component
    xb = bandpass(x, fs)
    f, p = _welch(xb, fs)

    centroid = spectral_centroid(f, p)
    env = envelope(xb, fs)

    return Descriptor(
        material_id=material_id,
        velocity_mm_s=velocity_mm_s,
        force_n=force_n,
        frequency_hz=centroid,
        intensity=rms(xb),
        pattern=spectral_entropy(f, p),
        duration_ms=envelope_period_ms(env, fs),
        spectral_rolloff_hz=spectral_rolloff(f, p),
        spectral_bandwidth_hz=spectral_bandwidth(f, p, centroid),
        zero_crossing_rate=zero_crossing_rate(xb, fs),
        env_depth=envelope_depth(env),
        flatness_db=spectral_flatness_db(f, p),
        crest=spectral_crest(f, p),
    )


def extract_many(recordings: Iterable[dict]) -> list[Descriptor]:
    """Map `extract` over an iterable of dicts.

    Each dict needs keys: accel, fs, and optionally material_id, velocity_mm_s,
    force_n. This is the shape `loader.py` yields.
    """
    out = []
    for r in recordings:
        out.append(extract(
            r["accel"], r["fs"],
            material_id=r.get("material_id", -1),
            velocity_mm_s=r.get("velocity_mm_s", float("nan")),
            force_n=r.get("force_n", float("nan")),
        ))
    return out


# --------------------------------------------------------------------------- #
# velocity normalisation
# --------------------------------------------------------------------------- #

# Reference condition the normalisation is expressed relative to.
REF_VELOCITY_MM_S = 40.0


def normalise(d: Descriptor) -> dict:
    """Recast a descriptor into quantities that should not depend on how the
    surface was explored.

    The raw descriptor is a property of *this stroke*, not of the material. A
    tool sliding at velocity v over a surface of spatial wavelength lambda
    excites f = v / lambda, so `frequency_hz` is guaranteed to move with
    velocity. The same is true of `duration_ms`, and `intensity` moves with
    normal force.

    Dividing the velocity back out gives candidate material constants:

        wavelength_mm     = v / frequency_hz        (spatial period of texture)
        burst_spacing_mm  = v * duration_ms / 1000  (spatial period of roughness bursts)
        roughness_index   = intensity / (force * sqrt(v / REF))
        pattern           unchanged -- already dimensionless

    IMPORTANT, and it must stay in the paper: on synthetic data this
    normalisation works by construction, because the generator was written with
    f = v / lambda in it. That proves the arithmetic, and nothing else. Whether
    real fabric measurements obey it closely enough for separability to survive
    pooling is an open empirical question, and answering it on the Cluster
    Haptic velocity grid is Objective 3's experiment. Do not report the
    synthetic result as evidence.
    """
    v = d.velocity_mm_s
    fo = d.force_n
    out = {
        "material_id": d.material_id,
        "velocity_mm_s": v,
        "force_n": fo,
        "pattern": d.pattern,
    }
    out["wavelength_mm"] = (
        v / d.frequency_hz
        if np.isfinite(v) and np.isfinite(d.frequency_hz) and d.frequency_hz > 0
        else float("nan")
    )
    out["burst_spacing_mm"] = (
        v * d.duration_ms / 1000.0
        if np.isfinite(v) and np.isfinite(d.duration_ms) else float("nan")
    )
    denom = fo * np.sqrt(v / REF_VELOCITY_MM_S) if (
        np.isfinite(fo) and np.isfinite(v) and fo > 0 and v > 0) else np.nan
    out["roughness_index"] = (
        d.intensity / denom if np.isfinite(denom) and denom > 0 else float("nan")
    )
    return out


NORMALISED_CORE = ("wavelength_mm", "roughness_index", "pattern", "burst_spacing_mm")


def normalise_frame(df):
    """Add the normalised columns to a descriptor DataFrame."""
    import pandas as pd
    rows = []
    for r in df.itertuples():
        rows.append(normalise(Descriptor(
            material_id=getattr(r, "material_id", -1),
            velocity_mm_s=r.velocity_mm_s,
            force_n=r.force_n,
            frequency_hz=r.frequency_hz,
            intensity=r.intensity,
            pattern=r.pattern,
            duration_ms=r.duration_ms,
        )))
    # 'pattern' is unchanged by normalisation and already present in df;
    # re-adding it would create a duplicate column, which silently turns
    # df['pattern'] into a DataFrame and breaks every downstream groupby.
    new_cols = [c for c in NORMALISED_CORE if c not in df.columns]
    norm = pd.DataFrame(rows, index=df.index)[new_cols]
    out = pd.concat([df, norm], axis=1)

    # Flag recordings whose centroid is pinned near a band edge. A surface with
    # a long spatial wavelength explored slowly excites a fundamental below
    # BAND_LOW_HZ (corduroy at 2.4 mm period and 20 mm/s sits at 8 Hz), the
    # band-pass removes it, and every descriptor for that recording describes
    # filter leakage rather than the texture. These must be excluded or
    # reported, never silently averaged in.
    # One octave from each corner. A 4th-order Butterworth has a transition
    # region, so a centroid reported within an octave of a corner is sitting on
    # the filter skirt rather than on real texture energy -- 1.5x was too loose
    # and let band-edge recordings through with nonsense wavelengths.
    edge_lo = BAND_LOW_HZ * 2.0
    edge_hi = BAND_HIGH_HZ * 0.5
    out["in_band"] = (out["frequency_hz"] > edge_lo) & (out["frequency_hz"] < edge_hi)
    return out


def to_frame(descriptors: list[Descriptor]):
    """Descriptors -> pandas DataFrame, for the separability analysis."""
    import pandas as pd
    return pd.DataFrame([d.as_dict() for d in descriptors])
