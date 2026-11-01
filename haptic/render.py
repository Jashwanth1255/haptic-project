"""Objective 4 -- map a haptic descriptor to a `navigator.vibrate()` pattern.

WHAT THIS CANNOT DO, AND DOES NOT TRY TO
----------------------------------------
`navigator.vibrate()` takes a list of millisecond on/off durations. It has no
amplitude control and no frequency control, and practical timing granularity on
Android is 10-20 ms, so the fastest possible on/off switching is about 50 Hz (25 Hz
at the 20 ms tick used by default). Texture energy lives at 20-1000 Hz.

**The carrier frequency therefore cannot be reproduced, and nothing here pretends
to.** The motor's own resonance sets the carrier the skin receives, whatever the
pattern says. Only three things can be encoded:

    intensity        as the share of time the motor is on
    envelope rate    as the timing of pulses, if it is below the switching limit
    coarseness       as a slow pulse rate standing in for the spectral centroid

Scheme C maps centroid onto pulse period, but that is a mapping, a code for
"finer" versus "coarser", not a reproduction: a 400 Hz texture does not become a
400 Hz vibration. Envelope modulation faster than the switching limit is lost as
well, and `fidelity` reports how much of it was.

For the same reason fidelity compares *envelope* spectra. Comparing raw
acceleration spectra would measure a gap that is impossible to close by
construction, and any number produced that way would mislead.

The schemes:

    A  fixed period, duty proportional to intensity
    B  pulse density tracking the amplitude envelope
    C  pulse period from the spectral centroid, scaled into the renderable band
    D  random control, matching a reference pattern's duration and duty

Descriptors from `signals.py` are scaffolding: scores computed on them say the
code works, not that a fabric can be felt.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal as sps
from scipy.stats import wasserstein_distance

from . import descriptor as D
from .descriptor import Descriptor

MIN_TICK_MS = 10            # finer timing than this is not dependable on Android
DEFAULT_TICK_MS = 20
MAX_PATTERN_ENTRIES = 99    # Chromium's limit as remembered; check on real devices
ENV_FS_HZ = 1000.0          # 1 ms per sample, matching vibrate()'s units

# First-order motor lag. An ASSUMED value, not a measurement: LRA and ERM motors
# differ and no device has been characterised. Report sensitivity to it.
ACTUATOR_TAU_MS = 25.0

# Scheme A
DUTY_PERIOD_MS = 200        # 10 ticks at 20 ms, so 10 distinct intensity levels
# Scheme C: period range for the centroid mapping
PERIOD_MIN_MS = 40
PERIOD_MAX_MS = 400

ENV_BAND_LOW_HZ = 0.5       # below this is the mean level, not modulation
SPECTRUM_FLOOR = 1e-4       # -40 dB of total in-band power
RENDERABLE_FRACTION = 0.5   # below this the intended envelope mostly cannot play


def switching_limit_hz(tick_ms: int = DEFAULT_TICK_MS) -> float:
    """Fastest on/off rate at this tick: one tick on, one tick off."""
    return 1000.0 / (2 * tick_ms)


@dataclass(frozen=True)
class RenderConfig:
    """`intensity_ref` is the intensity that maps to the motor fully on. It is a
    required choice: it depends on the corpus, see `intensity_reference`."""
    intensity_ref: float
    duration_ms: int = 1600     # whole periods of A: a cut-off one skews its duty
    tick_ms: int = DEFAULT_TICK_MS

    def __post_init__(self):
        if self.tick_ms < MIN_TICK_MS:
            raise ValueError(f"tick {self.tick_ms} ms is finer than the "
                             f"{MIN_TICK_MS} ms Android can be relied on for")
        if self.duration_ms % self.tick_ms:
            raise ValueError("duration_ms must be a whole number of ticks")
        if not self.intensity_ref > 0:
            raise ValueError("intensity_ref must be positive")

    @property
    def n_ticks(self) -> int:
        return self.duration_ms // self.tick_ms


def intensity_reference(descriptors, q: float = 95.0) -> float:
    """A corpus-wide full-scale intensity. A percentile, so one outlier does not
    push every other material down to a few ticks."""
    values = [d.intensity for d in descriptors if np.isfinite(d.intensity)]
    if not values:
        raise ValueError("no finite intensities")
    return float(np.percentile(values, q))


@dataclass(frozen=True)
class VibrationPattern:
    """Durations as `navigator.vibrate()` reads them: on, off, on, off, ..."""
    scheme: str
    durations: tuple[int, ...]
    tick_ms: int

    def __post_init__(self):
        if not self.durations:
            raise ValueError("empty pattern")
        if len(self.durations) > MAX_PATTERN_ENTRIES:
            raise ValueError(f"{len(self.durations)} entries exceeds the browser "
                             f"limit of {MAX_PATTERN_ENTRIES}")
        if any(d < 0 or d % self.tick_ms for d in self.durations):
            raise ValueError("every duration must be a non-negative whole number "
                             "of ticks")

    @property
    def total_ms(self) -> int:
        return sum(self.durations)

    @property
    def on_ms(self) -> int:
        return sum(self.durations[0::2])

    @property
    def duty(self) -> float:
        return self.on_ms / self.total_ms if self.total_ms else 0.0

    def binary(self, fs: float = ENV_FS_HZ) -> np.ndarray:
        """Motor command over time: 1 = on, 0 = off."""
        states = np.arange(len(self.durations)) % 2 == 0
        samples = np.round(np.array(self.durations) * fs / 1000).astype(int)
        return np.repeat(states.astype(float), samples)


def _from_ticks(ticks, tick_ms: int, scheme: str) -> VibrationPattern:
    """Run-length encode on/off ticks. A leading off tick needs a 0 ms pulse."""
    ticks = np.asarray(ticks, dtype=bool)
    edges = np.flatnonzero(np.diff(ticks)) + 1
    runs = np.diff(np.concatenate([[0], edges, [len(ticks)]])) * tick_ms
    durations = runs.tolist() if ticks[0] else [0, *runs.tolist()]
    return VibrationPattern(scheme, tuple(int(r) for r in durations), tick_ms)


def _level(d: Descriptor, cfg: RenderConfig) -> float:
    if not np.isfinite(d.intensity) or d.intensity < 0:
        raise ValueError(f"unusable intensity: {d.intensity}")
    return min(d.intensity / cfg.intensity_ref, 1.0)


def _periodic_ticks(n: int, period: int, on: int) -> np.ndarray:
    return (np.arange(n) % period) < on


def encode_a(d: Descriptor, cfg: RenderConfig) -> VibrationPattern:
    """A: fixed period, duty proportional to intensity. Encodes intensity only.

    Resolution is one tick per period. Anything audible gets at least one tick,
    so a faint material does not fall silent through rounding.
    """
    period = max(DUTY_PERIOD_MS // cfg.tick_ms, 2)
    level = _level(d, cfg)
    on = int(np.clip(round(level * period), 1, period)) if level > 0 else 0
    return _from_ticks(_periodic_ticks(cfg.n_ticks, period, on), cfg.tick_ms, "A")


def encode_b(d: Descriptor, cfg: RenderConfig,
             envelope: np.ndarray | None = None) -> VibrationPattern:
    """B: pulse density tracking the amplitude envelope.

    A first-order sigma-delta over ticks, so density follows the envelope and
    the long-run duty matches the intended level. The envelope is low-passed
    first: modulation above the switching limit cannot be played, and left in it
    would alias into a slower rhythm the texture does not have.

    `envelope` is at ENV_FS_HZ. Without one, `descriptor_envelope` is used.
    """
    env = _fit(descriptor_envelope(d, cfg) if envelope is None else envelope, cfg)
    level = _level(d, cfg)
    density = np.minimum(_tick_means(env, cfg) * level, 1.0)
    acc, ticks = 0.0, np.zeros(cfg.n_ticks, dtype=bool)
    for i, x in enumerate(density):
        acc += x
        if acc >= 1.0:
            acc -= 1.0
            ticks[i] = True
    return _from_ticks(ticks, cfg.tick_ms, "B")


def encode_c(d: Descriptor, cfg: RenderConfig) -> VibrationPattern:
    """C: pulse period from the spectral centroid. Encodes coarseness only.

    The centroid's position on a log scale across the analysis band is mapped to
    the period range: finer texture, shorter period. The pulse rate that results
    is a code for coarseness, and is not the texture's frequency.
    """
    f = d.frequency_hz
    if not np.isfinite(f) or f <= 0:
        raise ValueError(f"unusable frequency: {f}")
    pos = np.log(f / D.BAND_LOW_HZ) / np.log(D.BAND_HIGH_HZ / D.BAND_LOW_HZ)
    pos = float(np.clip(pos, 0.0, 1.0))
    low = max(PERIOD_MIN_MS, 2 * cfg.tick_ms)
    period_ms = PERIOD_MAX_MS * (low / PERIOD_MAX_MS) ** pos
    period = max(int(round(period_ms / cfg.tick_ms)), 2)
    return _from_ticks(_periodic_ticks(cfg.n_ticks, period, max(period // 2, 1)),
                       cfg.tick_ms, "C")


def encode_d(reference: VibrationPattern, seed: int) -> VibrationPattern:
    """D: control. The reference's ticks shuffled, so total duration and duty
    match exactly and all temporal structure is destroyed."""
    n_on = reference.on_ms // reference.tick_ms
    ticks = np.zeros(reference.total_ms // reference.tick_ms, dtype=bool)
    ticks[:n_on] = True
    np.random.default_rng(seed).shuffle(ticks)
    return _from_ticks(ticks, reference.tick_ms, "D")


# --------------------------------------------------------------------------- #
# intended envelopes
# --------------------------------------------------------------------------- #

def signal_envelope(x: np.ndarray, fs: float) -> np.ndarray:
    """Envelope of a recording at ENV_FS_HZ, by the same steps as `descriptor`."""
    xb = D.bandpass(np.asarray(x, dtype=float) - np.mean(x), fs)
    env = D.envelope(xb, fs)
    return np.maximum(sps.resample_poly(env, int(ENV_FS_HZ), int(fs)), 0.0)


def descriptor_envelope(d: Descriptor, cfg: RenderConfig) -> np.ndarray:
    """Sinusoidal envelope at the descriptor's modulation period, used when no
    recording is available. Mean 1, shape only: level comes from intensity."""
    if not np.isfinite(d.duration_ms) or d.duration_ms <= 0:
        raise ValueError(f"unusable duration_ms: {d.duration_ms}")
    depth = d.env_depth if np.isfinite(d.env_depth) else 0.8
    t = np.arange(cfg.duration_ms) / ENV_FS_HZ
    return 1.0 + float(np.clip(depth, 0, 1)) * np.sin(2 * np.pi * t * 1000 / d.duration_ms)


def _fit(env: np.ndarray, cfg: RenderConfig) -> np.ndarray:
    env = np.asarray(env, dtype=float)
    n = int(cfg.duration_ms * ENV_FS_HZ / 1000)
    if env.size < n:
        raise ValueError(f"envelope has {env.size} samples, need {n}")
    return env[:n]


def _tick_means(env: np.ndarray, cfg: RenderConfig) -> np.ndarray:
    """Envelope per tick, mean 1, with unplayable modulation removed first."""
    cutoff = 0.9 * switching_limit_hz(cfg.tick_ms)
    sos = sps.butter(4, cutoff / (ENV_FS_HZ / 2), btype="low", output="sos")
    smooth = np.maximum(sps.sosfiltfilt(sos, env), 0.0)
    per_tick = smooth.reshape(cfg.n_ticks, -1).mean(axis=1)
    mean = per_tick.mean()
    if mean <= 0:
        raise ValueError("envelope is silent")
    return per_tick / mean


# --------------------------------------------------------------------------- #
# fidelity
# --------------------------------------------------------------------------- #

def achievable_envelope(p: VibrationPattern, tau_ms: float = ACTUATOR_TAU_MS,
                        fs: float = ENV_FS_HZ) -> np.ndarray:
    """What the motor's amplitude does under this pattern: the on/off command
    through a first-order lag. See ACTUATOR_TAU_MS: tau is assumed."""
    alpha = 1.0 - np.exp(-1000.0 / (tau_ms * fs))
    return sps.lfilter([alpha], [1, alpha - 1], p.binary(fs))


def envelope_spectrum(env: np.ndarray, fs: float = ENV_FS_HZ, band=None):
    """Power spectrum of an envelope's modulation, normalised to sum to 1 in
    `band`, and the share of modulation power that lies inside `band`.

    Normalised because this compares the shape of the modulation. Level is
    scored separately, as `Fidelity.level_error`.
    """
    env = np.asarray(env, dtype=float)
    x = (env - env.mean()) * np.hanning(env.size)
    f = np.fft.rfftfreq(env.size, 1 / fs)
    p = np.abs(np.fft.rfft(x)) ** 2
    low, high = band or (ENV_BAND_LOW_HZ, fs / 2)
    mask = (f >= low) & (f <= high)
    total = p[f >= low].sum()
    inside = p[mask].sum()
    if total <= 0 or inside <= 0:
        return f[mask], np.zeros(mask.sum()), 0.0
    return f[mask], p[mask] / inside, float(inside / total)


def log_spectral_distance(p: np.ndarray, q: np.ndarray,
                          floor: float = SPECTRUM_FLOOR) -> float:
    """RMS difference in dB between two normalised spectra."""
    diff = 10 * np.log10((p + floor) / (q + floor))
    return float(np.sqrt(np.mean(diff ** 2)))


def spectral_wasserstein(f: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """Earth-mover distance in Hz between two normalised spectra: how far the
    modulation energy would have to move."""
    return float(wasserstein_distance(f, f, p, q))


@dataclass(frozen=True)
class Fidelity:
    scheme: str
    log_spectral_db: float
    wasserstein_hz: float
    level_error: float      # |duty - intended level|
    band_hz: tuple[float, float]
    in_band_fraction: float # share of intended modulation that can be played

    @property
    def renderable(self) -> bool:
        """False: most intended modulation is above the switching limit, so the
        two distances describe a remainder and not the envelope."""
        return self.in_band_fraction >= RENDERABLE_FRACTION

    def as_dict(self) -> dict:
        return {**asdict(self), "renderable": self.renderable}


def fidelity(intended_envelope: np.ndarray, pattern: VibrationPattern,
             level: float, tau_ms: float = ACTUATOR_TAU_MS) -> Fidelity:
    """Compare intended and achievable envelope spectra over the playable band."""
    achieved = achievable_envelope(pattern, tau_ms)
    intended = np.asarray(intended_envelope, dtype=float)
    if intended.size < achieved.size:
        raise ValueError(f"envelope has {intended.size} samples, need {achieved.size}")
    intended = intended[:achieved.size]
    if np.ptp(intended) == 0 or np.ptp(achieved) == 0:
        raise ValueError("an envelope with no modulation has no spectrum to compare")

    band = (ENV_BAND_LOW_HZ, switching_limit_hz(pattern.tick_ms))
    f, p, in_band = envelope_spectrum(intended, band=band)
    _, q, _ = envelope_spectrum(achieved, band=band)
    return Fidelity(pattern.scheme, log_spectral_distance(p, q),
                    spectral_wasserstein(f, p, q), abs(pattern.duty - level),
                    band, in_band)


def encode_all(d: Descriptor, cfg: RenderConfig, envelope: np.ndarray | None = None,
               seed: int = 0) -> dict[str, VibrationPattern]:
    """All four schemes. D shuffles B, the scheme it is the control for."""
    b = encode_b(d, cfg, envelope)
    return {"A": encode_a(d, cfg), "B": b, "C": encode_c(d, cfg),
            "D": encode_d(b, seed)}
