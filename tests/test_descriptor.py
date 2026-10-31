"""Verification: does the extractor recover parameters it was handed?

If these fail, nothing downstream can be trusted. Each test synthesises a signal
with a known property and checks the descriptor reports it.
"""
import numpy as np
import pytest

from haptic import descriptor as D
from haptic import signals as G


FS = G.FS_HZ


def _mat(wavelength_mm=0.25, periodicity=0.5, roughness=1.0, burst_per_mm=0.8):
    return G.SyntheticMaterial("t", wavelength_mm, periodicity,
                               roughness, burst_per_mm)


# ---------------------------------------------------------------- frequency
@pytest.mark.parametrize("wavelength_mm,velocity", [
    (0.22, 40.0), (0.50, 40.0), (0.30, 60.0), (0.25, 50.0), (0.18, 30.0),
])
def test_centroid_recovers_carrier(wavelength_mm, velocity):
    m = _mat(wavelength_mm)
    expected = m.carrier_hz(velocity)
    x = G.synthesize(m, velocity, 1.0, 2.0, seed=7)
    d = D.extract(x, FS, velocity_mm_s=velocity, force_n=1.0)
    assert abs(d.frequency_hz - expected) / expected < 0.12, (
        f"centroid {d.frequency_hz:.1f} vs expected {expected:.1f}")


# ----------------------------------------------------------------- duration
@pytest.mark.parametrize("burst_per_mm,velocity", [
    (0.8, 40.0), (1.2, 50.0), (0.6, 60.0), (1.5, 40.0),
])
def test_envelope_period_recovers_modulation(burst_per_mm, velocity):
    m = _mat(0.25, burst_per_mm=burst_per_mm)
    expected_ms = 1000.0 / m.envelope_hz(velocity)
    x = G.synthesize(m, velocity, 1.0, 2.0, seed=11)
    d = D.extract(x, FS, velocity_mm_s=velocity, force_n=1.0)
    assert abs(d.duration_ms - expected_ms) / expected_ms < 0.15, (
        f"period {d.duration_ms:.1f} ms vs expected {expected_ms:.1f} ms")


# ------------------------------------------------------------------ pattern
def test_pattern_is_monotone_in_periodicity():
    """More periodic surface => lower spectral entropy."""
    vals = []
    for per in (0.0, 0.3, 0.6, 0.9):
        reps = [D.extract(G.synthesize(_mat(0.25, per), 40.0, 1.0, 2.0, seed=s),
                          FS).pattern for s in range(4)]
        vals.append(np.mean(reps))
    assert all(a > b for a, b in zip(vals, vals[1:])), (
        f"pattern not monotone decreasing with periodicity: {vals}")


def test_pattern_is_bounded():
    for per in (0.0, 0.5, 1.0):
        d = D.extract(G.synthesize(_mat(0.25, per), 40.0, 1.0, 2.0, seed=3), FS)
        assert 0.0 <= d.pattern <= 1.0


# ---------------------------------------------------------------- intensity
def test_intensity_scales_with_force():
    weak = D.extract(G.synthesize(_mat(), 40.0, 0.5, 2.0, seed=5), FS).intensity
    strong = D.extract(G.synthesize(_mat(), 40.0, 1.0, 2.0, seed=5), FS).intensity
    assert strong > weak * 1.5


def test_intensity_scales_with_roughness():
    a = D.extract(G.synthesize(_mat(roughness=0.5), 40.0, 1.0, 2.0, seed=5), FS)
    b = D.extract(G.synthesize(_mat(roughness=2.0), 40.0, 1.0, 2.0, seed=5), FS)
    assert b.intensity > a.intensity * 2.0


# ------------------------------------------------------------ normalisation
def test_wavelength_is_velocity_invariant():
    """The point of normalisation: lambda should not move with velocity.

    Passes by construction on synthetic data -- the generator encodes
    f = v / lambda. This test proves the arithmetic, not the physics.
    """
    m = _mat(0.25)
    out = []
    for v in G.VELOCITIES_MM_S:
        d = D.extract(G.synthesize(m, v, 1.0, 2.0, seed=13), FS,
                      velocity_mm_s=v, force_n=1.0)
        out.append(D.normalise(d)["wavelength_mm"])
    out = np.array(out)
    cv = out.std() / out.mean()
    assert cv < 0.10, f"wavelength not invariant across velocity, CV={cv:.3f}"


def test_raw_frequency_is_not_velocity_invariant():
    """The complement: the raw descriptor *does* move with velocity.

    This is why pooling velocities destroys separability, and why Objective 3
    needs velocity conditioning.
    """
    m = _mat(0.25)
    out = [D.extract(G.synthesize(m, v, 1.0, 2.0, seed=13), FS).frequency_hz
           for v in G.VELOCITIES_MM_S]
    out = np.array(out)
    assert out.std() / out.mean() > 0.3


# ------------------------------------------------------------------- shapes
def test_accepts_three_axis_input():
    x = np.stack([G.synthesize(_mat(), 40.0, 1.0, 1.0, seed=s)
                  for s in (1, 2, 3)], axis=1)
    assert x.shape[1] == 3
    d = D.extract(x, FS)
    assert np.isfinite(d.frequency_hz)


def test_rejects_short_recording():
    with pytest.raises(ValueError):
        D.extract(np.zeros(16), FS)


def test_bandpass_rejects_impossible_band():
    with pytest.raises(ValueError):
        D.bandpass(np.zeros(1000), fs=30.0, low=20.0, high=1000.0)
