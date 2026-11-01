"""Objective 4 encoders and fidelity metric.

Descriptors and signals here are SYNTHETIC scaffolding: passing shows the code
behaves, not that a fabric can be felt through a phone.
"""
import numpy as np
import pytest

from haptic import descriptor as D
from haptic import render as R
from haptic import signals as G

CFG = R.RenderConfig(intensity_ref=2.0)


def _d(**kw):
    base = dict(material_id=0, velocity_mm_s=40.0, force_n=1.0, frequency_hz=200.0,
                intensity=1.0, pattern=0.5, duration_ms=200.0, env_depth=0.8)
    return D.Descriptor(**{**base, **kw})


def _env(hz, depth=0.8, n=1600):
    return 1 + depth * np.sin(2 * np.pi * hz * np.arange(n) / R.ENV_FS_HZ)


# ------------------------------ the constraint ------------------------------ #

def test_switching_limit_is_far_below_the_texture_band():
    assert R.switching_limit_hz(10) == 50.0
    assert R.switching_limit_hz(20) == 25.0
    assert R.switching_limit_hz(10) < D.BAND_HIGH_HZ / 10


def test_ticks_finer_than_android_can_deliver_are_refused():
    with pytest.raises(ValueError, match="finer"):
        R.RenderConfig(intensity_ref=1.0, tick_ms=5)


@pytest.mark.parametrize("scheme", "ABCD")
def test_no_pattern_switches_faster_than_the_limit(scheme):
    d = _d(frequency_hz=950.0, intensity=1.0)
    p = R.encode_all(d, CFG, _env(24))[scheme]
    positive = [x for x in p.durations if x > 0]
    assert min(positive) >= CFG.tick_ms


def test_carrier_is_not_encoded_by_a_or_b():
    """Two textures differing only in carrier give identical A and B patterns.
    This is the honest statement of the limit, as a test."""
    low, high = _d(frequency_hz=100.0), _d(frequency_hz=800.0)
    env = _env(5)
    assert R.encode_a(low, CFG) == R.encode_a(high, CFG)
    assert R.encode_b(low, CFG, env) == R.encode_b(high, CFG, env)


def test_c_is_a_mapping_and_not_the_frequency():
    """A 400 Hz texture gets a pulse rate under the limit, nowhere near 400 Hz."""
    p = R.encode_c(_d(frequency_hz=400.0), CFG)
    period_ms = p.durations[0] + p.durations[1]
    assert 1000 / period_ms <= R.switching_limit_hz(CFG.tick_ms)


# ---------------------------- pattern well-formed --------------------------- #

@pytest.mark.parametrize("scheme", "ABCD")
def test_every_scheme_fills_the_duration_on_the_tick_grid(scheme):
    p = R.encode_all(_d(), CFG, _env(5))[scheme]
    assert p.total_ms == CFG.duration_ms
    assert all(x % CFG.tick_ms == 0 for x in p.durations)
    assert len(p.durations) <= R.MAX_PATTERN_ENTRIES


def test_binary_reads_durations_as_on_off():
    p = R.VibrationPattern("X", (20, 40, 20), 20)
    assert p.binary().tolist() == [1.0] * 20 + [0.0] * 40 + [1.0] * 20
    assert p.duty == pytest.approx(0.5)


def test_leading_off_tick_needs_a_zero_pulse():
    assert R._from_ticks([False, True], 20, "X").durations == (0, 20, 20)


def test_pattern_longer_than_the_browser_limit_is_refused():
    with pytest.raises(ValueError, match="browser"):
        R.VibrationPattern("X", (20,) * 100, 20)


def test_off_grid_durations_are_refused():
    with pytest.raises(ValueError, match="whole number"):
        R.VibrationPattern("X", (25, 20), 20)


# --------------------------------- scheme A --------------------------------- #

@pytest.mark.parametrize("intensity, duty", [(0.4, 0.2), (1.0, 0.5), (1.6, 0.8)])
def test_a_duty_is_proportional_to_intensity(intensity, duty):
    assert R.encode_a(_d(intensity=intensity), CFG).duty == pytest.approx(duty, abs=0.03)


def test_a_clips_at_full_scale():
    assert R.encode_a(_d(intensity=50.0), CFG).durations == (CFG.duration_ms,)


def test_a_keeps_a_faint_material_audible_but_silent_stays_silent():
    assert R.encode_a(_d(intensity=0.001), CFG).on_ms > 0
    assert R.encode_a(_d(intensity=0.0), CFG).on_ms == 0


# --------------------------------- scheme B --------------------------------- #

def test_b_duty_matches_level():
    assert R.encode_b(_d(), CFG, _env(3)).duty == pytest.approx(0.5, abs=0.03)


def test_b_density_follows_the_envelope():
    """A slow envelope: ticks must bunch at its peak and thin out at its trough."""
    env = _env(1.0, depth=0.9)      # peak near 250 ms, trough near 750 ms
    p = R.encode_b(_d(intensity=1.0), R.RenderConfig(intensity_ref=2.0, duration_ms=1000), env)
    b = p.binary()
    assert b[50:450].mean() > b[550:950].mean() + 0.3


def test_b_filters_unplayable_modulation_instead_of_aliasing_it():
    """At 76 Hz the envelope must not come back as a slow false rhythm."""
    with_modulation = R.encode_b(_d(), CFG, _env(76.0)).binary()
    without = R.encode_b(_d(), CFG, _env(76.0, depth=0.0)).binary()
    changed_ticks = (with_modulation != without).sum() / CFG.tick_ms
    assert changed_ticks <= 4, "filter residue is a tick or two; aliasing is many"


def test_b_rejects_a_short_envelope():
    with pytest.raises(ValueError, match="samples"):
        R.encode_b(_d(), CFG, np.ones(10))


def test_b_uses_the_descriptor_period_without_a_recording():
    p = R.encode_b(_d(duration_ms=250.0), CFG)
    f, s, _ = R.envelope_spectrum(R.achievable_envelope(p), band=(0.5, 12))
    assert f[np.argmax(s)] == pytest.approx(4.0, abs=1.0)


# --------------------------------- scheme C --------------------------------- #

def test_c_finer_texture_gives_a_shorter_period():
    coarse = R.encode_c(_d(frequency_hz=30.0), CFG)
    fine = R.encode_c(_d(frequency_hz=800.0), CFG)
    assert coarse.durations[0] > fine.durations[0]


def test_c_clips_outside_the_band():
    assert R.encode_c(_d(frequency_hz=5000.0), CFG) == R.encode_c(
        _d(frequency_hz=1000.0), CFG)


def test_c_ignores_intensity():
    assert R.encode_c(_d(intensity=0.2), CFG) == R.encode_c(_d(intensity=1.9), CFG)


@pytest.mark.parametrize("field", ["frequency_hz"])
def test_c_refuses_nan(field):
    with pytest.raises(ValueError, match="unusable"):
        R.encode_c(_d(**{field: float("nan")}), CFG)


# --------------------------------- scheme D --------------------------------- #

def test_d_matches_duration_and_duty_exactly():
    ref = R.encode_b(_d(), CFG, _env(5))
    d = R.encode_d(ref, seed=1)
    assert d.total_ms == ref.total_ms and d.on_ms == ref.on_ms


def test_d_differs_by_seed_and_repeats_with_the_same_seed():
    ref = R.encode_a(_d(), CFG)
    assert R.encode_d(ref, 1) != R.encode_d(ref, 2)
    assert R.encode_d(ref, 1) == R.encode_d(ref, 1)


def test_d_destroys_periodic_structure():
    """A's single spectral line must not survive shuffling. Averaged over seeds,
    because one shuffle of 80 ticks can have a chance peak of its own."""
    ref = R.encode_a(_d(), CFG)
    band = (0.5, 25)
    _, a, _ = R.envelope_spectrum(R.achievable_envelope(ref), band=band)
    peaks = [R.envelope_spectrum(R.achievable_envelope(R.encode_d(ref, s)),
                                 band=band)[1].max() for s in range(30)]
    assert a.max() > 1.5 * np.mean(peaks)


# ------------------------------ fidelity metric ----------------------------- #

def test_identical_envelopes_score_zero():
    f, p, _ = R.envelope_spectrum(_env(5))
    assert R.log_spectral_distance(p, p) == 0
    assert R.spectral_wasserstein(f, p, p) == 0


def test_wasserstein_reads_a_frequency_shift_in_hz():
    f, p, _ = R.envelope_spectrum(_env(5), band=(0.5, 25))
    _, q, _ = R.envelope_spectrum(_env(9), band=(0.5, 25))
    assert R.spectral_wasserstein(f, p, q) == pytest.approx(4.0, abs=0.5)


def test_distance_grows_with_the_shift():
    f, p, _ = R.envelope_spectrum(_env(5), band=(0.5, 25))
    near = R.envelope_spectrum(_env(6), band=(0.5, 25))[1]
    far = R.envelope_spectrum(_env(15), band=(0.5, 25))[1]
    assert R.spectral_wasserstein(f, p, near) < R.spectral_wasserstein(f, p, far)


def test_spectra_are_scale_invariant_so_level_is_scored_separately():
    _, p, _ = R.envelope_spectrum(_env(5))
    _, q, _ = R.envelope_spectrum(7 * _env(5))
    assert R.log_spectral_distance(p, q) == pytest.approx(0, abs=1e-6)


def test_in_band_fraction_flags_unplayable_modulation():
    p = R.encode_a(_d(), CFG)
    assert R.fidelity(_env(5), p, 0.5).renderable
    high = R.fidelity(_env(76), p, 0.5)
    assert not high.renderable and high.in_band_fraction < 0.05


def test_level_error_is_duty_against_level():
    p = R.encode_a(_d(intensity=1.0), CFG)
    assert R.fidelity(_env(5), p, 0.8).level_error == pytest.approx(0.3)


def test_flat_envelope_or_flat_motor_is_refused():
    with pytest.raises(ValueError, match="no modulation"):
        R.fidelity(np.ones(1600), R.encode_a(_d(), CFG), 0.5)
    with pytest.raises(ValueError, match="no modulation"):
        R.fidelity(_env(5), R.VibrationPattern("X", (0, 1600), 20), 0.0)


def test_motor_lag_smooths_the_command():
    p = R.VibrationPattern("X", (100, 100), 20)
    env = R.achievable_envelope(p, tau_ms=25)
    assert 0 < env[10] < 1
    assert env[99] > 0.9 and env[199] < 0.15


def test_signal_envelope_finds_a_known_modulation():
    fs = 6000.0
    t = np.arange(int(2 * fs)) / fs
    x = np.sin(2 * np.pi * 300 * t) * (1 + 0.8 * np.sin(2 * np.pi * 7 * t))
    env = R.signal_envelope(x, fs)
    f, s, _ = R.envelope_spectrum(env, band=(0.5, 25))
    assert f[np.argmax(s)] == pytest.approx(7.0, abs=0.5)


# ------------------------------ schemes compared ---------------------------- #

def _synthetic_scores(scheme, seeds=range(6)):
    """Fabrics whose envelope is below the 25 Hz limit at 20 mm/s."""
    fabrics = [m for m in G.FABRIC_BANK if m.envelope_hz(20) < 18]
    cfg = R.RenderConfig(intensity_ref=2.5)
    out = []
    for m in fabrics:
        for s in seeds:
            x = G.synthesize(m, 20.0, seed=s)
            d = D.extract(x, G.FS_HZ)
            env = R.signal_envelope(x, G.FS_HZ)
            p = R.encode_all(d, cfg, env, seed=s)[scheme]
            if p.duty in (0.0, 1.0):
                continue
            out.append(R.fidelity(env, p, min(d.intensity / 2.5, 1.0)))
    return out


def test_envelope_tracking_beats_the_random_control():
    """Scheme B must beat its own shuffled control, or it tracks nothing."""
    b = np.median([f.wasserstein_hz for f in _synthetic_scores("B")])
    d = np.median([f.wasserstein_hz for f in _synthetic_scores("D")])
    assert b < 0.75 * d, f"B {b:.2f} Hz vs control {d:.2f} Hz"
