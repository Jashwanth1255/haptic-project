"""Audio band features for analysis M (invented signals)."""
import numpy as np

import make_audio118 as M


def test_no_empty_band_for_broadband_noise():
    fs = 44100
    x = np.random.default_rng(0).normal(size=fs)
    for prefix, (lo, hi) in M.RANGES.items():
        r = M.band_shares(x, fs, lo, hi, prefix)
        v = np.array([r[c] for c in M.columns(prefix)[:-1]])
        assert v.min() > -12, prefix          # log(1e-12) would mean an empty band


def test_tone_lands_in_its_band():
    fs = 44100
    t = np.arange(fs) / fs
    x = np.sin(2 * np.pi * 3000 * t) + 1e-3 * np.random.default_rng(1).normal(size=fs)
    r = M.band_shares(x, fs, 20.0, 8000.0, "a8k")
    edges = np.geomspace(20, 8000, M.N_BANDS + 1)
    k = np.searchsorted(edges, 3000) - 1
    v = [r[c] for c in M.columns("a8k")[:-1]]
    assert int(np.argmax(v)) == k
