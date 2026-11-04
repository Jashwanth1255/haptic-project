"""Cluster loader, on fake files laid out as the dataset documents them.

Both layouts: figshare (flat CSV folders) and the Hugging Face mirror (Parquet in
one folder per material)."""
import numpy as np
import pandas as pd
import pytest

from haptic import cli
from haptic import descriptor as D
from haptic import loader as L

FS_TRUE = 8600.0    # what the real mirror files hold


def _accel(seconds=3.0, freq=200.0, seed=0, jitter=True):
    """Unevenly sampled, like the documented sample rows."""
    rng = np.random.default_rng(seed)
    dt = 1 / FS_TRUE * (rng.uniform(0.6, 1.4, int(seconds * FS_TRUE)) if jitter else 1)
    t = np.cumsum(np.broadcast_to(dt, int(seconds * FS_TRUE))) + 0.001
    z = -1 + 0.05 * np.sin(2 * np.pi * freq * t) + 0.002 * rng.normal(size=t.size)
    return pd.DataFrame({"time": t, "X": 0.01 * rng.normal(size=t.size),
                         "Y": 0.01 * rng.normal(size=t.size), "Z": z})


def _position(velocity, seconds=3.0, ramp=0.6):
    """Still, then sliding at `velocity`, then still: 100 Hz."""
    t = np.arange(0, seconds, 0.01) + 0.0006
    moving = np.clip(t - ramp, 0, seconds - 2 * ramp)
    return pd.DataFrame({"time": t, "X": velocity * moving, "Y": np.full(t.size, 45.0)})


def _write(root, mid, direction, vel, force_mn, rep, layout, freq=200.0):
    name = f"{mid}_{direction}_{vel}_{force_mn}_{rep}"
    for signal, table in (("accel", _accel(freq=freq, seed=mid + rep)),
                          ("position", _position(vel))):
        if layout == "figshare":
            folder = root / "sensor_data" / signal
            folder.mkdir(parents=True, exist_ok=True)
            table.to_csv(folder / f"{name}.csv", index=False)
        else:   # the mirror stores integer nanoseconds as time_ns
            folder = root / "sensor_data" / signal / str(mid)
            folder.mkdir(parents=True, exist_ok=True)
            ns = (table.pop("time") * 1e9).round().astype("int64")
            table.insert(0, "time_ns", ns)
            table.to_parquet(folder / f"{name}.parquet")


@pytest.fixture(params=["figshare", "mirror"])
def dataset(tmp_path, request):
    for mid in (102, 103, 5):
        for direction in (0, 180):
            for rep in (0, 1):
                _write(tmp_path, mid, direction, 40, 1000, rep, request.param)
    (tmp_path / "sensor_data" / "accel" / "notes_1_2.csv").write_text("a\n1\n")
    return tmp_path


def test_names_parse_with_force_in_millinewtons():
    assert L.parse_name("102_135_40_500_1.csv") == {
        "material_id": 102, "direction_deg": 135, "velocity_mm_s": 40.0,
        "force_n": 0.5, "repeat": 1}


@pytest.mark.parametrize("name", ["102_0_20_1000_0.wav", "102_30_20_1000_0.csv",
                                  "102_0_25_1000_0.csv", "102_0_20_750_0.csv",
                                  "102_0_20_1000_2.csv", "102_20_1000.csv"])
def test_impossible_or_foreign_names_are_rejected(name):
    """The old loader read `102_0_20...` as velocity 0 and force 20 N, silently."""
    assert L.parse_name(name) is None


def test_discover_finds_only_cloth_accel_files(dataset):
    stubs = L.discover(dataset)
    assert len(stubs) == 2 * 2 * 2
    assert {s["material_id"] for s in stubs} == {102, 103}
    assert all(s["path"].parent.name in ("accel", "102", "103") for s in stubs)


def test_resampling_gives_a_uniform_grid_and_keeps_the_frequency():
    t = _accel(freq=300.0)
    grid, values, measured = L.resample(t)
    assert np.allclose(np.diff(grid), 1 / L.ACCEL_FS_HZ)
    assert measured == pytest.approx(FS_TRUE, rel=0.05)
    z = values[:, 2] - values[:, 2].mean()
    f = np.fft.rfftfreq(z.size, 1 / L.ACCEL_FS_HZ)[np.argmax(np.abs(np.fft.rfft(z)))]
    assert f == pytest.approx(300.0, abs=2)


def test_resampling_does_not_fold_content_above_the_new_nyquist():
    """A 4 kHz tone sampled at 8.6 kHz lies above the 3 kHz Nyquist of the 6 kHz
    grid. Plain interpolation folds it to 2 kHz; the anti-aliasing filter must not."""
    t = _accel(seconds=1.0, freq=4000.0, jitter=False)
    _, values, _ = L.resample(t)
    z = values[:, 2] - values[:, 2].mean()
    spec = np.abs(np.fft.rfft(z))
    f = np.fft.rfftfreq(z.size, 1 / L.ACCEL_FS_HZ)
    folded = spec[(f > 1900) & (f < 2100)].max()
    assert folded < 0.05 * 0.05 * z.size / 2


def test_time_in_nanoseconds_is_read_as_seconds():
    t = _accel(seconds=0.2, jitter=False)
    ns = t.assign(time_ns=(t.pop("time") * 1e9).astype("int64"))
    g1, v1, m1 = L.resample(_accel(seconds=0.2, jitter=False))
    g2, v2, m2 = L.resample(ns)
    assert m2 == pytest.approx(m1, rel=1e-6) and np.allclose(v1, v2, atol=1e-6)


def test_a_table_without_time_is_an_error():
    with pytest.raises(ValueError, match="no time column"):
        L.resample(pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0]}))


def test_resampling_survives_duplicate_and_unsorted_times():
    t = _accel(seconds=0.2, jitter=False)
    messy = pd.concat([t.iloc[::-1], t.iloc[:10]])
    grid, values, _ = L.resample(messy)
    assert np.all(np.diff(grid) > 0) and len(values) == len(grid)


def test_steady_window_skips_the_approach_and_the_stop():
    w = L.steady_window(_position(40.0), 40.0)
    assert w[0] == pytest.approx(0.6, abs=0.06) and w[1] == pytest.approx(2.4, abs=0.06)


def test_no_steady_sliding_gives_no_window():
    still = _position(40.0).assign(X=0.0)
    assert L.steady_window(still, 40.0) is None


def test_load_keeps_the_steady_stretch_and_reports_the_rate(dataset):
    stub = L.discover(dataset)[0]
    rec = L.load(stub)
    assert rec["accel"].shape[1] == 3
    assert rec["steady_s"] == pytest.approx(1.8, abs=0.1)
    assert len(rec["accel"]) == pytest.approx(rec["steady_s"] * L.ACCEL_FS_HZ, rel=0.01)
    assert rec["material_name"] in ("Velvet", "Corduroy")
    assert not rec["rate_off"]
    d = D.extract(rec["accel"], rec["fs"], material_id=rec["material_id"],
                  velocity_mm_s=rec["velocity_mm_s"], force_n=rec["force_n"])
    assert d.frequency_hz == pytest.approx(200.0, rel=0.15)


def test_a_missing_position_file_is_an_error_not_a_guess(dataset):
    stub = L.discover(dataset)[0]
    L._sibling(stub["path"], "position").unlink()
    with pytest.raises(ValueError, match="no position"):
        L.load(stub)


def test_repeat_index_puts_later_directions_in_the_upper_half():
    assert L.repeat_index(0, 0) == 0 and L.repeat_index(135, 1) == 7
    assert L.repeat_index(180, 0) == 8 and L.repeat_index(315, 1) == 15


def test_cloth_names_cover_every_cloth_id():
    assert set(L.CLOTH_NAMES) == set(L.CLOTH_IDS)
    assert L.CLOTH_NAMES[102] == "Velvet" and L.CLOTH_NAMES[117] == "Cotton linen"


def test_inspect_reports_counts_and_parsing(dataset):
    text = L.inspect(dataset)
    assert "parsed 12, unparsed 1" in text and "position file: found" in text
    assert "Velvet" in text


def test_gate_runs_on_the_real_layout_and_counts_skips(dataset, tmp_path):
    L._sibling(L.discover(dataset)[0]["path"], "position").unlink()
    assert cli.main(["gate", "--root", str(dataset), "--out", str(tmp_path / "d.csv")]) == 0
    import runlog
    run = runlog.load_run("o1_gate")
    assert run.info["synthetic"] is False
    assert run.metrics["n_skipped"] == 1
    df = pd.read_csv(tmp_path / "d.csv")
    assert {"direction_deg", "repeat", "steady_s"} <= set(df.columns)
