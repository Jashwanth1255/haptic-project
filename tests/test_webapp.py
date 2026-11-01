"""Objective 4 service. The classifier is a stub, so no model file is needed, and
every descriptor is synthetic scaffolding."""
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from haptic import render as R
from vision.textile import TEXTILE_CLASSES
from webapp import pipeline as P
from webapp.app import create_app
from webapp.sessions import SessionLog, summarise_latency


def _stub(cls="woven"):
    rest = [c for c in TEXTILE_CLASSES if c != cls]
    return lambda image: [(cls, 0.7), (rest[0], 0.2), (rest[1], 0.1)]


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (120, 60, 30)).save(buf, "PNG")
    return buf.getvalue()


def _unreadable(image):
    raise OSError("cannot identify")


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(classify=_stub(), log_dir=tmp_path))


def _post(client, data=None):
    return client.post("/api/analyse", files={"image": ("x.png", data or _png(),
                                                         "image/png")})


def test_page_states_its_limits(client):
    """These notices are part of the deliverable, so losing one is a failure."""
    page = client.get("/").text
    script = client.get("/static/app.js").text
    assert "cannot play the texture itself" in page
    assert "Synthetic data" in page
    assert "WebKit" in script
    assert "cannot confirm the motor" in script


@pytest.mark.parametrize("name", ["weave", "knit", "pleat", "felt"])
def test_samples_are_served(client, name):
    assert client.get(f"/static/samples/{name}.png").status_code == 200


def test_every_textile_class_has_a_fabric():
    assert set(P.CLASS_TO_FABRIC) == set(TEXTILE_CLASSES)


def test_analyse_returns_everything_the_page_shows(client):
    r = _post(client).json()
    assert r["predicted"]["class"] == "woven"
    assert len(r["predicted"]["top3"]) == 3
    assert set(r["descriptor"]) == {"frequency_hz", "intensity", "pattern", "duration_ms"}
    assert set(r["patterns"]) == set("ABCD")
    assert r["waveform"]["y"] and r["spectrum"]["f_hz"] and r["envelope"]["y"]
    assert set(r["timings_ms"]) == {"classify", "retrieve", "encode"}


def test_response_always_declares_itself_synthetic(client):
    assert _post(client).json()["synthetic"] is True


def test_patterns_are_valid_vibrate_arguments(client):
    for p in _post(client).json()["patterns"].values():
        d = p["durations"]
        assert all(isinstance(x, int) and x >= 0 for x in d)
        assert len(d) <= R.MAX_PATTERN_ENTRIES
        assert sum(d) == p["total_ms"]


def test_fidelity_is_on_envelopes_and_states_the_limit(client):
    r = _post(client).json()
    assert r["limits"]["switching_hz"] == 25.0
    scores = [f for f in r["fidelity"].values() if f]
    assert scores
    for f in scores:
        assert {"log_spectral_db", "wasserstein_hz", "renderable"} <= set(f)
        assert f["band_hz"][1] == 25.0


@pytest.mark.parametrize("cls", TEXTILE_CLASSES)
def test_every_class_can_be_served(tmp_path, cls):
    c = TestClient(create_app(classify=_stub(cls), log_dir=tmp_path))
    assert _post(c).status_code == 200


def test_empty_upload_is_refused(client):
    assert client.post("/api/analyse",
                       files={"image": ("x.png", b"", "image/png")}).status_code == 400


def test_oversize_upload_is_refused(client):
    assert _post(client, b"x" * (10 * 1024 * 1024 + 1)).status_code == 413


def test_unreadable_image_is_a_400_not_a_crash(tmp_path):
    c = TestClient(create_app(classify=_unreadable, log_dir=tmp_path))
    assert _post(c).status_code == 400


def test_session_keeps_device_fields_and_drops_the_rest(client, tmp_path):
    r = client.post("/api/session", json={
        "session_id": "s1", "device_model": "Pixel 8", "os": "Android",
        "os_version": "14", "user_agent": "UA", "ip": "1.2.3.4"})
    assert r.status_code == 200
    row = json.loads((tmp_path / "sessions.jsonl").read_text())
    assert row["device_model"] == "Pixel 8" and row["os"] == "Android"
    assert row["user_agent"] == "UA" and "ip" not in row and "logged_at" in row


def test_session_without_an_id_is_refused(client):
    assert client.post("/api/session", json={"os": "Android"}).status_code == 422


def test_long_strings_are_cut(tmp_path):
    row = SessionLog(tmp_path).session({"session_id": "s", "user_agent": "x" * 9000})
    assert len(row["user_agent"]) == 512


def test_latency_round_trip(client):
    for ms in (100.0, 200.0, 300.0):
        client.post("/api/latency", json={"session_id": "s", "machine_ms": ms,
                                          "select_to_ready_ms": ms, "autoplay": True,
                                          "end_to_end_ms": ms})
    s = client.get("/api/latency").json()
    assert s["n"] == 3
    assert s["machine_ms"]["median"] == 200.0


def test_human_wait_is_not_counted_in_autoplay_end_to_end():
    rows = [{"end_to_end_ms": 5000.0, "autoplay": False, "machine_ms": 90.0},
            {"end_to_end_ms": 100.0, "autoplay": True, "machine_ms": 100.0}]
    s = summarise_latency(rows)
    assert s["end_to_end_ms_autoplay"]["median"] == 100.0
    assert s["end_to_end_ms_autoplay"]["n"] == 1


def test_empty_latency_log_has_no_numbers():
    s = summarise_latency([])
    assert s["n"] == 0 and s["machine_ms"] is None


def test_real_classifier_ranks_all_twelve():
    pytest.importorskip("torch")
    if not P.DEFAULT_CHECKPOINT.exists():
        pytest.skip("weights are not in the repository")
    ranked = P.load_classifier()(_png())
    assert {c for c, _ in ranked} == set(TEXTILE_CLASSES)
    assert sum(p for _, p in ranked) == pytest.approx(1.0, abs=1e-4)
