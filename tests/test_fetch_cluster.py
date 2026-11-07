"""The downloader must survive dropped connections and rate limits, and still
fail loudly on a permanent error. No network: hf_hub_download is faked."""
from pathlib import Path

import huggingface_hub
import pytest

import fetch_cluster as F


def _fake(fail_first: int, exc):
    calls = {"n": 0}

    def download(repo, f, repo_type, revision, local_dir):
        calls["n"] += 1
        if calls["n"] <= fail_first:
            raise exc
        p = Path(local_dir) / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
        return str(p)
    return download, calls


ARGS = ["--no-images", "--directions", "0", "--repeats", "0"]


def test_retries_a_dropped_connection(tmp_path, monkeypatch):
    dl, calls = _fake(2, ConnectionError("Network error: Request middleware error"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", dl)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert F.main(["--out", str(tmp_path), *ARGS]) == 0
    n = len(F.expected_files(ids=F.CLOTH_IDS, images=False, directions=[0], repeats=[0]))
    assert calls["n"] == n + 2


def test_gives_up_after_repeated_network_failures(tmp_path, monkeypatch):
    dl, _ = _fake(10 ** 6, ConnectionError("down"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", dl)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(ConnectionError):
        F.main(["--out", str(tmp_path), *ARGS])


def test_threads_fetch_everything_and_retry(tmp_path, monkeypatch):
    dl, calls = _fake(3, ConnectionError("blip"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", dl)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert F.main(["--out", str(tmp_path), *ARGS, "--threads", "4"]) == 0
    n = len(F.expected_files(ids=F.CLOTH_IDS, images=False, directions=[0], repeats=[0]))
    assert calls["n"] == n + 3
