from webapp import demo


def test_lan_ip_is_a_routable_address_or_none():
    ip = demo.lan_ip()
    assert ip is None or (ip.count(".") == 3 and not ip.startswith("127."))


def test_missing_model_is_reported_not_crashed(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(demo.P, "DEFAULT_CHECKPOINT", tmp_path / "missing.pt")
    assert demo.main() == 1
    assert "Model not found" in capsys.readouterr().err
