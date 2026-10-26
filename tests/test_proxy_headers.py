"""`python -m kalos.portal` must not trust forwarded headers unless told which proxies to trust."""
from kalos.portal import __main__ as entry


def _run(monkeypatch, trusted):
    seen = {}
    monkeypatch.setattr(entry.uvicorn, "run", lambda *a, **k: seen.update(k))
    monkeypatch.setattr(entry.engine_lock, "acquire_engine_lock", lambda: object())
    if trusted is None:
        monkeypatch.delenv("KALOS_TRUSTED_PROXIES", raising=False)
    else:
        monkeypatch.setenv("KALOS_TRUSTED_PROXIES", trusted)
    monkeypatch.setattr("sys.argv", ["kalos.portal"])
    try:
        entry.main()
    except SystemExit:
        pass
    return seen


def test_forwarded_headers_ignored_by_default(monkeypatch):
    seen = _run(monkeypatch, None)
    assert seen.get("proxy_headers") is False


def test_forwarded_headers_only_from_named_proxies(monkeypatch):
    seen = _run(monkeypatch, "10.0.0.5")
    assert seen.get("proxy_headers") is True and seen.get("forwarded_allow_ips") == "10.0.0.5"
