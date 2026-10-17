"""Tenant id validation (P0 hardening): malformed tenant ids in the token config
are REJECTED at load (fail closed), never sanitised; and the per-tenant latest
cache filename is an injective function of the tenant id, so two distinct
tenants can never share (read or overwrite) one cache file."""
from __future__ import annotations

import hashlib
import json

import pytest

from kalos.portal.auth import (
    AuthConfigError,
    Authenticator,
    is_valid_tenant_id,
    validate_tenant_id,
)


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def _configure(monkeypatch, tenant) -> None:
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps([
        {"token_sha256": _sha("good"), "subject": "ok", "tenant": "acme", "scopes": ["read"]},
        {"token_sha256": _sha("tok"), "subject": "s", "tenant": tenant, "scopes": ["read"]},
    ]))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)


_BAD = [
    "acme/bio", "../etc", "acme bio", "acme.bio", "a" * 65, "tenant\n", "ténant",
    "acme\x00", ".", "..", "C:\\x", "acme:bio",
]
_GOOD = ["acme", "acme_bio", "acme-bio", "A9_z-", "a" * 64, "0"]


@pytest.mark.parametrize("tenant", _GOOD)
def test_well_formed_tenant_ids_are_accepted(tenant, monkeypatch):
    assert is_valid_tenant_id(tenant)
    assert validate_tenant_id(tenant) == tenant  # returned unchanged
    _configure(monkeypatch, tenant)
    assert Authenticator().principal_for("Bearer tok").tenant == tenant


@pytest.mark.parametrize("tenant", _BAD)
def test_malformed_tenant_ids_fail_the_whole_config_closed(tenant, monkeypatch):
    assert not is_valid_tenant_id(tenant)
    with pytest.raises(ValueError):
        validate_tenant_id(tenant)
    _configure(monkeypatch, tenant)
    auth = Authenticator()
    # Not sanitised, not skipped: the whole config is refused, so even the
    # VALID entry next to it does not authenticate.
    with pytest.raises(AuthConfigError, match="invalid 'tenant' id"):
        auth.principal_for("Bearer good")
    with pytest.raises(AuthConfigError):
        auth.enforces()


def test_invalid_tenant_error_does_not_echo_the_id(monkeypatch):
    _configure(monkeypatch, "evil/../../id")
    with pytest.raises(AuthConfigError) as exc:
        Authenticator().principal_for("Bearer good")
    assert "evil" not in str(exc.value)


def test_default_tenant_is_reserved_for_open_mode(monkeypatch):
    """'default' is the anonymous open-mode tenant; a provisioned token on it
    would inherit whatever anonymous callers stored there. The whole config is
    refused (fail closed), like any other invalid entry."""
    assert is_valid_tenant_id("default")  # still a well-formed STORAGE key
    _configure(monkeypatch, "default")
    auth = Authenticator()
    with pytest.raises(AuthConfigError, match="reserved tenant id 'default'"):
        auth.principal_for("Bearer good")
    with pytest.raises(AuthConfigError):
        auth.enforces()


def test_open_mode_keeps_the_anonymous_default_tenant(monkeypatch):
    monkeypatch.delenv("KALOS_AUTH_TOKENS", raising=False)
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    monkeypatch.delenv("KALOS_ENV", raising=False)
    p = Authenticator().principal_for(None)
    assert p.anonymous and p.tenant == "default"


@pytest.mark.parametrize("tenant", ["Default", "default2", "_default"])
def test_names_near_default_are_not_reserved(tenant, monkeypatch):
    _configure(monkeypatch, tenant)
    assert Authenticator().principal_for("Bearer tok").tenant == tenant


@pytest.mark.parametrize("tenant", ["", 5, None, ["acme"]])
def test_non_string_or_empty_tenants_are_rejected(tenant, monkeypatch):
    _configure(monkeypatch, tenant)
    with pytest.raises(AuthConfigError):
        Authenticator().principal_for("Bearer good")


pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal import app as portal  # noqa: E402


def test_invalid_tenant_config_is_a_500_over_http_never_a_login(monkeypatch):
    _configure(monkeypatch, "acme/bio")
    r = TestClient(portal.app).get("/api/latest", headers={"Authorization": "Bearer tok"})
    assert r.status_code == 500
    assert r.json()["detail"] == "authentication is misconfigured"


# --- latest cache file naming is injective ---------------------------------- #

@pytest.fixture
def latest_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(portal, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(portal, "_LATEST", {})
    return tmp_path / "latest"


def test_latest_path_is_injective_over_ids_the_old_sanitiser_merged(latest_dir):
    # Under the old scheme all of these collapsed onto `acme_bio.json`.
    names = {portal._latest_path(t).name for t in ["acme_bio", "acme-bio", "Acme_bio", "ACME_BIO"]}
    assert len(names) == 4
    # case-insensitive filesystems (macOS default) must not merge them either
    assert len({n.lower() for n in names}) == 4
    for n in names:
        assert set(n.removeprefix("t-").removesuffix(".json")) <= set("0123456789abcdef")


def test_latest_helpers_reject_ids_the_old_sanitiser_would_have_collided(latest_dir):
    """`acme/bio` was sanitised to `acme_bio` and so read/overwrote acme_bio's
    cache. It is now refused outright, and acme_bio's data is untouched."""
    portal._save_latest({"target": "x"}, "acme_bio.csv", tenant="acme_bio")
    with pytest.raises(ValueError):
        portal._save_latest({"target": "y"}, "evil.csv", tenant="acme/bio")
    with pytest.raises(ValueError):
        portal._load_latest("acme/bio")
    with pytest.raises(ValueError):
        portal._latest_path("acme/bio")
    portal._LATEST.clear()  # force a disk reload
    assert portal._load_latest("acme_bio")["dataset"] == "acme_bio.csv"


def test_similar_tenants_get_separate_cache_files_on_disk(latest_dir):
    portal._save_latest({"target": "a"}, "a.csv", tenant="acme_bio")
    portal._save_latest({"target": "b"}, "b.csv", tenant="acme-bio")
    portal._save_latest({"target": "c"}, "c.csv", tenant="Acme_bio")
    assert len(list(latest_dir.iterdir())) == 3
    portal._LATEST.clear()
    assert portal._load_latest("acme_bio")["dataset"] == "a.csv"
    assert portal._load_latest("acme-bio")["dataset"] == "b.csv"
    assert portal._load_latest("Acme_bio")["dataset"] == "c.csv"


# --- tenant is a REQUIRED keyword on the store and latest helpers ------------ #

import inspect  # noqa: E402

from kalos.portal.campaign import CampaignStore  # noqa: E402

_STORE_METHODS = ["seed", "get", "summary", "start", "set_result", "plan_fold",
                  "awaiting_recipes", "commit_fold"]


@pytest.mark.parametrize("name", _STORE_METHODS)
def test_store_methods_require_a_keyword_only_tenant(name):
    param = inspect.signature(getattr(CampaignStore, name)).parameters["tenant"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty  # no silent 'default'


@pytest.mark.parametrize("fn", ["_load_latest", "_save_latest", "_run_uploaded_sync"])
def test_latest_helpers_require_a_tenant(fn):
    param = inspect.signature(getattr(portal, fn)).parameters["tenant"]
    assert param.default is inspect.Parameter.empty


def test_omitting_tenant_is_a_type_error_not_the_default_tenant(tmp_path):
    store = CampaignStore(tmp_path)
    with pytest.raises(TypeError):
        store.get()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        portal._load_latest()  # type: ignore[call-arg]


def test_store_refuses_malformed_tenant_ids(tmp_path):
    import pandas as pd

    store = CampaignStore(tmp_path)
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        store.seed(df, "y", ["x"], tenant="acme/bio")
    with pytest.raises(ValueError):
        store.get(tenant="../default")
    store.seed(df, "y", ["x"], tenant="acme_bio")
    assert store.get(tenant="acme_bio") is not None
