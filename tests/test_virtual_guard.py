"""No virtual (simulated/synthetic) row reaches a model fit or an analysis unless
the caller passes `allow_origins=` explicitly (kalos/twin/origin.py,
docs/VIRTUAL_RUNS.md).

Three layers:
1. every fit entry point refuses a marked input, and accepts it with
   `allow_origins` (then labels the model with `data_origin`);
2. a seeded property test: random generators x random pandas transformations
   (subset, copy, reorder, concat with measured rows, CSV round trip, case and
   space changes to the marker) are always refused by `check_origin` and by the
   analysis;
3. every HTTP path that feeds the engine (`/api/run`, campaign upload, job submit,
   scale readout, start/result/reanalyze) refuses virtual rows.
"""
from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pandas as pd
import pytest

from kalos.portal.uploads import UploadRejected
from kalos.twin import generators as g
from kalos.twin.origin import (
    Origin,
    OriginArray,
    VirtualDataRefused,
    check_origin,
)

KEY = {"tenant": "acme", "campaign_id": "cmp_" + "c" * 32, "job_id": "vjb_" + "9" * 32}
ALL_GENERATORS = sorted(s.id for s in g.list_generators())


def _measured(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.random((n, 4))
    return pd.DataFrame({
        "x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2], "x4": X[:, 3],
        "y": 10 * np.exp(-4 * ((X - 0.6) ** 2).sum(axis=1)) + rng.normal(0, 0.1, n),
    })


def _batch(gen_id: str = "bench.gaussian_bump", n: int = 20):
    return g.generate(gen_id, None, n, **KEY)


# --- 1. fit entry points ------------------------------------------------------ #


def test_surrogate_fit_refuses_then_allows():
    from kalos.core.surrogate import Surrogate

    b = _batch()
    bounds = np.array([[0.0] * 4, [1.0] * 4])
    with pytest.raises(VirtualDataRefused, match=r"Surrogate\.fit"):
        Surrogate().fit(b.X, b.y, bounds=bounds)
    with pytest.raises(VirtualDataRefused):
        Surrogate().fit(np.asarray(b.X), b.y, bounds=bounds)  # either input marked
    with pytest.raises(VirtualDataRefused):  # allowing the wrong origin is not enough
        Surrogate().fit(b.X, b.y, bounds=bounds, allow_origins=["simulated"])
    s = Surrogate().fit(b.X, b.y, bounds=bounds, allow_origins=["synthetic"])
    assert s.data_origin is Origin.SYNTHETIC
    m = _measured()
    plain = Surrogate().fit(m[["x1", "x2", "x3", "x4"]].to_numpy(), m["y"].to_numpy(), bounds=bounds)
    assert plain.data_origin is Origin.MEASURED


def test_hybrid_multiobjective_feasibility_refuse():
    from kalos.core.feasibility import FeasibilityClassifier
    from kalos.core.mechanistic import HybridSurrogate
    from kalos.core.multiobjective import MultiObjectiveSurrogate

    b = _batch("bench.fed_batch", 12)
    bounds = np.array([[5.0, 0.0], [60.0, 0.05]])
    with pytest.raises(VirtualDataRefused, match=r"HybridSurrogate\.fit"):
        HybridSurrogate().fit(b.X, b.y, bounds)
    h = HybridSurrogate().fit(b.X, b.y, bounds, allow_origins=["synthetic"])
    assert h.data_origin is Origin.SYNTHETIC and h.residual.data_origin is Origin.MEASURED
    Y = OriginArray(np.column_stack([np.asarray(b.y), -np.asarray(b.y)]), "synthetic")
    with pytest.raises(VirtualDataRefused, match=r"MultiObjectiveSurrogate\.fit"):
        MultiObjectiveSurrogate().fit(np.asarray(b.X), Y, bounds)
    labels = OriginArray((np.asarray(b.y) > np.median(b.y)).astype(float), "synthetic")
    with pytest.raises(VirtualDataRefused, match=r"FeasibilityClassifier\.fit"):
        FeasibilityClassifier().fit(np.asarray(b.X), labels)


def test_scale_entry_points_refuse():
    from kalos.scale.candidates import MultiFidelitySurrogate, PhysicsMeanSurrogate
    from kalos.scale.readout import TargetSpec, build_readout
    from kalos.scale.transfer import ScaleUpTransferModel

    b = _batch("examples.synthetic_scaleup", 45)
    df = b.frame
    with pytest.raises(VirtualDataRefused, match=r"ScaleUpTransferModel\.fit"):
        ScaleUpTransferModel(process_columns=["ph_setpoint", "temperature_C"],
                             target_column="titer_g_per_L").fit(df)
    target = TargetSpec(scale_L=7500.0, agitation_rpm=150.0, airflow_L_per_min=300.0,
                        process_params={"ph_setpoint": 7.2, "temperature_C": 37.0})
    with pytest.raises(VirtualDataRefused, match="scale-up readout"):
        build_readout(df, "titer_g_per_L", ["ph_setpoint", "temperature_C"], target)
    X = OriginArray(np.random.default_rng(0).random((10, 2)), "synthetic")
    y = OriginArray(np.random.default_rng(1).random(10), "synthetic")
    for model in (PhysicsMeanSurrogate(["a", "b"], mean_feature_names=["a"]),
                  MultiFidelitySurrogate(["a", "b"], fidelity_feature_name="b")):
        cls = type(model)
        with pytest.raises(VirtualDataRefused, match=cls.__name__):
            model.fit(X, y, bounds=np.array([[0, 0], [1, 1]]))


def test_analyze_refuses_with_a_code_and_labels_an_allowed_run():
    from kalos.portal.analysis import _analyze

    df = _batch(n=20).frame
    with pytest.raises(UploadRejected) as info:
        _analyze(df, "y")
    assert info.value.code == "virtual_data_refused"
    assert info.value.to_body()["origins"] == ["synthetic"]
    with pytest.raises(UploadRejected):
        _analyze(df, "y", preflight_only=True)
    # pending recipes carrying an origin are refused too
    with pytest.raises(UploadRejected):
        _analyze(_measured(), "y", pending=[{"x1": 0.5, "origin": "simulated"}], preflight_only=True)
    allowed = _analyze(df.drop(columns=["y_true"]), "y", allow_origins=["synthetic"])
    assert allowed["data_origin"] == "synthetic"
    assert "data_origin" not in _analyze(_measured(), "y", preflight_only=True)


# --- 2. property test --------------------------------------------------------- #


def _virtual_positions(df: pd.DataFrame) -> np.ndarray:
    col = next(c for c in df.columns if str(c).strip().lower() == "origin")
    mask = df[col].map(lambda v: str(v).strip().lower() in ("simulated", "synthetic"))
    return np.flatnonzero(mask.to_numpy())


def _transform(df: pd.DataFrame, rng: np.random.Generator, kind: str) -> pd.DataFrame:
    """One transformation that keeps at least one virtual row."""
    if kind == "copy":
        return df.copy()
    if kind == "subset":
        idx = rng.choice(len(df), size=int(rng.integers(1, len(df) + 1)), replace=False)
        keep = int(rng.choice(_virtual_positions(df)))
        return df.iloc[np.unique(np.append(idx, keep))]
    if kind == "reorder":
        return df[list(rng.permutation(df.columns))]
    if kind == "concat_measured":
        m = _measured(30, int(rng.integers(0, 1000)))
        m = m.rename(columns={c: d for c, d in zip(m.columns, [c for c in df.columns if c != "origin"])})
        if rng.random() < 0.5:
            m.insert(0, "origin", "measured")
        virtual = df.iloc[_virtual_positions(df)]
        k = int(rng.integers(1, len(virtual) + 1))
        out = pd.concat([m, virtual.head(k)], ignore_index=True)
        return out.sample(frac=1.0, random_state=int(rng.integers(0, 1000)))
    if kind == "csv_roundtrip":
        return pd.read_csv(io.StringIO(df.to_csv(index=False)))
    if kind == "case":
        out = df.copy()
        col = next(c for c in out.columns if str(c).strip().lower() == "origin")
        out[col] = out[col].map(lambda v: f"  {str(v).upper()} ")
        return out.rename(columns={col: " Origin "})
    if kind == "drop_attrs":
        out = df.copy()
        out.attrs = {}
        return out
    if kind == "object_dtype":
        return df.astype(object)
    raise AssertionError(kind)


KINDS = ("copy", "subset", "reorder", "concat_measured", "csv_roundtrip", "case",
         "drop_attrs", "object_dtype")


def test_property_virtual_rows_never_pass_the_measured_path():
    """Seeded: 160 random (generator, chain of up to 3 transformations) cases. Every
    result still carries at least one virtual row, and every one is refused."""
    from kalos.portal.analysis import _analyze

    rng = np.random.default_rng(20261105)
    batches = {gid: g.generate(gid, None, 25, **KEY).frame for gid in ALL_GENERATORS}
    for case in range(160):
        gid = ALL_GENERATORS[case % len(ALL_GENERATORS)]
        df = batches[gid]
        chain = [KINDS[int(i)] for i in rng.integers(0, len(KINDS), size=int(rng.integers(1, 4)))]
        for kind in chain:
            df = _transform(df, rng, kind)
        with pytest.raises(VirtualDataRefused):
            check_origin(df, where="property")
        target = [c for c in df.columns if str(c).strip().lower() != "origin"][-1]
        with pytest.raises(UploadRejected) as info:
            _analyze(df, target, preflight_only=True)
        assert info.value.code == "virtual_data_refused", (gid, chain)


def test_measured_frames_with_a_measured_origin_column_pass():
    df = _measured()
    df.insert(0, "origin", "measured")
    assert check_origin(df, where="t") is Origin.MEASURED
    df2 = _measured()
    df2["Origin"] = ["E. coli BL21"] * len(df2)  # an unrelated meaning of "origin"
    assert check_origin(df2, where="t") is Origin.MEASURED


# --- 3. every HTTP path -------------------------------------------------------- #


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [{"token_sha256": _sha("w"), "subject": "bob@acme", "tenant": "acme",
           "scopes": ["read", "write"]}]
H = {"Authorization": "Bearer w"}


@pytest.fixture
def api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from kalos.portal import app as portal_module
    from kalos.portal import busy
    from kalos.portal import campaign as campaign_module
    from kalos.portal.campaign import CampaignStore, get_campaign_store

    busy.reset_limiter(total=2, per_tenant=1)
    store = CampaignStore(tmp_path)
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(TOKENS))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    portal_module.app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(portal_module.app), store
    finally:
        portal_module.app.dependency_overrides.pop(get_campaign_store, None)
        busy.reset_limiter()
        store.close()


def _virtual_csv() -> bytes:
    return _batch(n=20).frame.drop(columns=["y_true"]).to_csv(index=False).encode()


def test_http_upload_paths_refuse_virtual_rows(api):
    tc, store = api
    files = {"file": ("v.csv", _virtual_csv(), "text/csv")}
    r = tc.post("/api/run", files=files, data={"target": "y"}, headers=H)
    assert r.status_code == 400 and r.json()["code"] == "virtual_data_refused"
    cid = tc.post("/api/campaigns", json={"name": "c"}, headers=H).json()["id"]
    r = tc.post(f"/api/campaigns/{cid}/upload", files=files, data={"target": "y"}, headers=H)
    assert r.status_code == 400 and r.json()["code"] == "virtual_data_refused"
    r = tc.post(f"/api/campaigns/{cid}/jobs", files=files, data={"target": "y"}, headers=H)
    assert r.status_code == 400 and r.json()["code"] == "virtual_data_refused"
    assert store.get(tenant="acme", campaign_id=cid) is None  # nothing was seeded


def test_http_scale_readout_refuses_virtual_rows(api):
    tc, _ = api
    csv = _batch("examples.synthetic_scaleup", 45).frame.to_csv(index=False).encode()
    target = json.dumps({
        "scale_L": 7500.0, "agitation_rpm": 150.0, "airflow_L_per_min": 300.0,
        "target_column": "titer_g_per_L",
        "process_params": {"ph_setpoint": 7.2, "temperature_C": 37.0},
    })
    r = tc.post("/api/scale/readout", data={"target": target},
                files={"file": ("s.csv", csv, "text/csv")}, headers=H)
    assert r.status_code == 422 and r.json()["code"] == "virtual_data_refused"


def test_http_recipes_cannot_carry_an_origin(api):
    """start -> result -> reanalyze: the recipe keys are limited to the campaign's
    own columns, so an `origin` key is refused at start and never reaches a fold."""
    tc, store = api
    store.seed(_measured(), "y", ["x1", "x2", "x3", "x4"], tenant="acme")
    cid = store.default_campaign_id(tenant="acme")
    r = tc.post(f"/api/campaigns/{cid}/runs", headers=H, json={"recipes": [
        {"recipe": {"x1": 0.5, "x2": 0.5, "x3": 0.5, "x4": 0.5, "origin": "synthetic"}}]})
    assert r.status_code == 400
    assert store.get(tenant="acme", campaign_id=cid)["pending"] == []
