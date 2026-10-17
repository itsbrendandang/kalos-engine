"""CampaignStore.start type-checks recipe values against the campaign's own
dataset (review fix for #87): numeric column => finite number (numeric strings
refused), categorical column => one of its observed levels."""
from __future__ import annotations

import pandas as pd
import pytest

from kalos.portal.campaign import CampaignError, CampaignStore
from kalos.portal.recipe_values import CATEGORICAL, EMPTY, NUMERIC, check_recipe_values, column_kinds


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "pH": [6.0, 6.5, 7.0, 6.2, 6.8, 7.1],
        "rpm": [100, 200, 300, 150, 250, 220],
        "strain": ["A", "B", "A", "C", "B", "A"],
        "notes": [None] * 6,
        "titer": [1.0, 2.0, 3.0, 1.5, 2.5, 2.2],
    })


@pytest.fixture
def store(tmp_path):
    s = CampaignStore(tmp_path)
    s.seed(_df(), "titer", ["pH", "rpm", "strain"], tenant="acme")
    return s


def _start(store, recipe):
    return store.start([{"recipe": recipe}], tenant="acme")


def test_valid_numeric_and_categorical_values_start(store):
    (run,) = _start(store, {"pH": 6.4, "rpm": 180, "strain": "B"})
    assert run["recipe"] == {"pH": 6.4, "rpm": 180, "strain": "B"}


@pytest.mark.parametrize("value", ["7.3", "6.4", "nan", "", "high", True, None])
def test_numeric_columns_refuse_strings_bools_and_null(store, value):
    with pytest.raises(CampaignError, match="numeric columns are not finite JSON numbers"):
        _start(store, {"pH": value})


def test_numeric_column_accepts_the_exact_label_of_an_observed_value_as_that_number(store):
    """A numeric column declared categorical through roles gets its levels
    proposed as labels (`str(value)`); stored back as the number."""
    (run,) = _start(store, {"rpm": "200"})
    assert run["recipe"] == {"rpm": 200} and isinstance(run["recipe"]["rpm"], int)
    with pytest.raises(CampaignError):
        _start(store, {"rpm": "201"})  # not an observed value
    with pytest.raises(CampaignError):
        _start(store, {"rpm": " 200"})  # not the exact label


@pytest.mark.parametrize("value", ["D", "a", 1, 1.0, False])
def test_categorical_columns_refuse_unknown_levels_and_non_strings(store, value):
    with pytest.raises(CampaignError, match="not one of that column's levels"):
        _start(store, {"strain": value})


def test_categorical_accepts_null_and_strips_a_known_level(store):
    (run,) = _start(store, {"strain": None})
    assert run["recipe"] == {"strain": None}
    (run,) = _start(store, {"strain": " C "})
    assert run["recipe"] == {"strain": "C"}


def test_an_all_blank_column_accepts_only_null(store):
    _start(store, {"notes": None})
    with pytest.raises(CampaignError, match="no values"):
        _start(store, {"notes": "anything"})


def test_error_does_not_echo_the_value(store):
    with pytest.raises(CampaignError) as exc:
        _start(store, {"strain": "SECRET-LEVEL"})
    assert "SECRET" not in str(exc.value) and "strain" not in str(exc.value)


def test_a_rejected_batch_stores_nothing(store):
    with pytest.raises(CampaignError):
        store.start([{"recipe": {"pH": 6.4}}, {"recipe": {"pH": "6.4"}}], tenant="acme")
    assert store.get(tenant="acme")["pending"] == []


def test_column_kinds():
    rows = [{"a": 1, "b": "x", "c": None, "d": 2.5}, {"a": 2.0, "b": 3, "c": None, "d": float("nan")}]
    kinds = column_kinds(rows, ["a", "b", "c", "d"])
    assert kinds["a"].kind == NUMERIC
    assert kinds["b"].kind == CATEGORICAL and set(kinds["b"].levels) == {"x", "3"}
    assert kinds["c"].kind == EMPTY
    assert kinds["d"].kind == NUMERIC
    assert check_recipe_values({"b": "3"}, kinds) == ({"b": "3"}, None)


pytest.importorskip("fastapi")


def test_route_answers_400_with_the_store_message(tmp_path):
    from fastapi.testclient import TestClient

    from kalos.portal.app import app, get_campaign_store

    s = CampaignStore(tmp_path)
    s.seed(_df(), "titer", ["pH", "rpm", "strain"], tenant="default")
    app.dependency_overrides[get_campaign_store] = lambda: s
    try:
        r = TestClient(app).post("/api/campaign/start", json={"recipes": [{"recipe": {"pH": "7.3"}}]})
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
    assert r.status_code == 400
    assert "numeric strings are refused" in r.json()["error"]


# --- re-review: the store types columns the way the analysis does ------------- #

from kalos.portal.recipe_values import MAX_PENDING_RUNS, NUMERIC as _NUM  # noqa: E402


def test_a_mostly_numeric_column_with_one_text_cell_is_numeric():
    rows = [{"pH": v} for v in [6.0, 6.5, "ND", 7.0, 6.2, 6.8]]
    kinds = column_kinds(rows, ["pH"])
    assert kinds["pH"].kind == _NUM
    assert check_recipe_values({"pH": 6.41}, kinds) == ({"pH": 6.41}, None)


def test_a_single_unit_text_column_is_numeric_in_base_units():
    rows = [{"Temp": f"{t} C"} for t in [30.1, 31.0, 32.5, 33.0, 29.8, 34.6]]
    kinds = column_kinds(rows, ["Temp"])
    assert kinds["Temp"].kind == _NUM
    assert kinds["Temp"].low == pytest.approx(29.8) and kinds["Temp"].high == pytest.approx(34.6)
    assert check_recipe_values({"Temp": 31.7}, kinds)[1] is None


@pytest.mark.parametrize("value", [1e300, -1e300, 6.5e7])
def test_numeric_values_far_outside_the_observed_range_are_refused(store, value):
    with pytest.raises(CampaignError, match="observed span"):
        _start(store, {"pH": value})


def test_values_within_the_magnitude_bound_are_accepted(store):
    (run,) = _start(store, {"pH": 50.0})  # outside the box, inside the sanity bound
    assert run["recipe"] == {"pH": 50.0}


def test_runs_in_flight_are_capped_per_campaign(store):
    batch = [{"recipe": {"pH": 6.5}}] * 64
    for _ in range(MAX_PENDING_RUNS // 64):
        store.start(batch, tenant="acme")
    with pytest.raises(CampaignError, match=f"at most {MAX_PENDING_RUNS}"):
        store.start([{"recipe": {"pH": 6.5}}], tenant="acme")
    assert len(store.get(tenant="acme")["pending"]) == MAX_PENDING_RUNS


def _upload_then_start(tmp_path, df) -> tuple[int, dict]:
    from fastapi.testclient import TestClient

    from kalos.portal import app as portal
    from kalos.portal import campaign as campaign_module
    from kalos.portal.app import app, get_campaign_store

    s = CampaignStore(tmp_path)
    mp = pytest.MonkeyPatch()
    mp.setattr(portal, "_STATE_DIR", tmp_path)
    mp.setattr(portal, "_LATEST_DIR", tmp_path / "latest")
    mp.setattr(portal, "_LATEST", {})
    mp.setattr(campaign_module, "_STORE", s)
    app.dependency_overrides[get_campaign_store] = lambda: s
    try:
        c = TestClient(app)
        up = c.post("/api/run", files={"file": ("runs.csv", df.to_csv(index=False).encode(), "text/csv")})
        assert up.status_code == 200, up.text
        proposals = up.json()["proposals"]
        assert proposals
        r = c.post("/api/campaign/start", json={"recipes": proposals})
        return r.status_code, r.json()
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
        mp.undo()


def test_engine_proposals_start_end_to_end_with_text_cells_in_numeric_columns(tmp_path):
    """Regression: a pH column with one 'ND' cell and a temperature column in
    '30.1 C' text are numeric to the analysis, so the engine proposes numbers
    for them; the store must accept its own proposals."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 16
    methanol = rng.uniform(0, 4, n).round(3)
    ph = rng.uniform(5, 7, n).round(2)
    temp = rng.uniform(28, 34, n).round(1)
    df = pd.DataFrame({
        "Methanol": methanol,
        "pH": [str(v) for v in ph],
        "Temp": [f"{t} C" for t in temp],
        "lipase_titer": (1.5 * methanol + 0.2 * temp + rng.normal(0, 0.1, n)).round(3),
    })
    df.loc[3, "pH"] = "ND"
    status, body = _upload_then_start(tmp_path, df)
    assert status == 200, body
    assert len(body["started"]) >= 1
