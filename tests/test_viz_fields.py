"""VIZ_SPEC_V2 E3 / E5 / E12 (kalos/analysis/viz_fields.py).

Unit tests are torch-free (a fake surrogate); the integration tests run the real
`_analyze` on a small synthetic sheet (no real data) with a categorical whose
third level appears only on rows without a usable target (0 fitted runs).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalos.analysis import viz_fields as vf
from kalos.envconfig import ConfigError

# ---- E5: pairwise complete-case n ----------------------------------------


def test_pairwise_complete_n_counts_finite_pairs():
    cols = np.array([
        [1.0, 2.0, np.nan],
        [2.0, np.nan, 1.0],
        [3.0, 4.0, 5.0],
        [np.inf, 1.0, 2.0],
    ])
    n = vf.pairwise_complete_n(cols)
    assert n == [[3, 2, 2], [2, 3, 2], [2, 2, 3]]
    assert all(isinstance(v, int) for row in n for v in row)


def test_correlation_block_says_row_and_carries_n_matrix():
    from kalos.portal.viz_utils import _correlation_spearman

    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 2))
    y = X[:, 0] + rng.normal(scale=0.1, size=12)
    out = _correlation_spearman(X, y, ["a", "b"])
    assert out is not None and out["unit"] == "row"
    assert out["n_rows"] == 12
    assert out["n_matrix"] == [[12] * 3] * 3


# ---- E12: declared levels including 0-run levels -------------------------


def test_categorical_levels_lists_levels_without_runs_and_skips_blanks():
    df = pd.DataFrame({
        "medium": ["B", "A", " A ", "C", None, "na", ""],
        "y": [1.0, 2.0, 3.0, np.nan, 1.0, 2.0, 3.0],
    })
    assert vf.categorical_levels(df, ["medium", None, "absent", "medium"]) == {"medium": ["A", "B", "C"]}


# ---- E3: driver pairs and surfaces ---------------------------------------


def test_driver_pairs_order_and_k():
    drv = [{"name": n} for n in ("c", "medium", "a", "b", "d")]
    cont = ["a", "b", "c", "d"]
    assert vf.driver_pairs(drv, cont, 3) == [(2, 0), (2, 1), (0, 1)]  # (c,a) (c,b) (a,b)
    assert vf.driver_pairs(drv, cont, 6)[3:] == [(2, 3), (0, 3), (1, 3)]
    assert vf.driver_pairs(drv, cont, 0) == []
    assert vf.driver_pairs(drv[:2], cont, 3) == []  # one continuous driver: no pair


class _FakeGP:
    def __init__(self) -> None:
        self.calls = 0

    def posterior(self, X):
        self.calls += 1
        X = np.asarray(X, float)
        return X.sum(axis=1), 0.1 + np.abs(X[:, 0])


def _surface_args(k: int, first=None):
    X = np.array([[0.0, 1.0, 2.0], [1.0, 0.5, 1.0], [0.5, 0.2, 0.1]])
    y = np.array([1.0, 3.0, 2.0])
    return dict(
        X_fit=X, y_fit=y, cont_feats=["a", "b", "c"], kept_cats=[], cat_dims_map={},
        bounds=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]]),
        drv=[{"name": "b"}, {"name": "a"}, {"name": "c"}], incumbent=3.0,
        replicate_aware=False, first=first, k=k,
    )


def test_response_surfaces_share_the_grid_builder_and_reuse_the_first_pair():
    from kalos.portal.viz_utils import surface_grid

    gp = _FakeGP()
    args = _surface_args(3)
    first = surface_grid(gp, args["X_fit"], args["y_fit"], ["a", "b", "c"], [], {}, args["bounds"],
                         ("b", 1), ("a", 0), 3.0, False)
    calls_first = gp.calls
    out, why = vf.response_surfaces(gp, **{**args, "first": first})
    assert why is None
    assert [(s["x_feature"], s["y_feature"], s["pair_rank"]) for s in out] == [
        ("b", "a", 1), ("b", "c", 2), ("a", "c", 3)]
    assert {k: v for k, v in out[0].items() if k != "pair_rank"} == first
    assert gp.calls - calls_first == 2 * 21 * 21  # the first pair is not recomputed
    for s in out:
        assert np.asarray(s["z"]).shape == np.asarray(s["std"]).shape == (21, 21)
        assert (s["z_basis"], s["std_kind"], s["std_basis"]) == ("predicted", "gp_posterior_sd", "latent_mean")
        assert s["fixed_recipe"] == first["fixed_recipe"]


def test_response_surfaces_reason_when_fewer_pairs_or_off():
    out, why = vf.response_surfaces(_FakeGP(), **{**_surface_args(6)})
    assert len(out) == 3 and why == "3 of 6 pairs returned: only 3 pair(s) from the continuous drivers"
    out, why = vf.response_surfaces(_FakeGP(), **_surface_args(0))
    assert out == [] and "KALOS_VIZ_SURFACE_PAIRS=0" in why


def test_surface_pairs_setting_is_bounded(monkeypatch):
    monkeypatch.delenv(vf.SURFACE_PAIRS_ENV, raising=False)
    assert vf.surface_pairs_k() == vf.SURFACE_PAIRS_DEFAULT == 3
    monkeypatch.setenv(vf.SURFACE_PAIRS_ENV, "1")
    assert vf.surface_pairs_k() == 1
    monkeypatch.setenv(vf.SURFACE_PAIRS_ENV, "7")
    with pytest.raises(ConfigError):
        vf.surface_pairs_k()


def test_viz_fields_draws_no_random_state():
    """RNG order (reorg): nothing here may consume numpy/random state."""
    import random

    np.random.seed(5)
    random.seed(5)
    before = (np.random.get_state()[1].copy(), random.getstate())
    args = _surface_args(3)
    args.pop("first")
    args.pop("k")
    vf.viz_fields(_FakeGP(), df=pd.DataFrame({"m": ["x", "y"]}), level_columns=["m"],
                  response_surface=None, **args)
    assert (np.random.get_state()[1] == before[0]).all() and random.getstate() == before[1]


# ---- integration: real GP -------------------------------------------------

_CLONES = ["Jane Smith 0", "Bob Jones 1"]


def _sheet() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 24
    ph = rng.uniform(6.0, 7.5, n).round(3)
    methanol = rng.uniform(0.5, 3.5, n).round(3)
    temp = rng.uniform(28, 34, n).round(2)
    donor = [_CLONES[i % 2] for i in range(n)]
    titer = (3 + 1.2 * methanol - 1.5 * (ph - 6.8) ** 2 + 0.05 * temp + rng.normal(0, 0.1, n)).round(4)
    df = pd.DataFrame({"Donor_Name_1": donor, "Methanol": methanol, "pH": ph, "Temp": temp,
                       "lipase_titer": titer})
    # A third level that only appears on runs without a result: 0 fitted runs.
    extra = pd.DataFrame({"Donor_Name_1": ["Carol Day 2"] * 2, "Methanol": [1.0, 2.0], "pH": [6.5, 7.0],
                          "Temp": [30.0, 31.0], "lipase_titer": [np.nan, np.nan]})
    return pd.concat([df, extra], ignore_index=True)


def _roles():
    from kalos.domains import ColumnRoles

    return ColumnRoles.from_dict({
        "target": "lipase_titer", "features": ["Methanol", "pH", "Temp"],
        "categoricals": ["Donor_Name_1"],
    })


def analyze_viz_sheet(shared_analysis, anonymize: bool = False) -> dict:
    """Shared with tests/test_analysis_schema.py (same sheet, same xdist group)."""
    pytest.importorskip("torch")
    return shared_analysis(_sheet(), "lipase_titer", roles=_roles(), anonymize=anonymize)


@pytest.fixture
def analysis(shared_analysis) -> dict:
    return analyze_viz_sheet(shared_analysis)


@pytest.mark.xdist_group("viz_fields_sheet")
def test_real_analysis_carries_the_three_fields(analysis):
    assert analysis["categorical_levels"] == {"Donor_Name_1": ["Bob Jones 1", "Carol Day 2", "Jane Smith 0"]}
    # The model only knows the levels with runs; the 0-run level is listed, not proposed.
    assert all(p["recipe"]["Donor_Name_1"] in _CLONES for p in analysis["proposals"])

    corr = analysis["correlation"]
    assert corr["unit"] == "row" and corr["n_rows"] == analysis["n"]
    assert corr["n_matrix"] == [[analysis["n"]] * len(corr["labels"])] * len(corr["labels"])

    surfs = analysis["response_surfaces"]
    assert analysis["visualization_reasons"]["response_surfaces"] is None
    assert [s["pair_rank"] for s in surfs] == [1, 2, 3]
    assert {k: v for k, v in surfs[0].items() if k != "pair_rank"} == analysis["response_surface"]
    drv = [d["name"] for d in analysis["drivers"] if d["name"] in {"Methanol", "pH", "Temp"}]
    assert [(s["x_feature"], s["y_feature"]) for s in surfs] == [
        (drv[0], drv[1]), (drv[0], drv[2]), (drv[1], drv[2])]
    for s in surfs:
        assert s["fixed_recipe"] == analysis["response_surface"]["fixed_recipe"]
        assert all(v >= 0 for row in s["std"] for v in row)


@pytest.mark.xdist_group("viz_fields_sheet")
def test_anonymized_fields_alias_names_and_every_level(shared_analysis):
    out = analyze_viz_sheet(shared_analysis, anonymize=True)
    blob = json.dumps({"levels": out["categorical_levels"], "surfs": out["response_surfaces"]})
    for secret in ("Donor_Name_1", *_CLONES, "Carol Day 2"):
        assert secret not in blob
    (levels,) = out["categorical_levels"].values()
    assert len(levels) == 3 and all(v.startswith("level_") for v in levels)
    # Same pseudonym as the recipes use for the same level.
    used = {p["recipe"][k] for p in out["proposals"] for k in p["recipe"] if k.startswith("col_")}
    assert used <= set(levels)


# ---- docs/schemas/analysis.schema.json ------------------------------------

_SCHEMA = Path(__file__).resolve().parents[1] / "docs" / "schemas" / "analysis.schema.json"
_TYPES = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "null": type(None)}


def _check(value, schema: dict, path: str = "$") -> list[str]:
    """A stdlib subset of JSON Schema (type, const, enum, required, properties,
    additionalProperties-as-schema, items): enough for this partial schema."""
    errs: list[str] = []
    if "const" in schema and value != schema["const"]:
        errs.append(f"{path}: {value!r} != const {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in {schema['enum']!r}")
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        ok = any(isinstance(value, _TYPES[t]) and not (t in ("integer", "number") and isinstance(value, bool))
                 for t in types)
        if not ok:
            return [*errs, f"{path}: {type(value).__name__} is not {types}"]
    if isinstance(value, dict):
        errs += [f"{path}: missing {k}" for k in schema.get("required", []) if k not in value]
        for k, v in value.items():
            if k in schema.get("properties", {}):
                errs += _check(v, schema["properties"][k], f"{path}.{k}")
            elif isinstance(schema.get("additionalProperties"), dict):
                errs += _check(v, schema["additionalProperties"], f"{path}.{k}")
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            errs += _check(v, schema["items"], f"{path}[{i}]")
    return errs


def test_schema_checker_catches_violations():
    schema = json.loads(_SCHEMA.read_text())
    bad = {"correlation": {"unit": "recipe"}, "categorical_levels": {"m": [1]}, "response_surfaces": [{}]}
    errs = _check(bad, schema)
    assert any("const" in e for e in errs) and any("missing n_matrix" in e for e in errs)
    assert any("categorical_levels.m[0]" in e for e in errs) and any("missing pair_rank" in e for e in errs)


@pytest.mark.xdist_group("viz_fields_sheet")
def test_analysis_matches_the_schema(analysis):
    schema = json.loads(_SCHEMA.read_text())
    assert _check(json.loads(json.dumps(analysis)), schema) == []
