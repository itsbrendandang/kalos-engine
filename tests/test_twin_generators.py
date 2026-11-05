"""Origin labels and the generator registry (kalos/twin, docs/VIRTUAL_RUNS.md)."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from kalos.twin import generators as g
from kalos.twin.origin import (
    Origin,
    OriginArray,
    highest_origin,
    origins_of,
    parse_origin,
)

BUILTINS = {
    "bench.gaussian_bump", "bench.ackley", "bench.mixed_bump", "bench.bioprocess_like",
    "bench.fed_batch", "examples.synthetic_scaleup",
}
KEY = {"tenant": "acme", "campaign_id": "cmp_" + "a" * 32, "job_id": "vjb_" + "1" * 32}


def test_origin_values_and_taint_order():
    assert [o.value for o in Origin] == ["measured", "simulated", "synthetic"]
    assert parse_origin(" Synthetic ") is Origin.SYNTHETIC
    with pytest.raises(ValueError):
        parse_origin("real")
    assert highest_origin([]) is Origin.MEASURED
    assert highest_origin(["measured", "simulated"]) is Origin.SIMULATED
    assert highest_origin([Origin.SIMULATED, "synthetic", "measured"]) is Origin.SYNTHETIC


def test_origin_array_keeps_marker_through_slicing_and_arithmetic():
    a = OriginArray([[1.0, 2.0], [3.0, 4.0]], "simulated")
    assert a[:, :1].kalos_origin is Origin.SIMULATED
    assert (a * 2 - 1).kalos_origin is Origin.SIMULATED
    assert origins_of(a[1]) == {Origin.SIMULATED}
    # documented limit: np.asarray returns a plain ndarray
    assert origins_of(np.asarray(a)) == set()


def test_builtin_generators_are_registered_without_importing_torch():
    code = (
        "import sys; from kalos.twin.generators import list_generators;"
        "ids = [s.id for s in list_generators()];"
        "import json; print(json.dumps(sorted(ids))); print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    lines = out.stdout.strip().splitlines()
    assert set(json.loads(lines[0])) >= BUILTINS
    assert lines[1] == "False"


@pytest.mark.parametrize("gen_id", sorted(BUILTINS))
def test_every_builtin_generates_labelled_seeded_rows(gen_id):
    b = g.generate(gen_id, None, 7, **KEY)
    assert b.origin is Origin.SYNTHETIC
    assert len(b.rows) == 7 and [r.k for r in b.rows] == list(range(7))
    df = b.frame
    assert list(df["origin"].unique()) == ["synthetic"]
    assert df.attrs["kalos_origin"] == "synthetic"
    assert b.X.kalos_origin is Origin.SYNTHETIC and b.X.shape[0] == 7
    assert len(b.params_sha256) == 64 and b.params_sha256 == g.sha256_json(b.params)
    assert b.generator_version.startswith("1+")
    assert b.pcg64_state["bit_generator"] == "PCG64"
    assert b.library_versions["numpy"] == np.__version__
    assert b.twin_version_id is None


@pytest.mark.parametrize("gen_id", sorted(BUILTINS))
def test_golden_replay_same_inputs_same_rows(gen_id):
    a = g.generate(gen_id, None, 5, **KEY)
    b = g.generate(gen_id, None, 5, **KEY)
    assert [(r.design, r.outcome) for r in a.rows] == [(r.design, r.outcome) for r in b.rows]
    assert a.seed_entropy == b.seed_entropy and a.pcg64_state == b.pcg64_state
    other = g.generate(gen_id, None, 5, **{**KEY, "job_id": "vjb_" + "2" * 32})
    assert [r.design for r in other.rows] != [r.design for r in a.rows]


def test_golden_values_fed_batch():
    """Pinned values: a change in the seeding scheme or the generator shows here."""
    b = g.generate("bench.fed_batch", None, 2, **KEY)
    entropy = g.seed_entropy(tenant="acme", campaign_id=KEY["campaign_id"], twin_version=None,
                             job_id=KEY["job_id"])
    assert b.seed_entropy == entropy and entropy[2] == 0
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    first = rng.random(2)
    assert b.rows[0].design["S0_g_per_L"] == pytest.approx(5.0 + 55.0 * first[0])
    assert b.rows[0].design["feed_L_per_h"] == pytest.approx(0.05 * first[1])


def test_seed_depends_on_each_component():
    base = g.seed_entropy(tenant="a", campaign_id="c", twin_version=None, job_id="j")
    for change in ({"tenant": "b"}, {"campaign_id": "d"}, {"twin_version": "tv1"}, {"job_id": "k"}):
        args = {"tenant": "a", "campaign_id": "c", "twin_version": None, "job_id": "j", **change}
        assert g.seed_entropy(**args) != base


def test_params_are_validated():
    with pytest.raises(g.GeneratorError, match="does not take"):
        g.generate("bench.gaussian_bump", {"evil": 1}, 3, **KEY)
    with pytest.raises(g.GeneratorError, match="wrong type"):
        g.generate("bench.gaussian_bump", {"dim": "4"}, 3, **KEY)
    with pytest.raises(g.GeneratorError, match="wrong type"):
        g.generate("bench.gaussian_bump", {"dim": True}, 3, **KEY)
    with pytest.raises(g.GeneratorError, match="noise"):
        g.generate("bench.gaussian_bump", {"noise": 5.0}, 3, **KEY)
    with pytest.raises(g.GeneratorError, match="finite"):
        g.generate("bench.gaussian_bump", {"noise": float("nan")}, 3, **KEY)
    with pytest.raises(g.GeneratorError, match="positive"):
        g.generate("bench.fed_batch", {"mu_max": -1.0}, 3, **KEY)
    with pytest.raises(g.GeneratorError, match="unknown generator"):
        g.generate("nope", None, 3, **KEY)
    for bad_n in (0, g.MAX_ROWS_PER_BATCH + 1, 2.0, True):
        with pytest.raises(g.GeneratorError, match="n must be"):
            g.generate("bench.gaussian_bump", None, bad_n, **KEY)
    # plain Monod is allowed
    b = g.generate("bench.fed_batch", {"ki": None}, 2, **KEY)
    assert b.params["ki"] is None


def test_params_hash_is_stable_and_order_free():
    a = g.generate("bench.gaussian_bump", {"dim": 3, "noise": 0.2}, 2, **KEY)
    b = g.generate("bench.gaussian_bump", {"noise": 0.2, "dim": 3}, 2, **KEY)
    assert a.params_sha256 == b.params_sha256
    c = g.generate("bench.gaussian_bump", {"dim": 3, "noise": 0.25}, 2, **KEY)
    assert c.params_sha256 != a.params_sha256


def test_synthetic_refuses_twin_version_and_simulated_needs_one():
    with pytest.raises(g.GeneratorError, match="not tied"):
        g.generate("bench.gaussian_bump", None, 2, twin_version_id="tv_1", **KEY)

    @g.register_generator("test.simulated_only", version="1", origin="simulated",
                          description="test", outcome_columns=("y",))
    def _sim(params, rng, n):
        return pd.DataFrame({"x": rng.random(n), "y": rng.random(n)})

    try:
        with pytest.raises(g.GeneratorError, match="process model"):
            g.generate("test.simulated_only", None, 2, **KEY)
        b = g.generate("test.simulated_only", None, 2, twin_version_id="tv_1", **KEY)
        assert b.origin is Origin.SIMULATED and b.frame["origin"].eq("simulated").all()
        assert b.seed_entropy[2] != 0
    finally:
        g._REGISTRY.pop("test.simulated_only", None)


def test_self_registration_and_registry_rules():
    @g.register_generator("test.self_registered", version="2", description="t",
                          defaults={"a": 1.0}, outcome_columns=("y",))
    def _gen(params, rng, n):
        return pd.DataFrame({"x": rng.random(n), "y": params["a"] * rng.random(n)})

    try:
        spec = g.get_generator("test.self_registered")
        assert spec.origin is Origin.SYNTHETIC
        assert spec.source_files and spec.source_files[0].name == "test_twin_generators.py"
        assert spec.info()["version"].startswith("2+")
        with pytest.raises(g.GeneratorError, match="already registered"):
            g.register(g.GeneratorSpec(
                id="test.self_registered", version="1", origin=Origin.SYNTHETIC,
                description="clash", fn=lambda p, r, n: pd.DataFrame(),
            ))
        with pytest.raises(g.GeneratorError, match="origin"):
            g.register(g.GeneratorSpec(
                id="test.measured", version="1", origin=Origin.MEASURED,
                description="no", fn=_gen,
            ))
        with pytest.raises(g.GeneratorError, match="id must"):
            g.register(g.GeneratorSpec(
                id="bad id!", version="1", origin=Origin.SYNTHETIC, description="no", fn=_gen,
            ))
    finally:
        g._REGISTRY.pop("test.self_registered", None)


def test_generator_contract_violations_are_refused():
    @g.register_generator("test.bad_rows", version="1", description="t", outcome_columns=("y",))
    def _bad(params, rng, n):
        return pd.DataFrame({"x": rng.random(n + 1), "y": rng.random(n + 1)})

    @g.register_generator("test.own_origin", version="1", description="t", outcome_columns=("y",))
    def _own(params, rng, n):
        return pd.DataFrame({"origin": ["measured"] * n, "y": rng.random(n)})

    try:
        with pytest.raises(g.GeneratorError, match="did not return"):
            g.generate("test.bad_rows", None, 3, **KEY)
        with pytest.raises(g.GeneratorError, match="origin column"):
            g.generate("test.own_origin", None, 3, **KEY)
    finally:
        g._REGISTRY.pop("test.bad_rows", None)
        g._REGISTRY.pop("test.own_origin", None)


def test_discover_imports_only_registering_bench_modules(tmp_path, monkeypatch):
    """A bench module that mentions `register_generator` is imported by discovery;
    one that does not is never imported."""
    import importlib.util

    spec = importlib.util.find_spec("kalos.bench")
    locations = list(spec.submodule_search_locations)
    pkg = tmp_path / "fakebench"
    pkg.mkdir()
    (pkg / "registers.py").write_text(
        "from kalos.twin.generators import register_generator\n"
        "import pandas as pd\n"
        "@register_generator('test.discovered', version='1', description='t', outcome_columns=('y',))\n"
        "def _g(params, rng, n):\n    return pd.DataFrame({'x': rng.random(n), 'y': rng.random(n)})\n"
    )
    (pkg / "silent.py").write_text("raise RuntimeError('must not be imported')\n")
    monkeypatch.setattr(spec, "submodule_search_locations", [*locations, str(pkg)])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
    imported: list[str] = []

    def fake_import(name):
        imported.append(name)
        if name.startswith("kalos.bench."):
            path = pkg / f"{name.rsplit('.', 1)[1]}.py"
            if path.is_file():
                exec(compile(path.read_text(), str(path), "exec"), {})  # noqa: S102 - test fixture
            return None
        return __import__(name)

    monkeypatch.setattr(g.importlib, "import_module", fake_import)
    monkeypatch.setattr(g, "_DISCOVERED", False)
    try:
        g.discover()
        assert "kalos.bench.registers" in imported
        assert "kalos.bench.silent" not in imported
        assert "test.discovered" in g._REGISTRY
    finally:
        g._REGISTRY.pop("test.discovered", None)
