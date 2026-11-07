"""The tool table (kalos/skills/tools.py), static checks only: every tool is
bound to a real engine symbol, every skill tool is registered with a declared
label and output, entrypoints are tool symbols, extras and seeds line up."""
from __future__ import annotations

import dataclasses
import subprocess
import sys

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")

from kalos.skills import load_builtin
from kalos.skills.tools import DEFAULT_TIMEOUT_S, TOOLS, artifact_targets, check_tools


@pytest.fixture
def reg():
    return load_builtin()


def _with(reg, slug, **update):
    sk = reg.get(slug, "t0")
    return dataclasses.replace(sk, meta=sk.meta.model_copy(update=update))


def test_table_resolves_and_covers_every_skill(reg):
    assert check_tools(reg.all()) == []
    used = {t for s in reg.all() for t in s.meta.tools}
    assert used == set(TOOLS)
    assert len(TOOLS) == 30


def test_labels_and_artifact_targets(reg):
    for s in reg.all():
        for name in s.meta.tools:
            spec = TOOLS[name]
            js, tb = artifact_targets(s, spec)
            assert js is not None and s.meta.output(js).label == spec.label
            if spec.csv:
                assert s.meta.output(tb).kind in ("table", "csv")
    assert TOOLS["engine.propose_batch"].label == "predicted"
    assert TOOLS["engine.ode_model"].label == "simulated"


def test_entrypoints_are_own_tool_symbols(reg):
    for s in reg.all():
        if s.meta.status == "planned":
            assert s.meta.entrypoint is None
            continue
        assert s.meta.entrypoint in {TOOLS[t].symbol for t in s.meta.tools}, s.slug
    assert reg.get("identifier-anonymisation", "t0").meta.entrypoint == \
        "kalos.data.anonymizer:Anonymizer.anonymize_meta"


def test_timeouts_and_long_running_flags():
    assert all(t.timeout_s == DEFAULT_TIMEOUT_S == 1800 for t in TOOLS.values())
    long = {n for n, t in TOOLS.items() if t.long_running}
    assert long == {"engine.surrogate_fit", "engine.propose_batch", "engine.grouped_cv_report",
                    "engine.gp_shape_report", "engine.scale_build_readout"}


def test_check_tools_catches_each_problem(reg):
    def probs(slug, **u):
        return check_tools([_with(reg, slug, **u)])

    assert any("not registered" in p for p in probs("driver-ranking", tools=["engine.nope"]))
    assert any("emits simulated" in p for p in probs("driver-ranking", tools=["engine.ode_model"]))
    assert any("not the symbol of one of its tools" in p
               for p in probs("driver-ranking", entrypoint="kalos.core.drivers:benjamini_hochberg"))
    assert any("entrypoint" in p and "not defined" in p
               for p in probs("driver-ranking", entrypoint="kalos.core.drivers:nope"))
    assert any("needs extras ['ml']" in p for p in probs("bo-next-batch", requires_extras=[]))
    assert any("not an extra in pyproject" in p
               for p in probs("driver-ranking", requires_extras=["gpu"]))
    sk = reg.get("driver-ranking", "t0")
    no_seed = [p for p in sk.meta.parameters if p.name != "seed"]
    assert any("declare an int 'seed'" in p for p in probs("driver-ranking", parameters=no_seed))
    wrong = [p if p.name != "seed" else p.model_copy(update={"default": 7})
             for p in sk.meta.parameters]
    assert any("seed default 7" in p for p in probs("driver-ranking", parameters=wrong))
    two_json = [*sk.meta.outputs,
                sk.meta.outputs[0].model_copy(update={"name": "other"})]
    assert any("no single json output" in p for p in probs("driver-ranking", outputs=two_json))


def test_stochastic_skills_declare_seed(reg):
    for s in reg.all():
        for name in s.meta.tools:
            if TOOLS[name].args_seed_default is not None:
                assert any(p.name == "seed" for p in s.meta.parameters), s.slug


def test_importing_tools_and_runner_does_not_load_torch():
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; import kalos.skills.tools, kalos.skills.runner, kalos.skills.bindings;"
         "from kalos.skills import load_builtin; from kalos.skills.tools import check_tools;"
         "assert check_tools(load_builtin().all()) == [];"
         "print('torch' in sys.modules)"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"
