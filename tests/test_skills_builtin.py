"""The Kalos-curated skill set in kalos/skills/builtin/: loads cleanly, every
engine ref resolves at the cited line (static AST), the implementation split is
11 implemented / 3 partial / 5 planned, and importing kalos.skills never loads
torch."""
from __future__ import annotations

import re
import subprocess
import sys

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")

from kalos.skills import load_builtin
from kalos.skills.loader import check_engine_refs, load_dir
from kalos.skills.registry import BUILTIN_DIR

IMPLEMENTED = {
    "bo-next-batch", "campaign-status", "column-role-inference", "driver-ranking",
    "identifier-anonymisation", "replicate-noise-estimate", "run-sheet-checks",
    "scale-up-envelope-check", "search-box-definition", "target-readiness-check",
    "unit-normalisation",
}
PARTIAL = {"campaign-provenance-report", "trust-report", "virtual-run-simulation"}
PLANNED = {"dataset-licence-check", "distance-to-data", "hplc-peak-summary",
           "space-filling-design", "time-series-endpoint-extraction"}


@pytest.fixture(scope="module")
def skills():
    res = load_dir(BUILTIN_DIR)
    assert not res.errors, res.errors
    return {s.slug: s for s in res.skills}


def test_all_load_and_split(skills):
    assert set(skills) == IMPLEMENTED | PARTIAL | PLANNED
    by = {}
    for s in skills.values():
        by.setdefault(s.meta.implementation.status, set()).add(s.slug)
    assert by == {"implemented": IMPLEMENTED, "partial": PARTIAL, "needs-new-code": PLANNED}


def test_planned_skills_bind_no_tools(skills):
    for slug in PLANNED:
        m = skills[slug].meta
        assert m.status == "planned" and m.tools == [] and m.implementation.gaps
    for slug in IMPLEMENTED | PARTIAL:
        assert skills[slug].meta.tools, slug
        assert skills[slug].meta.status in ("active", "beta")


def test_partial_skills_list_gaps(skills):
    for slug in PARTIAL:
        assert skills[slug].meta.implementation.gaps, slug
    vr = skills["virtual-run-simulation"].meta
    assert vr.tools == ["engine.ode_model"]
    assert any("not bound" in g for g in vr.implementation.gaps)


def test_every_engine_ref_resolves_at_exact_line(skills):
    bad = {slug: check_engine_refs(s) for slug, s in skills.items()}
    assert not {k: v for k, v in bad.items() if v}
    assert sum(len(s.meta.implementation.engine) for s in skills.values()) >= 60


def test_run_sheet_slug_and_wording(skills):
    assert "run-sheet-checks" in skills and "run-sheet-validation" not in skills
    for s in skills.values():
        text = "\n".join((s.path / f).read_text(encoding="utf-8") for f in s.files)
        assert "run-sheet-validation" not in text, s.slug
        assert not re.search(r"\bvalidated\b", text, re.I), s.slug
        assert not re.search(r"complian|21 CFR|GxP-ready", text, re.I), s.slug


def test_curated_ownership_and_licence(skills):
    for s in skills.values():
        assert s.meta.owner == "system" and s.meta.visibility == "system"
        assert s.meta.licence.spdx == "Apache-2.0"
        assert s.source == "kalos"


def test_simulation_outputs_never_measured(skills):
    for s in skills.values():
        if s.meta.category == "simulation":
            assert s.meta.output_labels <= {"simulated", "synthetic"}


def test_registry_cached_builtin_has_no_errors():
    reg = load_builtin()
    assert not reg.errors
    assert len(reg.all()) == 19


def _fresh(code: str) -> str:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          check=True).stdout.strip()


def test_import_kalos_skills_does_not_load_torch():
    out = _fresh("import sys, kalos.skills; print('torch' in sys.modules)")
    assert out == "False", "importing kalos.skills must never load torch"


def test_loading_and_checking_builtins_does_not_load_torch():
    out = _fresh(
        "import sys\n"
        "from kalos.skills import load_builtin\n"
        "from kalos.skills.loader import check_engine_refs\n"
        "r = load_builtin()\n"
        "assert all(not check_engine_refs(s) for s in r.all())\n"
        "r.search('propose next experiments', 't0')\n"
        "print('torch' in sys.modules)"
    )
    assert out == "False"
