"""Deterministic routing (kalos/skills/registry.py search): for every built-in
skill, >= 2 queries route to it and 1 adjacent query routes to its neighbour
instead. The search is server-side and deterministic so this table is a test,
not a hope; embedding search may come later but must keep this suite green."""
from __future__ import annotations

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")

from kalos.skills import load_builtin

# The 11 cases first checked against the drafts (run-sheet slug renamed since).
LEGACY = {
    "how noisy are replicates": "replicate-noise-estimate",
    "will it hold at 200 L scale": "scale-up-envelope-check",
    "titer from chromatogram": "hplc-peak-summary",
    "can we use this public dataset commercially": "dataset-licence-check",
    "propose next experiments": "bo-next-batch",
    "which columns are factors": "column-role-inference",
    "first round starting design": "space-filling-design",
    "set the input ranges for the search box": "search-box-definition",
    "extrapolation far from data": "distance-to-data",
    "simulate fed-batch": "virtual-run-simulation",
    "export for tech transfer": "campaign-provenance-report",
}

POS = {
 "bo-next-batch": ["propose next experiments", "suggest the next batch of runs with bayesian optimisation"],
 "campaign-provenance-report": ["export for tech transfer", "export the campaign with provenance and audit trail"],
 "campaign-status": ["where are we in the campaign and what is the next step", "campaign status and runs in flight"],
 "column-role-inference": ["which columns are factors", "guess the role of each column in my sheet"],
 "dataset-licence-check": ["can we use this public dataset commercially", "check the licence terms of a dataset"],
 "distance-to-data": ["extrapolation far from data", "is this recipe at the edge of range, nearest run distance"],
 "driver-ranking": ["which factors drive titer the most", "rank the drivers of the response"],
 "hplc-peak-summary": ["titer from chromatogram", "integrate HPLC peaks and calibration"],
 "identifier-anonymisation": ["anonymise batch identifiers before sharing", "replace sample ids with aliases"],
 "replicate-noise-estimate": ["how noisy are replicates", "estimate the replicate noise floor"],
 "run-sheet-checks": ["check my run sheet for duplicate runs and impossible values", "data quality findings on an uploaded run sheet"],
 "scale-up-envelope-check": ["will it hold at 200 L scale", "predict titer at a larger vessel scale"],
 "search-box-definition": ["set the input ranges for the search box", "define bounds and constraints for each factor"],
 "space-filling-design": ["first round starting design", "latin hypercube design with no data yet"],
 "target-readiness-check": ["is the target column usable or constant", "does the response vary enough to model"],
 "time-series-endpoint-extraction": ["compute IVCD and mu_max from time courses", "extract endpoints from sampled trajectories"],
 "trust-report": ["can I trust the model", "cross-validation calibration and promotion gates verdict"],
 "unit-normalisation": ["convert mixed units to base units", "normalise g/L and mg/mL in one column"],
 "virtual-run-simulation": ["simulate fed-batch", "kinetic Monod simulation of a culture"],
}
NEG = {  # query near skill K that must route to V instead
 "bo-next-batch": ("first round starting design", "space-filling-design"),
 "campaign-provenance-report": ("where are we in the campaign and what is the next step", "campaign-status"),
 "campaign-status": ("export the campaign with provenance and audit trail", "campaign-provenance-report"),
 "column-role-inference": ("convert mixed units to base units", "unit-normalisation"),
 "dataset-licence-check": ("anonymise batch identifiers before sharing", "identifier-anonymisation"),
 "distance-to-data": ("can I trust the model", "trust-report"),
 "driver-ranking": ("which columns are factors", "column-role-inference"),
 "hplc-peak-summary": ("compute IVCD and mu_max from time courses", "time-series-endpoint-extraction"),
 "identifier-anonymisation": ("check the licence terms of a dataset", "dataset-licence-check"),
 "replicate-noise-estimate": ("is the target column usable or constant", "target-readiness-check"),
 "run-sheet-checks": ("how noisy are replicates", "replicate-noise-estimate"),
 "scale-up-envelope-check": ("simulate fed-batch", "virtual-run-simulation"),
 "search-box-definition": ("is this recipe at the edge of range, nearest run distance", "distance-to-data"),
 "space-filling-design": ("propose next experiments", "bo-next-batch"),
 "target-readiness-check": ("check my run sheet for duplicate runs and impossible values", "run-sheet-checks"),
 "time-series-endpoint-extraction": ("integrate HPLC peaks and calibration", "hplc-peak-summary"),
 "trust-report": ("extrapolation far from data", "distance-to-data"),
 "unit-normalisation": ("guess the role of each column in my sheet", "column-role-inference"),
 "virtual-run-simulation": ("will it hold at 200 L scale", "scale-up-envelope-check"),
}


def _top(q: str) -> str | None:
    res = load_builtin().search(q, "t0", include_beta=True, include_planned=True)
    return res[0][1].slug if res else None


def test_table_covers_every_skill():
    slugs = {s.slug for s in load_builtin().all()}
    assert set(POS) == slugs and set(NEG) == slugs
    assert all(len(v) >= 2 for v in POS.values())


@pytest.mark.parametrize("q,slug", list(LEGACY.items()))
def test_legacy_routing(q, slug):
    assert _top(q) == slug


@pytest.mark.parametrize("slug,q", [(k, q) for k, qs in POS.items() for q in qs])
def test_positive_routing(slug, q):
    assert _top(q) == slug


@pytest.mark.parametrize("slug,q,other", [(k, q, v) for k, (q, v) in NEG.items()])
def test_adjacent_negative_routing(slug, q, other):
    top = _top(q)
    assert top != slug and top == other


def test_search_is_deterministic_and_ordered():
    reg = load_builtin()
    a = [(sc, s.slug) for sc, s in reg.search("noise data runs", "t0", include_beta=True,
                                              include_planned=True)]
    b = [(sc, s.slug) for sc, s in reg.search("noise data runs", "t0", include_beta=True,
                                              include_planned=True)]
    assert a == b and len(a) > 3
    assert a == sorted(a, key=lambda x: (-x[0], x[1]))


def test_stopwords_only_query_matches_nothing():
    assert load_builtin().search("the and for with", "t0") == []
