"""The assistant's local index (kalos/assist/index.py) on a temp state dir."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kalos.assist.index import AssistIndex, campaign_chunks, tokens
from kalos.assist.paths import tenant_key
from kalos.assist.state import clear_refusal, latest_refusal, record_refusal
from kalos.portal.campaign import CampaignStore

ANALYSIS = {
    "target": "lipase_titer",
    "features": ["Methanol", "pH"],
    "n": 8,
    "timestamp": 100,
    "provenance": {"input_sha256": "ab" * 32},
    "validation": {"findings": [
        {"check": "outliers", "severity": "warning", "message": "Row 3 is far from the rest.", "column": "pH"},
        {"check": "outliers", "severity": "warning", "message": "Row 5 is far from the rest.", "column": "pH"},
    ]},
    "promotion": {"passed": False, "failures": ["surrogate_spearman=0.100 < min 0.3"], "not_judged": []},
    "cv_spearman": 0.1,
    "drivers": [{"name": "Methanol", "rho": 0.62, "ci95": [0.2, 0.9]}],
    "proposals": [{"recipe": {"Methanol": 2.5, "pH": 6.1}, "pred": 3.3, "std": 0.4}],
}


def _df(n: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"Methanol": rng.uniform(0, 4, n).round(3), "pH": rng.uniform(5, 7, n).round(2),
                         "lipase_titer": rng.uniform(1, 3, n).round(3)})


@pytest.fixture
def store(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "CAMPAIGN_LOOP.md").write_text("# Loop\n\nRecord a result for every started run.\n\n"
                                           "## Reanalyze\n\nFolding measured runs refits the model.\n")
    monkeypatch.setenv("KALOS_ASSIST_DOCS_DIR", str(docs))
    skills = tmp_path / "skills" / "replicate-noise"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("---\nschema: kalos-skill/1\nname: Replicate noise\n"
                                     "summary: Estimate run-to-run spread from repeats. Not for single runs.\n---\n"
                                     "# What it does\n\nReads repeated recipes and reports their spread.\n")
    monkeypatch.setenv("KALOS_ASSIST_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("KALOS_ASSIST_EMBED", "0")
    return CampaignStore(tmp_path / "state")


def _seed(store: CampaignStore, tenant: str = "acme", name: str = "Lipase") -> tuple[str, str]:
    cid = store.create_campaign(tenant=tenant, name=name, created_by="t")["id"]
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant=tenant, campaign_id=cid,
               analysis=ANALYSIS, dataset="runs.csv")
    run = store.start([{"recipe": {"Methanol": 2.5, "pH": 6.1}, "pred": 3.3, "std": 0.4}],
                      tenant=tenant, campaign_id=cid)[0]
    return cid, run["id"]


def test_round_trip_on_a_temp_state_dir(store):
    cid, _run_id = _seed(store)
    index = AssistIndex(store, "acme")
    status = index.refresh()
    assert status["chunks"] > 0 and status["retriever"] == "bm25" and status["built_at"]
    assert index.path == store.state_dir / "assist" / tenant_key("acme") / "index.json"
    assert index.path.exists()

    kinds = {c.kind for c in index.chunks()}
    assert kinds >= {"sheet", "finding", "round", "analysis", "doc", "skill"}

    # a fresh instance reads the same file
    again = AssistIndex(store, "acme")
    assert {c.chunk_id for c in again.chunks()} == {c.chunk_id for c in index.chunks()}

    hits = again.search("which inputs went with lipase titer", campaign_id=cid)
    assert hits and hits[0].chunk.campaign_id in (cid, None)
    assert any(h.chunk.source_id == "drivers" for h in hits)

    sheet = next(c for c in index.chunks() if c.kind == "sheet")
    assert sheet.source_id == "ab" * 32
    findings = sorted(c.source_id for c in index.chunks() if c.kind == "finding")
    assert findings == ["outliers", "outliers.2"]
    doc = [c for c in index.chunks() if c.kind == "doc"]
    assert {c.source_id for c in doc} == {"docs/CAMPAIGN_LOOP.md"}
    assert {c.source_id for c in index.chunks() if c.kind == "skill"} == {"replicate-noise"}
    proposals = next(c for c in index.chunks() if c.source_id == "proposals")
    assert "Predicted lipase_titer 3.3" in proposals.text and "not measured" in proposals.text


def test_incremental_refresh_touches_only_the_changed_campaign(store):
    cid, run_id = _seed(store)
    _seed(store, name="Other")
    index = AssistIndex(store, "acme")
    index.refresh()
    first = index.path.read_bytes()
    assert index.refresh()["chunks"] > 0
    assert index.path.read_bytes() == first  # nothing changed, nothing written

    store.set_result(run_id, 2.75, tenant="acme", campaign_id=cid)
    index.refresh(campaign_ids=[cid])
    data = json.loads(index.path.read_text())
    assert data["changed_groups"] == 1
    rounds = [c for c in index.chunks() if c.kind == "round" and c.campaign_id == cid]
    assert any("Measured lipase_titer 2.75" in c.text for c in rounds)


def test_nothing_relevant_gives_no_hits(store):
    _seed(store)
    index = AssistIndex(store, "acme")
    index.refresh()
    assert index.search("zebra quantum saxophone") == []
    assert index.search("the and of") == []  # stop words only


def test_campaign_filter_keeps_docs_and_drops_other_campaigns(store):
    a, _ = _seed(store, name="Alpha")
    b, _ = _seed(store, name="Beta")
    index = AssistIndex(store, "acme")
    index.refresh()
    hits = index.search("proposed recipes predicted", campaign_id=a, k=20)
    assert hits and all(h.chunk.campaign_id in (a, None) for h in hits)
    assert b not in {h.chunk.campaign_id for h in hits}


def test_tenant_isolation(store):
    _seed(store, tenant="acme", name="AcmeSecretZeus")
    _seed(store, tenant="globex", name="GlobexSecretHera")
    a, g = AssistIndex(store, "acme"), AssistIndex(store, "globex")
    a.refresh()
    g.refresh()
    assert a.dir != g.dir
    assert "GlobexSecretHera" not in a.path.read_text()
    assert "AcmeSecretZeus" not in g.path.read_text()
    assert all("Hera" not in h.chunk.text for h in a.search("GlobexSecretHera campaign", k=50))


def test_fake_embedder_is_used_and_vectors_persist(store):
    _seed(store)

    class Fake:
        name = "fake-local"

        def encode(self, texts):
            out = []
            for t in texts:
                v = np.zeros(8, dtype=np.float32)
                for tok in tokens(t):
                    v[hash(tok) % 8] += 1.0
                n = np.linalg.norm(v) or 1.0
                out.append(v / n)
            return np.stack(out)

    index = AssistIndex(store, "acme", embedder=Fake())
    status = index.refresh()
    assert status["retriever"] == "embeddings"
    assert index.vec_path.exists()
    assert index.search("lipase titer methanol", min_similarity=0.0)


def test_refusal_note_round_trip(store):
    cid, _ = _seed(store)
    assert latest_refusal(store, "acme", cid) is None
    record_refusal(store, "acme", cid, {"error": "x", "code": "too_many_fit_rows",
                                        "n_rows": 2400, "max_fit_rows": 2000, "sheet": "secret"})
    found = latest_refusal(store, "acme", cid)
    assert found is not None
    assert found["code"] == "too_many_fit_rows"
    assert found["context"] == {"n_rows": 2400, "max_fit_rows": 2000}  # numbers only, no names
    assert latest_refusal(store, "globex", cid) is None
    clear_refusal(store, "acme", cid)
    assert latest_refusal(store, "acme", cid) is None


def test_chunks_never_call_an_input_a_cause():
    chunks = campaign_chunks({"id": "cmp_" + "1" * 32, "name": "x"}, None, ANALYSIS)
    drivers = next(c for c in chunks if c.source_id == "drivers")
    assert "went with, not a cause" in drivers.text


def test_builtin_bm25_matches_without_rank_bm25(store, monkeypatch):
    import sys

    _seed(store)
    index = AssistIndex(store, "acme")
    index.refresh()
    with_lib = [h.chunk.chunk_id for h in index.search("lipase titer drivers methanol")]
    monkeypatch.setitem(sys.modules, "rank_bm25", None)  # import now raises ImportError
    without = [h.chunk.chunk_id for h in index.search("lipase titer drivers methanol")]
    assert without and set(without) & set(with_lib)  # same question, overlapping answers


def test_campaign_state_chunks_for_the_simple_view(store):
    cid, run_id = _seed(store)
    store.set_result(run_id, 9.5, tenant="acme", campaign_id=cid)
    chunks = {c.source_id: c for c in campaign_chunks(
        store.campaign_info(tenant="acme", campaign_id=cid), store.get(tenant="acme", campaign_id=cid),
        {**ANALYSIS, "proposals": [{"recipe": {"Methanol": 2.5}, "pred": 3.3, "std": 0.4, "mode": "explore"}]},
    ) if c.kind == "analysis"}
    assert {"verdict", "drivers", "proposals", "best_run", "round_delta", "promotion"} <= set(chunks)
    verdict = chunks["verdict"].text
    assert "Not yet reliable" in verdict and "surrogate_spearman" in verdict and "Why this matters:" in verdict
    assert "went with, not a cause" in chunks["drivers"].text
    assert "Higher Methanol went with higher lipase_titer" in chunks["drivers"].text
    assert "(New ground)" in chunks["proposals"].text and "Predicted lipase_titer 3.3" in chunks["proposals"].text
    best = chunks["best_run"].text
    assert "Measured lipase_titer 9.5" in best and run_id in best and "Methanol = 2.5" in best
    assert f"Run {run_id}: Predicted lipase_titer 3.3, Measured 9.5." in chunks["round_delta"].text


def test_every_number_in_a_state_chunk_sits_next_to_a_label(store):
    cid, run_id = _seed(store)
    store.set_result(run_id, 9.5, tenant="acme", campaign_id=cid)
    chunks = campaign_chunks(store.campaign_info(tenant="acme", campaign_id=cid),
                             store.get(tenant="acme", campaign_id=cid), ANALYSIS)
    for c in chunks:
        if c.source_id in ("best_run", "proposals"):
            assert "Measured" in c.text or "Predicted" in c.text, c.text


def test_no_recipes_gives_a_refusal_chunk():
    chunks = campaign_chunks({"id": "cmp_" + "2" * 32, "name": "x"}, None,
                             {**ANALYSIS, "proposals": [], "proposal_notice": None})
    refusal = next(c for c in chunks if c.source_id == "refusal")
    assert "empty_batch" in refusal.text and "Why this matters:" in refusal.text


def test_real_analysis_shape_provenance_list_and_run_record(store):
    """Real analyses carry `provenance` as the per-column list and the sheet hash
    in `run.input_sha256` (kalos/portal/pipeline.py)."""
    cid = store.create_campaign(tenant="acme", name="Real", created_by="t")["id"]
    real = {**ANALYSIS, "provenance": [{"name": "Methanol", "status": "kept_feature"},
                                       {"name": "Batch", "status": "dropped_id"}],
            "run": {"input_sha256": "ef" * 32}, "proposal_provenance": {"input_sha256": "ef" * 32}}
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=cid, analysis=real)
    index = AssistIndex(store, "acme")
    index.refresh()
    sheet = next(c for c in index.chunks() if c.kind == "sheet")
    assert sheet.source_id == "ef" * 32
    assert "Column 'Batch': dropped_id." in sheet.text


def test_one_bad_campaign_does_not_break_the_index(store, monkeypatch):
    import kalos.assist.index as index_module

    _seed(store, name="Good")
    real = index_module.campaign_chunks

    def flaky(info, *a, **k):
        if info["name"] == "Bad":
            raise AttributeError("odd record")
        return real(info, *a, **k)

    _seed(store, name="Bad")
    monkeypatch.setattr(index_module, "campaign_chunks", flaky)
    index = AssistIndex(store, "acme")
    assert index.refresh()["chunks"] > 0
    assert any("Good" in c.text for c in index.chunks())


def test_a_refresh_with_nothing_changed_writes_nothing(store):
    """sec-assist nit: read-scope routes (ask, status) refresh the index; with no
    change they must not rewrite index.json."""
    cid, run_id = _seed(store)
    index = AssistIndex(store, "acme")
    index.refresh()
    before = (index.path.read_bytes(), index.path.stat().st_mtime_ns)
    index.refresh()
    index.refresh(campaign_ids=[cid])
    assert (index.path.read_bytes(), index.path.stat().st_mtime_ns) == before
    store.set_result(run_id, 2.0, tenant="acme", campaign_id=cid)
    index.refresh(campaign_ids=[cid])
    assert index.path.read_bytes() != before[0]
