"""Retrieval-quality benchmark for the assistant's local index.

This is a MEASUREMENT test, not a pass/fail feature test. It builds a realistic
multi-campaign index on a temp state dir, runs a labelled set of bioprocess
questions whose relevant chunk kind/ids are known, and reports retrieval metrics
(recall@k, precision@1, MRR) for the BM25 retriever. The embedding retriever is
measured too when a local sentence-transformers model is already on disk;
otherwise that arm is skipped (never downloaded), per the index's offline rule.

Run just this file with `-s` to see the numbers:

    pytest tests/test_assist_retrieval_bench.py -s

The thresholds here are deliberately loose floors (a regression guard), not
targets. The printed table is the artefact: it tells us whether a scoring or
embedding change actually improved retrieval before we ship it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.assist.index import AssistIndex
from kalos.portal.campaign import CampaignStore

# --- fixtures: a realistic two-campaign tenant --------------------------------- #

def _df(seed: int, n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "glucose": rng.uniform(0, 40, n).round(2),
        "temperature": rng.uniform(28, 38, n).round(1),
        "pH": rng.uniform(6.5, 7.4, n).round(2),
        "titer": rng.uniform(1, 5, n).round(3),
    })


def _analysis(target: str, feats: list[str], *, sha: str) -> dict:
    return {
        "target": target,
        "features": feats,
        "n": 12,
        "timestamp": 100,
        "provenance": {"input_sha256": sha},
        "validation": {"findings": [
            {"check": "outliers", "severity": "warning",
             "message": "Row 4 is far from the rest.", "column": "pH"},
            {"check": "missing", "severity": "warning",
             "message": "glucose has 2 missing values.", "column": "glucose"},
        ]},
        "promotion": {"passed": False,
                      "failures": ["surrogate_spearman=0.12 < min 0.3"], "not_judged": []},
        "cv_spearman": 0.12,
        "drivers": [{"name": "glucose", "rho": 0.71, "ci95": [0.3, 0.9]},
                    {"name": "temperature", "rho": -0.4, "ci95": [-0.7, -0.1]}],
        "proposals": [{"recipe": {"glucose": 22.0, "temperature": 32.0, "pH": 7.0},
                       "pred": 4.2, "std": 0.5}],
    }


@pytest.fixture
def seeded_store(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "CAMPAIGN_LOOP.md").write_text(
        "# Loop\n\nRecord a result for every started run.\n\n"
        "## Reanalyze\n\nFolding measured runs refits the model and proposes a new batch.\n")
    (docs / "SCALE_READOUT.md").write_text(
        "# Scale readout\n\nThe scale-up readout predicts performance at a larger scale "
        "and refuses when its backtest does not support the prediction.\n")
    monkeypatch.setenv("KALOS_ASSIST_DOCS_DIR", str(docs))
    monkeypatch.setenv("KALOS_ASSIST_EMBED", "0")  # BM25 arm; embed arm handled separately
    monkeypatch.delenv("KALOS_ASSIST_SKILLS_DIR", raising=False)

    store = CampaignStore(tmp_path / "state")
    ids: dict[str, str] = {}
    for key, name, target, feats, seed in [
        ("ferment", "CHO fermentation", "titer", ["glucose", "temperature", "pH"], 1),
        ("media", "Media optimization", "titer", ["glucose", "pH"], 2),
    ]:
        cid = store.create_campaign(tenant="acme", name=name, created_by="t")["id"]
        store.seed(_df(seed), target, feats, tenant="acme", campaign_id=cid,
                   analysis=_analysis(target, feats, sha=f"{seed:02x}" * 32), dataset="runs.csv")
        store.start([{"recipe": {"glucose": 22.0, "temperature": 32.0, "pH": 7.0},
                      "pred": 4.2, "std": 0.5}], tenant="acme", campaign_id=cid)
        ids[key] = cid
    return store, ids


# --- the labelled query set ---------------------------------------------------- #
# Each item: a natural question, and the chunk KIND we expect in the top-k.
# Grounded in what the index actually stores (sheet/finding/round/analysis/doc).

_QUERIES: list[tuple[str, str]] = [
    ("which inputs went with titer", "sheet"),
    ("what columns were used for the target", "sheet"),
    ("what warnings did the sheet checks find", "finding"),
    ("were there any outliers in pH", "finding"),
    ("which columns had missing values", "finding"),
    ("what are the strongest drivers of titer", "analysis"),
    ("did the model pass its promotion gate", "analysis"),
    ("what recipes were proposed next", "analysis"),
    ("what is the predicted titer for the next batch", "analysis"),
    ("how do I record a result for a started run", "doc"),
    ("what does reanalyze do", "doc"),
    ("when does the scale readout refuse to predict", "doc"),
]


def _metrics(index: AssistIndex, cid: str, k: int = 6) -> dict[str, float]:
    recall_hits = 0
    p_at_1 = 0
    rr_sum = 0.0
    per_kind: dict[str, list[int]] = {}
    for question, want_kind in _QUERIES:
        hits = index.search(question, campaign_id=cid, k=k)
        kinds = [h.chunk.kind for h in hits]
        found = want_kind in kinds
        rank = kinds.index(want_kind) + 1 if found else 0
        recall_hits += int(found)
        p_at_1 += int(bool(kinds) and kinds[0] == want_kind)
        rr_sum += (1.0 / rank) if rank else 0.0
        per_kind.setdefault(want_kind, []).append(int(found))
    n = len(_QUERIES)
    return {
        "recall_at_k": recall_hits / n,
        "precision_at_1": p_at_1 / n,
        "mrr": rr_sum / n,
        "n": float(n),
        **{f"recall_{kind}": sum(v) / len(v) for kind, v in per_kind.items()},
    }


def test_bm25_retrieval_quality_report(seeded_store, capsys):
    store, ids = seeded_store
    index = AssistIndex(store, "acme", embedder=False)  # force BM25
    status = index.refresh()
    assert status["retriever"] == "bm25"

    m = _metrics(index, ids["ferment"])
    with capsys.disabled():
        print("\n=== assist retrieval benchmark (BM25) ===")
        print(f"queries: {int(m['n'])}  recall@6: {m['recall_at_k']:.3f}  "
              f"P@1: {m['precision_at_1']:.3f}  MRR: {m['mrr']:.3f}")
        for kind in ("sheet", "finding", "analysis", "doc"):
            key = f"recall_{kind}"
            if key in m:
                print(f"  recall[{kind}]: {m[key]:.3f}")

    # Loose regression floors, not targets. If a scoring change drops below
    # these, the benchmark fails and we look at the printed table.
    assert m["recall_at_k"] >= 0.70, f"recall@6 regressed: {m}"
    assert m["mrr"] >= 0.40, f"MRR regressed: {m}"


def test_retrieval_stays_within_the_campaign(seeded_store):
    """A query scoped to one campaign must not surface the other campaign's
    sheet/finding/analysis chunks (docs are shared and allowed)."""
    store, ids = seeded_store
    index = AssistIndex(store, "acme", embedder=False)
    index.refresh()
    hits = index.search("which inputs went with titer", campaign_id=ids["ferment"], k=6)
    other = ids["media"]
    leaked = [h.chunk for h in hits if h.chunk.campaign_id == other]
    assert leaked == [], f"cross-campaign leak: {[c.kind for c in leaked]}"


def test_empty_when_nothing_matches(seeded_store):
    """An off-topic question returns no hits, so the caller abstains rather
    than citing an irrelevant chunk."""
    store, ids = seeded_store
    index = AssistIndex(store, "acme", embedder=False)
    index.refresh()
    hits = index.search("quarterly revenue forecast for the sales team",
                        campaign_id=ids["ferment"], k=6)
    assert hits == []
