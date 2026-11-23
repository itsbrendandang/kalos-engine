"""Paper excerpts in the assistant (KALOS_ASSIST_LITERATURE_DIR): what a pack may
hold, how it is indexed, and what an answer citing a paper may say."""
from __future__ import annotations

import json

import pytest

from kalos.assist.chat import LITERATURE_NOTE, Labelled, build_prompt, check_answer
from kalos.assist.index import AssistIndex, Chunk, Hit, literature_chunks
from kalos.portal.campaign import CampaignStore

GUIDE = {"step": "Record a result for every started run.", "why": "", "why_it_matters": ""}


def _item(i: int = 1, **kw: object) -> dict:
    return {"id": f"lit-{i:03d}", "citation": "Smith J. Glutamine feeds in CHO fed-batch. 2020",
            "title": "Glutamine feeds in CHO fed-batch", "doi": "10.1/x", "location": "Smith2020 pages 2-3",
            "excerpt": "Lower glutamine feeds reduced ammonia. Titer rose by 30% at day 14.", **kw}


def _pack(items: list[dict], **kw: object) -> str:
    return json.dumps({"format": "kalos-literature/1", "producer": "research-agent", "items": items, **kw})


# --- packs ----------------------------------------------------------------------- #


def test_a_pack_becomes_untrusted_literature_chunks_and_bad_files_are_refused_whole(tmp_path):
    (tmp_path / "a.json").write_text(_pack([_item(1), _item(1), _item(2, excerpt="Ammonia slows growth. " * 120)]))
    (tmp_path / "b.json").write_text(_pack([_item(3)], secret_field="x"))         # unknown field
    (tmp_path / "c.json").write_text(json.dumps({"format": "other/1", "items": []}))
    (tmp_path / "d.json").write_text(_pack([_item(4, excerpt="")]))               # empty excerpt
    chunks, refused = literature_chunks(sorted(tmp_path.glob("*.json")))
    assert {c.source_id for c in chunks} == {"lit-001", "lit-002"}              # duplicate id once
    assert all(c.kind == "literature" and not c.trusted_text and c.kalos_text() == "" for c in chunks)
    assert all(c.campaign_id is None and len(c.text) <= 1200 for c in chunks)
    assert len([c for c in chunks if c.source_id == "lit-002"]) > 1             # long excerpt split
    assert chunks[0].title == "Smith J. Glutamine feeds in CHO fed-batch. 2020 - Smith2020 pages 2-3"
    assert [r["file"] for r in refused] == ["b.json", "c.json", "d.json"]
    assert all("kalos-literature/1" in r["reason"] for r in refused)


@pytest.fixture
def store(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "LOOP.md").write_text("# Loop\n\nRecord a result for every started run.\n")
    lit = tmp_path / "literature"
    lit.mkdir()
    (lit / "pack.json").write_text(_pack([_item(1)]))
    (lit / "broken.json").write_text("{not json")
    monkeypatch.setenv("KALOS_ASSIST_DOCS_DIR", str(docs))
    monkeypatch.setenv("KALOS_ASSIST_LITERATURE_DIR", str(lit))
    monkeypatch.setenv("KALOS_ASSIST_EMBED", "0")
    return CampaignStore(tmp_path / "state")


def test_the_index_holds_the_pack_reports_refusals_and_follows_changes(store, tmp_path):
    index = AssistIndex(store, "acme", embedder=False)
    status = index.refresh()
    assert status["literature"] == {"items": 1, "refused": [
        {"file": "broken.json", "reason": status["literature"]["refused"][0]["reason"]}]}
    hits = index.search("glutamine ammonia")
    assert hits and hits[0].chunk.kind == "literature"
    (tmp_path / "literature" / "pack.json").write_text(_pack([_item(1), _item(2, excerpt="Lactate shift.")]))
    assert index.refresh()["literature"]["items"] == 2


def test_without_the_setting_there_is_no_literature(store, monkeypatch):
    monkeypatch.delenv("KALOS_ASSIST_LITERATURE_DIR")
    index = AssistIndex(store, "acme", embedder=False)
    assert index.refresh()["literature"] == {"items": 0, "refused": []}
    assert index.search("glutamine ammonia") == []


# --- answers --------------------------------------------------------------------- #


def literature_chunks_from(item: dict) -> Chunk:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "p.json"
        p.write_text(_pack([item]))
        return literature_chunks([p])[0][0]


def _paper() -> Chunk:
    return literature_chunks_from(_item(1))


def _sheet() -> Chunk:
    return Chunk("c9", "round", "1", "cmp_" + "1" * 32, "Round 1", "Measured lipase_titer 3.3 in run R1-1.")


def test_a_sentence_citing_a_paper_states_no_number_but_may_name_its_year():
    lab = [Labelled("S1", Hit(_paper(), 1.0)), Labelled("S2", Hit(_sheet(), 1.0))]
    out = check_answer(
        "Smith 2020 reports that lower glutamine feeds reduced ammonia [S1]. "
        "Titer rose by 30% at day 14 in that paper [S1]. "
        "Your run reached 3.3 while a paper saw a rise [S1][S2]. "
        "Measured lipase_titer was 3.3 [S2].", lab, GUIDE)
    assert out.answer == ("Smith 2020 reports that lower glutamine feeds reduced ammonia. "
                          "Measured lipase_titer was 3.3.")
    assert out.drop_reasons == {"literature_number": 2}
    paper = next(s for s in out.sources if s["kind"] == "literature")
    assert paper == {"kind": "literature", "id": "lit-001", "campaign_id": None,
                     "quote": "Lower glutamine feeds reduced ammonia.",
                     "citation": "Smith J. Glutamine feeds in CHO fed-batch. 2020 - Smith2020 pages 2-3"}
    sheet = next(s for s in out.sources if s["kind"] == "round")
    assert "citation" not in sheet


def test_a_year_is_allowed_only_from_the_cited_papers_citation():
    lab = [Labelled("S1", Hit(_paper(), 1.0))]
    assert check_answer("Smith 2019 reports lower ammonia [S1].", lab, GUIDE).drop_reasons == {
        "literature_number": 1}


def test_a_paper_never_supports_a_claim_word():
    paper = literature_chunks_from(_item(1, excerpt="Glutamine limitation causes lower ammonia and will raise titer."))
    out = check_answer("Glutamine limitation causes lower ammonia [S1].", [Labelled("S1", Hit(paper, 1.0))], GUIDE)
    assert out.rejected and out.answer == ""


def test_the_prompt_changes_only_when_a_paper_was_retrieved():
    with_paper, _ = build_prompt("glutamine?", [Hit(_paper(), 1.0)], GUIDE)
    without, _ = build_prompt("titer?", [Hit(_sheet(), 1.0)], GUIDE)
    assert with_paper.startswith(LITERATURE_NOTE)
    assert without.startswith("Sources:\n\n") and "literature" not in without

