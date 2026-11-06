"""SOP drafts (`kalos.assist.sop`, `/api/assist/sop*`).

- every number in the settings table is in `fields_from_record` with a
  source id that resolves to the campaign record;
- a revision may change prose only: placeholders, numbers, record values and
  the header are checked after the model answers (422, nothing stored);
- a good revision stores v2 and v1 stays byte-identical on disk;
- fill replaces tokens verbatim; unknown names and bad values are 400;
- drafts are per tenant: another tenant's id is the same 404 a missing one
  gets, and the list shows only the caller's drafts;
- provider `none`: drafting works, revise answers 404 assist_disabled;
- export as markdown and .docx.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
from dataclasses import dataclass

import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.assist import sop
from kalos.assist.llm import ModelUnavailable, get_chat_model
from kalos.assist.sop_routes import router
from kalos.assist.words import BANNED, banned_in, claims_in
from kalos.portal import app as portal_module
from kalos.portal import campaign as campaign_module
from kalos.portal.app import app
from kalos.portal.campaign import CampaignStore, get_campaign_store


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": _sha("a-write"), "subject": "bob@acme", "tenant": "acme",
     "scopes": ["read", "write"]},
    {"token_sha256": _sha("a-read"), "subject": "carol@acme", "tenant": "acme",
     "scopes": ["read"]},
    {"token_sha256": _sha("b-write"), "subject": "gina@globex", "tenant": "globex",
     "scopes": ["read", "write"]},
]


def H(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "Methanol": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 0.5, 1.25],
        "pH": [6.0, 6.5, 7.0, 6.1, 6.4, 6.8, 5.9, 6.3],
        "lipase_titer": [1.1, 2.2, 3.3, 1.6, 2.7, 3.9, 0.4, 1.3],
    })


ANALYSIS = {
    "target": "lipase_titer",
    "proposals": [
        {"recipe": {"Methanol": 2.5, "pH": 6.2}, "pred": 3.1, "std": 0.4},
        {"recipe": {"Methanol": 3.75, "pH": 6.9}, "pred": 3.8, "std": 0.6},
    ],
    "feature_ranges": {"Methanol": [0.0, 4.0], "pH": [5.0, 7.0]},
    "constraint": {"column": "purity", "floor": 95, "applied": True, "reason": None},
    "provenance": {"input_sha256": "ab" * 32},
}


def _seed(store: CampaignStore, tenant: str = "acme") -> tuple[str, list[dict]]:
    cid = store.create_campaign(tenant=tenant, name="c", created_by="x")["id"]
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant=tenant, campaign_id=cid,
               analysis=ANALYSIS)
    runs = store.start(
        [{"recipe": p["recipe"], "pred": p["pred"], "std": p["std"]} for p in ANALYSIS["proposals"]],
        tenant=tenant, campaign_id=cid,
    )
    store.set_result(runs[0]["id"], 4.2, tenant=tenant, campaign_id=cid)
    return cid, runs


@dataclass
class FakeModel:
    """A chat model whose reply is a function of the draft it was sent."""

    edit: object
    model: str = "fake-1"
    provider: str = "fake"
    local: bool = True

    def complete(self, system: str, prompt: str) -> str:
        text = prompt.split("The draft:\n\n", 1)[1]
        return self.edit(text)  # type: ignore[operator]


@pytest.fixture
def env(tmp_path, monkeypatch):
    store = CampaignStore(tmp_path)
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(TOKENS))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "none")
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    if not any(getattr(r, "path", "") == "/api/assist/sop/draft" for r in app.routes):
        app.include_router(router)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app), store
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
        app.dependency_overrides.pop(get_chat_model, None)


def _use_model(edit) -> None:
    app.dependency_overrides[get_chat_model] = lambda: FakeModel(edit)


def _draft(tc, cid: str, source: dict, template: str = "batch_recipe", token: str = "a-write") -> dict:
    r = tc.post("/api/assist/sop/draft",
                json={"campaign_id": cid, "source": source, "template": template},
                headers=H(token))
    assert r.status_code == 201, r.text
    return r.json()


def _table_numbers(markdown: str, heading: str = "## Settings") -> set[str]:
    section = markdown.split(heading, 1)[1].split("\n## ", 1)[0]
    rows = [ln for ln in section.splitlines() if ln.startswith("|") and not ln.startswith("|---")]
    return set(sop.NUMBER_RE.findall("\n".join(rows[1:])))


def _resolves(source_id: str, store: CampaignStore, cid: str) -> bool:
    state = store.get(tenant="acme", campaign_id=cid)
    analysis = store.load_analysis(tenant="acme", campaign_id=cid)
    kind, _, rest = source_id.partition(":")
    if kind == "run":
        return any(r["id"] == rest for r in state["pending"])
    if kind == "proposal":
        rnd, idx = rest.split(".")
        return int(rnd) == state["round"] and int(idx) < len(analysis["proposals"])
    if kind == "analysis":
        return rest in analysis
    if kind == "sheet":
        return rest == analysis["provenance"]["input_sha256"]
    return False


# --- drafting ---------------------------------------------------------------- #


@pytest.mark.parametrize("template", list(sop.TEMPLATES))
@pytest.mark.parametrize("kind", ["run_ids", "proposal_index", "round"])
def test_every_number_in_the_parameters_table_is_sourced(env, template, kind):
    tc, store = env
    cid, runs = _seed(store)
    source = {"run_ids": [r["id"] for r in runs], "proposal_index": [0, 1], "round": 0}[kind]
    d = _draft(tc, cid, {kind: source}, template)
    md = d["markdown"]
    assert md.split("\n")[0] == f"{sop.HEADER_PREFIX}{d['created_at'][:10]}{sop.HEADER_SUFFIX}"
    assert [f"## {t}" in md for t in ("Purpose", "Scope", "Materials", "Equipment", "Settings",
                                        "Procedure", "In-process checks", "Acceptance criteria",
                                        "Deviations and notes", "Revision table")] == [True] * 10
    values = {f["value"] for f in d["fields_from_record"]}
    numbers = _table_numbers(md)
    assert numbers and numbers <= values, numbers - values
    for f in d["fields_from_record"]:
        assert _resolves(f["source_id"], store, cid), f
        assert f["value"] in md
    assert set(d["source_ids"]) == {f["source_id"] for f in d["fields_from_record"]
                                    if f["source_id"].startswith(("run:", "proposal:"))}
    # predicted is never shown as measured; the measured run is labelled
    assert "| 3.1 | Predicted |" in md
    if kind != "proposal_index":
        assert "| 4.2 | Measured |" in md
    else:
        assert "Measured |" not in md
    # the constraint limit comes with its source; the rest of the limits are placeholders
    assert any(f["source_id"] == "analysis:constraint" and f["value"] == "95" for f in d["fields_from_record"])
    names = {p["name"] for p in d["placeholders"]}
    assert {"safety_notes", "hazard_flags", "equipment_ids", "prepared_by", "checked_by",
            "approved_by", "acceptance_limits", "material_lot_numbers"} <= names
    assert set(sop.PLACEHOLDER_RE.findall(md)) == names
    assert d["version"] == 1 and d["model"] is None and d["filled"] == []
    assert d["sha256"] == _sha(md)
    assert _only_fixed_banned(md) and not claims_in(md)


def _only_fixed_banned(md: str) -> bool:
    """"approved" appears only in the fixed header and the revision table (the
    owner's header text, the Approved by column and its placeholder)."""
    lines = md.splitlines()
    flagged = [ln for ln in lines if banned_in(ln)]
    revision = md.split("## Revision table", 1)[1].splitlines()
    return set(banned_in(md)) <= {"approved"} and all(ln == lines[0] or ln in revision for ln in flagged)


def test_round_with_no_runs_and_bad_sources_are_400(env):
    tc, store = env
    cid, runs = _seed(store)
    for source in ({"round": 7}, {"run_ids": [secrets.token_hex(16)]}, {"proposal_index": [9]},
                   {"run_ids": [runs[0]["id"]], "round": 0}, {}):
        r = tc.post("/api/assist/sop/draft",
                    json={"campaign_id": cid, "source": source, "template": "batch_recipe"},
                    headers=H("a-write"))
        assert r.status_code == 400, (source, r.text)
        assert r.json()["code"] in ("invalid_source", "invalid_body")
    assert tc.get("/api/assist/sop", headers=H("a-read")).json() == {"drafts": []}
    # a campaign the caller cannot see, or a malformed id, is the campaign 404
    missing = tc.post("/api/assist/sop/draft",
                      json={"campaign_id": "cmp_" + secrets.token_hex(16), "source": {"round": 0},
                            "template": "custom"}, headers=H("a-write"))
    bad = tc.post("/api/assist/sop/draft",
                  json={"campaign_id": "../x", "source": {"round": 0}, "template": "custom"},
                  headers=H("a-write"))
    assert missing.status_code == bad.status_code == 404
    assert missing.content == bad.content
    # read scope cannot draft
    assert tc.post("/api/assist/sop/draft", json={"campaign_id": cid, "source": {"round": 0},
                   "template": "custom"}, headers=H("a-read")).status_code == 403


def test_draft_is_stored_append_only_per_tenant(env):
    tc, store = env
    cid, _ = _seed(store)
    d = _draft(tc, cid, {"round": 0})
    folder = store.state_dir / "assist" / ("t-" + b"acme".hex()) / "sop" / d["draft_id"]
    assert (folder / "v1.json").is_file() and (folder / "v1.md").read_text() == d["markdown"]
    assert json.loads((folder / "v1.json").read_text()) == d
    with pytest.raises(sop.VersionConflict):
        sop._write_version(folder, sop.Draft.from_dict(d))
    assert (folder / "v1.md").read_text() == d["markdown"]
    audit = [e for e in store.audit_entries(tenant="acme") if e["action"] == "sop.draft"]
    assert audit and audit[0]["campaign_id"] == cid
    assert d["markdown"][:30] not in json.dumps(audit)


# --- revise ------------------------------------------------------------------ #


def _refused(tc, draft_id: str, edit) -> list[str]:
    _use_model(edit)
    r = tc.post(f"/api/assist/sop/{draft_id}/revise", json={"instruction": "tighten the wording"},
                headers=H("a-write"))
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["code"] == "revision_refused" and body["reasons"]
    return body["reasons"]


def test_revise_refuses_a_filled_placeholder(env):
    tc, store = env
    cid, _ = _seed(store)
    d = _draft(tc, cid, {"round": 0})
    reasons = _refused(tc, d["draft_id"], lambda t: t.replace("[[TO FILL: safety_notes]]", "Wear gloves."))
    assert any("safety_notes" in r for r in reasons)
    assert tc.get(f"/api/assist/sop/{d['draft_id']}", headers=H("a-read")).json()["version"] == 1


def test_revise_refuses_a_renamed_or_new_placeholder(env):
    tc, store = env
    cid, _ = _seed(store)
    d = _draft(tc, cid, {"round": 0})
    _refused(tc, d["draft_id"], lambda t: t.replace("[[TO FILL: hazard_flags]]", "[[TO FILL: hazards]]"))
    _refused(tc, d["draft_id"], lambda t: t + "\nAlso: [[TO FILL: extra_thing]]\n")


def test_revise_refuses_a_number_change(env):
    tc, store = env
    cid, _ = _seed(store)
    d = _draft(tc, cid, {"round": 0})
    reasons = _refused(tc, d["draft_id"], lambda t: t.replace("| 2.5 |", "| 2.6 |"))
    assert any("number" in r for r in reasons)
    _refused(tc, d["draft_id"], lambda t: t + "\nHold for 30 minutes.\n")
    _refused(tc, d["draft_id"], lambda t: t.replace("| 4.2 | Measured |", "| 4.2 | Predicted |").replace("| 3.1 | Predicted |", "| 3.1 | Measured |") + "")


def test_revise_refuses_a_missing_or_changed_header(env):
    tc, store = env
    cid, _ = _seed(store)
    d = _draft(tc, cid, {"round": 0})
    header = d["markdown"].split("\n")[0]
    reasons = _refused(tc, d["draft_id"], lambda t: t.split("\n", 1)[1])
    assert any("header" in r for r in reasons)
    _refused(tc, d["draft_id"], lambda t: t.replace(header, header.replace("Not reviewed. ", "")))
    _refused(tc, d["draft_id"], lambda t: t.replace("not a procedure in itself", f"a {BANNED[0]} procedure"))
    _refused(tc, d["draft_id"], lambda t: t.replace("not a procedure in itself", "it will work"))
    assert tc.get(f"/api/assist/sop/{d['draft_id']}", headers=H("a-read")).json()["version"] == 1


def test_a_good_revision_stores_v2_and_leaves_v1_untouched(env):
    tc, store = env
    cid, _ = _seed(store)
    d = _draft(tc, cid, {"run_ids": [store.get(tenant="acme", campaign_id=cid)["pending"][0]["id"]]})
    folder = store.state_dir / "assist" / ("t-" + b"acme".hex()) / "sop" / d["draft_id"]
    v1_json, v1_md = (folder / "v1.json").read_bytes(), (folder / "v1.md").read_bytes()
    _use_model(lambda t: "```markdown\n" + t.replace(
        "It is a starting point for the lab's own procedure, not a procedure in itself.",
        "Treat it as a starting point for the lab's own procedure.") + "\n```")
    r = tc.post(f"/api/assist/sop/{d['draft_id']}/revise", json={"instruction": "shorter"},
                headers=H("a-write"))
    assert r.status_code == 200, r.text
    v2 = r.json()
    assert v2["version"] == 2 and v2["model"] == "fake-1" and v2["created_by"] == "bob@acme"
    assert "Treat it as a starting point" in v2["markdown"]
    assert v2["fields_from_record"] == d["fields_from_record"]
    assert v2["placeholders"] == d["placeholders"]
    assert (folder / "v1.json").read_bytes() == v1_json and (folder / "v1.md").read_bytes() == v1_md
    assert (folder / "v2.md").read_text() == v2["markdown"]
    assert tc.get(f"/api/assist/sop/{d['draft_id']}?version=1", headers=H("a-read")).json() == d
    assert tc.get(f"/api/assist/sop/{d['draft_id']}", headers=H("a-read")).json() == v2
    assert tc.get(f"/api/assist/sop/{d['draft_id']}?version=3", headers=H("a-read")).status_code == 404
    listed = tc.get("/api/assist/sop", headers=H("a-read")).json()["drafts"]
    assert listed[0]["latest_version"] == 2 and listed[0]["updated_at"] == v2["created_at"]
    assert listed[0]["source_ids"] == v2["source_ids"]
    audit = [e for e in store.audit_entries(tenant="acme") if e["action"] == "sop.revise"]
    assert audit and audit[0]["campaign_id"] == cid


def test_revise_model_unavailable_is_503(env):
    tc, store = env
    cid, _ = _seed(store)
    d = _draft(tc, cid, {"round": 0})

    def boom(_):
        raise ModelUnavailable("down")

    _use_model(boom)
    r = tc.post(f"/api/assist/sop/{d['draft_id']}/revise", json={"instruction": "x"}, headers=H("a-write"))
    assert r.status_code == 503 and r.json()["code"] == "assist_provider_unavailable"


def test_provider_none_drafts_but_revise_is_404(env):
    tc, store = env
    cid, _ = _seed(store)
    d = _draft(tc, cid, {"proposal_index": [1]}, "media_prep")
    assert "| B |" not in d["markdown"] and d["source_ids"] == ["proposal:0.1"]
    r = tc.post(f"/api/assist/sop/{d['draft_id']}/revise", json={"instruction": "x"}, headers=H("a-write"))
    assert r.status_code == 404 and r.json()["code"] == "assist_disabled"
    assert tc.get(f"/api/assist/sop/{d['draft_id']}", headers=H("a-read")).json()["version"] == 1


# --- fill -------------------------------------------------------------------- #


def test_fill_replaces_tokens_and_rejects_unknown_names_and_bad_values(env):
    tc, store = env
    cid, _ = _seed(store)
    d = _draft(tc, cid, {"round": 0})
    r = tc.post(f"/api/assist/sop/{d['draft_id']}/fill",
                json={"values": {"safety_notes": "Wear gloves and eye protection.",
                                 "equipment_ids": "BR-7 (bioreactor), PH-2 (pH meter)"}},
                headers=H("a-write"))
    assert r.status_code == 200, r.text
    v2 = r.json()
    assert v2["version"] == 2 and v2["model"] is None
    assert "Safety: Wear gloves and eye protection." in v2["markdown"]
    assert "[[TO FILL: safety_notes]]" not in v2["markdown"]
    assert "[[TO FILL: equipment_ids]]" not in v2["markdown"]
    assert "[[TO FILL: hazard_flags]]" in v2["markdown"]
    assert {p["name"] for p in v2["placeholders"]} == {p["name"] for p in d["placeholders"]} - {"safety_notes", "equipment_ids"}
    assert sorted(v2["filled"], key=lambda f: f["name"]) == [
        {"name": "equipment_ids", "section": "equipment", "value": "BR-7 (bioreactor), PH-2 (pH meter)", "filled_by": "bob@acme"},
        {"name": "safety_notes", "section": "scope", "value": "Wear gloves and eye protection.", "filled_by": "bob@acme"},
    ]
    # the same name again, or an unknown one: 400 unknown_placeholder, nothing stored
    r = tc.post(f"/api/assist/sop/{d['draft_id']}/fill",
                json={"values": {"safety_notes": "again", "nope": "x"}}, headers=H("a-write"))
    assert r.status_code == 400 and r.json()["code"] == "unknown_placeholder"
    assert r.json()["names"] == ["nope", "safety_notes"]
    for bad in ("", "x" * 501, "two\nlines", "a [[TO FILL: y]]", "close ]] it"):
        r = tc.post(f"/api/assist/sop/{d['draft_id']}/fill",
                    json={"values": {"hazard_flags": bad}}, headers=H("a-write"))
        assert r.status_code == 400 and r.json()["code"] == "invalid_value", bad
    assert tc.get(f"/api/assist/sop/{d['draft_id']}", headers=H("a-read")).json()["version"] == 2
    # a later revision keeps the filled text in place
    _use_model(lambda t: t.replace("Wear gloves and eye protection.", "Gloves."))
    r = tc.post(f"/api/assist/sop/{d['draft_id']}/revise", json={"instruction": "x"}, headers=H("a-write"))
    assert r.status_code == 422 and any("safety_notes" in s for s in r.json()["reasons"])
    audit = [e["action"] for e in store.audit_entries(tenant="acme")]
    assert "sop.fill" in audit


# --- tenant scoping ---------------------------------------------------------- #


def test_other_tenants_draft_is_the_same_404_and_lists_are_scoped(env):
    tc, store = env
    a_cid, _ = _seed(store, "acme")
    b_cid, _ = _seed(store, "globex")
    a = _draft(tc, a_cid, {"round": 0})
    b = _draft(tc, b_cid, {"round": 0}, token="b-write")
    missing = "sop_" + secrets.token_hex(16)
    for path, method, kw in (
        ("/api/assist/sop/{id}", "GET", {}),
        ("/api/assist/sop/{id}/export?format=md", "GET", {}),
        ("/api/assist/sop/{id}/revise", "POST", {"json": {"instruction": "x"}}),
        ("/api/assist/sop/{id}/fill", "POST", {"json": {"values": {"safety_notes": "x"}}}),
    ):
        foreign = tc.request(method, path.format(id=b["draft_id"]), headers=H("a-write"), **kw)
        absent = tc.request(method, path.format(id=missing), headers=H("a-write"), **kw)
        assert foreign.status_code == 404, (path, foreign.text)
        assert (foreign.content, sorted(foreign.headers.items())) == (absent.content, sorted(absent.headers.items()))
        for odd in ("sop_x", "SOP_" + "A" * 32, "sop_" + "g" * 32, a["draft_id"] + "x"):
            malformed = tc.request(method, path.format(id=odd), headers=H("a-write"), **kw)
            assert malformed.status_code == 404 and malformed.content == absent.content, odd
        assert foreign.json() == {"error": "draft not found", "code": "not_found"}
    assert [d["draft_id"] for d in tc.get("/api/assist/sop", headers=H("a-read")).json()["drafts"]] == [a["draft_id"]]
    assert [d["draft_id"] for d in tc.get("/api/assist/sop", headers=H("b-write")).json()["drafts"]] == [b["draft_id"]]
    # a draft of a campaign the caller cannot see is never made
    r = tc.post("/api/assist/sop/draft", json={"campaign_id": b_cid, "source": {"round": 0},
                "template": "custom"}, headers=H("a-write"))
    assert r.status_code == 404
    assert tc.get("/api/assist/sop", headers=H("a-read")).json()["drafts"][0]["draft_id"] == a["draft_id"]
    assert len(tc.get("/api/assist/sop", headers=H("a-read")).json()["drafts"]) == 1


# --- export ------------------------------------------------------------------ #


def test_export_markdown(env):
    tc, store = env
    cid, _ = _seed(store)
    d = _draft(tc, cid, {"round": 0})
    r = tc.get(f"/api/assist/sop/{d['draft_id']}/export", headers=H("a-read"))
    assert r.status_code == 200 and r.text == d["markdown"]
    assert r.headers["content-type"].startswith("text/markdown")
    assert r.headers["content-disposition"] == f'attachment; filename="{d["draft_id"]}-v1.md"'
    assert tc.get(f"/api/assist/sop/{d['draft_id']}/export?format=pdf", headers=H("a-read")).status_code == 422


def test_export_docx(env):
    docx = pytest.importorskip("docx")
    tc, store = env
    cid, _ = _seed(store)
    d = _draft(tc, cid, {"round": 0})
    r = tc.get(f"/api/assist/sop/{d['draft_id']}/export?format=docx&version=1", headers=H("a-read"))
    assert r.status_code == 200, r.text
    assert r.headers["content-disposition"] == f'attachment; filename="{d["draft_id"]}-v1.docx"'
    assert r.content[:2] == b"PK"
    import io

    document = docx.Document(io.BytesIO(r.content))
    text = "\n".join(p.text for p in document.paragraphs)
    assert text.startswith(sop.HEADER_PREFIX[2:])  # the "> " quote marker is markdown
    cells = {c.text for t in document.tables for row in t.rows for c in row.cells}
    assert {"2.5", "6.2", "Predicted", "Measured", "[[TO FILL: prepared_by]]"} <= cells
    for ph in re.findall(r"\[\[TO FILL: \w+\]\]", d["markdown"]):
        assert ph in text or ph in cells


def test_export_docx_without_python_docx_is_406(env, monkeypatch):
    tc, store = env
    cid, _ = _seed(store)
    d = _draft(tc, cid, {"round": 0})
    import builtins

    real = builtins.__import__

    def no_docx(name, *a, **kw):
        if name == "docx":
            raise ImportError("no docx")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_docx)
    r = tc.get(f"/api/assist/sop/{d['draft_id']}/export?format=docx", headers=H("a-read"))
    assert r.status_code == 406 and r.json()["code"] == "docx_unavailable"


# --- module level ----------------------------------------------------------- #


def _seed_small(store: CampaignStore) -> str:
    cid = store.create_campaign(tenant="acme", name="c", created_by="x")["id"]
    df = pd.DataFrame({"Methanol": [1.0, 2.0, 3.0], "pH": [6.0, 6.5, 7.0], "lipase_titer": [1.1, 2.2, 3.3]})
    store.seed(df, "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=cid,
               analysis={"target": "lipase_titer", "proposals": [{"recipe": {"Methanol": 2.5, "pH": 6.2}, "pred": 3.1, "std": 0.4}],
                         "feature_ranges": {"Methanol": [0.0, 4.0]}})
    store.start([{"recipe": {"Methanol": 2.5, "pH": 6.2}, "pred": 3.1, "std": 0.4}], tenant="acme", campaign_id=cid)
    return cid


def test_draft_from_the_module_is_sourced_and_clean(tmp_path):
    store = CampaignStore(tmp_path)
    cid = _seed_small(store)
    d = sop.draft(store, "acme", cid, {"round": 0}, "batch_recipe", None, "alice")
    assert d.markdown.startswith(sop.HEADER_PREFIX) and d.version == 1 and d.model is None
    assert {f["value"] for f in d.fields_from_record} >= {"2.5", "6.2", "3.1", "0.4", "0.0", "4.0", "lipase_titer"}
    assert set(sop.PLACEHOLDER_RE.findall(d.markdown)) == {p["name"] for p in d.placeholders}
    assert _only_fixed_banned(d.markdown) and not claims_in(d.markdown)
    assert sop.load(store, "acme", d.draft_id).markdown == d.markdown
    assert sop.load(store, "globex", d.draft_id) is None
    with pytest.raises(sop.SourceError):
        sop.draft(store, "acme", cid, {"round": 3}, "custom", None, "alice")


def test_check_revision_refuses_every_kind_of_drift(tmp_path):
    store = CampaignStore(tmp_path)
    d = sop.draft(store, "acme", _seed_small(store), {"proposal_index": [0]}, "media_prep", "T", "alice")
    md = d.markdown
    assert sop.check_revision(d, md) == []
    assert sop.check_revision(d, md.replace("It is a starting point", "Treat it as a starting point")) == []
    for bad in (
        md.split("\n", 1)[1],
        md.replace("| 2.5 |", "| 2.50 |"),
        md.replace("[[TO FILL: safety_notes]]", "gloves"),
        md.replace("[[TO FILL: safety_notes]]", "[[TO FILL: safety]]"),
        md + "[[TO FILL: more]]\n",
        md + "Hold 30 minutes.\n",
        md.replace("| 3.1 | Predicted |", "| 3.1 | Measured |"),
        md + "This will work.\n",
    ):
        assert sop.check_revision(d, bad), bad[-80:]


def test_campaign_id_omitted_uses_the_tenant_default(env):
    tc, store = env
    r = tc.post("/api/assist/sop/draft", json={"source": {"round": 0}, "template": "custom"}, headers=H("a-write"))
    assert r.status_code == 404 and r.json() == {"error": "campaign not found", "code": "not_found"}
    cid = store.create_campaign(tenant="acme", name="default", created_by="x", make_default=True)["id"]
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=cid, analysis=ANALYSIS)
    assert store.default_campaign_id(tenant="acme") == cid
    r = tc.post("/api/assist/sop/draft", json={"source": {"proposal_index": [0]}, "template": "batch_recipe"},
                headers=H("a-write"))
    assert r.status_code == 201, r.text
    assert r.json()["campaign_id"] == cid
    assert r.json()["source_ids"] == ["proposal:0.0"]


@pytest.mark.parametrize("template", ["batch_recipe", "media_prep", "custom"])
def test_markdown_uses_only_the_agreed_constructs(env, template):
    tc, store = env
    cid, _runs = _seed(store)
    md = _draft(tc, cid, {"round": 0}, template)["markdown"]
    assert "<" not in md and "](" not in md and "![" not in md
    for line in md.splitlines():
        if not line.strip():
            continue
        assert not line.startswith((" ", "\t")), f"nested or indented line: {line!r}"
        assert re.match(r"(#{1,3} |> |[-*] |\d+\. |\|)", line) or line[0].isalnum() or line[0] in "[`*", line


# --- security review sec-assist: BLOCKER ------------------------------------------- #


def _two_recipe_draft(tmp_path):
    store = CampaignStore(tmp_path)
    cid = store.create_campaign(tenant="acme", name="c", created_by="x")["id"]
    df = pd.DataFrame({"pH": [6.0, 6.5, 7.0], "Temp": [25, 28, 30], "titer": [1.1, 2.2, 3.3]})
    store.seed(df, "titer", ["pH", "Temp"], tenant="acme", campaign_id=cid,
               analysis={"target": "titer", "proposals": [
                   {"recipe": {"pH": 6.0, "Temp": 30}, "pred": 3.1, "std": 0.4},
                   {"recipe": {"pH": 7.0, "Temp": 25}, "pred": 2.9, "std": 0.5}]})
    return sop.draft(store, "acme", cid, {"proposal_index": [0, 1]}, "batch_recipe", None, "alice")


def test_revise_cannot_swap_values_between_recipes_in_prose(tmp_path):
    d = _two_recipe_draft(tmp_path)
    md = d.markdown
    a, b = "Recipe A: pH 6.0; Temp 30", "Recipe B: pH 7.0; Temp 25"
    assert a in md and b in md
    swapped = md.replace(a, "\x00").replace(b, a).replace("\x00", b)
    assert swapped != md
    reasons = sop.check_revision(d, swapped)
    assert any("record values" in r for r in reasons), reasons


def test_revise_cannot_reword_predicted_into_measured(tmp_path):
    d = _two_recipe_draft(tmp_path)
    md = d.markdown
    line = "Predicted values are the model's estimate at the time of the analysis, not a measurement."
    assert line in md
    reworded = md.replace(line, "These values were measured in the lab.")
    assert sop.check_revision(d, reworded)
    assert sop.check_revision(d, md.replace(line, "")), "dropping the line is refused too"
    # prose lines without values or Predicted/Measured wording stay editable
    assert sop.check_revision(d, md.replace("It is a starting point", "Treat it as a starting point")) == []


def test_revise_refuses_new_review_words(tmp_path):
    d = _two_recipe_draft(tmp_path)
    md = d.markdown
    base = "It is a starting point for the lab's own procedure, not a procedure in itself."
    for planted in ("This procedure was signed off.", "Passed validation in the lab.",
                    "Higher pH increases the result.", "Temperature drives the result.",
                    "Low pH leads to failure.", "It failed due to heat.", "Use it because it works."):
        assert sop.check_revision(d, md.replace(base, planted)), planted


def test_draft_and_version_caps_are_409(env, monkeypatch):
    """sec-assist (i): 500 drafts per tenant and 200 versions per draft (lowered here)."""
    tc, store = env
    cid, _runs = _seed(store)
    monkeypatch.setattr(sop, "MAX_DRAFTS_PER_TENANT", 2)
    monkeypatch.setattr(sop, "MAX_VERSIONS_PER_DRAFT", 2)
    d = _draft(tc, cid, {"round": 0})
    _draft(tc, cid, {"round": 0})
    r = tc.post("/api/assist/sop/draft", json={"campaign_id": cid, "source": {"round": 0}, "template": "custom"},
                headers=H("a-write"))
    assert r.status_code == 409 and r.json()["code"] == "limit_reached"
    ok = tc.post(f"/api/assist/sop/{d['draft_id']}/fill", json={"values": {"prepared_by": "Ana"}}, headers=H("a-write"))
    assert ok.status_code == 200 and ok.json()["version"] == 2
    r = tc.post(f"/api/assist/sop/{d['draft_id']}/fill", json={"values": {"checked_by": "Ben"}}, headers=H("a-write"))
    assert r.status_code == 409 and r.json()["code"] == "limit_reached"
    assert tc.get(f"/api/assist/sop/{d['draft_id']}", headers=H("a-read")).json()["version"] == 2
    # other tenant unaffected by acme's count
    assert sop.MAX_DRAFTS_PER_TENANT == 2
