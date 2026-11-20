"""Lineage: where a campaign (or a run) came from, and what came of it.

Voyager linked experiments parent -> child (``ExperimentLink``) and walked the
links with a recursive query to show a whole lineage. In Kalos an experiment is a
CAMPAIGN (rounds of runs), an iteration is a ROUND, and moving up the scale track
(well plate -> flask -> bench -> pilot) is usually a NEW campaign at the next
stage, so a link joins two campaigns, or a run of one to a run of another:

- ``scale_up`` / ``scale_down``: the child repeats the parent at another scale;
- ``next_round``: a run of a later round continues one of an earlier round;
- ``follow_up``: the child campaign was started from the parent's findings;
- ``replicate``: the child repeats the parent at the same scale;
- ``derived``: anything else, said in the note.

Each end may carry the stage key it was at (``S1``..``S6`` of the campaign's
stage track, `kalos.portal.stages`). It is RECORDED (what the team says), never
computed.

Rules this module enforces (the HTTP layer adds tenant resolution: both
campaigns must be the caller's, so a link can never cross tenants):

- no self-link, and a link inside one campaign joins two of its RUNS;
- no cycles: across campaigns the campaign graph stays acyclic; inside a campaign
  the run graph does. A cycle would make "where did this come from" unanswerable;
- no duplicate active link (the same two ends and relation);
- traversal is bounded: depth (`MAX_DEPTH`), nodes (`MAX_NODES`) and links
  (`MAX_LINKS_RETURNED`); a cut-off says so (`truncated`) instead of silently
  returning less. Voyager's unbounded recursive CTE is not ported as is.

Links are append-only; withdrawing one is a retraction with a reason.
"""
from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from kalos.records.projects import RecordsValidationError, clean_text
from kalos.storage.migrations.v9 import LINK_RELATIONS

RELATIONS: tuple[str, ...] = LINK_RELATIONS
RELATION_DESCRIPTIONS = {
    "next_round": "a run of a later round continues one of an earlier round",
    "scale_up": "the child repeats the parent at a larger scale",
    "scale_down": "the child repeats the parent at a smaller scale (a scale-down model)",
    "follow_up": "the child was started from the parent's findings",
    "replicate": "the child repeats the parent at the same scale",
    "derived": "another relation, described in the note",
}
MAX_NOTE_CHARS = 500
MAX_REASON_CHARS = 500
MAX_RUN_ID_CHARS = 128
MAX_LINKS_PER_TENANT = 20_000
DEFAULT_DEPTH = 6
MAX_DEPTH = 32
MAX_NODES = 500
MAX_LINKS_RETURNED = 2000
STAGE_KEY_PATTERN = re.compile(r"S?[A-Za-z0-9_-]{1,16}")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


@dataclass(frozen=True)
class LinkFields:
    parent_campaign_id: str
    parent_run_id: str  # '' = the whole campaign
    child_campaign_id: str
    child_run_id: str
    relation: str
    parent_stage: str | None
    child_stage: str | None
    note: str | None


def _run_id(value: object, what: str) -> str:
    if value is None or value == "":
        return ""
    if not isinstance(value, str) or RUN_ID_PATTERN.fullmatch(value) is None:
        raise RecordsValidationError(
            "invalid_run_id", f"{what} must be a run id (letters, digits, _ . : -; up to 128)"
        )
    return value


def _stage(value: object, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or STAGE_KEY_PATTERN.fullmatch(value) is None:
        raise RecordsValidationError("invalid_stage", f"{what} must be a stage key such as S2")
    return value


def link_fields(
    *, parent_campaign_id: str, child_campaign_id: str, relation: object,
    parent_run_id: object = None, child_run_id: object = None,
    parent_stage: object = None, child_stage: object = None, note: object = None,
) -> LinkFields:
    """Validate a link's own fields (not the graph: see `check_new_link`)."""
    if relation not in RELATIONS:
        raise RecordsValidationError("invalid_relation", f"relation must be one of: {', '.join(RELATIONS)}")
    p_run, c_run = _run_id(parent_run_id, "parent_run_id"), _run_id(child_run_id, "child_run_id")
    if parent_campaign_id == child_campaign_id:
        if not p_run or not c_run:
            raise RecordsValidationError(
                "self_link", "a link inside one campaign must join two of its runs "
                "(parent_run_id and child_run_id)",
            )
        if p_run == c_run:
            raise RecordsValidationError("self_link", "a run cannot be linked to itself")
    clean_note = clean_text(note, "note", MAX_NOTE_CHARS, required=False, allow_newlines=True)
    if relation == "derived" and clean_note is None:
        raise RecordsValidationError("note_required", "a 'derived' link needs a note saying how")
    return LinkFields(
        parent_campaign_id=parent_campaign_id, parent_run_id=p_run,
        child_campaign_id=child_campaign_id, child_run_id=c_run, relation=str(relation),
        parent_stage=_stage(parent_stage, "parent_stage"), child_stage=_stage(child_stage, "child_stage"),
        note=clean_note,
    )


def check_reason(reason: object) -> str:
    clean = clean_text(reason, "reason", MAX_REASON_CHARS, required=True, allow_newlines=True)
    assert clean is not None
    return clean


def _reaches(edges: Iterable[tuple[str, str]], start: str, goal: str) -> bool:
    out: dict[str, list[str]] = {}
    for a, b in edges:
        out.setdefault(a, []).append(b)
    seen, queue = {start}, deque([start])
    while queue:
        node = queue.popleft()
        if node == goal:
            return True
        for nxt in out.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def check_new_link(link: LinkFields, active: list[dict[str, Any]]) -> None:
    """Refuse a duplicate or a cycle, given the tenant's ACTIVE links touching the
    graph (the repository passes them inside the write transaction)."""
    for row in active:
        if (row["parent_campaign_id"], row["parent_run_id"], row["child_campaign_id"],
                row["child_run_id"], row["relation"]) == (
                link.parent_campaign_id, link.parent_run_id, link.child_campaign_id,
                link.child_run_id, link.relation):
            raise RecordsValidationError("duplicate_link", "this link already exists")
    if link.parent_campaign_id != link.child_campaign_id:
        cross = [(r["parent_campaign_id"], r["child_campaign_id"]) for r in active
                 if r["parent_campaign_id"] != r["child_campaign_id"]]
        if _reaches(cross, link.child_campaign_id, link.parent_campaign_id):
            raise RecordsValidationError(
                "cycle", "this link would make a campaign its own ancestor; lineage must not loop"
            )
        return
    inner = [(r["parent_run_id"], r["child_run_id"]) for r in active
             if r["parent_campaign_id"] == r["child_campaign_id"] == link.parent_campaign_id]
    if _reaches(inner, link.child_run_id, link.parent_run_id):
        raise RecordsValidationError(
            "cycle", "this link would make a run its own ancestor; lineage must not loop"
        )


def lineage_view(
    start_campaign_id: str, links: list[dict[str, Any]], *, depth: int = DEFAULT_DEPTH,
) -> dict[str, Any]:
    """Ancestors and descendants of a campaign over the active cross-campaign
    links, breadth first, up to `depth` hops each way and `MAX_NODES` campaigns;
    then every active link (cross or within) among the campaigns reached, up to
    `MAX_LINKS_RETURNED`. `truncated` names each bound that cut the answer."""
    depth = max(1, min(int(depth), MAX_DEPTH))
    cross = [r for r in links if r["parent_campaign_id"] != r["child_campaign_id"]]
    parents: dict[str, set[str]] = {}
    children: dict[str, set[str]] = {}
    for r in cross:
        children.setdefault(r["parent_campaign_id"], set()).add(r["child_campaign_id"])
        parents.setdefault(r["child_campaign_id"], set()).add(r["parent_campaign_id"])
    truncated: list[str] = []
    level: dict[str, int] = {start_campaign_id: 0}

    def walk(neighbours: dict[str, set[str]], sign: int) -> None:
        frontier = deque([(start_campaign_id, 0)])
        while frontier:
            node, d = frontier.popleft()
            for nxt in sorted(neighbours.get(node, ())):
                if nxt in level:
                    continue
                if d + 1 > depth:
                    if "depth" not in truncated:
                        truncated.append("depth")
                    continue
                if len(level) >= MAX_NODES:
                    if "nodes" not in truncated:
                        truncated.append("nodes")
                    return
                level[nxt] = sign * (d + 1)
                frontier.append((nxt, d + 1))

    walk(parents, -1)
    walk(children, +1)
    reached = set(level)
    among = [r for r in links if r["parent_campaign_id"] in reached and r["child_campaign_id"] in reached]
    among.sort(key=lambda r: int(r["id"]))
    if len(among) > MAX_LINKS_RETURNED:
        among = among[:MAX_LINKS_RETURNED]
        truncated.append("links")
    nodes = [
        {"campaign_id": cid, "generation_offset": off,
         "role": "start" if off == 0 else ("ancestor" if off < 0 else "descendant")}
        for cid, off in sorted(level.items(), key=lambda kv: (kv[1], kv[0]))
    ]
    return {
        "campaign_id": start_campaign_id,
        "depth": depth,
        "nodes": nodes,
        "links": among,
        "truncated": truncated,
        "bounds": {"max_depth": MAX_DEPTH, "max_nodes": MAX_NODES, "max_links": MAX_LINKS_RETURNED},
    }


def relation_catalogue() -> list[dict[str, str]]:
    return [{"relation": r, "description": RELATION_DESCRIPTIONS[r]} for r in RELATIONS]


__all__ = [
    "DEFAULT_DEPTH",
    "MAX_DEPTH",
    "MAX_LINKS_PER_TENANT",
    "MAX_NODES",
    "RELATIONS",
    "LinkFields",
    "check_new_link",
    "check_reason",
    "lineage_view",
    "link_fields",
    "relation_catalogue",
]
