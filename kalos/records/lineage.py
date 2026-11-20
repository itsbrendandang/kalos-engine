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
  returning less. It walks one hop at a time (one indexed query per hop and
  direction), never loading the tenant's whole graph. Voyager's unbounded
  recursive CTE is not ported as is.

Links are append-only; withdrawing one is a retraction with a reason.
"""
from __future__ import annotations

import re
from collections.abc import Callable
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


# One hop: given a frontier of node ids, the active links leaving it (down) or
# entering it (up). The storage layer answers each hop with one indexed query.
Hop = Callable[[list[str]], list[dict[str, Any]]]


def reachable(start: str, goal: str, step: Callable[[list[str]], set[str]], *, max_nodes: int) -> bool:
    """Whether `goal` can be reached from `start`, walking one hop per call of
    `step(frontier) -> next nodes`. Exhaustive up to `max_nodes` visited nodes
    (the graph's size bound), so a cycle check is never cut short by depth."""
    seen, frontier = {start}, [start]
    while frontier:
        if goal in frontier:
            return True
        nxt = sorted(n for n in step(frontier) if n not in seen)
        seen.update(nxt)
        if len(seen) > max_nodes:  # pragma: no cover - bounded by MAX_LINKS_PER_TENANT
            raise RecordsValidationError("graph_too_large", "the lineage graph is too large to check")
        frontier = nxt
    return False


def check_new_link(
    link: LinkFields, *, duplicate: bool, campaign_children: Hop, run_children: Hop,
) -> None:
    """Refuse a duplicate or a cycle. `campaign_children(ids)` returns the
    active cross-campaign links whose parent is in `ids`; `run_children(run_ids)`
    the active links inside the link's campaign whose parent run is in
    `run_ids`. Called inside the write transaction."""
    if duplicate:
        raise RecordsValidationError("duplicate_link", "this link already exists")
    if link.parent_campaign_id != link.child_campaign_id:
        def step(ids: list[str]) -> set[str]:
            return {r["child_campaign_id"] for r in campaign_children(ids)}

        if reachable(link.child_campaign_id, link.parent_campaign_id, step, max_nodes=MAX_LINKS_PER_TENANT + 1):
            raise RecordsValidationError(
                "cycle", "this link would make a campaign its own ancestor; lineage must not loop"
            )
        return

    def run_step(runs: list[str]) -> set[str]:
        return {r["child_run_id"] for r in run_children(runs)}

    if reachable(link.child_run_id, link.parent_run_id, run_step, max_nodes=MAX_LINKS_PER_TENANT + 1):
        raise RecordsValidationError(
            "cycle", "this link would make a run its own ancestor; lineage must not loop"
        )


def lineage_view(
    start_campaign_id: str, *, parents_of: Hop, children_of: Hop,
    links_among: Callable[[list[str]], list[dict[str, Any]]], depth: int = DEFAULT_DEPTH,
) -> dict[str, Any]:
    """Ancestors and descendants of a campaign over active cross-campaign links,
    breadth first, one query per hop each way (`parents_of(frontier)` returns
    the links entering the frontier, `children_of(frontier)` those leaving it),
    up to `depth` hops and `MAX_NODES` campaigns; then every active link (cross
    or within) among the campaigns reached, via `links_among(ids)` (which
    returns at most MAX_LINKS_RETURNED + 1). `truncated` names each bound that
    cut the answer."""
    depth = max(1, min(int(depth), MAX_DEPTH))
    truncated: list[str] = []
    level: dict[str, int] = {start_campaign_id: 0}

    def walk(hop: Hop, far_end: str, sign: int) -> None:
        frontier, d = [start_campaign_id], 0
        while frontier:
            found = sorted({r[far_end] for r in hop(frontier)} - set(level))
            if not found:
                return
            if d + 1 > depth:
                if "depth" not in truncated:
                    truncated.append("depth")
                return
            nxt: list[str] = []
            for node in found:
                if len(level) >= MAX_NODES:
                    if "nodes" not in truncated:
                        truncated.append("nodes")
                    return
                level[node] = sign * (d + 1)
                nxt.append(node)
            frontier, d = nxt, d + 1

    walk(parents_of, "parent_campaign_id", -1)
    walk(children_of, "child_campaign_id", +1)
    among = sorted(links_among(sorted(level)), key=lambda r: int(r["id"]))
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


def in_memory_hops(links: list[dict[str, Any]]) -> dict[str, Any]:
    """`parents_of` / `children_of` / `links_among` / `campaign_children` /
    `run_children` over a list of links (tests and small callers)."""
    cross = [r for r in links if r["parent_campaign_id"] != r["child_campaign_id"]]

    def among(ids: list[str]) -> list[dict[str, Any]]:
        s = set(ids)
        return [r for r in links if r["parent_campaign_id"] in s and r["child_campaign_id"] in s]

    return {
        "parents_of": lambda ids: [r for r in cross if r["child_campaign_id"] in set(ids)],
        "children_of": lambda ids: [r for r in cross if r["parent_campaign_id"] in set(ids)],
        "links_among": among,
        "run_children": lambda runs: [r for r in links if r["parent_campaign_id"] == r["child_campaign_id"]
                                      and r["parent_run_id"] in set(runs)],
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
