"""Score observations against the truth key (PROTOCOL.md section 5).

Inputs: `evaluator/truth.json` and `observations/<tier>/`. Output: a scorecard
dict (written as JSON) and a markdown rendering. Also the CI regression check
against a stored baseline.
"""
from __future__ import annotations

import json
import math
import platform
import re
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .bases import REPO_ROOT
from .cases import load_truth
from .catalogue import (
    ANY_CODE,
    FAULT_COLUMN,
    PROTOCOL_VERSION,
    SEVERITY_RANK,
    SUITE_SEED,
    Matcher,
    catch_detectors,
    clean_by_id,
    fault_by_id,
)
from .runner import load_observations
from .signals import Signal, keys

OUTCOME_RANK = {"missed": 0, "noticed": 1, "caught": 2}
Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> list[float] | None:
    """Wilson score interval for k successes in n trials (None when n == 0)."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def matches(m: Matcher, s: Signal, *, columns: Iterable[str] = (), rows: Iterable[int] = ()) -> bool:
    if s["source"] != m.source:
        return False
    if m.codes != (ANY_CODE,) and s["code"] not in m.codes:
        return False
    if SEVERITY_RANK.get(s["severity"], 0) < SEVERITY_RANK[m.min_severity]:
        return False
    if m.column == FAULT_COLUMN:
        if s["column"] is None or s["column"] not in set(columns):
            return False
    elif m.column is not None and s["column"] != m.column:
        return False
    if m.rows and not set(s["rows"]) & set(rows):
        return False
    return not (m.message and not re.search(m.message, s["message"], re.IGNORECASE))


def _brief(s: Signal) -> str:
    col = f" [{s['column']}]" if s["column"] else ""
    return f"{s['source']}:{s['code']}{col} ({s['severity']})"


def _score_fault(rec: Mapping[str, Any], obs: Mapping[str, Any], parent: Mapping[str, Any] | None) -> dict[str, Any]:
    fault = fault_by_id(rec["id"])
    cols, rows = rec.get("columns") or [], rec.get("rows") or []
    signals: list[Signal] = list(obs.get("signals") or [])
    caught = [(m, s) for m in fault.caught_if for s in signals if matches(m, s, columns=cols, rows=rows)]
    harm = [_brief(s) + (f": {s['message']}" if s["message"] else "")
            for m in fault.harm_if for s in signals if matches(m, s, columns=cols, rows=rows)]
    noticed: list[str] = []
    truncated = False
    if not caught:
        noticed += [_brief(s) for m in fault.noticed_if for s in signals
                    if matches(m, s, columns=cols, rows=rows)]
        parent_keys = keys(parent.get("signals") or []) if parent else set()
        for s in signals:
            if SEVERITY_RANK.get(s["severity"], 0) < 1:
                continue
            if (s["source"], s["code"], s["column"]) not in parent_keys:
                noticed.append(_brief(s) + " (not on the clean parent)")
            elif set(s["rows"]) & set(rows):
                noticed.append(_brief(s) + " (names a planted row)")
        # A matcher that failed only because the engine's row list was capped.
        for m in fault.caught_if:
            for s in signals:
                if m.rows and s.get("n_rows", 0) > len(s["rows"]) and matches(
                    Matcher(m.source, m.codes, m.column, False, m.min_severity, m.message), s,
                    columns=cols, rows=rows,
                ):
                    truncated = True
    status = obs.get("status", "missing")
    if status in ("error", "missing", "not_run"):
        outcome = "missed"
    else:
        outcome = "caught" if caught else ("noticed" if noticed else "missed")
    return {
        "case": rec["case"], "id": rec["id"], "kind": "fault", "family": fault.family,
        "title": fault.title, "surface": fault.surface, "status": status,
        "error": obs.get("error"), "seconds": obs.get("seconds"),
        "outcome": outcome,
        "caught_by": sorted({_brief(s) + (" [planned code]" if m.planned else "") for m, s in caught}),
        "noticed_by": sorted(set(noticed)),
        "harm": sorted(set(harm)),
        "row_list_truncated": truncated,
        "expected": fault.expected, "detectable_from_input": fault.detectable_from_input,
        "owner_module": fault.owner_module, "owner_plan": fault.owner_plan,
        "summary": obs.get("summary"),
    }


def _score_clean(rec: Mapping[str, Any], obs: Mapping[str, Any]) -> dict[str, Any]:
    clean = clean_by_id(rec["id"])
    signals: list[Signal] = list(obs.get("signals") or [])
    status = obs.get("status", "missing")
    watch = rec.get("watch_rows")
    fired: list[str] = []
    for det in catch_detectors():
        for s in signals:
            if not matches(det, s):
                continue
            if watch is not None and not (set(s["rows"]) & set(watch)):
                continue
            fired.append(f"{det.source}:{'|'.join(det.codes)}")
    refused = [_brief(s) for s in signals if s["source"] == "refusal"]
    det_keys = {(d.source, c) for d in catch_detectors() for c in d.codes}
    standing = sorted({
        f"{s['source']}:{s['code']}" for s in signals
        if SEVERITY_RANK.get(s["severity"], 0) >= 1 and (s["source"], s["code"]) not in det_keys
        and s["source"] != "refusal"
    })
    false_alarm = bool(fired or refused or status == "error")
    return {
        "case": rec["case"], "id": rec["id"], "kind": "clean", "surface": clean.surface,
        "status": status, "error": obs.get("error"), "seconds": obs.get("seconds"),
        "false_alarm": false_alarm, "detectors_fired": sorted(set(fired)), "refusals": refused,
        "standing_warnings": standing, "expected": clean.expected, "summary": obs.get("summary"),
    }


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(  # noqa: S603 - fixed git argv, no shell
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=10,  # noqa: S607
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "torch", "botorch", "gpytorch"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def _rate(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "wilson95": wilson(k, n)}


def score_suite(out: Path, tier: str) -> dict[str, Any]:
    truth = load_truth(out)["cases"]
    observations = load_observations(out, tier)
    faults: list[dict[str, Any]] = []
    cleans: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for case, rec in sorted(truth.items()):
        if tier not in rec["tiers"]:
            continue
        if rec.get("skipped"):
            skipped.append({"case": case, "id": rec["id"], "reason": rec["skipped"]})
            continue
        obs = observations.get(case, {"status": "missing", "signals": []})
        if rec["kind"] == "fault":
            parent = observations.get(rec["parent_case"])
            faults.append(_score_fault(rec, obs, parent))
        else:
            cleans.append(_score_clean(rec, obs))
    counts = Counter(f["outcome"] for f in faults)
    detectable = [f for f in faults if f["detectable_from_input"]]
    by_family: dict[str, dict[str, int]] = {}
    for f in faults:
        fam = by_family.setdefault(f["family"], {"caught": 0, "noticed": 0, "missed": 0})
        fam[f["outcome"]] += 1
    per_detector = Counter(d for c in cleans for d in c["detectors_fired"])
    n_fa = sum(c["false_alarm"] for c in cleans)
    return {
        "suite": "kalos planted-fault honesty suite",
        "label": "synthetic planted faults",
        "protocol_version": PROTOCOL_VERSION, "suite_seed": SUITE_SEED, "tier": tier,
        "commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions": _versions(),
        "totals": {
            "faults_scored": len(faults), "caught": counts["caught"], "noticed": counts["noticed"],
            "missed": counts["missed"], "clean_scored": len(cleans), "clean_false_alarm": n_fa,
            "skipped": len(skipped),
            "crashed": sum(1 for x in faults + cleans if x["status"] == "error"),
        },
        "rates": {
            "caught": _rate(counts["caught"], len(faults)),
            "caught_or_noticed": _rate(counts["caught"] + counts["noticed"], len(faults)),
            "caught_detectable": _rate(sum(f["outcome"] == "caught" for f in detectable), len(detectable)),
            "clean_false_alarm": _rate(n_fa, len(cleans)),
        },
        "by_family": by_family,
        "false_alarms_by_detector": dict(sorted(per_detector.items())),
        "faults": sorted(faults, key=lambda f: (f["family"], f["id"])),
        "clean": sorted(cleans, key=lambda c: c["id"]),
        "skipped": skipped,
    }


def _fmt_rate(r: Mapping[str, Any]) -> str:
    if not r["n"]:
        return "n/a"
    lo, hi = r["wilson95"]
    return f"{r['k']}/{r['n']} ({r['rate']:.0%}; Wilson 95% {lo:.2f}-{hi:.2f})"


def _cell(items: Sequence[str], limit: int = 3) -> str:
    if not items:
        return ""
    shown = "; ".join(items[:limit]) + (f"; +{len(items) - limit} more" if len(items) > limit else "")
    return shown.replace("|", "/")


def to_markdown(card: Mapping[str, Any]) -> str:
    t, r = card["totals"], card["rates"]
    lines = [
        f"# Planted-fault scorecard ({card['tier']} tier): {card['label']}",
        "",
        f"Protocol v{card['protocol_version']}, seed {card['suite_seed']}, commit "
        f"`{(card.get('commit') or 'unknown')[:10]}` on `{card.get('branch')}`, {card['generated_at']}.",
        "",
        f"- Caught: {_fmt_rate(r['caught'])}",
        f"- Caught or noticed: {_fmt_rate(r['caught_or_noticed'])}",
        f"- Caught, faults detectable from the input: {_fmt_rate(r['caught_detectable'])}",
        f"- Clean cases with a false alarm: {_fmt_rate(r['clean_false_alarm'])}",
        f"- Missed: {t['missed']}; crashed: {t['crashed']}; skipped (base unavailable): {t['skipped']}",
        "",
        "## Faults",
        "",
        "| Fault | Family | Outcome | Evidence | Harm on a miss | Owner (module / plan) |",
        "|---|---|---|---|---|---|",
    ]
    for f in card["faults"]:
        evidence = f["caught_by"] or f["noticed_by"]
        if f["status"] == "error":
            evidence = [f"crashed: {f['error']}"]
        if f["row_list_truncated"]:
            evidence = [*evidence, "right reason fired but its 20-row list omits the planted rows"]
        owner = "" if f["outcome"] == "caught" else f"{f['owner_module']} / {f['owner_plan']}"
        outcome = f["outcome"] + ("" if f["detectable_from_input"] else " (blind spot)")
        lines.append(
            f"| `{f['id']}` | {f['family']} | **{outcome}** | {_cell(evidence)} | "
            f"{_cell(f['harm'], 2)} | {owner.replace('|', '/')} |"
        )
    lines += [
        "",
        "## Clean cases (false alarms)",
        "",
        "| Clean case | Surface | False alarm | Detectors fired / refusals | Standing warnings (not counted) |",
        "|---|---|---|---|---|",
    ]
    for c in card["clean"]:
        fired = c["detectors_fired"] + c["refusals"] + ([f"crashed: {c['error']}"] if c["status"] == "error" else [])
        lines.append(
            f"| `{c['id']}` | {c['surface']} | {'**yes**' if c['false_alarm'] else 'no'} | "
            f"{_cell(fired)} | {_cell(c['standing_warnings'], 4)} |"
        )
    if card["skipped"]:
        lines += ["", "Skipped: " + ", ".join(f"`{s['id']}` ({s['reason']})" for s in card["skipped"])]
    lines += ["", "Synthetic planted faults only; see PROTOCOL.md section 8 for what this does not claim.", ""]
    return "\n".join(lines)


def write_scorecard(out: Path, tier: str, card: Mapping[str, Any]) -> tuple[Path, Path]:
    out = Path(out)
    jpath, mpath = out / f"scorecard_{tier}.json", out / f"scorecard_{tier}.md"
    jpath.write_text(json.dumps(card, indent=1, sort_keys=False, default=str) + "\n", encoding="utf-8")
    mpath.write_text(to_markdown(card), encoding="utf-8")
    return jpath, mpath


def gate_view(card: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a scorecard the CI gate compares (stable across machines)."""
    return {
        "protocol_version": card["protocol_version"], "tier": card["tier"],
        "faults": {f["id"]: f["outcome"] for f in card["faults"]},
        "clean": {c["id"]: sorted(c["detectors_fired"] + c["refusals"]) for c in card["clean"]},
    }


def regressions(baseline: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """What got worse: a fault outcome that dropped, or a clean case that gained
    a false-alarm detector. Cases missing from either side are not compared."""
    problems = []
    for fid, was in baseline["faults"].items():
        now = current["faults"].get(fid)
        if now is not None and OUTCOME_RANK[now] < OUTCOME_RANK[was]:
            problems.append(f"{fid}: {was} -> {now}")
    for cid, was_fired in baseline["clean"].items():
        now_fired = current["clean"].get(cid)
        if now_fired is None:
            continue
        new = sorted(set(now_fired) - set(was_fired))
        if new:
            problems.append(f"{cid}: new false alarm(s) {new}")
    return problems


__all__ = ["gate_view", "matches", "regressions", "score_suite", "to_markdown", "wilson", "write_scorecard"]
