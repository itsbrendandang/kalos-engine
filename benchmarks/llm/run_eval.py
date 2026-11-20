"""Score local language models on Kalos's language tasks (benchmarks/llm/README.md).

    .venv/bin/python benchmarks/llm/run_eval.py                 # every config
    .venv/bin/python benchmarks/llm/run_eval.py --configs cascade --tasks assistant

Needs a local Ollama with the models pulled (`ollama list`). Runs every task
through the ENGINE's own code paths (the cascade, the validators, the offline
fallbacks), so what is scored is what a user would be shown. Writes
results/eval_<date>.json (every item, raw replies included) and prints the
markdown table that RESULTS.md records.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

os.environ.setdefault("KALOS_ASSIST_EMBED", "0")
os.environ.setdefault("KALOS_ASSIST_DOCS_DIR", str(ROOT / "docs"))

import llm_cases as cases  # noqa: E402
import numpy as np  # noqa: E402

from kalos.assist.chat import ask, split_sentences  # noqa: E402
from kalos.assist.index import Chunk, Hit, _bm25_scores, doc_chunks, tokens  # noqa: E402
from kalos.assist.llm import OllamaChat  # noqa: E402
from kalos.assist.setup_suggest import suggest  # noqa: E402
from kalos.assist.tasks import classify_intent, plain_wording  # noqa: E402
from kalos.normalize.llm import (  # noqa: E402
    _OLLAMA_JSON_INSTRUCTION,
    _SYSTEM_PROMPT,
    _GenerateTier,
    offline_plan,
    plan_validator,
)
from kalos.normalize.payload import build_payload  # noqa: E402
from kalos.providers.cascade import Cascade  # noqa: E402
from kalos.providers.grounding import number_keys  # noqa: E402
from kalos.providers.ollama_provider import ollama_url  # noqa: E402

CONFIGS: dict[str, tuple[str, str]] = {
    "offline": ("", ""),  # no model: the engine's own deterministic paths (the floor)
    "phi4-mini": ("phi4-mini", "phi4-mini"),
    "qwen2.5:7b": ("qwen2.5:7b", "qwen2.5:7b"),
    "qwen2.5:14b": ("qwen2.5:14b", "qwen2.5:14b"),
    "cascade": ("phi4-mini", "qwen2.5:7b"),
    "cascade-14b": ("phi4-mini", "qwen2.5:14b"),
}
TASKS = ("header_mapping", "setup", "assistant", "intent", "wording")
_NEGATION = re.compile(r"\b(no|not|cannot|can't|does not|doesn't|don't|isn't|unknown|none)\b", re.I)
_LABEL = re.compile(r"\[(S\d{1,2}|G)\]")


class Recorder:
    """Wraps a tier model and records every call (reply text, seconds)."""

    def __init__(self, tier: Any) -> None:
        self.tier, self.provider, self.model, self.local = tier, tier.provider, tier.model, True
        self.calls: list[dict[str, Any]] = []

    def _rec(self, fn: Any, system: str, prompt: str) -> str:
        t0 = time.perf_counter()
        try:
            reply = fn(system, prompt)
        except Exception as exc:
            self.calls.append({"model": self.model, "error": type(exc).__name__,
                               "seconds": time.perf_counter() - t0})
            raise
        self.calls.append({"model": self.model, "reply": str(reply)[:3000], "seconds": time.perf_counter() - t0})
        return str(reply)

    def complete(self, system: str, prompt: str) -> str:
        return self._rec(self.tier.complete, system, prompt)

    def complete_json(self, system: str, prompt: str) -> str:
        fn = getattr(self.tier, "complete_json", self.tier.complete)
        return self._rec(fn, system, prompt)


def make_cascade(fast: str, quality: str, *, generate: bool = False) -> tuple[Cascade | None, list[Recorder]]:
    url = ollama_url()
    if not fast:
        return None, []
    def tier(m: str) -> Any:
        return _GenerateTier(m, url) if generate else OllamaChat(model=m, url=url)
    if fast == quality:
        one = Recorder(tier(fast))
        return Cascade(fast=one, quality=one, escalate=False), [one]  # type: ignore[arg-type]
    f, q = Recorder(tier(fast)), Recorder(tier(quality))
    return Cascade(fast=f, quality=q), [f, q]  # type: ignore[arg-type]


def drain(recs: list[Recorder]) -> list[dict[str, Any]]:
    out = [c for r in recs for c in r.calls]
    for r in recs:
        r.calls = []
    return out


# --- retrieval over the fixture (the BM25 branch of AssistIndex.search) ---------- #


def load_fixture() -> dict[str, Any]:
    data = json.loads((HERE / "fixtures" / "assistant_campaign.json").read_text())
    data["chunk_objs"] = [Chunk(**{**c, "user_terms": tuple(c.get("user_terms") or ())}) for c in data["chunks"]]
    data["doc_objs"] = doc_chunks(sorted((ROOT / "docs").glob("*.md")))
    return data


def retrieve(fx: dict[str, Any], query: str, k: int = 6) -> list[Hit]:
    """`AssistIndex.search` without the file: same BM25, same 1.6x weight on
    the campaign's own records, same shared-word rule."""
    pool: list[Chunk] = fx["chunk_objs"] + fx["doc_objs"]
    q = tokens(query)
    corpus = [tokens(c.title + " " + c.text) for c in pool]
    scores = _bm25_scores(corpus, q) * np.asarray([1.0 if c.kind in ("doc", "skill") else 1.6 for c in pool])
    qset = set(q)
    return [Hit(pool[i], float(scores[i])) for i in np.argsort(-scores, kind="stable")
            if qset & set(corpus[i]) and scores[i] > 0][:k]


# --- tasks ------------------------------------------------------------------------ #


def _unit_ok(got: str | None, accepted: list[str | None]) -> bool:
    norm = None if got is None else got.strip().lower().replace("°", "").replace(" ", "") or None
    acc = {None if a is None else a.lower().replace("°", "").replace(" ", "") for a in accepted}
    return norm in acc


def run_header_mapping(cfg: tuple[str, str]) -> list[dict[str, Any]]:
    cascade, recs = make_cascade(*cfg, generate=True)
    items = []
    for sheet in cases.sheets():
        df = sheet["df"]
        payload, dropped = build_payload(df, max_sample=5)
        validate = plan_validator(df, payload, dropped)
        t0 = time.perf_counter()
        if cascade is None:
            from kalos.providers.cascade import CascadeResult

            result = CascadeResult(offline_plan(df), "offline", [], offline=True)
        else:
            result = cascade.run(
            "header_mapping", f"{_SYSTEM_PROMPT}\n\n{_OLLAMA_JSON_INSTRUCTION}",
            f"Input payload:\n{json.dumps(payload, sort_keys=True)}", validate,
            lambda df=df: offline_plan(df), json_mode=True,
        )
        seconds = time.perf_counter() - t0
        plan = {c.raw_name: c for c in result.value.columns}
        roles = units = roles_ok = units_ok = 0
        for col, exp in sheet["columns"].items():
            roles += 1
            roles_ok += plan[col].role in exp["norm"]
            if "unit" in exp:
                units += 1
                units_ok += _unit_ok(plan[col].unit_token, exp["unit"])
        shown_notes = " ".join(c.note for c in result.value.columns)
        allowed = number_keys(json.dumps(payload))
        items.append({
            "sheet": sheet["name"], "path": result.provider_path, "seconds": seconds,
            "roles": roles, "roles_ok": roles_ok, "units": units, "units_ok": units_ok,
            "numbers_shown_unsupported": 0 if result.offline else len(number_keys(shown_notes) - allowed),
            "attempts": result.report()["attempts"], "calls": drain(recs),
        })
    return items


def run_setup(cfg: tuple[str, str]) -> list[dict[str, Any]]:
    cascade, recs = make_cascade(*cfg)
    items = []
    for sheet in cases.sheets():
        t0 = time.perf_counter()
        out = suggest(sheet["df"], target=sheet["target"], cascade=cascade)
        seconds = time.perf_counter() - t0
        by = {s["column"]: s for s in out["suggestions"]}
        roles = units = roles_ok = units_ok = 0
        for col, exp in sheet["columns"].items():
            if col == sheet["target"] or col not in by:
                continue  # the target's role is the user's choice, not the model's
            roles += 1
            roles_ok += by[col]["role"] in exp["setup"]
            if "unit" in exp:
                units += 1
                units_ok += _unit_ok(by[col]["unit"], exp["unit"])
        from kalos.assist.setup_suggest import column_summaries

        summaries, _ = column_summaries(sheet["df"])
        allowed = number_keys(json.dumps(summaries, default=str))
        shown = " ".join(s["reason"] for s in out["suggestions"] if s["reason_source"] == "model")
        items.append({
            "sheet": sheet["name"], "path": out["provider_path"], "seconds": seconds,
            "roles": roles, "roles_ok": roles_ok, "units": units, "units_ok": units_ok,
            "numbers_shown_unsupported": len(number_keys(shown) - allowed - number_keys(" ".join(by))),
            "attempts": out["attempts"], "calls": drain(recs),
        })
    return items


def _raw_answer_stats(reply: str, labelled_texts: dict[str, str], guide_text: str) -> dict[str, int]:
    sentences = split_sentences(reply)
    valid = 0
    unsupported_numbers = 0
    for s in sentences:
        labels = _LABEL.findall(s)
        ok = bool(labels) and all(lab == "G" or lab in labelled_texts for lab in labels)
        valid += ok
        cited = " ".join(labelled_texts.get(lab, "") for lab in labels) + (guide_text if "G" in labels else "")
        from kalos.assist.chat import _value_numbers

        unsupported_numbers += len(_value_numbers(_LABEL.sub("", s)) - number_keys(cited))
    return {"sentences": len(sentences), "valid_citations": valid, "numbers_unsupported": unsupported_numbers}


def run_assistant(cfg: tuple[str, str], fx: dict[str, Any]) -> list[dict[str, Any]]:
    cascade, recs = make_cascade(*cfg)
    guide = fx["guide"]
    guide_src = " ".join([guide.get("step", ""), guide.get("why", ""), guide.get("why_it_matters", "")])
    known_ids = {(c.kind, c.source_id) for c in fx["chunk_objs"] + fx["doc_objs"]}
    items = []
    for case in cases.QUESTIONS:
        hits = retrieve(fx, case["q"])
        t0 = time.perf_counter()
        try:
            if cascade is None:
                from kalos.assist.chat import abstain

                out = abstain(guide)
            else:
                out = ask(case["q"], hits, guide, cascade)
        except Exception as exc:  # noqa: BLE001 - recorded as unavailable
            out = {"answer": "", "sources": [], "abstained": True, "provider_path": f"error:{type(exc).__name__}"}
        seconds = time.perf_counter() - t0
        answer = out["answer"]
        if case.get("refuse"):
            # Refused: abstained, or opened by saying the record does not hold it.
            first = (split_sentences(answer) or [""])[0]
            correct = bool(out["abstained"] or _NEGATION.search(first))
        else:
            low = answer.lower()
            correct = (
                (not out["abstained"] or bool(case.get("guide_ok")))
                and all(any(alt.lower() in low for alt in group) for group in case["expect"])
                and not any(f.lower() in low for f in case.get("forbid", []))
            )
        shown_texts = " ".join(h.chunk.text for h in hits) + " " + guide_src
        numbers_shown = 0 if out["abstained"] else len(
            number_keys(re.sub(r"(?:\b(?:recipe|run|round|step)\s*|#)(10|[1-9])(?![\d.,eE])", " ", answer, flags=re.I))
            - number_keys(shown_texts))
        labelled = {f"S{i + 1}": h.chunk.text for i, h in enumerate(hits)}
        calls = drain(recs)
        raw = [_raw_answer_stats(c.get("reply", ""), labelled, guide_src) for c in calls if "reply" in c]
        items.append({
            "q": case["q"], "refuse": bool(case.get("refuse")), "correct": correct,
            "abstained": out["abstained"], "path": out.get("provider_path"), "seconds": seconds,
            "answer": answer[:600], "n_hits": len(hits),
            "sources_valid": all((s["kind"], s["id"]) in known_ids for s in out["sources"]),
            "n_sources": len(out["sources"]),
            "numbers_shown_unsupported": numbers_shown,
            "raw": raw, "calls": calls,
        })
    return items


def run_intent(cfg: tuple[str, str]) -> list[dict[str, Any]]:
    cascade, recs = make_cascade(*cfg)
    items = []
    for q, want in cases.INTENTS:
        t0 = time.perf_counter()
        r = classify_intent(q, cascade)
        items.append({"q": q, "want": want, "got": r.value, "correct": r.value == want, "path": r.provider_path,
                      "seconds": time.perf_counter() - t0, "calls": drain(recs)})
    return items


def run_wording(cfg: tuple[str, str]) -> list[dict[str, Any]]:
    cascade, recs = make_cascade(*cfg)
    items = []
    for case in cases.WORDING:
        t0 = time.perf_counter()
        r = plain_wording(case["message"], cascade, columns=case["columns"])
        items.append({
            "message": case["message"], "shown": r.value, "accepted": not r.offline, "path": r.provider_path,
            "numbers_shown_unsupported": len(number_keys(r.value) - number_keys(case["message"])),
            "seconds": time.perf_counter() - t0, "calls": drain(recs),
        })
    return items


# --- scoring ---------------------------------------------------------------------- #


def _pct(a: float, b: float) -> float | None:
    return None if b == 0 else round(100.0 * a / b, 1)


def _p(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))], 2)


def summarize(res: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    secs: list[float] = []
    shown_bad = 0
    raw_numbers = 0
    if "header_mapping" in res:
        it = res["header_mapping"]
        out["header_role_acc"] = _pct(sum(i["roles_ok"] for i in it), sum(i["roles"] for i in it))
        out["header_unit_acc"] = _pct(sum(i["units_ok"] for i in it), sum(i["units"] for i in it))
        out["header_offline"] = sum("offline" in i["path"] for i in it)
        shown_bad += sum(i["numbers_shown_unsupported"] for i in it)
        raw_numbers += sum(a["reason"] == "mismatched_number" for i in it for a in i["attempts"])
        secs += [i["seconds"] for i in it]
    if "setup" in res:
        it = res["setup"]
        out["setup_role_acc"] = _pct(sum(i["roles_ok"] for i in it), sum(i["roles"] for i in it))
        out["setup_unit_acc"] = _pct(sum(i["units_ok"] for i in it), sum(i["units"] for i in it))
        out["setup_offline"] = sum("offline" in i["path"] for i in it)
        shown_bad += sum(i["numbers_shown_unsupported"] for i in it)
        raw_numbers += sum(a["reason"] == "mismatched_number" for i in it for a in i["attempts"])
        secs += [i["seconds"] for i in it]
    if "assistant" in res:
        it = res["assistant"]
        ans = [i for i in it if not i["refuse"]]
        ref = [i for i in it if i["refuse"]]
        out["answer_acc"] = _pct(sum(i["correct"] for i in ans), len(ans))
        out["correct_refusals"] = _pct(sum(i["correct"] for i in ref), len(ref))
        raw = [r for i in it for r in i["raw"]]
        out["citation_valid_raw"] = _pct(sum(r["valid_citations"] for r in raw), sum(r["sentences"] for r in raw))
        out["citation_valid_shown"] = _pct(sum(i["sources_valid"] for i in it if i["n_sources"]),
                                           sum(1 for i in it if i["n_sources"]))
        raw_numbers += sum(r["numbers_unsupported"] for r in raw)
        shown_bad += sum(i["numbers_shown_unsupported"] for i in it)
        out["answer_p50_s"] = _p([i["seconds"] for i in it], 0.5)
        out["answer_p95_s"] = _p([i["seconds"] for i in it], 0.95)
        secs += [i["seconds"] for i in it]
    if "intent" in res:
        it = res["intent"]
        out["intent_acc"] = _pct(sum(i["correct"] for i in it), len(it))
        secs += [i["seconds"] for i in it]
    if "wording" in res:
        it = res["wording"]
        out["wording_accepted"] = _pct(sum(i["accepted"] for i in it), len(it))
        shown_bad += sum(i["numbers_shown_unsupported"] for i in it)
        secs += [i["seconds"] for i in it]
    out["hallucinated_numbers_raw"] = raw_numbers
    out["hallucinated_numbers_shown"] = shown_bad
    out["latency_p50_s"] = _p(secs, 0.5)
    out["latency_p95_s"] = _p(secs, 0.95)
    out["total_s"] = round(sum(secs), 1)
    parts = [out.get(k) for k in ("header_role_acc", "setup_role_acc", "answer_acc", "correct_refusals", "intent_acc")]
    parts = [p for p in parts if p is not None]
    out["score"] = round(statistics.fmean(parts), 1) if parts else None
    return out


def warm(models: set[str]) -> dict[str, float]:
    """Load each model once (cold start is reported, not counted)."""
    out = {}
    for m in sorted(models):
        t0 = time.perf_counter()
        try:
            OllamaChat(model=m, url=ollama_url()).complete("Reply with OK.", "OK?")
        except Exception as exc:  # noqa: BLE001
            print(f"warm-up failed for {m}: {exc}", file=sys.stderr)
        out[m] = round(time.perf_counter() - t0, 2)
    return out


def machine() -> dict[str, Any]:
    def sh(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()  # noqa: S603
        except Exception:  # noqa: BLE001
            return ""
    return {
        "platform": platform.platform(),
        "chip": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_gb": round(int(sh(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30, 1),
        "ollama": sh(["ollama", "--version"]),
        "git": sh(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]),
    }


def table(summary: dict[str, dict[str, Any]]) -> str:
    cols = [
        ("score", "Score"), ("header_role_acc", "Header roles %"), ("header_unit_acc", "Header units %"),
        ("setup_role_acc", "Setup roles %"), ("setup_unit_acc", "Setup units %"),
        ("answer_acc", "Answers %"), ("correct_refusals", "Refusals %"), ("intent_acc", "Intent %"),
        ("wording_accepted", "Wording kept %"), ("citation_valid_raw", "Citations valid raw %"),
        ("citation_valid_shown", "Citations valid shown %"), ("hallucinated_numbers_raw", "Bad numbers raw"),
        ("hallucinated_numbers_shown", "Bad numbers shown"), ("answer_p50_s", "Answer p50 s"),
        ("latency_p50_s", "All p50 s"), ("latency_p95_s", "All p95 s"), ("total_s", "Total s"),
    ]
    lines = ["| Config | " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for name, s in summary.items():
        lines.append(f"| {name} | " + " | ".join("n/a" if s.get(k) is None else str(s[k]) for k, _ in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    fx = load_fixture()
    started = datetime.now(UTC)
    cold = warm({m for c in args.configs for m in CONFIGS[c] if m})
    results: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    for name in args.configs:
        cfg = CONFIGS[name]
        warm({m for m in cfg if m})  # this config's models resident before timing
        res: dict[str, list[dict[str, Any]]] = {}
        for task in args.tasks:
            print(f"[{name}] {task} ...", file=sys.stderr, flush=True)
            if task == "header_mapping":
                res[task] = run_header_mapping(cfg)
            elif task == "setup":
                res[task] = run_setup(cfg)
            elif task == "assistant":
                res[task] = run_assistant(cfg, fx)
            elif task == "intent":
                res[task] = run_intent(cfg)
            elif task == "wording":
                res[task] = run_wording(cfg)
        results[name] = res
        summary[name] = summarize(res)
        print(json.dumps({name: summary[name]}), file=sys.stderr, flush=True)
    report = {
        "started": started.strftime("%Y-%m-%dT%H:%M:%SZ"), "machine": machine(), "cold_load_s": cold,
        "configs": {k: {"fast": CONFIGS[k][0], "quality": CONFIGS[k][1]} for k in args.configs},
        "tasks": args.tasks, "summary": summary, "items": results,
    }
    out = Path(args.out) if args.out else HERE / "results" / f"eval_{started.strftime('%Y%m%d')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(table(summary))
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
