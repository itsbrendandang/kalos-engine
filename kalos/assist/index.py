"""A local search index over one tenant's own data, plus the docs.

What goes in (each chunk keeps its source kind and id, so an answer can cite it):
  - `sheet`: the sheet a campaign analysed (target, inputs, row counts, columns
    left out). id = the sheet's input sha256 (the campaign id when unknown).
  - `finding`: each sheet check finding. id = its code, `.n` when repeated.
  - `round`: each round's started runs with their recipe and recorded result,
    and the best measured value per round. id = the round number.
  - `analysis`: the model checks, the cross-check numbers, the noise estimate,
    the drivers, the proposed recipes (predicted values labelled as predicted),
    and the latest refusal. id = the section key.
  - `doc`: docs/*.md split by heading. id = `docs/<file>.md`.
  - `skill`: SKILL.md files, when a skills folder exists. id = the slug.

Retrieval is local only. With `sentence-transformers` installed AND its model
already on disk (`KALOS_ASSIST_EMBED_MODEL`, loaded with local_files_only, never
downloaded), chunks are ranked by embedding similarity; otherwise by BM25
(`rank_bm25` when installed, else the same Okapi formula built in), so it
always works offline.

Stored under `<state dir>/assist/<tenant key>/index.json` (+ `vectors.npz`).
The index is grouped (one group per campaign, one for docs, one for skills),
each with a fingerprint; `refresh` rebuilds only groups whose fingerprint
changed, which is how uploads and recorded results update it incrementally.
The index is derived data: deleting it loses nothing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from kalos.assist.paths import tenant_dir
from kalos.assist.retrieval_synonyms import expand_query_tokens

log = logging.getLogger("kalos.assist")

INDEX_VERSION = 2
MAX_CHUNK_CHARS = 1200
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or", "our", "so", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "was", "we", "were", "what", "when", "where", "which", "who", "why", "with", "you", "your", "should", "would", "could", "about", "any", "all"]
)


def tokens(text: str) -> list[str]:
    """Lower-case word tokens with stop words removed; `a_b` also yields `a`, `b`."""
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower().replace("_", " _ ").replace("-", " ")):
        if tok not in _STOP and tok != "_":
            out.append(tok)
    return out


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of text and where it came from."""

    chunk_id: str  # stable: hash of kind, source id and text
    kind: str  # sheet | finding | round | analysis | doc | skill
    source_id: str  # the id a source reference carries (see module docstring)
    campaign_id: str | None
    title: str
    text: str
    # Strings in `text` the lab wrote (campaign name, column names, labels). A
    # claim word inside one of them never supports a claim in an answer.
    user_terms: tuple[str, ...] = ()
    # False for text the lab or a third party wrote as a whole (a tenant skills
    # folder): it can be quoted, but supports no claim word.
    trusted_text: bool = True

    def kalos_text(self) -> str:
        """The chunk text with every lab-written string removed: what Kalos
        itself wrote. Claim words are checked against this only."""
        if not self.trusted_text:
            return ""
        return strip_terms(self.text, self.user_terms)


def strip_terms(text: str, terms: Iterable[str], *, keep_claims: bool = False) -> str:
    """`text` with each of `terms` removed as a whole word or phrase
    (case-insensitive, longest first). A one-letter label such as 'A' removes
    only a standalone 'A', never the letter inside other words."""
    from kalos.assist.words import claims_in

    for term in sorted({t.strip() for t in terms if t and t.strip()}, key=len, reverse=True):
        if keep_claims and claims_in(term) == [re.sub(r"\s+", " ", term.lower())]:
            # In an ANSWER, a lab term that IS a claim word ("increases", "leads
            # to") stays, so the claim word is still checked. A longer name that
            # merely contains one is a mention and is removed. (In a source every
            # lab term is removed: lab text supports nothing.)
            continue
        text = re.sub(r"(?<![\w])" + re.escape(term) + r"(?![\w])", " ", text, flags=re.IGNORECASE)
    return text


def _user_terms(info: dict[str, Any], state: dict[str, Any] | None, a: dict[str, Any]) -> tuple[str, ...]:
    """Every string in one campaign's record that the lab wrote."""
    terms: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            terms.add(value.strip()[:200])

    add(info.get("name"))
    add(a.get("dataset"))
    add(a.get("target"))
    add((state or {}).get("target"))
    for key in ("features", "categorical_features", "targets", "auto_categorical"):
        for v in a.get(key) or []:
            add(v)
    for v in (state or {}).get("features") or []:
        add(v)
    columns = a.get("columns") or a.get("column_provenance") or a.get("provenance") or []
    for col in columns if isinstance(columns, list) else []:
        if isinstance(col, dict):
            add(col.get("name"))
    for f in ((a.get("validation") or {}).get("findings") or []):
        if isinstance(f, dict):
            add(f.get("column"))
    recipes = [p.get("recipe") for p in a.get("proposals") or [] if isinstance(p, dict)]
    recipes += [r.get("recipe") for r in (state or {}).get("pending") or [] if isinstance(r, dict)]
    for row in (state or {}).get("base_rows") or []:
        recipes.append(row)
    for recipe in recipes:
        if isinstance(recipe, dict):
            for k, v in recipe.items():
                add(k)
                add(v)
    return tuple(sorted(terms))


def _chunk(kind: str, source_id: str, campaign_id: str | None, title: str, text: str) -> Chunk:
    digest = hashlib.sha256(f"{kind}\x00{source_id}\x00{campaign_id}\x00{text}".encode()).hexdigest()
    return Chunk("c" + digest[:12], kind, source_id, campaign_id, title, text.strip())


# --- embeddings (optional, local only) ----------------------------------------- #


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """L2-normalised vectors, one row per text."""
        ...


class _SentenceTransformerEmbedder:
    def __init__(self, name: str) -> None:
        from sentence_transformers import SentenceTransformer  # optional extra

        self.name = name
        self._model = SentenceTransformer(name, local_files_only=True)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER: dict[str, Embedder | None] = {}


def load_embedder() -> Embedder | None:
    """The local embedding model, or None (not installed, not on disk, or turned
    off with KALOS_ASSIST_EMBED=0). Never downloads anything. Cached per process."""
    if (os.environ.get("KALOS_ASSIST_EMBED") or "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    name = (os.environ.get("KALOS_ASSIST_EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2").strip()
    with _EMBEDDER_LOCK:
        if name not in _EMBEDDER:
            try:
                _EMBEDDER[name] = _SentenceTransformerEmbedder(name)
            except Exception:  # noqa: BLE001 - absent package/model means BM25
                log.info("assist: no local embedding model available; using BM25")
                _EMBEDDER[name] = None
        return _EMBEDDER[name]


# --- BM25 ----------------------------------------------------------------------- #


def _bm25_scores(corpus: list[list[str]], query: list[str]) -> np.ndarray:
    """Okapi BM25 scores of `query` against each tokenised document."""
    if not corpus:
        return np.zeros(0)
    try:
        from rank_bm25 import BM25Okapi  # the `assist` extra

        return np.asarray(BM25Okapi(corpus).get_scores(query), dtype=float)
    except ImportError:
        pass
    k1, b = 1.5, 0.75
    n = len(corpus)
    avgdl = sum(len(d) for d in corpus) / n or 1.0
    df: Counter[str] = Counter()
    for doc in corpus:
        df.update(set(doc))
    scores = np.zeros(n)
    for i, doc in enumerate(corpus):
        tf = Counter(doc)
        for term in query:
            if term not in tf:
                continue
            idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            f = tf[term]
            scores[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(doc) / avgdl))
    return scores


# --- what goes in ---------------------------------------------------------------- #


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _recipe_text(recipe: dict[str, Any]) -> str:
    return ", ".join(f"{k} = {_fmt(v)}" for k, v in recipe.items())


def input_sha256(analysis: dict[str, Any] | None) -> str | None:
    """The analysed sheet's sha256: `run.input_sha256` (the run record), else
    `proposal_provenance.input_sha256`. (`provenance` is the per-column list.)"""
    for key in ("run", "proposal_provenance", "provenance"):
        block = (analysis or {}).get(key)
        sha = block.get("input_sha256") if isinstance(block, dict) else None
        if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha):
            return sha
    return None


def campaign_chunks(
    info: dict[str, Any],
    state: dict[str, Any] | None,
    analysis: dict[str, Any] | None,
    refusal: dict[str, Any] | None = None,
) -> list[Chunk]:
    """The chunks for one campaign, from its record only, each carrying the
    lab-written strings it may contain (`Chunk.user_terms`)."""
    from dataclasses import replace

    terms = _user_terms(info, state, analysis or {})
    return [replace(c, user_terms=terms) for c in _campaign_chunks(info, state, analysis, refusal)]


def _campaign_chunks(
    info: dict[str, Any],
    state: dict[str, Any] | None,
    analysis: dict[str, Any] | None,
    refusal: dict[str, Any] | None = None,
) -> list[Chunk]:
    cid = str(info["id"])
    name = str(info.get("name") or "campaign")
    out: list[Chunk] = []
    a = analysis or {}
    target = a.get("target") or (state or {}).get("target")

    if a:
        sha = input_sha256(a) or cid
        lines = [f"Run sheet for campaign {name!r}" + (f" ({a.get('dataset')})" if a.get("dataset") else "") + "."]
        if target:
            lines.append(f"Target (the result Kalos aims to raise): {target}.")
        if a.get("features"):
            lines.append("Inputs used by the model: " + ", ".join(map(str, a["features"])) + ".")
        if a.get("categorical_features"):
            lines.append("Inputs read as labels: " + ", ".join(map(str, a["categorical_features"])) + ".")
        if a.get("n") is not None:
            lines.append(f"Runs the model was fit on: {a['n']}.")
        if a.get("n_dropped_incomplete"):
            lines.append(f"Runs left out for blank or unreadable cells: {a['n_dropped_incomplete']}.")
        columns = a.get("columns") or a.get("column_provenance") or a.get("provenance") or []
        for col in columns if isinstance(columns, list) else []:
            if isinstance(col, dict) and col.get("name"):
                lines.append(f"Column {col['name']!r}: {col.get('status') or col.get('role') or col.get('kind')}.")
        out.append(_chunk("sheet", str(sha), cid, f"Sheet: {name}", "\n".join(lines)[:MAX_CHUNK_CHARS * 2]))

        seen: Counter[str] = Counter()
        for f in ((a.get("validation") or {}).get("findings") or []):
            if not isinstance(f, dict):
                continue
            code = str(f.get("code") or f.get("check") or "finding")
            seen[code] += 1
            fid = code if seen[code] == 1 else f"{code}.{seen[code]}"
            col = f" on column {f['column']!r}" if f.get("column") else ""
            text = f"Sheet check {code} ({f.get('severity', 'info')}){col}: {f.get('message', '')}"
            out.append(_chunk("finding", fid, cid, f"Finding: {code}", text))

        promo = a.get("promotion")
        if isinstance(promo, dict):
            verdict = "passed" if promo.get("passed") else "not passed"
            parts = [f"Model checks for campaign {name!r}: {verdict}."]
            for fail in promo.get("failures") or []:
                parts.append(f"Check not passed: {fail}.")
            for nj in promo.get("not_judged") or []:
                parts.append(f"Not judged: {nj}.")
            if promo.get("meaning"):
                parts.append(str(promo["meaning"]))
            out.append(_chunk("analysis", "promotion", cid, "Model checks", " ".join(parts)))
        if a.get("cv_spearman") is not None:
            ci = a.get("cv_ci95")
            text = (f"Cross-check of the model on runs it was not shown (campaign {name!r}): "
                    f"rank agreement {a['cv_spearman']}" + (f", range {ci}" if ci else "")
                    + f", over {a.get('cv_n_groups')} groups.")
            out.append(_chunk("analysis", "cv", cid, "Model cross-check", text))
        noise = a.get("noise")
        if isinstance(noise, dict):
            keys = ("best_single", "best_reproducible", "noise_sd", "n_recipes", "n_replicated")
            text = f"Measured spread for campaign {name!r}: " + ", ".join(
                f"{k.replace('_', ' ')} {_fmt(noise[k])}" for k in keys if noise.get(k) is not None
            ) + "."
            out.append(_chunk("analysis", "noise", cid, "Run-to-run spread", text))
        verdict_chunk = _verdict_chunk(cid, name, a)
        if verdict_chunk is not None:
            out.append(verdict_chunk)
        drivers = [d for d in a.get("drivers") or [] if isinstance(d, dict) and d.get("name")]
        if drivers:
            ordered = sorted(drivers, key=lambda d: -abs(float(d.get("rho") or 0.0)))
            rows = []
            for d in ordered[:8]:
                rho = d.get("rho")
                way = "higher" if isinstance(rho, (int, float)) and rho >= 0 else "lower"
                sure = "clear signal" if d.get("significant") else "could be chance"
                rows.append(f"Higher {d['name']} went with {way} {target} (strength {rho} on a scale "
                            f"from -1 to 1; {sure})")
            text = (f"What went with {target} in campaign {name!r}, strongest first "
                    "(went with, not a cause):\n" + "\n".join(rows))
            out.append(_chunk("analysis", "drivers", cid, "What went with the target: which input matters most, important inputs, drivers, influence", text))
        proposals = [p for p in a.get("proposals") or [] if isinstance(p, dict)]
        if proposals:
            rows = []
            for i, p in enumerate(proposals):
                mode = {"exploit": "Best bet", "explore": "New ground"}.get(str(p.get("mode")), "")
                rows.append(
                    f"Recipe {i + 1}" + (f" ({mode})" if mode else "") + f": {_recipe_text(p.get('recipe') or {})}; "
                    f"Predicted {target} {_fmt(p.get('pred'))}"
                    + (f" (model spread {_fmt(p.get('std'))}, Predicted)" if p.get("std") is not None else "")
                )
            text = (f"Proposed recipes for campaign {name!r} (Predicted values, not measured; "
                    "Best bet = the model expects a high value, New ground = tries a recipe the model is unsure about):\n"
                    + "\n".join(rows))
            out.append(_chunk("analysis", "proposals", cid, "Proposed recipes: what to run next, next round, suggested runs", text[:MAX_CHUNK_CHARS * 3]))

    refusal_chunk = _refusal_chunk(cid, name, a, refusal)
    if refusal_chunk is not None:
        out.append(refusal_chunk)
    if state and target:
        out.extend(_best_and_delta_chunks(cid, name, state, a, str(target)))

    if state:
        rounds: dict[int, list[dict[str, Any]]] = {}
        for run in state.get("pending") or []:
            r = run.get("round")
            rounds.setdefault(int(r) if isinstance(r, int) else int(state.get("round") or 0), []).append(run)
        for r, runs in sorted(rounds.items()):
            rows = []
            for run in runs:
                result = run.get("result")
                measured = f"Measured {target} {_fmt(result)}" if result is not None else "no result recorded yet"
                rows.append(f"Run {run.get('id')}: {_recipe_text(run.get('recipe') or {})}; {measured}")
            text = f"Round {r + 1} of campaign {name!r} (round number {r}), started runs:\n" + "\n".join(rows)
            out.append(_chunk("round", str(r), cid, f"Round {r + 1}", text[:MAX_CHUNK_CHARS * 3]))
        for h in state.get("history") or []:
            if isinstance(h, dict) and h.get("best") is not None:
                text = (f"After round number {h.get('round')} of campaign {name!r}: best measured {target} "
                        f"{_fmt(h['best'])} across {h.get('n_base')} runs.")
                out.append(_chunk("round", str(h.get("round")), cid, "Best so far", text))
    return out


def _verdict_chunk(cid: str, name: str, a: dict[str, Any]) -> Chunk | None:
    """`analysis:verdict`: the model's reliability state in plain words, the
    reasons (the guide's own sentences) and the glossary why-line."""
    from kalos.assist.guide import _gate_blockers, _trusted
    from kalos.assist.why import WHY

    trusted = _trusted(a)
    if trusted is None:
        return None
    if trusted:
        state, why, reasons = "Reliable", WHY["reliable"], []
    else:
        blockers = _gate_blockers(a)
        unmeasured = all(b["why"] == WHY["unmeasured"] for b in blockers)
        state = "Not enough runs to tell" if unmeasured else "Not yet reliable"
        why = WHY["insufficient" if unmeasured else "not-yet"]
        reasons = [f"{b['text']} (check {b['code']})" for b in blockers]
    text = f"Model verdict for campaign {name!r}: {state}."
    if reasons:
        text += " Reasons: " + " ".join(reasons)
    text += f" Why this matters: {why}"
    return _chunk("analysis", "verdict", cid, "Model verdict: can I trust the model, is it reliable, why not", text)


def _refusal_chunk(
    cid: str, name: str, a: dict[str, Any], refusal: dict[str, Any] | None,
) -> Chunk | None:
    """`analysis:refusal`: why there are no recipes to run: the last refusal
    (with its code), or a published analysis that proposed nothing."""
    from kalos.assist.why import why_for

    if refusal and refusal.get("code"):
        code = str(refusal["code"])
        text = (f"The last sheet for campaign {name!r} was not read into a model (reason code {code}). "
                f"Why this matters: {why_for(code)}")
        return _chunk("analysis", "refusal", cid, "Why there are no recipes", text)
    if a and not (a.get("proposals") or []):
        notice = a.get("proposal_notice") if isinstance(a.get("proposal_notice"), dict) else {}
        code = str((notice or {}).get("code") or "empty_batch")
        text = (f"The last analysis of campaign {name!r} proposed no recipes (code {code}). "
                f"Why this matters: {why_for(code)}")
        return _chunk("analysis", "refusal", cid, "Why there are no recipes", text)
    return None


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _best_and_delta_chunks(
    cid: str, name: str, state: dict[str, Any], a: dict[str, Any], target: str,
) -> list[Chunk]:
    """`analysis:best_run` (best measured run so far, with its settings) and
    `analysis:round_delta` (what changed since the round before)."""
    out: list[Chunk] = []
    features = [str(f) for f in state.get("features") or []]
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for row in state.get("base_rows") or []:
        if isinstance(row, dict) and _num(row.get(target)) is not None:
            candidates.append((float(_num(row.get(target)) or 0.0), "from the run sheet or an earlier round",
                               {k: row.get(k) for k in features if k in row}))
    for run in state.get("pending") or []:
        if isinstance(run, dict) and _num(run.get("result")) is not None:
            r = run.get("round")
            where = f"run {run.get('id')}" + (f", round number {r}" if isinstance(r, int) else "")
            candidates.append((float(_num(run.get("result")) or 0.0), where, dict(run.get("recipe") or {})))
    if candidates:
        value, where, recipe = max(candidates, key=lambda c: c[0])
        text = (f"Best measured run so far in campaign {name!r}: Measured {target} {_fmt(value)} ({where}). "
                f"Its settings: {_recipe_text(recipe)}.")
        out.append(_chunk("analysis", "best_run", cid, "Best measured run: top result, highest value, why is it best", text))

    history = [h for h in state.get("history") or [] if isinstance(h, dict)]
    lines: list[str] = []
    if len(history) >= 2:
        before, after = history[-2], history[-1]
        lines.append(
            f"After round number {after.get('round')}: best Measured {target} {_fmt(after.get('best'))}, "
            f"before it {_fmt(before.get('best'))} (Measured); runs the model learns from went from "
            f"{before.get('n_base')} to {after.get('n_base')}."
        )
    elif history:
        lines.append(f"No round has been added to the model yet; best Measured {target} so far "
                     f"{_fmt(history[-1].get('best'))} across {history[-1].get('n_base')} runs.")
    for run in state.get("pending") or []:
        if isinstance(run, dict) and _num(run.get("result")) is not None and _num(run.get("pred")) is not None:
            lines.append(f"Run {run.get('id')}: Predicted {target} {_fmt(run.get('pred'))}, "
                         f"Measured {_fmt(run.get('result'))}.")
    if lines:
        text = f"What changed in campaign {name!r}:\n" + "\n".join(lines)
        out.append(_chunk("analysis", "round_delta", cid, "What changed since the last round: new results, difference, compared with before", text))
    return out


def _split_markdown(text: str) -> list[tuple[str, str]]:
    """(heading, body) sections, each at most MAX_CHUNK_CHARS (long ones split on
    paragraphs)."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        if line.startswith("#"):
            sections.append((line.lstrip("#").strip(), []))
        else:
            sections[-1][1].append(line)
    out: list[tuple[str, str]] = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        piece = ""
        for para in re.split(r"\n\s*\n", body):
            if piece and len(piece) + len(para) > MAX_CHUNK_CHARS:
                out.append((heading, piece.strip()))
                piece = ""
            piece += para + "\n\n"
        if piece.strip():
            out.append((heading, piece.strip()[: MAX_CHUNK_CHARS * 2]))
    return out


def docs_dir() -> Path | None:
    raw = os.environ.get("KALOS_ASSIST_DOCS_DIR")
    path = Path(raw) if raw else Path(__file__).resolve().parents[2] / "docs"
    return path if path.is_dir() else None


def skills_dirs() -> list[Path]:
    """Folders holding `<slug>/SKILL.md`: the engine's built-in set when present
    (kalos/skills/builtin) and KALOS_ASSIST_SKILLS_DIR."""
    dirs = [Path(__file__).resolve().parents[1] / "skills" / "builtin"]
    if os.environ.get("KALOS_ASSIST_SKILLS_DIR"):
        dirs.append(Path(os.environ["KALOS_ASSIST_SKILLS_DIR"]))
    return [d for d in dirs if d.is_dir()]


def _files_fingerprint(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        st = p.stat()
        h.update(f"{p}\x00{st.st_size}\x00{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def _doc_files() -> list[Path]:
    d = docs_dir()
    return sorted(p for p in d.glob("*.md") if p.is_file() and not p.is_symlink()) if d else []


def _skill_files() -> list[Path]:
    files: list[Path] = []
    for d in skills_dirs():
        files.extend(p for p in d.glob("*/SKILL.md") if p.is_file() and not p.is_symlink())
    return sorted(files)


def doc_chunks(files: list[Path]) -> list[Chunk]:
    out = []
    for path in files:
        rel = f"docs/{path.name}"
        for heading, body in _split_markdown(path.read_text(encoding="utf-8", errors="replace")):
            out.append(_chunk("doc", rel, None, f"{rel} - {heading}" if heading else rel, body))
    return out


def skill_chunks(files: list[Path]) -> list[Chunk]:
    """SKILL.md files: the frontmatter is skipped except name and summary; the
    body is split by heading like a doc."""
    out: list[Chunk] = []
    builtin = (Path(__file__).resolve().parents[1] / "skills" / "builtin").resolve()
    for path in files:
        slug = path.parent.name
        trusted = builtin in path.resolve().parents
        text = path.read_text(encoding="utf-8", errors="replace")
        head = ""
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                front, text = text[3:end], text[end + 4:]
                keep = [ln for ln in front.splitlines() if ln.startswith(("name:", "summary:"))]
                head = "\n".join(keep)
        pieces = ([_chunk("skill", slug, None, f"Skill {slug}", head)] if head else []) + [
            _chunk("skill", slug, None, f"Skill {slug} - {heading}", body)
            for heading, body in _split_markdown(text)
        ]
        from dataclasses import replace

        out.extend(replace(c, trusted_text=trusted) for c in pieces)
    return out


# --- the index ------------------------------------------------------------------- #


@dataclass
class Hit:
    chunk: Chunk
    score: float


_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_BUILDING: set[str] = set()


def _lock_for(path: Path) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(path), threading.Lock())


class AssistIndex:
    """One tenant's index. Cheap to construct; reads its file lazily."""

    def __init__(self, store: Any, tenant: str, *, embedder: Embedder | bool | None = True) -> None:
        self.store = store
        self.tenant = tenant
        self.dir = tenant_dir(store.state_dir, tenant)
        self.path = self.dir / "index.json"
        self.vec_path = self.dir / "vectors.npz"
        self._embedder: Embedder | None = (
            load_embedder() if embedder is True else (embedder or None)
        )

    # -- persistence
    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict) or data.get("version") != INDEX_VERSION:
            return {}
        return data

    def _save(self, data: dict[str, Any], vectors: dict[str, np.ndarray]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, self.path)
        if vectors:
            ids = sorted(vectors)
            vtmp = self.dir / "vectors.tmp.npz"
            np.savez(vtmp, ids=np.asarray(ids), vecs=np.stack([vectors[i] for i in ids]))
            os.replace(vtmp, self.vec_path)
        elif self.vec_path.exists():
            self.vec_path.unlink()

    def _load_vectors(self) -> dict[str, np.ndarray]:
        if not self.vec_path.exists():
            return {}
        try:
            with np.load(self.vec_path, allow_pickle=False) as z:
                return {str(i): v for i, v in zip(z["ids"], z["vecs"])}
        except (OSError, ValueError, KeyError):
            return {}

    # -- groups
    def _campaign_groups(self, only: set[str] | None) -> dict[str, tuple[str, Any]]:
        """{group key: (fingerprint, loader)} for this tenant's campaigns."""
        from kalos.assist.state import latest_refusal

        groups: dict[str, tuple[str, Any]] = {}
        for info in self.store.list_campaigns(tenant=self.tenant, include_archived=True):
            cid = info["id"]
            key = f"campaign:{cid}"
            if only is not None and cid not in only:
                groups[key] = ("", None)  # keep as is
                continue
            analysis = self.store.load_analysis(tenant=self.tenant, campaign_id=cid)
            refusal = latest_refusal(self.store, self.tenant, cid)
            fp = hashlib.sha256(json.dumps([
                info.get("updated_at"), info.get("name"),
                self.store.generation_of(tenant=self.tenant, campaign_id=cid),
                (analysis or {}).get("updated"), (analysis or {}).get("timestamp"),
                (refusal or {}).get("at"), (refusal or {}).get("code"),
            ], default=str).encode()).hexdigest()

            def loader(info: dict[str, Any] = info, analysis: Any = analysis, refusal: Any = refusal) -> list[Chunk]:
                state = self.store.get(tenant=self.tenant, campaign_id=info["id"])
                try:
                    return campaign_chunks(info, state, analysis, refusal)
                except Exception:  # noqa: BLE001 - one odd record must not break the index
                    log.exception("assist: could not index one campaign; it is left out")
                    return []

            groups[key] = (fp, loader)
        docs, skills = _doc_files(), _skill_files()
        groups["docs"] = (_files_fingerprint(docs), lambda: doc_chunks(docs))
        groups["skills"] = (_files_fingerprint(skills), lambda: skill_chunks(skills))
        return groups

    def refresh(self, *, campaign_ids: Iterable[str] | None = None, force: bool = False) -> dict[str, Any]:
        """Rebuild the groups whose fingerprint changed (all of them with `force`).
        `campaign_ids` limits the campaign groups looked at (the upload/record
        hooks pass the one campaign that changed). Returns the status."""
        lock = _lock_for(self.path)
        with lock:
            _BUILDING.add(str(self.path))
            try:
                data = {} if force else self._load()
                old_groups: dict[str, Any] = data.get("groups") or {}
                vectors = {} if force else self._load_vectors()
                only = set(campaign_ids) if campaign_ids is not None else None
                new_groups: dict[str, Any] = {}
                changed = 0
                for key, (fp, loader) in self._campaign_groups(only).items():
                    old = old_groups.get(key)
                    if loader is None:  # not looked at this time: keep
                        if old is not None:
                            new_groups[key] = old
                        continue
                    if old is not None and old.get("fingerprint") == fp and not force:
                        new_groups[key] = old
                        continue
                    chunks = loader()
                    new_groups[key] = {"fingerprint": fp, "chunks": [asdict(c) for c in chunks]}
                    changed += 1
                live = {c["chunk_id"] for g in new_groups.values() for c in g["chunks"]}
                retriever = "bm25"
                embed_name = None
                if self._embedder is not None:
                    missing = [c for g in new_groups.values() for c in g["chunks"] if c["chunk_id"] not in vectors]
                    try:
                        if missing:
                            vecs = self._embedder.encode([c["text"] for c in missing])
                            for c, v in zip(missing, vecs):
                                vectors[c["chunk_id"]] = np.asarray(v, dtype=np.float32)
                        retriever, embed_name = "embeddings", self._embedder.name
                    except Exception:  # noqa: BLE001 - fall back to BM25, never fail
                        log.exception("assist: embedding failed; using BM25")
                        vectors = {}
                vectors = {k: v for k, v in vectors.items() if k in live} if retriever == "embeddings" else {}
                if (
                    not force and changed == 0 and data and self.path.exists()
                    and set(new_groups) == set(old_groups) and data.get("retriever") == retriever
                ):
                    # Nothing changed: a read (ask, status) leaves the file alone.
                    return self._status_of(data)
                data = {
                    "version": INDEX_VERSION,
                    "built_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "retriever": retriever,
                    "embed_model": embed_name,
                    "groups": new_groups,
                    "changed_groups": changed,
                }
                self._save(data, vectors)
                return self._status_of(data)
            finally:
                _BUILDING.discard(str(self.path))

    def rebuild(self) -> dict[str, Any]:
        return self.refresh(force=True)

    # -- reading
    def _status_of(self, data: dict[str, Any]) -> dict[str, Any]:
        groups = data.get("groups") or {}
        return {
            "chunks": sum(len(g.get("chunks") or []) for g in groups.values()),
            "built_at": data.get("built_at"),
            "building": str(self.path) in _BUILDING,
            "retriever": data.get("retriever") or ("embeddings" if self._embedder else "bm25"),
        }

    def status(self) -> dict[str, Any]:
        return self._status_of(self._load())

    def chunks(self) -> list[Chunk]:
        groups = self._load().get("groups") or {}
        return [
            Chunk(**{**c, "user_terms": tuple(c.get("user_terms") or ())})
            for g in groups.values() for c in g.get("chunks") or []
        ]

    def search(
        self, query: str, *, campaign_id: str | None = None, k: int = 6, min_similarity: float = 0.35,
    ) -> list[Hit]:
        """The `k` most relevant chunks. Campaign chunks are limited to
        `campaign_id` when given (docs and skills always qualify). A chunk is
        relevant only if it shares a word with the question (BM25) or, with
        embeddings, its similarity reaches `min_similarity`. Empty when nothing
        is relevant: the caller abstains."""
        if not self.path.exists():
            self.refresh()
        pool = [c for c in self.chunks() if campaign_id is None or c.campaign_id in (None, campaign_id)]
        q = tokens(query)
        if not pool or not q:
            return []
        data = self._load()
        if data.get("retriever") == "embeddings" and self._embedder is not None:
            # The embedding path uses the raw query text (semantics handle
            # synonyms on their own), so it is left untouched.
            vectors = self._load_vectors()
            if all(c.chunk_id in vectors for c in pool):
                qv = self._embedder.encode([query])[0]
                sims = np.asarray([float(np.dot(vectors[c.chunk_id], qv)) for c in pool])
                order = np.argsort(-sims)
                return [Hit(pool[i], float(sims[i])) for i in order[:k] if sims[i] >= min_similarity]
        # BM25 is lexical, so expand the query with canonical bioprocess terms
        # for any lay/synonym token ("sugar" -> "glucose"). Query-side only:
        # stored chunks are unchanged, so citations and text are unaffected.
        q = expand_query_tokens(q)
        corpus = [tokens(c.title + " " + c.text) for c in pool]
        scores = _bm25_scores(corpus, q)
        # The lab's own records first: a doc must match clearly better to outrank them.
        scores = scores * np.asarray([1.0 if c.kind in ("doc", "skill") else 1.6 for c in pool])
        qset = set(q)
        hits = [
            Hit(pool[i], float(scores[i]))
            for i in np.argsort(-scores)
            if qset & set(corpus[i]) and scores[i] > 0
        ]
        return hits[:k]


def refresh_quietly(store: Any, tenant: str, campaign_id: str | None) -> None:
    """Best-effort incremental refresh after an upload or a recorded result.
    Never raises: the index is derived data, and a failure here must not fail
    the request that changed the campaign."""
    if (os.environ.get("KALOS_ASSIST_INDEX_ON_WRITE") or "1").strip() in {"0", "false", "off", "no"}:
        return
    try:
        index = AssistIndex(store, tenant)
        if not index.path.exists():
            return  # built on first use; nothing to keep up to date yet
        cid = campaign_id or store.default_campaign_id(tenant=tenant)
        index.refresh(campaign_ids=[cid] if cid else [])
    except Exception:  # noqa: BLE001 - derived data, never fails the write
        log.exception("assist: index refresh after a write failed")


__all__ = [
    "AssistIndex",
    "Chunk",
    "Embedder",
    "Hit",
    "campaign_chunks",
    "doc_chunks",
    "load_embedder",
    "refresh_quietly",
    "skill_chunks",
    "tokens",
]
