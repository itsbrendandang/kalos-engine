"""Semantic search over one campaign: runs, columns, notes and the assistant index.

`search(store, tenant, campaign_id, query)` ranks the campaign's own records by
meaning when a local embedding model is available, and by shared words
otherwise:

- `embeddings`: `nomic-embed-text` (Apache-2.0) through Ollama
  (`KALOS_EMBED_MODEL`), used only when the assistant's provider is `ollama`
  (`KALOS_LLM_PROVIDER`). Text is never sent to a remote provider for
  embedding: with `anthropic` or `none`, search is keyword search.
- `keyword`: BM25 over the same documents (`kalos.assist.index`). Used when no
  model is configured or reachable, with `fallback_reason` saying why.

Nothing here generates text: a hit is a record the engine already holds, quoted
from the record, so there is nothing for a model to get wrong except the order.

Documents (all from the caller's tenant and this campaign only):
  - `run`: each started run: its id, round, recipe, measured result, and the
    reason given when it was started;
  - `row`: each row of the uploaded sheet (capped at `MAX_ROWS`);
  - `column`: each column of the sheet, with what Kalos did with it;
  - `note`: the reason recorded with each corrected result;
  - the assistant index chunks for this campaign (`sheet`, `finding`, `round`,
    `analysis`, `refusal`), with their kind.

Bounds: at most `MAX_DOCUMENTS` documents per search, each clipped to
`MAX_DOC_CHARS`; a query of at most `MAX_QUERY_CHARS`; at most `MAX_K` hits.
Vectors are cached per tenant in `<state>/assist/<tenant>/embeddings.sqlite`,
keyed by the sha256 of (model, text), and the cache keeps at most
`KALOS_EMBED_CACHE_MAX` vectors (least recently used are dropped). The cache
is derived data: deleting it loses nothing.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from kalos.assist.index import AssistIndex, _bm25_scores, _fmt, _recipe_text, tokens
from kalos.assist.paths import tenant_dir

log = logging.getLogger("kalos.assist.semantic")

DEFAULT_EMBED_MODEL = "nomic-embed-text"
EMBED_MODEL_ENV = "KALOS_EMBED_MODEL"
SEARCH_ENV = "KALOS_SEMANTIC_SEARCH"
CACHE_MAX_ENV = "KALOS_EMBED_CACHE_MAX"
DEFAULT_CACHE_MAX = 20_000
MAX_DOCUMENTS = 3000
MAX_ROWS = 1000
MAX_DOC_CHARS = 1000
MAX_QUERY_CHARS = 500
MAX_K = 25
SNIPPET_CHARS = 300
_BATCH = 64
# nomic-embed-text is trained with these task prefixes.
_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


@dataclass(frozen=True)
class Document:
    kind: str
    id: str
    title: str
    text: str


# --- embeddings ------------------------------------------------------------------- #


class SearchEmbedder(Protocol):
    name: str

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        """L2-normalised vectors, one row per text."""
        ...


class OllamaEmbedder:
    """`nomic-embed-text` (or `KALOS_EMBED_MODEL`) through the local Ollama."""

    def __init__(self, model: str | None = None) -> None:
        self.name = model or embed_model()

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        from kalos.providers.ollama_provider import OllamaProvider

        prefix = _QUERY_PREFIX if query else _DOC_PREFIX
        rows: list[list[float]] = []
        for i in range(0, len(texts), _BATCH):
            rows.extend(OllamaProvider().embed([prefix + t for t in texts[i:i + _BATCH]], model=self.name))
        vecs = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1.0, norms)


def embed_model() -> str:
    return (os.environ.get(EMBED_MODEL_ENV) or DEFAULT_EMBED_MODEL).strip()


def embedder_or_reason() -> tuple[OllamaEmbedder | None, str | None]:
    """The embedder to use, or None and the reason search is keyword-only."""
    if (os.environ.get(SEARCH_ENV) or "1").strip().lower() in {"0", "false", "no", "off"}:
        return None, "semantic search is turned off (KALOS_SEMANTIC_SEARCH=0)"
    from kalos.assist.llm import assist_provider

    if assist_provider() != "ollama":
        return None, "no local embedding model is configured (KALOS_LLM_PROVIDER is not ollama)"
    return OllamaEmbedder(), None


def _cache_max() -> int:
    raw = (os.environ.get(CACHE_MAX_ENV) or "").strip()
    try:
        value = int(raw) if raw else DEFAULT_CACHE_MAX
    except ValueError:
        value = DEFAULT_CACHE_MAX
    return max(100, min(value, 1_000_000))


_CACHE_LOCK = threading.Lock()


class EmbeddingCache:
    """Vectors by sha256(model, text), per tenant, size-bounded (LRU)."""

    def __init__(self, directory: Path) -> None:
        self.path = Path(directory) / "embeddings.sqlite"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, model TEXT NOT NULL, "
            "dim INTEGER NOT NULL, vec BLOB NOT NULL, used_at REAL NOT NULL)"
        )
        return conn

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()

    def get_many(self, model: str, texts: list[str]) -> dict[str, np.ndarray]:
        keys = [self.key(model, t) for t in texts]
        out: dict[str, np.ndarray] = {}
        with _CACHE_LOCK, self._connect() as conn:
            for i in range(0, len(keys), 500):
                part = keys[i:i + 500]
                marks = ",".join("?" * len(part))
                for key, dim, blob in conn.execute(
                    f"SELECT key, dim, vec FROM vectors WHERE key IN ({marks})", part  # noqa: S608 - only '?' marks
                ):
                    vec = np.frombuffer(blob, dtype=np.float32)
                    if vec.shape == (dim,):
                        out[key] = vec
            if out:
                now = time.time()
                conn.executemany("UPDATE vectors SET used_at = ? WHERE key = ?", [(now, k) for k in out])
        return out

    def put_many(self, model: str, items: Iterable[tuple[str, np.ndarray]]) -> None:
        now = time.time()
        rows = [(self.key(model, t), model, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes(), now)
                for t, v in items]
        if not rows:
            return
        with _CACHE_LOCK, self._connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?, ?)", rows)
            limit = _cache_max()
            (count,) = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()
            if count > limit:
                conn.execute(
                    "DELETE FROM vectors WHERE key IN (SELECT key FROM vectors ORDER BY used_at ASC LIMIT ?)",
                    (count - limit,),
                )

    def size(self) -> int:
        if not self.path.exists():
            return 0
        with _CACHE_LOCK, self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])


# --- documents -------------------------------------------------------------------- #


def _clip(text: str) -> str:
    return " ".join(text.split())[:MAX_DOC_CHARS]


def _screened(row: dict[str, Any]) -> dict[str, Any]:
    """`row` without identity columns (operator, sample name, ...): the same
    header pre-screen the normalize payload uses (kalos/normalize/payload.py),
    so no identity value reaches an embedding model."""
    from kalos.normalize.payload import _is_identity_header

    return {k: v for k, v in row.items() if not _is_identity_header(str(k))}


def _is_identity_column(name: str) -> bool:
    from kalos.normalize.payload import _is_identity_header

    return _is_identity_header(name)


def campaign_documents(store: Any, tenant: str, campaign_id: str) -> list[Document]:
    """The searchable records of one campaign (tenant-scoped by the store)."""
    state = store.get(tenant=tenant, campaign_id=campaign_id) or {}
    analysis = store.load_analysis(tenant=tenant, campaign_id=campaign_id) or {}
    target = str(state.get("target") or analysis.get("target") or "target")
    docs: list[Document] = []

    for run in state.get("pending") or []:
        if not isinstance(run, dict) or not run.get("id"):
            continue
        result = run.get("result")
        measured = f"Measured {target} {_fmt(result)}" if result is not None else "no result recorded yet"
        parts = [f"Run {run['id']} (round {run.get('round')}): {_recipe_text(_screened(run.get('recipe') or {}))}; {measured}."]
        if run.get("reason"):
            parts.append(f"Reason given: {run['reason']}.")
        if run.get("mode"):
            parts.append(f"Mode: {run['mode']}.")
        docs.append(Document("run", str(run["id"]), f"Run {str(run['id'])[:8]}", _clip(" ".join(parts))))
        if int(run.get("result_version") or 1) > 1:
            for entry in store.result_history(str(run["id"]), tenant=tenant, campaign_id=campaign_id):
                if entry.get("reason"):
                    docs.append(Document(
                        "note", f"{run['id']}.v{entry.get('version')}", f"Correction to run {str(run['id'])[:8]}",
                        _clip(f"Result for run {run['id']} changed from {_fmt(entry.get('previous_value'))} "
                              f"to {_fmt(entry.get('value'))}. Reason: {entry['reason']}"),
                    ))

    for i, row in enumerate((state.get("base_rows") or [])[:MAX_ROWS]):
        if isinstance(row, dict):
            docs.append(Document("row", f"row:{i + 1}", f"Sheet row {i + 1}",
                                 _clip(f"Sheet row {i + 1}: {_recipe_text(_screened(row))}.")))

    columns: dict[str, str] = {}
    listed = analysis.get("columns") or analysis.get("column_provenance") or analysis.get("provenance")
    for col in listed if isinstance(listed, list) else []:
        if isinstance(col, dict) and col.get("name"):
            bits = [str(col.get(k)) for k in ("status", "role", "kind", "unit", "reason") if col.get(k)]
            columns[str(col["name"])] = "; ".join(bits)
    for name in [*(state.get("features") or []), *(analysis.get("features") or [])]:
        columns.setdefault(str(name), "input used by the model")
    columns.setdefault(target, "the target (the measured result)")
    for name, what in columns.items():
        if _is_identity_column(name):
            continue
        docs.append(Document("column", name, f"Column {name}", _clip(f"Column {name}: {what}.")))

    try:
        index = AssistIndex(store, tenant, embedder=False)
        index.refresh(campaign_ids=[campaign_id])  # cheap when nothing changed
        for chunk in index.chunks():
            if chunk.campaign_id == campaign_id:
                docs.append(Document(chunk.kind, chunk.source_id, chunk.title, _clip(chunk.text)))
    except Exception:  # noqa: BLE001 - the index is derived data; search the records alone
        log.exception("semantic search: the assistant index could not be read")
    return docs[:MAX_DOCUMENTS]


# --- search ----------------------------------------------------------------------- #


def _hit(doc: Document, score: float) -> dict[str, Any]:
    return {"kind": doc.kind, "id": doc.id, "title": doc.title, "text": doc.text[:SNIPPET_CHARS],
            "score": round(float(score), 4)}


def keyword_search(docs: list[Document], query: str, k: int) -> list[dict[str, Any]]:
    q = tokens(query)
    if not docs or not q:
        return []
    corpus = [tokens(d.title + " " + d.text) for d in docs]
    scores = _bm25_scores(corpus, q)
    qset = set(q)
    return [_hit(docs[i], scores[i]) for i in np.argsort(-scores, kind="stable")
            if scores[i] > 0 and qset & set(corpus[i])][:k]


def embedding_search(docs: list[Document], query: str, k: int, embedder: SearchEmbedder,
                     cache: EmbeddingCache) -> list[dict[str, Any]]:
    texts = [d.title + "\n" + d.text for d in docs]
    cached = cache.get_many(embedder.name, texts)
    keys = [cache.key(embedder.name, t) for t in texts]
    missing = [t for t, key in zip(texts, keys) if key not in cached]
    if missing:
        fresh = embedder.encode(missing)
        cache.put_many(embedder.name, zip(missing, fresh))
        for t, v in zip(missing, fresh):
            cached[cache.key(embedder.name, t)] = v
    matrix = np.stack([cached[key] for key in keys])
    qv = embedder.encode([query], query=True)[0]
    sims = matrix @ qv
    order = np.argsort(-sims, kind="stable")[:k]
    return [_hit(docs[i], sims[i]) for i in order]


def search(store: Any, tenant: str, campaign_id: str, query: str, *, k: int = 10,
           embedder: SearchEmbedder | bool | None = True) -> dict[str, Any]:
    """Ranked hits for `query` over one campaign's records. `embedder=True`
    resolves the configured one; None/False forces keyword search."""
    query = query.strip()[:MAX_QUERY_CHARS]
    k = max(1, min(int(k), MAX_K))
    docs = campaign_documents(store, tenant, campaign_id)
    reason: str | None
    model: SearchEmbedder | None
    if embedder is True:
        model, reason = embedder_or_reason()
    else:
        model = embedder if not isinstance(embedder, bool) and embedder is not None else None
        reason = None if model is not None else "keyword search was requested"
    if model is not None and docs:
        try:
            hits = embedding_search(docs, query, k, model, EmbeddingCache(tenant_dir(store.state_dir, tenant)))
            return {"retriever": "embeddings", "model": model.name, "fallback_reason": None,
                    "n_documents": len(docs), "hits": hits}
        except Exception as exc:  # noqa: BLE001 - any model failure: keyword search
            log.info("semantic search: embeddings failed (%s); keyword search", type(exc).__name__)
            reason = "the local embedding model did not answer (is it pulled? `ollama pull nomic-embed-text`)"
    return {"retriever": "keyword", "model": None, "fallback_reason": reason,
            "n_documents": len(docs), "hits": keyword_search(docs, query, k)}


__all__ = [
    "DEFAULT_EMBED_MODEL",
    "MAX_K",
    "MAX_QUERY_CHARS",
    "Document",
    "EmbeddingCache",
    "OllamaEmbedder",
    "SearchEmbedder",
    "campaign_documents",
    "embedder_or_reason",
    "keyword_search",
    "search",
]
