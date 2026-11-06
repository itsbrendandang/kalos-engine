"""A small, offline bioprocess synonym map for query-side expansion.

The assistant's BM25 retriever matches on shared words, so a question worded in
lay or synonym terms ("how much sugar") never reaches a chunk that stores the
canonical term ("glucose"). The retrieval benchmark measures this exactly:
literal queries reach recall@6 = 1.000, paraphrased ones only 0.500, and the
drop is concentrated in the `sheet` and `finding` chunks (sugar->glucose,
unusual->outlier). See `tests/test_assist_retrieval_bench.py` and the README's
"Assistant retrieval quality" section.

This map closes that gap on the QUERY side only: when a query token is a known
lay term, its canonical bioprocess term(s) are added to the query before BM25
scores it. Nothing is added to the stored chunks, so:
  - it cannot change what a chunk says or how it is cited;
  - it never downloads anything and needs no model (pure dict lookup);
  - it is conservative - only well-established, unambiguous equivalences are
    listed, so expansion cannot pull in an unrelated chunk.

The map is deliberately small and curated, not learned. Each entry maps a lay
or alternate term to the canonical term(s) the engine actually stores. Keep it
one-directional (lay -> canonical) and unambiguous; a term with two plausible
meanings is left out rather than guessed.
"""
from __future__ import annotations

# lay / alternate term -> canonical term(s) the engine stores.
# All keys and values are single lower-case tokens (post-tokenization).
_SYNONYMS: dict[str, tuple[str, ...]] = {
    # sugars / carbon sources
    "sugar": ("glucose",),
    "sugars": ("glucose",),
    "dextrose": ("glucose",),
    "carbon": ("glucose",),
    # temperature
    "heat": ("temperature",),
    "temp": ("temperature",),
    "warmth": ("temperature",),
    # pH / acidity
    "acidity": ("ph",),
    "alkalinity": ("ph",),
    "basicity": ("ph",),
    # oxygen
    "oxygen": ("do", "dissolved"),
    "aeration": ("do", "dissolved"),
    # agitation
    "stirring": ("agitation", "rpm"),
    "mixing": ("agitation", "rpm"),
    "speed": ("agitation", "rpm"),
    # outcome / target
    "yield": ("titer",),
    "output": ("titer",),
    "production": ("titer",),
    "productivity": ("titer",),
    # findings vocabulary
    "unusual": ("outlier", "outliers"),
    "anomaly": ("outlier", "outliers"),
    "anomalies": ("outlier", "outliers"),
    "weird": ("outlier", "outliers"),
    "gaps": ("missing",),
    "blanks": ("missing",),
    "blank": ("missing",),
    "empty": ("missing",),
    "flagged": ("outlier", "outliers", "missing"),
    # drivers / importance
    "factors": ("drivers", "driver"),
    "factor": ("drivers", "driver"),
    "influences": ("drivers", "driver"),
    "drive": ("drivers", "driver"),
    # recipes / experiments
    "experiment": ("run", "recipe", "recipes"),
    "experiments": ("run", "recipe", "recipes"),
    "trial": ("run", "recipe", "recipes"),
    # model-quality vocabulary
    "quality": ("promotion", "gate"),
    "check": ("promotion", "gate", "validation"),
    # feeding / media
    "fed": ("feed", "glucose"),
    "feeding": ("feed",),
    "nutrient": ("media", "feed"),
    "nutrients": ("media", "feed"),
}


def expand_query_tokens(query_tokens: list[str]) -> list[str]:
    """Return `query_tokens` plus the canonical term(s) for any known lay term,
    de-duplicated and order-preserving. Tokens with no synonym pass through
    unchanged. Added terms never replace the originals, so an already-canonical
    query is unaffected."""
    out: list[str] = []
    seen: set[str] = set()
    for tok in query_tokens:
        if tok not in seen:
            out.append(tok)
            seen.add(tok)
        for canon in _SYNONYMS.get(tok, ()):  # expansion, not replacement
            if canon not in seen:
                out.append(canon)
                seen.add(canon)
    return out


__all__ = ["expand_query_tokens"]
