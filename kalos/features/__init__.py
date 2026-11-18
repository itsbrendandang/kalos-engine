"""Feature sources. KmerEmbedder needs no extra deps; ESM2Embedder and ESMCEmbedder need transformers."""
from .protein import (
    ESM2Embedder,
    ESMCEmbedder,
    KmerEmbedder,
    build_feature_table,
    clean_sequence,
)

__all__ = ["ESM2Embedder", "ESMCEmbedder", "KmerEmbedder", "build_feature_table", "clean_sequence"]
