"""Protein-sequence embeddings as surrogate features.

Embed the expressed product's amino-acid sequence so one surrogate can transfer
across products/strains. Three backends behind one interface:

  - `ESMCEmbedder` — ESMC (ESM Cambrian), the current ESM protein language
    model from Biohub (formerly EvolutionaryScale), via Hugging Face
    transformers. MIT-licensed open weights; default is the 300M variant.
  - `ESM2Embedder` — the earlier ESM-2 model (the same family NVIDIA BioNeMo
    serves). Small and fast; kept for existing feature tables.
  - Both run on Apple MPS or CPU and mean-pool the residue embeddings to one
    vector per sequence.
  - `KmerEmbedder` — a dependency-free hashed k-mer stand-in that runs with no
    model download, for tests and offline use.

Trust boundary: an embedder only ever sees a PUBLIC protein sequence, never
proprietary process data.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np
import pandas as pd

_NON_AA = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")


def clean_sequence(sequence: str) -> str:
    if not isinstance(sequence, str):
        raise TypeError("protein sequence must be a string of amino-acid letters")
    seq = _NON_AA.sub("", sequence.strip().upper())
    if not seq:
        raise ValueError("sequence has no valid amino-acid residues")
    return seq


class ProteinEmbedder(Protocol):
    dim: int

    def embed(self, sequence: str) -> np.ndarray: ...


class KmerEmbedder:
    """Deterministic hashed k-mer counts (L2-normalized). No model, no download."""

    def __init__(self, k: int = 3, dim: int = 64) -> None:
        if k < 1 or dim < 1:
            raise ValueError("k and dim must be >= 1")
        self.k = k
        self.dim = dim

    def _bucket(self, kmer: str) -> int:
        return int.from_bytes(hashlib.blake2b(kmer.encode(), digest_size=8).digest(), "big") % self.dim

    def embed(self, sequence: str) -> np.ndarray:
        seq = clean_sequence(sequence)
        vec = np.zeros(self.dim, dtype=float)
        for i in range(len(seq) - self.k + 1):
            vec[self._bucket(seq[i : i + self.k])] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


# Immutable commit pins for the default checkpoints.
#
# `from_pretrained("facebook/esm2_t6_8M_UR50D")` with no `revision` resolves to
# whatever that repo's `main` branch points at TODAY. A branch name is mutable:
# if the upstream repo is ever re-uploaded or its weights re-quantized, every
# embedding this class produces changes, silently, with no diff in our own code.
# That breaks the platform's reproducibility contract - a result must be
# reproducible from committed code and config - and it breaks it in the worst
# way, by changing model features under a fixed random seed while every test
# still passes.
#
# So each default is pinned to an explicit commit SHA. Bumping a pin is a
# deliberate, reviewable act: change the SHA, re-run the embedding tests, and
# note the change, because it invalidates every previously computed embedding.
#
# ESM-2: verified against https://huggingface.co/api/models/facebook/esm2_t6_8M_UR50D
# on 2026-08-08: sha c731040fcd8d73dceaa04b0a8e6329b345b0f5df, upstream
# lastModified 2023-03-21.
DEFAULT_ESM2_MODEL = "facebook/esm2_t6_8M_UR50D"
DEFAULT_ESM2_REVISION = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"

# ESMC: verified against https://huggingface.co/api/models/biohub/ESMC-300M on
# 2026-11-18: sha 96a2d8d61654ac0c08881be44da6c867686449b9, upstream
# lastModified 2026-11-06. Loaded with transformers' native `esmc` model
# (transformers >= 5.16), never the repo's remote code.
DEFAULT_ESMC_MODEL = "biohub/ESMC-300M"
DEFAULT_ESMC_REVISION = "96a2d8d61654ac0c08881be44da6c867686449b9"


class _HFProteinEmbedder:
    """Shared Hugging Face loader: pinned revision, mean-pooled residues."""

    _default_model: str
    _default_revision: str
    _max_length: int

    def __init__(self, model: str, device: str | None, revision: str | None) -> None:
        # Argument validation first, before the heavy optional imports: a caller
        # who mispaired model and revision should get a fast, clear error rather
        # than one that only surfaces on machines with the `protein` extra
        # installed, or a 404 raised deep inside transformers.
        if revision == self._default_revision and model != self._default_model:
            # A commit SHA is repo-specific, so carrying the default pin over to
            # a different checkpoint can never resolve.
            raise ValueError(
                f"revision {self._default_revision!r} is the pin for "
                f"{self._default_model!r} and does not exist in {model!r}; "
                "pass that model's own commit SHA, or revision=None to "
                "track its branch."
            )

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model_name = model
        # Recorded so a run's provenance can state exactly which weights produced
        # its features, not merely which model name was requested.
        self.revision = revision
        # Never run Python shipped with a checkpoint, and never unpickle weights:
        # both would execute code from the model repo on this machine.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, revision=revision, trust_remote_code=False
        )
        self.model = (
            AutoModel.from_pretrained(
                model, revision=revision, trust_remote_code=False, use_safetensors=True
            )
            .to(device)
            .eval()
        )
        self.dim = int(self.model.config.hidden_size)

    def embed(self, sequence: str) -> np.ndarray:
        torch = self._torch
        seq = clean_sequence(sequence)
        with torch.no_grad():
            # truncate to the model context (BOS + residues + EOS) so a long
            # sequence degrades, not crashes.
            enc = self.tokenizer(
                seq, return_tensors="pt", truncation=True, max_length=self._max_length
            ).to(self.device)
            out = self.model(**enc).last_hidden_state[0]  # (L, H) incl. BOS/EOS
            emb = out[1:-1].mean(dim=0)  # mean-pool real residues
        return emb.float().cpu().numpy()


class ESM2Embedder(_HFProteinEmbedder):
    """ESM-2 protein language model (Hugging Face transformers) on MPS/CPU.

    Default model is the small `esm2_t6_8M_UR50D` (8M params, 320-dim, ~30 MB) so
    it runs fast locally; swap `model` for a larger ESM-2 for stronger features.

    `revision` pins the checkpoint to an immutable commit (see
    `DEFAULT_ESM2_REVISION` above for why this is not optional). Pass
    `revision=None` to deliberately track the upstream branch instead, and
    accept that embeddings then depend on when you ran the code rather than on
    what you committed. When swapping `model` for a larger ESM-2, pass that
    model's own commit SHA as `revision` - carrying the default pin over to a
    different repo would raise, since the SHA does not exist there.
    """

    _default_model = DEFAULT_ESM2_MODEL
    _default_revision = DEFAULT_ESM2_REVISION
    # ESM-2's context: BOS + 1022 residues + EOS = 1024.
    _max_length = 1024

    def __init__(
        self,
        model: str = DEFAULT_ESM2_MODEL,
        device: str | None = None,
        *,
        revision: str | None = DEFAULT_ESM2_REVISION,
    ) -> None:
        super().__init__(model, device, revision)


class ESMCEmbedder(_HFProteinEmbedder):
    """ESMC (ESM Cambrian) protein language model (Hugging Face transformers).

    The current ESM model (Biohub, MIT license; https://github.com/Biohub/esm).
    Default is `biohub/ESMC-300M` (960-dim, ~1.2 GB download), which runs on
    CPU; `biohub/ESMC-600M` and `biohub/ESMC-6B` give stronger features and
    want a GPU. Needs transformers >= 5.16, which ships the native `esmc` model.

    `revision` works as in `ESM2Embedder`: the default is pinned to an
    immutable commit, and a larger variant needs its own commit SHA.
    """

    _default_model = DEFAULT_ESMC_MODEL
    _default_revision = DEFAULT_ESMC_REVISION
    # ESMC was trained to a 2048-token context (BOS + 2046 residues + EOS).
    _max_length = 2048

    def __init__(
        self,
        model: str = DEFAULT_ESMC_MODEL,
        device: str | None = None,
        *,
        revision: str | None = DEFAULT_ESMC_REVISION,
    ) -> None:
        super().__init__(model, device, revision)


def build_feature_table(
    products: dict[str, str], embedder: ProteinEmbedder, prefix: str = "prot_emb_"
) -> pd.DataFrame:
    """One embedding row per product. Join onto run data on `product` to add the
    protein embedding as extra surrogate features."""
    rows: list[dict] = []
    for name, seq in products.items():
        e = embedder.embed(seq)
        rows.append({"product": name, **{f"{prefix}{i}": float(e[i]) for i in range(len(e))}})
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_ESM2_MODEL",
    "DEFAULT_ESM2_REVISION",
    "DEFAULT_ESMC_MODEL",
    "DEFAULT_ESMC_REVISION",
    "ESM2Embedder",
    "ESMCEmbedder",
    "KmerEmbedder",
    "ProteinEmbedder",
    "build_feature_table",
    "clean_sequence",
]
