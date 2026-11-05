"""The provenance "run record": a versioned, typed `run` block attached to every
analysis result (docs/CAMPAIGNS_API.md, "Run record").

It answers "what exactly produced this answer?": which bytes went in (sha256 and
size), how they were read (sheet), what was asked for (target, declared roles,
profile, anonymize), how the engine was configured (seed, torch threads, build
sha, library versions, validation mode), and what it saw (rows, features). The
gate decisions (promotion, proposal_gating, validation) are referenced by their
key in the result, never copied, so the record cannot drift from the decision it
describes.

Built once in `kalos.portal.pipeline.compute_upload_analysis`, after the parse
and before publishing, so the sync routes and the background-job worker attach
an identical block for the same bytes (both call the same function). Nothing in
it is a timestamp or a process id; that is what makes "same bytes, same stack,
same thread count -> same record" hold.

The separate `proposal_provenance` block (`proposal_provenance` below) does
carry a timestamp; its `provenance_sha256` excludes it.

Stored inside the existing analyses JSON blob (no migration). Rows written
before this block existed have no `run` key at all.

PRIVACY: with `anonymize=True` the record omits `filename` and `sheet` entirely
(keys absent, not null). Only the hash remains. The stored `dataset` label is
also not the filename then (see `pipeline.publish_upload_analysis`).

Torch-free: importing this module never loads torch, botorch or gpytorch. The
library versions come from `importlib.metadata`, not from importing the libs.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

RUN_SCHEMA_VERSION = 1

# What produced the analysis: an uploaded run sheet (`upload`, both the sync
# route and the job worker), or a re-analysis of a campaign's folded base
# dataset (`reanalyze`, kalos/portal/campaign_routes.py). For `reanalyze` the
# input is the folded dataset serialized as CSV (`fold_bytes`), not an upload.
KINDS = ("upload", "reanalyze")

# The reference gate decisions the record points at, by key in the analysis
# result. The values are key names, not decisions: a consumer follows the key.
GATE_KEYS: dict[str, str] = {
    "promotion": "promotion",
    "proposal_gating": "proposal_gating",
    "validation": "validation",
}

# Libraries whose versions decide the numbers. Looked up by distribution name.
LIBRARY_DISTRIBUTIONS: tuple[str, ...] = ("numpy", "pandas", "torch", "botorch")


def _library_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for dist in LIBRARY_DISTRIBUTIONS:
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def build_sha() -> str:
    """The commit the engine was built from (`KALOS_BUILD_SHA`, set at image
    build time from `github.sha`). "unknown" when the build did not say."""
    return os.environ.get("KALOS_BUILD_SHA", "").strip() or "unknown"


@dataclass(frozen=True)
class RunRecord:
    schema_version: int
    input_sha256: str
    input_bytes: int
    sheet: str | None
    filename: str | None
    target: str | None
    profile: str
    roles_declared: dict[str, Any] | None
    anonymize: bool
    seed: int
    torch_threads: int | None
    torch_threads_actual: int | None
    max_fit_rows: int
    sheet_index: int | None
    sheet_hash: str | None
    build_sha: str
    library_versions: dict[str, str | None]
    validation_mode: str
    n_rows: int
    n_features: int
    kind: str = "upload"
    pending_runs: int = 0
    gates: dict[str, str] = field(default_factory=lambda: dict(GATE_KEYS))
    # Aggregate counts of what the fit excluded (see `drops_summary`). Counts only:
    # no column names, so the block is safe under `anonymize`.
    data_drops: dict[str, Any] | None = None
    header_row: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """The JSON-ready block. Under `anonymize`, `filename` and `sheet` are
        omitted (keys absent), so the stored record cannot carry them."""
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "input_sha256": self.input_sha256,
            "input_bytes": self.input_bytes,
            "target": self.target,
            "profile": self.profile,
            "roles_declared": (
                dict(self.roles_declared) if self.roles_declared is not None else None
            ),
            "anonymize": self.anonymize,
            "seed": self.seed,
            # None = unpinned (KALOS_TORCH_THREADS=0: torch keeps its own default).
            "torch_threads": self.torch_threads,
            # What torch reported when the fit ran; None if torch was never loaded.
            "torch_threads_actual": self.torch_threads_actual,
            "max_fit_rows": self.max_fit_rows,
            # The analysed sheet: its position in the workbook and a salted hash of
            # its name (the same hash the alias uses). Both null for a CSV.
            "sheet_index": self.sheet_index,
            "sheet_hash": self.sheet_hash,
            "build_sha": self.build_sha,
            "library_versions": dict(self.library_versions),
            "validation_mode": self.validation_mode,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "kind": self.kind,
            # Runs in the incubator the batch was told about (0 for an upload,
            # which starts a campaign with none in flight).
            "pending_runs": self.pending_runs,
            "gates": dict(self.gates),
            "data_drops": dict(self.data_drops) if self.data_drops is not None else None,
            # Which sheet row held the column names (0 = the first row).
            "header_row": self.header_row,
        }
        if not self.anonymize:
            out["filename"] = self.filename
            out["sheet"] = self.sheet
        return out


def fold_bytes(df: Any) -> bytes:
    """The input of a `reanalyze` record: the folded base dataset as CSV. Two
    campaigns with the same folded rows give the same bytes, hence the same hash."""
    return df.to_csv(index=False).encode("utf-8")


def build(
    raw_bytes: bytes,
    filename: str | None,
    sheet: str | None,
    target: str | None,
    roles_json: str,
    anonymize: bool,
    profile: str,
    torch_threads: int,
    seed: int,
    n_rows: int,
    n_features: int,
    *,
    kind: str = "upload",
    pending_runs: int = 0,
    sheet_index: int | None = None,
    sheet_hash: str | None = None,
    torch_threads_actual: int | None = None,
    data_drops: dict[str, Any] | None = None,
    header_row: int | None = None,
) -> RunRecord:
    """Assemble the record for one analysis. `raw_bytes` is the exact input the
    parser saw (an upload) or `fold_bytes(df)` (a reanalyze); the hash is over
    those bytes, not over a re-serialized frame."""
    if kind not in KINDS:
        raise ValueError(f"unknown run kind {kind!r}")
    # Deferred: kalos.portal.analysis imports cleanly without torch, but this
    # module should not pull the analysis stack in at import time. Reading the
    # mode through analysis.validation_mode keeps one definition of the rule.
    from kalos.portal.analysis import validation_mode
    from kalos.portal.uploads import MAX_FIT_ROWS  # deferred, as for validation_mode

    roles: dict[str, Any] | None = None
    if roles_json.strip():
        parsed = json.loads(roles_json)
        roles = parsed if isinstance(parsed, dict) else None

    return RunRecord(
        schema_version=RUN_SCHEMA_VERSION,
        input_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        input_bytes=len(raw_bytes),
        sheet=sheet,
        filename=filename,
        target=target,
        profile=profile,
        roles_declared=roles,
        anonymize=bool(anonymize),
        seed=int(seed),
        # 0 is the config's "do not pin" value; it is recorded as null, not 0.
        torch_threads=int(torch_threads) if int(torch_threads) > 0 else None,
        torch_threads_actual=torch_threads_actual,
        max_fit_rows=int(MAX_FIT_ROWS),
        sheet_index=sheet_index,
        sheet_hash=sheet_hash,
        build_sha=build_sha(),
        library_versions=_library_versions(),
        validation_mode=validation_mode(),
        n_rows=int(n_rows),
        n_features=int(n_features),
        kind=kind,
        pending_runs=int(pending_runs),
        data_drops=data_drops,
        header_row=header_row,
    )


def drops_summary(result: dict[str, Any]) -> dict[str, Any]:
    """The `data_drops` block of the run record, taken from an analysis result:
    the runs excluded from the fit, censored runs excluded, and cells by reason.
    Per-column detail stays in the result and its findings (it carries column
    names, which the record never does)."""
    drops = result.get("data_drops") or {}
    return {
        "rows_dropped": int(drops.get("rows_dropped", result.get("n_dropped_incomplete", 0)) or 0),
        "censored_rows_excluded": int(
            drops.get("censored_rows_excluded", result.get("n_censored_excluded", 0)) or 0
        ),
        "cells_by_reason": dict(drops.get("cells_by_reason") or {}),
        "auto_categorical_count": len(result.get("auto_categorical") or []),
    }


# --- proposal provenance (E5-lite) ---------------------------------------- #

PROVENANCE_SCHEMA_VERSION = 1

# The libraries named in the proposal provenance block. Versions only; nothing
# is imported. No model artefact (weights, state_dict, pickle) is stored: every
# analysis refits from the rows, so these fields plus the input bytes, the
# config and the seed are what a refit needs.
PROVENANCE_LIBRARIES: tuple[str, ...] = (
    "torch", "botorch", "gpytorch", "linear-operator", "numpy", "scipy", "pandas",
)

# The run-record fields that configure the fit (not the data, not the stack).
# `config_sha256` is the SHA-256 of these values as canonical JSON. The values
# themselves are in the `run` block; only their names are repeated here.
CONFIG_FIELDS: tuple[str, ...] = (
    "kind", "target", "roles_declared", "profile", "anonymize", "seed",
    "torch_threads", "max_fit_rows", "validation_mode", "sheet_index",
    "header_row", "pending_runs",
)


def _canonical_sha256(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def proposal_provenance(
    run: dict[str, Any], *, engine_version: str, timestamp: int | float | None,
    proposal_fit_restarts: int | None = None,
) -> dict[str, Any]:
    """The `proposal_provenance` block of an analysis result: what produced its
    proposals. Built from the `run` block (`RunRecord.to_dict()`), so it repeats
    the input hash and seed rather than recomputing them.

    `provenance_sha256` hashes every field except `timestamp` and `created_at`:
    the same input bytes, config, seed, engine and library versions give the
    same value. It is a reproducibility key, not a signature.
    """
    from kalos.versions import PINNED_ML, library_versions, public_version

    libs = library_versions(PROVENANCE_LIBRARIES)
    drift = {k: libs.get(k) for k, v in PINNED_ML.items() if public_version(libs.get(k)) != v}
    config = {k: run.get(k) for k in CONFIG_FIELDS}
    # The effective number of hyperparameter starts of the proposal fit (from the
    # result's `model_diagnostics`; decided from n and d only). Part of the config.
    config["proposal_fit_restarts"] = proposal_fit_restarts
    body: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "engine_version": str(engine_version),
        "build_sha": run.get("build_sha") or build_sha(),
        "library_versions": libs,
        # Installed torch/botorch/gpytorch that differ from the pins in
        # kalos/versions.py ({} when they match).
        "pin_drift": drift,
        "seed": run.get("seed"),
        "input_sha256": run.get("input_sha256"),
        "config_sha256": _canonical_sha256(config),
        "config_fields": [*CONFIG_FIELDS, "proposal_fit_restarts"],
        "model_artifact_stored": False,
    }
    ts = int(timestamp) if timestamp is not None else None
    return {
        **body,
        "provenance_sha256": _canonical_sha256(body),
        "timestamp": ts,
        "created_at": (
            datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            if ts is not None else None
        ),
    }


def provenance_ref(analysis: dict[str, Any] | None) -> dict[str, Any] | None:
    """The short reference a started run carries to the published analysis it was
    started against: None when there is no analysis or it predates the block."""
    block = (analysis or {}).get("proposal_provenance")
    if not isinstance(block, dict):
        return None
    keys = ("provenance_sha256", "input_sha256", "config_sha256", "engine_version",
            "seed", "timestamp")
    return {k: block.get(k) for k in keys}
