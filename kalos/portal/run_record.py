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

Stored inside the existing analyses JSON blob (no migration). Rows written
before this block existed read as `run: null`.

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
from importlib import metadata
from typing import Any

RUN_SCHEMA_VERSION = 1

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
    torch_threads: int
    build_sha: str
    library_versions: dict[str, str | None]
    validation_mode: str
    n_rows: int
    n_features: int
    gates: dict[str, str] = field(default_factory=lambda: dict(GATE_KEYS))

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
            "torch_threads": self.torch_threads,
            "build_sha": self.build_sha,
            "library_versions": dict(self.library_versions),
            "validation_mode": self.validation_mode,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "gates": dict(self.gates),
        }
        if not self.anonymize:
            out["filename"] = self.filename
            out["sheet"] = self.sheet
        return out


def build(
    raw_bytes: bytes,
    filename: str,
    sheet: str | None,
    target: str | None,
    roles_json: str,
    anonymize: bool,
    profile: str,
    torch_threads: int,
    seed: int,
    n_rows: int,
    n_features: int,
) -> RunRecord:
    """Assemble the record for one upload. `raw_bytes` is the exact input the
    parser saw; the hash is over those bytes, not over a re-serialized frame."""
    # Deferred: kalos.portal.analysis imports cleanly without torch, but this
    # module should not pull the analysis stack in at import time. Reading the
    # mode through analysis.validation_mode keeps one definition of the rule.
    from kalos.portal.analysis import validation_mode

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
        torch_threads=int(torch_threads),
        build_sha=build_sha(),
        library_versions=_library_versions(),
        validation_mode=validation_mode(),
        n_rows=int(n_rows),
        n_features=int(n_features),
    )
