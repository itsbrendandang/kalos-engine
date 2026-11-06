"""Skill run receipts (`kalos-skill-receipt/1`) and the per-run directories
they live in.

Every tool call gets a fresh run directory
`<root>/<tenant>/<receipt_id>/` holding `receipt.json` and the call's output
artifacts. `root` defaults to `$KALOS_STATE_DIR/skill_runs`. KALOS_STATE_DIR is
required; there is no ~/.kalos fallback.

A receipt moves through a small state machine, forward only:

    running -> analysis_complete -> terminal (result pass | fail)
    running -> terminal (fail)            runtime fault, timeout, block, refusal

A presentation or report step (`kalos.skills.runner.finalize`) may only start
from `analysis_complete`. A terminal receipt is never written again.

A receipt says which bytes ran (skill slug, version, content hash; tool and
engine symbol; engine build and library versions), on what (canonical
arguments and their sha256; sha256 of each input; the seed) and what came out
(per artifact: path, sha256, bytes, label). It never holds input data: inputs
are hashed, never copied. An error carries a machine `error_code` and a fixed
generic message, never the engine's exception text (which can quote cell
values). Kalos-original code.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .schema import DataLabel

RECEIPT_SCHEMA = "kalos-skill-receipt/1"
RECEIPT_FILE = "receipt.json"
RECEIPT_ID_RE = re.compile(r"^skr_[0-9a-f]{32}$")
_TENANT_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,128}$")

State = Literal["running", "analysis_complete", "terminal"]
RunStatus = Literal["completed", "blocked", "refused", "error"]
ErrorCode = Literal["timeout", "input_unreadable", "invalid_result", "output_too_small",
                    "label_mismatch", "path_escape", "internal"]
ERROR_MESSAGES: dict[str, str] = {
    "timeout": "the tool did not finish within its time limit; any late result is discarded",
    "input_unreadable": "the input could not be read",
    "invalid_result": "the engine result did not match the tool's result model",
    "output_too_small": "an output artifact is smaller than its declared minimum size",
    "label_mismatch": "the result label contradicts the skill's declared outputs",
    "path_escape": "an output path resolved outside the run directory",
    "internal": "the engine raised an unexpected error",
}
_NEXT: dict[str, frozenset[str]] = {
    "running": frozenset({"analysis_complete", "terminal"}),
    "analysis_complete": frozenset({"terminal"}),
    "terminal": frozenset(),
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Artifact(_Strict):
    name: str  # the declared skill output this artifact is
    path: str  # relative to the run directory
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    label: DataLabel


class Blocker(_Strict):
    code: str
    message: str = Field(max_length=2000)
    rows: list[int] | None = None


class ErrorInfo(_Strict):
    error_code: ErrorCode
    exception_type: str | None = None
    message: str
    receipt_id: str


class SkillRunReceipt(_Strict):
    schema_id: Literal["kalos-skill-receipt/1"] = "kalos-skill-receipt/1"
    receipt_id: str = Field(pattern=RECEIPT_ID_RE.pattern)
    skill_slug: str
    skill_version: str
    skill_content_hash: str
    tool: str
    symbol: str
    args_json: str
    args_sha256: str
    input_sha256s: list[str] = []
    seed: int | None = None
    build_sha: str
    library_versions: dict[str, str | None] = {}
    state: State
    result: Literal["pass", "fail"] | None = None
    status: RunStatus | None = None
    summary: str = ""
    outputs: list[Artifact] = []
    failures: list[Blocker] = []
    error: ErrorInfo | None = None
    tenant: str
    subject: str | None = None
    campaign_id: str | None = None
    started_at: str
    updated_at: str
    finished_at: str | None = None
    duration_ms: int | None = None


def new_receipt_id() -> str:
    return "skr_" + uuid.uuid4().hex


def utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds")


def _finite(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> bytes:
    """Stable bytes for hashing: sorted keys, no whitespace, non-finite floats
    as null (a missing cell and a NaN cell are the same input)."""
    return json.dumps(_finite(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=str).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def table_sha256(rows: list[dict[str, Any]]) -> str:
    """Hash of a table in the form run records use (`fold_bytes`: CSV without
    the index), so the same rows give the same hash here and in a run record."""
    import pandas as pd

    from kalos.portal.run_record import fold_bytes

    return sha256_hex(fold_bytes(pd.DataFrame(rows)))


def frame_sha256(df: Any) -> str:
    from kalos.portal.run_record import fold_bytes

    return sha256_hex(fold_bytes(df))


def engine_build_sha() -> str:
    from kalos.portal.run_record import build_sha

    return build_sha()


def library_versions() -> dict[str, str | None]:
    """Installed versions of the numeric stack, read from package metadata (no
    import, so this never loads torch)."""
    from importlib import metadata

    from kalos.portal.run_record import LIBRARY_DISTRIBUTIONS

    out: dict[str, str | None] = {}
    for dist in (*LIBRARY_DISTRIBUTIONS, "scipy", "scikit-learn", "gpytorch", "kalos"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


class StateDirNotSet(RuntimeError):
    """`KALOS_STATE_DIR` is required for skill runs; there is no home-directory
    fallback, so runs never land in a developer's real ~/.kalos by accident."""


def default_root() -> Path:
    state = os.environ.get("KALOS_STATE_DIR", "").strip()
    if not state:
        raise StateDirNotSet(
            "KALOS_STATE_DIR is not set: skill runs and receipts need an explicit state "
            "directory (set KALOS_STATE_DIR, or pass RunStore(root=...))")
    return Path(state) / "skill_runs"


class TransitionError(RuntimeError):
    """A receipt write that would move the state machine backwards or touch a
    terminal receipt."""


class RunStore:
    """Per-tenant run directories, each with one receipt.json."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_root()

    def tenant_dir(self, tenant: str) -> Path:
        if not _TENANT_RE.match(tenant) or tenant in (".", ".."):
            raise ValueError("invalid tenant id for a run path")
        return self.root / tenant

    def run_dir(self, tenant: str, receipt_id: str) -> Path:
        if not RECEIPT_ID_RE.match(receipt_id):
            raise ValueError("invalid receipt id")
        return self.tenant_dir(tenant) / receipt_id

    def create_run(self, receipt: SkillRunReceipt) -> Path:
        """Make the fresh run directory (must not exist) and write the first
        (`running`) receipt."""
        if receipt.state != "running":
            raise TransitionError("a run starts in state running")
        d = self.run_dir(receipt.tenant, receipt.receipt_id)
        d.parent.mkdir(parents=True, exist_ok=True)
        d.mkdir(exist_ok=False)
        self._write(d, receipt)
        return d

    def update(self, receipt: SkillRunReceipt) -> None:
        d = self.run_dir(receipt.tenant, receipt.receipt_id)
        prev = self.read(d)
        if receipt.state not in _NEXT[prev.state]:
            raise TransitionError(f"{prev.state} -> {receipt.state} is not allowed")
        self._write(d, receipt)

    @staticmethod
    def read(run_dir: Path) -> SkillRunReceipt:
        return SkillRunReceipt.model_validate_json(
            (Path(run_dir) / RECEIPT_FILE).read_text(encoding="utf-8"))

    @staticmethod
    def _write(d: Path, receipt: SkillRunReceipt) -> None:
        data = json.dumps(receipt.model_dump(), indent=1, sort_keys=True) + "\n"
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(data)
            os.replace(tmp, d / RECEIPT_FILE)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def get(self, tenant: str, receipt_id: str) -> SkillRunReceipt | None:
        """The receipt, or None when absent or owned by another tenant."""
        if not RECEIPT_ID_RE.match(receipt_id):
            return None
        d = self.run_dir(tenant, receipt_id)
        if not (d / RECEIPT_FILE).is_file():
            return None
        return self.read(d)

    def list_for(self, tenant: str, *, skill: str | None = None) -> list[SkillRunReceipt]:
        d = self.tenant_dir(tenant)
        if not d.is_dir():
            return []
        out = [self.read(p) for p in d.glob("skr_*") if (p / RECEIPT_FILE).is_file()]
        out = [r for r in out if skill is None or r.skill_slug == skill]
        out.sort(key=lambda r: (r.started_at, r.receipt_id))
        return out


__all__ = [
    "ERROR_MESSAGES",
    "RECEIPT_FILE",
    "RECEIPT_SCHEMA",
    "Artifact",
    "Blocker",
    "ErrorInfo",
    "RunStore",
    "SkillRunReceipt",
    "StateDirNotSet",
    "TransitionError",
    "canonical_json",
    "engine_build_sha",
    "frame_sha256",
    "library_versions",
    "new_receipt_id",
    "sha256_hex",
    "sha256_json",
    "table_sha256",
]
