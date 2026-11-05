"""Skill run receipts: one JSON record per tool call, written whether the call
succeeded, was refused, or failed.

A receipt says which bytes ran (skill slug, version and folder `content_hash`;
tool name and engine symbol; engine `build_sha`), on what (sha256 of the
validated arguments and, for table inputs, of the input data in the same CSV
form `kalos.portal.run_record.fold_bytes` uses), with which seed, and what came
out (one entry per output with its sha256 and its data label: measured,
predicted, simulated, synthetic, derived or metadata). It never stores the input
data or the output values themselves, only their hashes.

A refusal is a successful, receipted outcome (`outcome="refused"` with the
engine's reason), not an error to retry around.

Storage: `<root>/<tenant>/<receipt_id>.json`, root defaulting to
`$KALOS_STATE_DIR/skill_receipts` (`~/.kalos/skill_receipts`). Files are written
atomically (temp file + rename) and never overwritten. Kalos-original code.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .schema import DataLabel

RECEIPT_SCHEMA = "kalos-skill-receipt/1"
RECEIPT_ID_RE = re.compile(r"^skr_[0-9a-f]{32}$")
_TENANT_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,128}$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OutputRecord(_Strict):
    name: str
    label: DataLabel
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Refusal(_Strict):
    code: str
    reason: str = Field(max_length=2000)


class SkillRunReceipt(_Strict):
    schema_id: Literal["kalos-skill-receipt/1"] = "kalos-skill-receipt/1"
    receipt_id: str = Field(pattern=RECEIPT_ID_RE.pattern)
    skill_slug: str
    skill_version: str
    skill_content_hash: str
    tool: str
    symbol: str
    args_sha256: str
    input_sha256: str | None = None
    input_rows: int | None = None
    seed: int | None = None
    build_sha: str
    outcome: Literal["ok", "refused", "error"]
    outputs: list[OutputRecord] = []
    refusal: Refusal | None = None
    tenant: str
    subject: str | None = None
    campaign_id: str | None = None
    started_at: str
    finished_at: str
    duration_ms: int


def new_receipt_id() -> str:
    return "skr_" + uuid.uuid4().hex


def utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds")


def canonical_json(obj: Any) -> bytes:
    """Stable bytes for hashing: sorted keys, no whitespace, NaN refused
    (non-finite floats must have been mapped to None before this)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=str).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def table_sha256(rows: list[dict[str, Any]]) -> str:
    """Hash of a table input in the form run records use (`fold_bytes`: the
    frame as CSV without the index), so the same rows give the same hash here
    and in a run record."""
    import pandas as pd

    from kalos.portal.run_record import fold_bytes

    return sha256_hex(fold_bytes(pd.DataFrame(rows)))


def engine_build_sha() -> str:
    from kalos.portal.run_record import build_sha

    return build_sha()


def default_root() -> Path:
    return Path(os.environ.get("KALOS_STATE_DIR", Path.home() / ".kalos")) / "skill_receipts"


class ReceiptStore:
    """Append-only, per-tenant receipt files."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_root()

    def _dir(self, tenant: str) -> Path:
        if not _TENANT_RE.match(tenant):
            raise ValueError("invalid tenant id for a receipt path")
        return self.root / tenant

    def write(self, receipt: SkillRunReceipt) -> Path:
        d = self._dir(receipt.tenant)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{receipt.receipt_id}.json"
        if path.exists():
            raise FileExistsError(f"receipt {receipt.receipt_id} already written")
        data = json.dumps(receipt.model_dump(by_alias=True), indent=1, sort_keys=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(data + "\n")
            os.link(tmp, path)  # fails if `path` appeared meanwhile: never overwrite
        finally:
            os.unlink(tmp)
        return path

    def get(self, tenant: str, receipt_id: str) -> SkillRunReceipt | None:
        """The receipt, or None when absent or owned by another tenant."""
        if not RECEIPT_ID_RE.match(receipt_id):
            return None
        path = self._dir(tenant) / f"{receipt_id}.json"
        if not path.is_file():
            return None
        return SkillRunReceipt.model_validate_json(path.read_text(encoding="utf-8"))

    def list_for(self, tenant: str, *, skill: str | None = None) -> list[SkillRunReceipt]:
        d = self._dir(tenant)
        if not d.is_dir():
            return []
        out = []
        for p in sorted(d.glob("skr_*.json")):
            r = SkillRunReceipt.model_validate_json(p.read_text(encoding="utf-8"))
            if skill is None or r.skill_slug == skill:
                out.append(r)
        out.sort(key=lambda r: (r.started_at, r.receipt_id))
        return out


__all__ = [
    "RECEIPT_SCHEMA",
    "OutputRecord",
    "ReceiptStore",
    "Refusal",
    "SkillRunReceipt",
    "canonical_json",
    "engine_build_sha",
    "new_receipt_id",
    "sha256_json",
    "table_sha256",
]
