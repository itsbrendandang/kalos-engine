"""The one runner for skill tools. Skills never carry their own runner.

`invoke(skill, tool, args, ctx)`:

1. checks the skill lists the tool, is not `planned`, and the caller holds the
   skill's and the tool's scope; validates the arguments (bad arguments stop
   here, with no receipt);
2. creates a fresh run directory `<root>/<tenant>/<receipt_id>/` and writes a
   `running` receipt;
3. calls the adapter on a worker thread with the tool's timeout;
4. validates the result, checks its label against the skill's declared
   outputs, writes the artifacts (JSON, plus a CSV for table outputs) inside
   the run directory, checks each against its declared `min_bytes`, and moves
   the receipt to `analysis_complete`;
5. moves the receipt to `terminal` with result `pass` (unless `hold=True`, which
   leaves it at `analysis_complete` for `finalize`).

Every non-completed outcome is a receipted `terminal`/`fail`:

- `blocked`: the data cannot support a result (too few rows, a blocking
  finding, a failed fit);
- `refused`: the request is not allowed or not well-formed for this tool;
- `error`: a runtime fault, with an `error_code` and a fixed message. The
  engine's exception text is never stored.

Blocked and refused runs are results, not exceptions, and are never retried.

The returned `ToolStatus` is small JSON: status, summary, artifacts (path,
sha256, bytes, label), blockers. Data never travels inline; read the artifacts.

Timeouts: a Python thread cannot be stopped. On timeout the receipt becomes
terminal/fail (`error_code=timeout`), the runner stops waiting and discards any
late result; the computation may keep using CPU until it ends. Tools marked
`long_running` belong on the killable job-worker subprocess once a run route
exists. v1 starts no child processes; any future subprocess runner must pass a
minimal, explicit environment.

Kalos-original code.
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import dataclasses
import io
import json
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from . import bindings as b
from .bindings import ToolContext, ToolRefusal
from .loader import Skill
from .receipts import (
    ERROR_MESSAGES,
    Artifact,
    Blocker,
    ErrorInfo,
    RunStore,
    SkillRunReceipt,
    canonical_json,
    engine_build_sha,
    library_versions,
    new_receipt_id,
    sha256_hex,
    sha256_json,
    utcnow,
)
from .tools import TOOLS, ToolSpec, artifact_targets

# Exception classes (matched by name anywhere in the MRO, so this module never
# imports them, or torch) that mean the engine refused or could not support the
# input. Everything else is a runtime fault.
_BLOCKED_BY_NAME = {"UploadRejected": "upload_rejected", "FitError": "fit_failed"}
_REFUSED_BY_NAME = {"CampaignNotFound": "not_found", "CampaignArchived": "archived",
                    "CampaignError": "campaign_error"}
_UNREADABLE_NAMES = frozenset({"ParserError", "EmptyDataError", "UnicodeDecodeError",
                               "BadZipFile", "InvalidFileException"})


class ToolCallError(Exception):
    """The call could not start (planned skill, unknown tool, scope, bad
    arguments). Raised before any run directory or receipt exists."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ToolStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["completed", "blocked", "refused", "error"]
    summary: str
    artifacts: list[Artifact] = []
    blockers: list[Blocker] = []
    error: ErrorInfo | None = None
    receipt_id: str


@dataclasses.dataclass(frozen=True)
class RunResult:
    status: ToolStatus
    receipt: SkillRunReceipt
    run_dir: Path

    def load(self, name: str) -> Any:
        """Read one artifact back (JSON parsed; CSV as a list of row dicts)."""
        art = next(a for a in self.status.artifacts if a.name == name)
        path = self.run_dir / art.path
        if path.suffix == ".csv":
            return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
        return json.loads(path.read_text(encoding="utf-8"))


def guard_measured_write(label: str) -> None:
    """Honesty invariant: only `measured` values may be written into a campaign's
    measured data. Any write path calls this before storing a skill result. No
    v1 tool writes campaign data."""
    if label != "measured":
        raise ToolRefusal("label_mismatch",
                          f"a {label} result can never be written as measured data")


def _classify(exc: BaseException) -> tuple[str, str, str, list[int] | None]:
    """(status, code, message, rows) for an exception from an adapter."""
    if isinstance(exc, ToolRefusal):
        return exc.kind, exc.code, exc.reason[:2000], exc.rows
    names = [c.__name__ for c in type(exc).__mro__]
    for n in names:
        if n in _BLOCKED_BY_NAME:
            code = getattr(exc, "code", None) or _BLOCKED_BY_NAME[n]
            msg = getattr(exc, "message", None) or ERROR_MESSAGES.get(code, "blocked")
            # UploadRejected messages are already client-facing engine text
            # (no cell values); a FitError's is numeric internals - keep generic.
            if n == "FitError":
                msg = "the model fit failed; no fallback result is produced"
            return "blocked", str(code), str(msg)[:2000], None
        if n in _REFUSED_BY_NAME:
            return "refused", _REFUSED_BY_NAME[n], "campaign not available", None
    return "error", "input_unreadable" if set(names) & _UNREADABLE_NAMES else "internal", "", None


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: "" if r.get(k) is None else r.get(k) for k in cols})
    return buf.getvalue().encode("utf-8")


def _write_artifact(run_dir: Path, filename: str, data: bytes) -> Path:
    path = (run_dir / filename).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError:
        raise _Fault("path_escape") from None
    if path.exists():
        raise _Fault("path_escape")
    path.write_bytes(data)
    return path


class _Fault(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def invoke(skill: Skill, tool: str, args: dict[str, Any], ctx: ToolContext, *,
           timeout_s: float | None = None, hold: bool = False) -> RunResult:
    m = skill.meta
    if m.status == "planned":
        raise ToolCallError("planned", f"skill {m.slug} is planned; it has no tools yet")
    if tool not in m.tools:
        raise ToolCallError("tool_not_in_skill", f"{tool} is not a tool of {m.slug}")
    spec: ToolSpec | None = TOOLS.get(tool)
    if spec is None:
        raise ToolCallError("unknown_tool", f"{tool} is not registered")
    needed = {m.scope, spec.scope}
    if not needed <= set(ctx.scopes):
        raise ToolCallError("forbidden", f"requires scope(s) {sorted(needed)}")
    try:
        parsed = spec.args_model.model_validate(args)
    except ValidationError as exc:
        raise ToolCallError("invalid_args",
                            f"invalid arguments for {tool}: {exc.error_count()} error(s)") from exc

    store: RunStore = ctx.runs if ctx.runs is not None else RunStore()
    dumped = parsed.model_dump(mode="json")
    args_json = canonical_json(dumped).decode("utf-8")
    cid = dumped.get("campaign_id")
    seed = dumped.get("seed") if isinstance(dumped.get("seed"), int) else None
    rid = new_receipt_id()
    t0 = time.monotonic()
    started = utcnow()
    receipt = SkillRunReceipt(
        receipt_id=rid, skill_slug=m.slug, skill_version=m.version,
        skill_content_hash=skill.content_hash, tool=tool, symbol=spec.symbol,
        args_json=args_json, args_sha256=sha256_hex(args_json.encode("utf-8")), seed=seed,
        build_sha=engine_build_sha(), library_versions=library_versions(), state="running",
        tenant=ctx.tenant, subject=ctx.subject,
        campaign_id=cid if isinstance(cid, str) else None, started_at=started,
        updated_at=started)
    run_dir = store.create_run(receipt)
    call_ctx = dataclasses.replace(ctx, input_sha256s=[])

    def done(**upd: Any) -> RunResult:
        nonlocal receipt
        now = utcnow()
        receipt = receipt.model_copy(update={
            "input_sha256s": list(call_ctx.input_sha256s), "updated_at": now, **upd})
        if receipt.state == "terminal":
            receipt = receipt.model_copy(update={
                "finished_at": now, "duration_ms": int((time.monotonic() - t0) * 1000)})
        store.update(receipt)
        status = ToolStatus(status=receipt.status or "completed", summary=receipt.summary,
                            artifacts=receipt.outputs, blockers=receipt.failures,
                            error=receipt.error, receipt_id=rid)
        return RunResult(status=status, receipt=receipt, run_dir=run_dir)

    def fault(code: str, exc_type: str | None = None) -> RunResult:
        err = ErrorInfo(error_code=code, exception_type=exc_type,  # type: ignore[arg-type]
                        message=ERROR_MESSAGES[code], receipt_id=rid)
        return done(state="terminal", result="fail", status="error",
                    summary=f"{tool}: {ERROR_MESSAGES[code]}", error=err)

    # --- run ------------------------------------------------------------------ #
    adapter = getattr(b, spec.adapter)
    ex = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="kalos-skill-tool")
    try:
        fut = ex.submit(adapter, parsed, call_ctx)
        try:
            data = fut.result(timeout=timeout_s if timeout_s is not None else spec.timeout_s)
        except cf.TimeoutError:
            # Stop waiting; whatever the thread returns later is never read.
            return fault("timeout")
        except Exception as exc:  # noqa: BLE001 - classified, always receipted
            status, code, msg, rows = _classify(exc)
            if status == "error":
                return fault(code, type(exc).__name__)
            return done(state="terminal", result="fail", status=status,
                        summary=f"{tool}: {status} ({code})",
                        failures=[Blocker(code=code, message=msg, rows=rows)])
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    # --- check and write outputs ------------------------------------------------ #
    try:
        result = spec.result_model.model_validate(data)
    except ValidationError:
        return fault("invalid_result")
    if spec.label not in m.output_labels:
        return fault("label_mismatch")
    json_name, csv_name = artifact_targets(skill, spec)
    if json_name is None or (spec.csv and csv_name is None):
        return fault("label_mismatch")
    out = result.model_dump(mode="json")
    blobs: list[tuple[str, str, bytes]] = []
    if spec.csv:
        assert csv_name is not None
        blobs.append((csv_name, f"{csv_name}.csv", _csv_bytes(out.pop(spec.csv))))
    blobs.append((json_name, f"{json_name}.json",
                  json.dumps(out, sort_keys=True, indent=1, allow_nan=False).encode("utf-8")))
    artifacts: list[Artifact] = []
    try:
        for name, filename, blob in blobs:
            decl = m.output(name)
            assert decl is not None
            if len(blob) < decl.min_bytes:
                raise _Fault("output_too_small")
            _write_artifact(run_dir, filename, blob)
            artifacts.append(Artifact(name=name, path=filename, sha256=sha256_hex(blob),
                                      bytes=len(blob), label=decl.label))
    except _Fault as f:
        return fault(f.code)
    summary = f"{tool}: completed, {len(artifacts)} artifact(s) labelled {spec.label}"
    res = done(state="analysis_complete", status="completed", summary=summary,
               outputs=artifacts)
    if hold:
        return res
    return done(state="terminal", result="pass")


def finalize(run: RunResult, *, present: Any = None, store: RunStore | None = None) -> RunResult:
    """The presentation/report step. Only starts from `analysis_complete`.
    `present(run_dir, receipt) -> list[(name, filename, bytes, label)]` may add
    report artifacts; the receipt then becomes terminal/pass."""
    store = store or RunStore(run.run_dir.parent.parent)
    receipt = store.read(run.run_dir)
    if receipt.state != "analysis_complete":
        raise ToolCallError("bad_state", f"finalize needs analysis_complete, not {receipt.state}")
    outputs = list(receipt.outputs)
    for name, filename, blob, label in (present(run.run_dir, receipt) if present else []):
        try:
            _write_artifact(run.run_dir, filename, blob)
        except (_Fault, OSError):
            # An escaping, clashing or unwritable report path fails the run: the
            # receipt becomes terminal/fail and keeps only the analysis outputs.
            now = utcnow()
            err = ErrorInfo(error_code="path_escape", message=ERROR_MESSAGES["path_escape"],
                            receipt_id=receipt.receipt_id)
            receipt = receipt.model_copy(update={
                "state": "terminal", "result": "fail", "status": "error", "error": err,
                "summary": f"{receipt.tool}: {ERROR_MESSAGES['path_escape']}",
                "updated_at": now, "finished_at": now})
            store.update(receipt)
            status = run.status.model_copy(update={"status": "error", "error": err,
                                                   "summary": receipt.summary})
            return RunResult(status=status, receipt=receipt, run_dir=run.run_dir)
        outputs.append(Artifact(name=name, path=filename, sha256=sha256_hex(blob),
                                bytes=len(blob), label=label))
    now = utcnow()
    receipt = receipt.model_copy(update={"state": "terminal", "result": "pass", "outputs": outputs,
                                         "updated_at": now, "finished_at": now})
    store.update(receipt)
    status = run.status.model_copy(update={"artifacts": outputs})
    return RunResult(status=status, receipt=receipt, run_dir=run.run_dir)


class Verification(BaseModel):
    ok: bool
    problems: list[str]


def verify_receipt(run_dir: Path, skill: Skill | None) -> Verification:
    """Passes only if the receipt is terminal/pass, the skill's current content
    hash equals the one the run used, and every output file is still there with
    the recorded sha256 and size (and inside the run directory)."""
    problems: list[str] = []
    run_dir = Path(run_dir)
    try:
        r = RunStore.read(run_dir)
    except (OSError, ValidationError, ValueError):
        return Verification(ok=False, problems=["receipt missing or unreadable"])
    if r.state != "terminal" or r.result != "pass":
        problems.append(f"state is {r.state}/{r.result}, not terminal/pass")
    if skill is None:
        problems.append(f"skill {r.skill_slug} not found")
    elif skill.slug != r.skill_slug or skill.content_hash != r.skill_content_hash:
        problems.append("skill content hash changed since the run")
    for a in r.outputs:
        p = (run_dir / a.path).resolve()
        try:
            p.relative_to(run_dir.resolve())
        except ValueError:
            problems.append(f"{a.path}: outside the run directory")
            continue
        if not p.is_file():
            problems.append(f"{a.path}: missing")
            continue
        blob = p.read_bytes()
        if len(blob) != a.bytes or sha256_hex(blob) != a.sha256:
            problems.append(f"{a.path}: changed since the run")
    return Verification(ok=not problems, problems=problems)


__all__ = [
    "RunResult",
    "ToolCallError",
    "ToolContext",
    "ToolRefusal",
    "ToolStatus",
    "Verification",
    "finalize",
    "guard_measured_write",
    "invoke",
    "sha256_json",
    "verify_receipt",
]
