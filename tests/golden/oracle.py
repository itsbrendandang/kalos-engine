"""Golden oracle (reorg guard R02): the engine's observable output, pinned.

`run_scenario()` drives the real FastAPI app over HTTP (TestClient, a fresh
state directory, one torch thread, deterministic ids) through a fixed scenario
and returns one record per case:

- `run/<sheet>`: `POST /api/run` on each of the four committed synthetic sheets;
- `campaign/<sheet>`: the same sheet through a campaign (setup where the sheet
  needs one, upload, `GET .../analysis`);
- `fold/round1`, `fold/round2`: a two-round campaign fold (start runs from the
  proposals, record results, reanalyze), with the analysis after each round;
- `scale/readout_json` (and the HTML readout's status/headers/hash);
- `http/<NN>-<method>-<path>`: status, headers (minus volatile ones) and body for
  every other route the scenario exercises (campaign CRUD, setup, stages, gates,
  clones, exports, history, operating ranges, jobs (queued then cancelled; no
  worker), the legacy single-campaign routes, the assistant (offline), probes).

Records are canonical JSON: key order is the response's own (so a reordering
is a diff), floats round-trip exactly, and only values that differ between two
identical runs are scrubbed (wall-clock times, durations, content hashes of
time-stamped bodies). Ids are deterministic (uuid4 / token_hex are replaced by
counters before the engine is imported), so ids are compared too.

Byte-identical on the SAME pinned stack and platform only (decision D7): the
records live under `tests/golden/<env key>/`, and the CI image's key is the one
that gates merges (.github/workflows/golden.yml). See scripts/golden.py.
"""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import os
import platform
import re
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = ROOT / "tests" / "golden"
EXAMPLES = ROOT / "examples"
ASSIST_DOCS = GOLDEN_DIR / "assist_docs"
TENANT_TOKEN = "golden-admin"

# Keys whose values are wall-clock or elapsed time. Scrubbed wherever they appear,
# as is every key ending in `_at` (an epoch or ISO time by convention here).
VOLATILE_KEYS = frozenset({
    "updated",
    "timestamp", "created_at", "updated_at", "started_at", "finished_at", "completed_at", "queued_at",
    "decided_at", "recorded_at", "measured_at", "archived_at", "deleted_at", "last_activity_at",
    "fit_seconds", "elapsed_s", "elapsed_seconds", "duration_s", "duration_ms", "wall_seconds",
    "heartbeat_at", "built_at", "generated_at", "at", "ts", "time", "when", "server_time",
    "uptime_s",
})
# Response headers that vary per request (or hash a body that carries a time).
VOLATILE_HEADERS = frozenset({"date", "x-request-id", "content-length", "x-kalos-content-sha256", "etag",
                              "x-kalos-request-id", "server-timing"})
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")


# --- environment key ------------------------------------------------------------- #

def env_key() -> str:
    """The platform and pinned stack the records are valid for. Changes when the
    platform, the Python minor version or any numeric library version changes."""
    import botorch
    import gpytorch
    import numpy
    import pandas
    import scipy
    import sklearn
    import torch

    machine = platform.machine().lower().replace("amd64", "x86_64").replace("aarch64", "arm64")
    parts = [
        sys.platform, machine, f"py{sys.version_info.major}.{sys.version_info.minor}",
        f"torch{torch.__version__.split('+')[0]}", f"botorch{botorch.__version__}",
        f"gpytorch{gpytorch.__version__}", f"numpy{numpy.__version__}", f"scipy{scipy.__version__}",
        f"pandas{pandas.__version__}", f"sklearn{sklearn.__version__}",
    ]
    return "-".join(parts)


def env_details() -> dict[str, Any]:
    import torch

    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            cpu = next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")), cpu)
    except OSError:
        pass
    pins = {k: os.environ[k] for k in ("MKL_CBWR", "OPENBLAS_CORETYPE", "ATEN_CPU_CAPABILITY") if k in os.environ}
    return {"key": env_key(), "platform": platform.platform(), "python": platform.python_version(),
            "torch_threads": torch.get_num_threads(), "cpu": cpu, "kernel_pins": pins}


# --- canonical form -------------------------------------------------------------- #

def scrub(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: scrub(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if key is not None and (key in VOLATILE_KEYS or key.endswith("_at")) and value is not None:
        return "<volatile>"
    if isinstance(value, str) and _ISO.fullmatch(value):
        return "<timestamp>"
    return value


def canonical(record: Any) -> str:
    """The bytes compared. Key order is preserved (not sorted) on purpose."""
    return json.dumps(record, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


# --- the four sheets --------------------------------------------------------------- #

def _csv(df: Any) -> bytes:
    return str(df.to_csv(index=False)).encode()


def sheet_bioprocess_subset() -> tuple[bytes, str]:
    import pandas as pd

    df = pd.read_csv(EXAMPLES / "synthetic_bioprocess" / "synthetic_bioprocess.csv").head(30)
    return _csv(df[["pH", "temperature_C", "glucose_gL", "titer_g_L"]]), "titer_g_L"


def sheet_scaleup() -> tuple[bytes, str]:
    import pandas as pd

    df = pd.read_csv(EXAMPLES / "synthetic_scaleup" / "synthetic_scaleup.csv")
    return _csv(df), "titer_g_per_L"


def sheet_categorical_constrained() -> tuple[bytes, str]:
    """24 runs: two continuous inputs and a 3-level medium; the campaign case
    adds bounds and a linear constraint (a + b <= 12) through PUT setup."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(11)
    a = rng.uniform(0, 10, 24).round(2)
    b = rng.uniform(1, 6, 24).round(2)
    medium = np.array(["CD-A", "CD-B", "CD-C"])[np.arange(24) % 3]
    effect = {"CD-A": 0.0, "CD-B": 0.8, "CD-C": -0.5}
    y = 2 + 0.4 * a - 0.15 * (b - 3.5) ** 2 + np.array([effect[m] for m in medium]) + rng.normal(0, 0.15, 24)
    return _csv(pd.DataFrame({"a": a, "b": b, "medium": medium, "titer": y.round(3)})), "titer"


def sheet_minimize() -> tuple[bytes, str]:
    """24 runs, goal minimize (set through PUT setup on the campaign case)."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(23)
    x1 = rng.uniform(0, 4, 24).round(3)
    x2 = rng.uniform(5, 7, 24).round(2)
    y = 1.0 + 0.6 * (x1 - 2.2) ** 2 + 0.9 * (x2 - 6.1) ** 2 + rng.normal(0, 0.08, 24)
    return _csv(pd.DataFrame({"feed": x1, "pH": x2, "impurity_pct": y.round(3)})), "impurity_pct"


SHEETS: dict[str, Callable[[], tuple[bytes, str]]] = {
    "bioprocess_subset": sheet_bioprocess_subset,
    "scaleup": sheet_scaleup,
    "categorical_constrained": sheet_categorical_constrained,
    "minimize": sheet_minimize,
}
SETUPS: dict[str, dict[str, Any]] = {
    "categorical_constrained": {
        "goal": "maximize",
        "bounds": {"a": {"lo": 0, "hi": 10}},
        "linear_constraints": [{"coefficients": {"a": 1, "b": 1}, "op": "<=", "rhs": 12}],
        "target_unit": "g/L",
    },
    "minimize": {"goal": "minimize", "target_unit": "%"},
}


# --- the scenario -------------------------------------------------------------- #

def _deterministic_ids() -> None:
    """Replace the id sources before kalos is imported. Only ids change: uuid4
    and token_hex become counters, so every id in every response is stable."""
    import secrets
    import uuid

    counter = itertools.count(1)
    uuid.uuid4 = lambda: uuid.UUID(int=next(counter))  # type: ignore[assignment]
    secrets.token_hex = lambda nbytes=None: f"{next(counter):0{2 * (nbytes or 32)}x}"  # type: ignore[assignment]


class _Recorder:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.records: dict[str, Any] = {}
        self.n = 0
        # A fixed request id: a middleware that would otherwise mint one per request
        # (feat/voyager-records' X-Request-ID) must not consume the id counter.
        self.headers = {"Authorization": f"Bearer {TENANT_TOKEN}", "X-Request-ID": "golden"}

    def call(self, method: str, url: str, *, name: str | None = None, store: bool = True,
             **kwargs: Any) -> Any:
        response = self.client.request(method, url, headers={**self.headers, **kwargs.pop("headers", {})},
                                       **kwargs)
        if store:
            self.n += 1
            label = name or f"http/{self.n:02d}-{method}-{_slug(url)}"
            self.records[label] = _response_record(method, url, response)
        return response


def _slug(url: str) -> str:
    url = re.sub(r"cmp_[0-9a-f]{32}", "CID", url)
    url = re.sub(r"job_[0-9a-f]{32}", "JOB", url)
    url = re.sub(r"sop_[0-9a-f]{32}", "SOP", url)
    url = re.sub(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])", "RUN", url)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", url.strip("/"))[:80]


# Routes whose numbers are NOT reproducible run to run (the demo simulators draw
# unseeded random data; found by this oracle, see REORG_PLAN R02 notes). Their
# status, headers and the SHAPE of the body (keys and value types) are pinned.
SHAPE_ONLY = ("/api/single", "/api/multi")


def shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return type(value).__name__


def _response_record(method: str, url: str, response: Any) -> dict[str, Any]:
    ctype = response.headers.get("content-type", "")
    headers = {k: v for k, v in sorted(response.headers.items()) if k.lower() not in VOLATILE_HEADERS}
    record: dict[str, Any] = {"request": f"{method} {url}", "status": response.status_code, "headers": headers}
    if "json" in ctype and url.split("?")[0] in SHAPE_ONLY:
        record["body_shape"] = shape(response.json())
    elif "json" in ctype:
        record["body"] = scrub(response.json())
    elif ctype.startswith("text/"):
        # Rendered text (the HTML readout): pinned verbatim, render times scrubbed.
        record["body_text"] = _ISO.sub("<timestamp>", response.text)
    else:
        body = response.content
        record["body_sha256"] = hashlib.sha256(body).hexdigest()
        record["body_bytes"] = len(body)
    return record


def _upload(rec: _Recorder, url: str, sheet: str, *, name: str | None = None, **form: Any) -> Any:
    raw, target = SHEETS[sheet]()
    files = {"file": (f"{sheet}.csv", io.BytesIO(raw), "text/csv")}
    return rec.call("POST", url, name=name, files=files, data={"target": target, **form})



def run_scenario() -> dict[str, Any]:
    """Run the whole scenario in THIS process. Call it in a fresh interpreter
    (scripts/golden.py does): it sets env vars, replaces the id sources and
    imports the app."""
    import tempfile

    state = tempfile.mkdtemp(prefix="kalos-golden-")
    tokens = [{"token_sha256": hashlib.sha256(TENANT_TOKEN.encode()).hexdigest(), "subject": "golden@lab",
               "tenant": "golden", "scopes": ["read", "write", "admin"]}]
    os.environ.update({
        "KALOS_STATE_DIR": state, "KALOS_TORCH_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "KALOS_AUTH_TOKENS": json.dumps(tokens), "KALOS_ASSIST_DOCS_DIR": str(ASSIST_DOCS),
        "KALOS_LLM_PROVIDER": "none", "KALOS_ASSIST_INDEX_ON_WRITE": "0",
        # Provenance carries the build SHA (env, else `git rev-parse`): pinned, or
        # every commit would change the scale readout.
        "KALOS_BUILD_SHA": "golden-build",
    })
    for var in ("KALOS_AUTH_TOKENS_FILE", "KALOS_JWT_PUBLIC_KEYS_FILE", "KALOS_ENV", "KALOS_ASSIST_SKILLS_DIR",
                "KALOS_ANON_SALT", "ANTHROPIC_API_KEY", "KALOS_GIT_SHA"):
        os.environ.pop(var, None)
    _deterministic_ids()

    import torch
    from fastapi.testclient import TestClient

    from kalos.portal.app import app

    torch.set_num_threads(1)
    # No lifespan (no `with`): the job runner never starts, so queued jobs stay
    # queued and the job routes are deterministic.
    rec = _Recorder(TestClient(app))
    _probes(rec)
    for sheet in SHEETS:
        _upload(rec, "/api/run", sheet, name=f"run/{sheet}")
    cids = {sheet: _campaign_case(rec, sheet) for sheet in SHEETS}
    _fold(rec, cids["bioprocess_subset"])
    _campaign_routes(rec, cids["categorical_constrained"])
    _scale(rec)
    _legacy(rec)
    _jobs(rec, cids["minimize"])
    _assist(rec, cids["bioprocess_subset"])
    return rec.records


def _probes(rec: _Recorder) -> None:
    rec.call("GET", "/healthz", headers={"Authorization": ""})
    rec.call("GET", "/readyz", headers={"Authorization": ""})
    rec.call("GET", "/api/providers")
    rec.call("GET", "/api/latest")
    rec.call("GET", "/api/campaigns")


def _campaign_case(rec: _Recorder, sheet: str) -> str:
    created = rec.call("POST", "/api/campaigns", json={"name": f"golden {sheet}"})
    cid = str(created.json()["id"])
    if sheet in SETUPS:
        rec.call("PUT", f"/api/campaigns/{cid}/setup", json=SETUPS[sheet])
    _upload(rec, f"/api/campaigns/{cid}/upload", sheet)
    rec.call("GET", f"/api/campaigns/{cid}/analysis", name=f"campaign/{sheet}")
    return cid


def _fold(rec: _Recorder, cid: str) -> None:
    """Two rounds: start the top two proposals, record results near the model's
    prediction (a plausible lab result), correct one, reanalyze."""
    for rnd in (1, 2):
        analysis = rec.call("GET", f"/api/campaigns/{cid}/analysis", store=False).json()
        proposals = analysis.get("proposals", [])[:2]
        started = rec.call("POST", f"/api/campaigns/{cid}/runs",
                           json={"recipes": [{"recipe": p["recipe"]} for p in proposals]}).json()
        for i, run in enumerate(started.get("started", [])):
            value = round(float(proposals[i]["pred"]) + 0.05 * (i + 1) - 0.02 * rnd, 3)
            rec.call("POST", f"/api/campaigns/{cid}/runs/{run['id']}/result", json={"value": value})
        if rnd == 2 and started.get("started"):
            first = started["started"][0]["id"]
            corrected = round(float(proposals[0]["pred"]) + 0.01, 3)
            rec.call("POST", f"/api/campaigns/{cid}/runs/{first}/result",
                     json={"value": corrected, "reason": "re-assayed, dilution corrected"})
        rec.call("POST", f"/api/campaigns/{cid}/reanalyze")
        rec.call("GET", f"/api/campaigns/{cid}/analysis", name=f"fold/round{rnd}")
    rec.call("GET", f"/api/campaigns/{cid}")
    rec.call("GET", f"/api/campaigns/{cid}/results/history")
    rec.call("GET", f"/api/campaigns/{cid}/export/rows")
    rec.call("POST", f"/api/campaigns/{cid}/operating-ranges", json={"threshold": 0.6, "probability": 0.8})
    rec.call("GET", f"/api/campaigns/{cid}/operating-ranges")
    # One more measured run, then a reanalyze JOB: queued (no worker runs), then cancelled.
    analysis = rec.call("GET", f"/api/campaigns/{cid}/analysis", store=False).json()
    recipes = [{"recipe": p["recipe"]} for p in analysis.get("proposals", [])[:1]]
    started = rec.call("POST", f"/api/campaigns/{cid}/runs", json={"recipes": recipes}).json().get("started", [])
    for run in started:
        rec.call("POST", f"/api/campaigns/{cid}/runs/{run['id']}/result", json={"value": 1.9})
    job = rec.call("POST", f"/api/campaigns/{cid}/reanalyze/jobs").json()
    job_id = (job.get("job") or job).get("id")
    if job_id:
        rec.call("POST", f"/api/jobs/{job_id}/cancel")


def _campaign_routes(rec: _Recorder, cid: str) -> None:
    rec.call("GET", f"/api/campaigns/{cid}")
    rec.call("PATCH", f"/api/campaigns/{cid}", json={"name": "golden categorical (renamed)"})
    rec.call("GET", f"/api/campaigns/{cid}/setup")
    rec.call("GET", f"/api/campaigns/{cid}/stages")
    rec.call("PUT", f"/api/campaigns/{cid}/stages", json={
        "stages": [{"key": "S1", "label": "Screen"},
                   {"key": "S2", "label": "Micro", "min_L": 0.0001, "max_L": 0.003},
                   {"key": "S3", "label": "Bioreactor", "min_L": 2, "max_L": 50, "nominal_L": [2, 10, 50]}],
        "current_stage": "S2"})
    rec.call("PUT", f"/api/campaigns/{cid}/gates/S1", json={"decision": "go", "note": "screen cleared"})
    rec.call("GET", f"/api/campaigns/{cid}/gates")
    rec.call("GET", f"/api/campaigns/{cid}/gates/S1/history")
    rec.call("GET", f"/api/campaigns/{cid}/clones")
    rec.call("GET", f"/api/campaigns/{cid}/clones/summary")
    rec.call("PUT", f"/api/campaigns/{cid}/clones/C1", json={"status": "lead", "host": "CHO-K1"})
    rec.call("POST", f"/api/campaigns/{cid}/clones/C1/titer-history",
             json={"basis": "harvest_titer", "value": 3.2, "unit": "g/L", "measured_on": "2026-01-02"})
    rec.call("GET", f"/api/campaigns/{cid}/clones/C1/titer-history")
    rec.call("GET", f"/api/campaigns/{cid}/export/rows")
    rec.call("GET", f"/api/campaigns/{cid}/results/history")
    rec.call("GET", "/api/campaigns?summary=true")
    rec.call("GET", "/api/campaigns/cmp_" + "0" * 32)  # not found
    rec.call("PUT", f"/api/campaigns/{cid}/setup", json={"goal": "sideways"})  # refused


SCALE_TARGET = {
    "scale_L": 7500.0, "agitation_rpm": 150.0, "airflow_L_per_min": 300.0, "target_column": "titer_g_per_L",
    "process_params": {"ph_setpoint": 7.2, "temperature_C": 37.0},
}


def _scale(rec: _Recorder) -> None:
    target = json.dumps(SCALE_TARGET)
    _upload(rec, "/api/scale/readout.json", "scaleup", name="scale/readout_json", target=target)
    _upload(rec, "/api/scale/readout", "scaleup", name="scale/readout_html", target=target)
    _upload(rec, "/api/scale/readout?format=json", "scaleup", target=target)


def _legacy(rec: _Recorder) -> None:
    """The single-campaign routes act on the tenant's default campaign, which the
    last `POST /api/run` seeded."""
    campaign = rec.call("GET", "/api/campaign").json()
    proposals = (campaign.get("analysis") or {}).get("proposals") or campaign.get("proposals") or []
    if proposals:
        started = rec.call("POST", "/api/campaign/start", json={"recipes": [{"recipe": proposals[0]["recipe"]}]})
        for run in started.json().get("started", []):
            rec.call("POST", "/api/campaign/result", json={"id": run["id"], "value": 2.5})
        rec.call("GET", "/api/campaign")
    rec.call("GET", "/api/single")
    rec.call("GET", "/api/multi")


def _jobs(rec: _Recorder, cid: str) -> None:
    created = _upload(rec, f"/api/campaigns/{cid}/jobs", "minimize")
    job = created.json().get("job", created.json())
    job_id = job.get("id") or job.get("job_id")
    rec.call("GET", f"/api/campaigns/{cid}/jobs")
    if job_id:
        rec.call("GET", f"/api/jobs/{job_id}")
        rec.call("POST", f"/api/jobs/{job_id}/cancel")
        rec.call("GET", f"/api/jobs/{job_id}")


def _assist(rec: _Recorder, cid: str) -> None:
    rec.call("GET", "/api/assist/index/status")
    rec.call("GET", f"/api/assist/next?campaign_id={cid}")
    rec.call("POST", "/api/assist/ask", json={"question": "Which input matters most?", "campaign_id": cid})
    rec.call("GET", "/api/assist/sop")
    runs = rec.call("GET", f"/api/campaigns/{cid}/results/history", store=False).json()
    run_ids = sorted({r["run_id"] for r in runs.get("history", runs.get("results", [])) if "run_id" in r})[:1]
    source = {"run_ids": run_ids} if run_ids else {"proposal_index": [0]}
    rec.call("POST", "/api/assist/sop/draft", json={"campaign_id": cid, "source": source, "template": "batch_recipe"})


def iter_cases(records: dict[str, Any]) -> Iterator[tuple[str, str]]:
    for name, record in records.items():
        yield name, canonical(record)
