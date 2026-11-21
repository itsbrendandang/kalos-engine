"""Build `tests/fixtures/portal_v7.db`: a REAL schema-v7 portal database.

It was produced by running the v7 engine (kalos-engine `origin/main` at 4bb477d,
schema v7) end to end over HTTP in a temporary state directory:

    git worktree add /tmp/kalos-v7 4bb477d
    cd /tmp/kalos-v7 && KALOS_TORCH_THREADS=1 python <this file> <out.db>

Nothing in it is hand-written SQL: two tenants upload a small run sheet (real
analyses), start the engine's proposals, record results (one with a correction
and a reason), and re-analyze (real folds). `acme` ends at round 2 with three
runs still in flight (two measured, one awaiting); `globex` ends at round 1 with
two runs in flight. The rows of rounds 0 and 1 that v7 folded have lost their
recorded prediction, band and run id: that is the history schema v8 stops losing.

`tests/test_schema_v8.py` copies this file and migrates the copy; the file in the
repository is never opened for writing. Synthetic data only.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": _sha("acme-admin"), "subject": "alice@acme", "tenant": "acme",
     "scopes": ["read", "write", "admin"]},
    {"token_sha256": _sha("globex-admin"), "subject": "gina@globex", "tenant": "globex",
     "scopes": ["read", "write", "admin"]},
]


def _truth(m: float, ph: float) -> float:
    return 1.5 * m - 0.8 * (ph - 6.0) ** 2 + 2.0


def _sheet(n: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    m = rng.uniform(0, 4, n).round(3)
    ph = rng.uniform(5, 7, n).round(2)
    y = np.array([_truth(a, b) for a, b in zip(m, ph)]) + rng.normal(0, 0.15, n)
    buf = io.StringIO()
    pd.DataFrame({"Methanol": m, "pH": ph, "lipase_titer": y.round(3)}).to_csv(buf, index=False)
    return buf.getvalue().encode()


def main(out: Path) -> None:
    state = Path(tempfile.mkdtemp(prefix="kalos-v7-fixture-"))
    os.environ["KALOS_STATE_DIR"] = str(state)
    os.environ["KALOS_AUTH_TOKENS"] = json.dumps(TOKENS)
    os.environ.setdefault("KALOS_TORCH_THREADS", "1")
    from fastapi.testclient import TestClient

    from kalos.portal import migrate
    from kalos.portal.app import app

    assert migrate.SCHEMA_VERSION == 7, "run this against the v7 engine"
    tc = TestClient(app)
    rng = np.random.default_rng(7)

    def h(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def ok(resp):
        assert resp.status_code < 300, (resp.status_code, resp.text[:500])
        return resp.json()

    def round_trip(token: str, cid: str, *, k: int, fold: bool, awaiting: int = 0,
                   correct_first: bool = False) -> None:
        analysis = ok(tc.get(f"/api/campaigns/{cid}/analysis", headers=h(token)))
        props = [p for p in analysis.get("proposals", []) if isinstance(p, dict)][:k]
        assert props, "the analysis made no proposals"
        recipes = [{"recipe": p["recipe"], "pred": p.get("pred"), "std": p.get("std"),
                    "mode": p.get("mode"), "reason": "fixture"} for p in props]
        runs = ok(tc.post(f"/api/campaigns/{cid}/runs", headers=h(token),
                          json={"recipes": recipes}))["started"]
        for i, run in enumerate(runs[: len(runs) - awaiting]):
            r = run["recipe"]
            value = round(_truth(float(r["Methanol"]), float(r["pH"])) + rng.normal(0, 0.15), 3)
            ok(tc.post(f"/api/campaigns/{cid}/runs/{run['id']}/result", headers=h(token),
                       json={"value": value}))
            if correct_first and i == 0:
                ok(tc.post(f"/api/campaigns/{cid}/runs/{run['id']}/result", headers=h(token),
                           json={"value": round(value + 0.2, 3), "reason": "re-read the plate"}))
        if fold:
            ok(tc.post(f"/api/campaigns/{cid}/reanalyze", headers=h(token)))

    for token, name, seed, rounds in (("acme-admin", "Lipase media", 0, 2),
                                      ("globex-admin", "Feed screen", 1, 1)):
        cid = ok(tc.post("/api/campaigns", headers=h(token), json={"name": name}))["id"]
        ok(tc.post(f"/api/campaigns/{cid}/upload", headers=h(token),
                   files={"file": ("runs.csv", _sheet(14, seed), "text/csv")},
                   data={"target": "lipase_titer"}))
        for r in range(rounds):
            round_trip(token, cid, k=3, fold=True, correct_first=(r == 0))
        round_trip(token, cid, k=3 if rounds == 2 else 2, fold=False,
                   awaiting=1 if rounds == 2 else 0)

    src = sqlite3.connect(str(state / "portal.db"))
    out.unlink(missing_ok=True)
    dst = sqlite3.connect(str(out))
    src.backup(dst)
    dst.execute("PRAGMA journal_mode=DELETE")
    dst.execute("VACUUM")
    dst.close()
    src.close()
    print(f"wrote {out} ({out.stat().st_size} bytes) from {state}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
