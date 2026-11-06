"""POST /api/preflight: the column reading and the sheet check, no fit, nothing
stored (docs/CAMPAIGNS_API.md, "Preflight")."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.portal.app import app, get_campaign_store
from kalos.portal.campaign import CampaignStore

ROOT = Path(__file__).resolve().parents[1]
CSV = (
    b"Row,Strain,Glucose_gL,Titer,,Temp_C\n"
    + b"".join(
        f"{i},{['WT', 'dPEP4', 'clone7'][i % 3]},{[10, 20, 30, 40, 50][i % 5]},"
        f"{10 + i % 7},{'abc'[i % 3]},{[28, 30, 32][i % 3]}\n".encode()
        for i in range(1, 17)
    )
)


@pytest.fixture
def client(tmp_path):
    store = CampaignStore(tmp_path)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


def _post(c, path="/api/preflight", data=None, body=CSV):
    return c.post(path, files={"file": ("messy.csv", io.BytesIO(body), "text/csv")},
                  data={"target": "Titer", **(data or {})})


def test_preflight_reads_the_columns_and_stores_nothing(client):
    r = _post(client)
    assert r.status_code == 200, r.text
    body = r.json()
    json.dumps(body, allow_nan=False)
    assert body["preflight"] is True and body["target"] == "Titer"
    kinds = {c["name"]: c["kind"] for c in body["columns"]}
    assert kinds["Row"] == "row_number" and kinds["Strain"] == "category"
    assert "Row" not in body["features"] and "Strain" in body["categorical_features"]
    assert [c["name"] for c in body["columns"]] == [p["name"] for p in body["provenance"]]
    codes = {f["code"] for f in body["validation"]["findings"]}
    assert "headerless_column" in codes and "row_number_column" in codes
    assert "proposals" not in body and "model_optimum" not in body
    assert client.get("/api/latest").json() == {"has_data": False}


def test_preflight_matches_the_upload_reading(client):
    pytest.importorskip("botorch")
    pre = _post(client).json()
    full = _post(client, "/api/run").json()
    assert pre["columns"] == full["columns"]
    assert pre["features"] == full["features"]


def test_preflight_refuses_like_an_upload(client):
    r = _post(client, body=b"a,Titer\n1,2\n2,3\n")
    assert r.status_code == 400 and r.json()["code"] == "too_few_rows"
    bad = _post(client, data={"roles": json.dumps({"target": "Titer", "allow_beyond": 3})})
    assert bad.status_code == 400 and bad.json()["code"] == "invalid_allow_beyond"


def test_preflight_does_not_import_torch(tmp_path):
    code = (
        "import io, sys\n"
        "from fastapi.testclient import TestClient\n"
        "from kalos.portal.app import app\n"
        f"body = {CSV!r}\n"
        "r = TestClient(app).post('/api/preflight', files={'file': ('m.csv', io.BytesIO(body), 'text/csv')},"
        " data={'target': 'Titer'})\n"
        "assert r.status_code == 200, r.text\n"
        "print(sorted(m for m in ('torch', 'botorch') if m in sys.modules))\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("KALOS_")}
    env["KALOS_STATE_DIR"] = str(tmp_path)
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "[]"
