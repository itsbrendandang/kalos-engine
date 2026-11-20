"""Reorg guard R02: the golden oracle (tests/golden/oracle.py, scripts/golden.py).

- the full check: the scenario's records equal tests/golden/<env key>/ byte for
  byte. Runs here when goldens exist for this machine's env key; on GitHub
  Actions it runs in its own workflow (.github/workflows/golden.yml), not in
  the serial pytest job, so it is not run twice;
- fast unit checks of the canonical form (scrubbing, shapes, slugs) and of the
  recorded goldens themselves (manifest hashes match the files, every recorded
  env has a change log).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("golden_cli", ROOT / "scripts" / "golden.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _oracle() -> ModuleType:
    spec = importlib.util.spec_from_file_location("golden_oracle", GOLDEN / "oracle.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _recorded() -> list[Path]:
    return sorted(p for p in GOLDEN.iterdir() if (p / "MANIFEST.json").exists())


def test_scrub_removes_only_times() -> None:
    oracle = _oracle()
    body = {"updated": 1795154519.32, "result_recorded_at": 1.0, "created_at": None, "pred": 0.596,
            "stamp": "2026-11-20T05:20:08+00:00", "label": "2026 batch", "rows": [{"exported_at": 3.0, "x": 1}]}
    assert oracle.scrub(body) == {"updated": "<volatile>", "result_recorded_at": "<volatile>", "created_at": None,
                                  "pred": 0.596, "stamp": "<timestamp>", "label": "2026 batch",
                                  "rows": [{"exported_at": "<volatile>", "x": 1}]}


def test_canonical_keeps_key_order_and_exact_floats() -> None:
    oracle = _oracle()
    a = oracle.canonical({"b": 0.1 + 0.2, "a": 1})
    assert a.index('"b"') < a.index('"a"')
    assert json.loads(a)["b"] == 0.1 + 0.2
    with pytest.raises(ValueError):
        oracle.canonical({"x": float("nan")})


def test_shape_and_slug() -> None:
    oracle = _oracle()
    assert oracle.shape({"a": [1.5, 2.0], "b": {"c": "x"}, "d": []}) == {"a": ["float"], "b": {"c": "str"}, "d": []}
    url = "/api/campaigns/cmp_" + "a" * 32 + "/runs/" + "b" * 32 + "/result"
    assert oracle._slug(url) == "api_campaigns_CID_runs_RUN_result"


def test_recorded_goldens_are_consistent() -> None:
    recorded = _recorded()
    assert recorded, "no golden records committed"
    for env_dir in recorded:
        manifest = json.loads((env_dir / "MANIFEST.json").read_text(encoding="utf-8"))
        assert manifest["env"]["key"] == env_dir.name
        assert manifest["reasons"], f"{env_dir.name}: the manifest must say why it was recorded"
        assert manifest["env"]["torch_threads"] == 1
        files = {p.relative_to(env_dir).with_suffix("").as_posix() for p in env_dir.rglob("*.json")} - {"MANIFEST"}
        assert files == set(manifest["cases"]), env_dir.name
        for name, digest in manifest["cases"].items():
            text = (env_dir / f"{name}.json").read_text(encoding="utf-8")
            assert hashlib.sha256(text.encode()).hexdigest() == digest, name


def test_record_refuses_an_empty_reason() -> None:
    with pytest.raises(SystemExit):
        _script().main(["record", "--reason", "  "])


@pytest.mark.slow
@pytest.mark.serial
def test_golden_oracle_is_byte_identical(tmp_path: Path) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        pytest.skip("on CI the oracle runs in .github/workflows/golden.yml")
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "golden.py"), "check",
                           "--actual", str(tmp_path / "actual"), "--summary-json", str(tmp_path / "s.json")],
                          cwd=ROOT, capture_output=True, text=True, check=False, timeout=3600)
    assert (tmp_path / "s.json").exists(), proc.stdout[-3000:] + proc.stderr[-3000:]
    summary = json.loads((tmp_path / "s.json").read_text())
    if not summary["recorded_for_env"]:
        pytest.skip(f"no goldens recorded for {summary['env_key']} (CI's linux key gates merges)")
    assert proc.returncode == 0, proc.stdout[-6000:] + proc.stderr[-3000:]
