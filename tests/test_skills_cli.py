"""`kalos skills ...` / `python -m kalos.skills ...` (kalos/skills/cli.py):
exit codes and JSON output shape."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")

from kalos.skills.cli import kalos_main, main
from kalos.skills.loader import tree_hash
from kalos.skills.registry import BUILTIN_DIR


def test_validate_builtins_clean_json(capsys):
    assert main(["validate", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and out["problems"] == []
    assert len(out["skills"]) == 19
    row = out["skills"][0]
    assert set(row) == {"slug", "category", "kind", "status", "implementation", "refs", "tools",
                        "content_hash", "ok"}


def test_validate_reports_problems_exit_1(tmp_path, capsys):
    shutil.copytree(BUILTIN_DIR / "driver-ranking", tmp_path / "driver-ranking")
    shutil.copytree(BUILTIN_DIR / "trust-report", tmp_path / "trust-report")
    p = tmp_path / "driver-ranking" / "SKILL.md"
    p.write_text(p.read_text().replace("kalos/core/drivers.py:95", "kalos/core/drivers.py:96"))
    q = tmp_path / "trust-report" / "SKILL.md"
    q.write_text(q.read_text().replace("engine.check_gates", "engine.not_a_tool"))
    assert main(["validate", str(tmp_path), "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    msgs = {(x["where"], x["message"]) for x in out["problems"]}
    assert any(w == "driver-ranking" and "ref says 96" in m for w, m in msgs)
    assert any(w == "trust-report" and "not registered" in m for w, m in msgs)


def test_validate_schema_error_exit_1(tmp_path, capsys):
    shutil.copytree(BUILTIN_DIR / "driver-ranking", tmp_path / "driver-ranking")
    p = tmp_path / "driver-ranking" / "SKILL.md"
    p.write_text(p.read_text().replace("kind: locked", "kind: locked\nsurprise: 1"))
    assert main(["validate", str(tmp_path)]) == 1
    assert "surprise" in capsys.readouterr().out


def test_validate_foreign_corpus(tmp_path, capsys):
    shutil.copytree(BUILTIN_DIR / "driver-ranking", tmp_path / "skills" / "driver-ranking")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "x.md").write_text((tmp_path / "skills" / "driver-ranking" / "SKILL.md").read_text())
    assert main(["validate", str(tmp_path / "skills"), "--foreign-corpus", str(corpus)]) == 1
    assert "shared with the foreign corpus" in capsys.readouterr().out
    (corpus / "x.md").write_text("entirely different words that never appear in any skill file\n")
    assert main(["validate", str(tmp_path / "skills"), "--foreign-corpus", str(corpus)]) == 0


def test_list_search_hash_tools(capsys):
    assert main(["list", "--json", "--category", "trust"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["slug"] for r in rows} == {"trust-report", "distance-to-data"}
    assert main(["search", "titer from chromatogram", "--all", "--limit", "1"]) == 0
    assert "hplc-peak-summary" in capsys.readouterr().out
    assert main(["hash", str(BUILTIN_DIR / "driver-ranking")]) == 0
    assert capsys.readouterr().out.strip() == tree_hash(BUILTIN_DIR / "driver-ranking")[0]
    assert main(["hash", str(BUILTIN_DIR)]) == 2
    assert main(["tools", "--json"]) == 0
    tools = json.loads(capsys.readouterr().out)
    assert len(tools) == 30 and {"name", "symbol", "label", "needs_torch"} <= set(tools[0])


def test_kalos_entry_point_and_usage(capsys):
    assert kalos_main(["skills", "validate"]) == 0
    assert kalos_main([]) == 2
    assert kalos_main(["other"]) == 2
    with pytest.raises(SystemExit) as e:
        main(["nope"])
    assert e.value.code == 2


def test_python_dash_m_and_no_torch():
    r = subprocess.run([sys.executable, "-c",
                        "import sys, runpy; sys.argv=['x','validate'];"
                        "from kalos.skills.cli import main; rc=main(['validate']);"
                        "print('torch' in sys.modules, rc)"],
                       capture_output=True, text=True, check=True)
    assert r.stdout.strip().splitlines()[-1] == "False 0"
    r2 = subprocess.run([sys.executable, "-m", "kalos.skills", "validate", "--json"],
                        capture_output=True, text=True)
    assert r2.returncode == 0 and json.loads(r2.stdout)["ok"] is True
