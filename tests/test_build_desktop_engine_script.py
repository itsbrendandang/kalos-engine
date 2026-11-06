"""scripts/build_desktop_engine.sh: the guards that run without building
anything, and the pins and smoke gates the build relies on."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_desktop_engine.sh"


def _run(out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(SCRIPT), str(out)], capture_output=True, text=True,
                          timeout=60, cwd=ROOT)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_refuses_to_delete_a_folder_that_is_not_an_earlier_bundle(tmp_path):
    out = tmp_path / "Documents"
    out.mkdir()
    (out / "thesis.docx").write_text("precious")
    r = _run(out)
    assert r.returncode == 2, r.stderr
    assert "refusing to delete it" in r.stderr
    assert (out / "thesis.docx").read_text() == "precious"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_refuses_a_file_as_the_output(tmp_path):
    out = tmp_path / "notes.txt"
    out.write_text("keep")
    r = _run(out)
    assert r.returncode == 2 and out.read_text() == "keep"


def test_python_is_pinned_to_a_full_version():
    text = SCRIPT.read_text()
    m = re.search(r'PY_VERSION="\$\{KALOS_BUNDLE_PYTHON:-([^}]+)\}"', text)
    assert m and re.fullmatch(r"\d+\.\d+\.\d+", m.group(1)), m


def test_dependencies_are_installed_with_hashes():
    text = SCRIPT.read_text()
    assert "--no-hashes" not in text
    assert "--require-hashes" in text


def test_the_smoke_gate_needs_401_without_token_and_a_clean_sigterm_exit():
    text = SCRIPT.read_text()
    assert '[ "$UNAUTH" != "401" ]' in text
    assert '[ "$EXIT_CODE" != "0" ]' in text
