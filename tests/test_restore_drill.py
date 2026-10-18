"""scripts/restore_drill.sh and the RUNBOOK restore procedure (deploy/RUNBOOK.md).

The drill's whole point is the volume-root ownership bug: chowning only the
two .db files leaves SQLite unable to create -wal/-shm on a fresh volume. The
static tests below pin that the script and the runbook both chown the root,
and the docker test (opt-in, needs a daemon and the backup image) reproduces
the failure for real.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRILL = ROOT / "scripts" / "restore_drill.sh"
RUNBOOK = ROOT / "deploy" / "RUNBOOK.md"


def test_drill_script_exists_and_is_valid_sh():
    assert DRILL.is_file()
    assert os.access(DRILL, os.X_OK)
    subprocess.run(["sh", "-n", str(DRILL)], check=True)


def test_drill_chowns_the_volume_root_not_only_the_files():
    """The regression: `chown ... /state/*.db` alone is the bug. The root
    directory must be handed to the engine uid as well."""
    text = DRILL.read_text()
    assert re.search(r"chown \$\{ENGINE_UID\}:\$\{ENGINE_UID\} /state\n", text), (
        "the volume root (/state) must be chowned to the engine uid"
    )
    assert "ENGINE_UID=10001" in text


def test_drill_verifies_wal_files_are_created_as_the_engine_user():
    text = DRILL.read_text()
    assert "--user \"$ENGINE_UID:$ENGINE_UID\"" in text
    assert "journal_mode=WAL" in text
    assert "portal.db-wal" in text and "portal.db-shm" in text


def test_drill_never_touches_the_live_volume_or_reads_secrets():
    text = DRILL.read_text()
    # the scratch volume name is generated, and the live one is refused
    assert "kalos-restore-drill-" in text
    assert "kalos-state" in text  # only in the refusal guard
    for secret_word in ("KALOS_AUTH", "KALOS_ANON_SALT", "TOKEN", "API_KEY", ".env"):
        assert secret_word not in text, secret_word


def test_drill_refuses_a_missing_backup_cleanly(tmp_path):
    """Exit 1 with a one-line reason, before any docker call, when no
    snapshot exists - the check runs first so an empty host cannot half-restore."""
    env = {**os.environ, "BACKUP_DIR": str(tmp_path)}
    proc = subprocess.run(
        ["sh", str(DRILL)], capture_output=True, text=True, env=env, timeout=60
    )
    assert proc.returncode != 0
    assert "no experiments snapshot" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_runbook_restore_chowns_the_volume_root():
    text = RUNBOOK.read_text()
    assert "chown 10001:10001 /state /state/experiments.db /state/portal.db" in text
    assert "chown 10001:10001 /state/experiments.db /state/portal.db\n" not in text


def test_runbook_documents_the_backup_image_build_and_the_drill():
    text = RUNBOOK.read_text()
    assert "docker build -f deploy/backup/Containerfile" in text
    assert "scripts/restore_drill.sh" in text


@pytest.mark.skipif(
    os.environ.get("KALOS_TEST_DOCKER") != "1" or shutil.which("docker") is None,
    reason="set KALOS_TEST_DOCKER=1 (needs a docker daemon and kalos-backup:local)",
)
def test_drill_passes_end_to_end_on_a_fresh_volume(tmp_path):
    import sqlite3

    for sub, name in (("experiments", "experiments-20260101T000000Z.db"),
                      ("portal", "portal-20260101T000000Z.db")):
        (tmp_path / sub).mkdir()
        con = sqlite3.connect(tmp_path / sub / name)
        con.execute("CREATE TABLE t (x)")
        con.commit()
        con.close()
    env = {**os.environ, "BACKUP_DIR": str(tmp_path)}
    proc = subprocess.run(
        ["sh", str(DRILL)], capture_output=True, text=True, env=env, timeout=600
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "PASSED" in proc.stdout
