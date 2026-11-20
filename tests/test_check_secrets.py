"""Secret guard (scripts/check_secrets.py): what it flags, what it lets through,
that it never prints a value, and that scripts/check.sh stops on a staged .env.

Secret-shaped test values are assembled at runtime (`"sk-" + "ant-" + ...`) so
this file itself never contains one and the guard passes on its own diff.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_secrets.py"
_spec = importlib.util.spec_from_file_location("check_secrets", SCRIPT)
assert _spec is not None and _spec.loader is not None
cs = importlib.util.module_from_spec(_spec)
sys.modules["check_secrets"] = cs
_spec.loader.exec_module(cs)

A40 = "Ab3dE5gH7jK9mN1pQ3sT5vW7yZ9bC2eF4hJ6kL8"  # 40 secret-looking chars, no placeholder words
DASH5 = "-" * 5

# (type, a line that must be flagged as that type, a placeholder line that must not be flagged)
CASES: list[tuple[str, str, str]] = [
    ("private key block", f"{DASH5}BEGIN RSA PRIVATE KEY{DASH5}", f"{DASH5}BEGIN PRIVATE KEY{DASH5} ... (example)"),
    ("Anthropic API key", "ANTHROPIC=" + "sk-" + "ant-" + A40, "ANTHROPIC=" + "sk-" + "ant-" + "your-key-here-xxxx"),
    ("OpenAI-style API key", "key: " + "sk-" + A40, "key: " + "sk-" + "<your-openai-key-goes-here>"),
    ("GitHub token", "gh_token " + "ghp" + "_" + A40, "gh_token " + "ghp" + "_" + "x" * 36),
    ("GitLab token", "glpat" + "-" + A40[:24], "glpat" + "-" + "REPLACE_WITH_TOKEN_123456"),
    ("Slack token", "xoxb" + "-" + A40[:20], "xoxb" + "-" + "your-slack-bot-token"),
    ("AWS access key id", "aws " + "AKIA" + "Q3T5V7X9Z1B3D5F7", "aws " + "AKIA" + "Q3T5V7X9Z1B3D5F7 # example"),
    ("Google API key", "maps=" + "AIza" + A40[:35], "maps=" + "AIza" + A40[:35] + "  # sample"),
    ("bearer token", "Authorization: " + "Bearer " + A40, "Authorization: " + "Bearer ${ENGINE_TOKEN}"),
    ("private JWK component", '{"kty": "OKP", "d"' + ': "' + A40 + '"}', '{"d"' + ': "' + "<base64url-private-scalar>" + '"}'),
    ("credentialed database URL", "DB=postgres" + "ql://kalos:" + "hunter2x@db:5432/k", "DB=postgres" + "ql://kalos:${PGPASS}@db:5432/k"),
    ("named Kalos secret", "BETTER_AUTH" + "_SECRET=" + A40, "BETTER_AUTH" + "_SECRET=<generate-with-openssl>"),
    ("named Kalos secret", "KALOS_ANON" + "_SALT: " + A40[:16], "KALOS_ANON" + "_SALT: ${KALOS_ANON_SALT}"),
    ("named Kalos secret", "GOOGLE_CLIENT" + "_SECRET='" + A40[:24] + "'", "GOOGLE_CLIENT" + "_SECRET=''"),
    ("secret behind NEXT_PUBLIC_", "NEXT_PUBLIC" + "_ENGINE_TOKEN=abc", "NEXT_PUBLIC" + "_API_URL=http://127.0.0.1:8050"),
    ("KALOS_DEV_NO_AUTH enabled", "KALOS_DEV" + "_NO_AUTH=1", "KALOS_DEV" + "_NO_AUTH=0"),
    ("KALOS_DEV_NO_AUTH enabled", '"KALOS_DEV' + '_NO_AUTH": "true"', '"KALOS_DEV' + '_NO_AUTH": "false"'),
    ("secret assignment", 'api_key = "' + A40[:20] + '"', 'api_key = "your-api-key-here"'),
    ("secret assignment", "ENGINE_SECRET=" + A40[:20], "ENGINE_SECRET=changeme-before-deploying"),
    ("secret assignment", "  password: " + A40[:16], "  password: <set-in-env>"),
]


@pytest.mark.parametrize(("kind", "positive", "negative"), CASES, ids=[f"{k}-{i}" for i, (k, _, _) in enumerate(CASES)])
def test_each_pattern_flags_its_line_and_skips_its_placeholder(kind: str, positive: str, negative: str) -> None:
    assert cs.classify(positive) == kind
    assert cs.classify(negative) is None


def test_every_pattern_type_has_a_case() -> None:
    assert {k for k, _, _ in cs.PATTERNS} == {k for k, _, _ in CASES}


@pytest.mark.parametrize(
    "line",
    [
        "token = make_token(tenant)",  # unquoted Python call
        "unit_token=item.unit_token,",
        "token_id=verified.token_id,",
        '_API_KEY_ENV_VAR = "ANTHROPIC' + '_API_KEY"',  # names where a secret lives
        "- KALOS_ENGINE_JWT_PRIVATE_KEY_FILE=/run/secrets/engine-jwt.pem",
        'export const ENGINE_TOKEN_AUDIENCE = "kalos-engine";',
        "Never put a secret behind NEXT_PUBLIC_.",
        "api_key = " + '"' + A40 + '"  # secret-guard: allow (reviewed test vector)',
    ],
)
def test_code_and_docs_are_not_flagged(line: str) -> None:
    assert cs.classify(line) is None


def _diff(path_header: str, *added: str, start: int = 1) -> str:
    body = "\n".join("+" + a for a in added)
    return (
        f"diff --git a/x b/x\nindex 0000000..1111111 100644\n--- a/x\n{path_header}\n"
        f"@@ -0,0 +{start},{len(added)} @@\n{body}\n"
    )


def test_plus_plus_plus_b_path_is_parsed_with_removeprefix() -> None:
    # lstrip("b/") would strip every leading 'b' and '/', giving "ackend/..." and "in/...".
    secret = "api_key = " + '"' + A40 + '"'
    diff = _diff("+++ b/backend/settings.py", "x = 1", secret, start=7) + _diff("+++ b/bin/run.sh", secret)
    assert [str(h) for h in cs.scan_diff(diff)] == [
        "backend/settings.py:8 (secret assignment)",
        "bin/run.sh:1 (secret assignment)",
    ]


def test_added_line_starting_with_plus_plus_is_content_not_a_header() -> None:
    secret = "++ api_key = " + '"' + A40 + '"'
    hits = cs.scan_diff(_diff("+++ b/notes.md", secret))
    assert [str(h) for h in hits] == ["notes.md:1 (secret assignment)"]


def test_deleted_files_and_removed_lines_are_not_scanned() -> None:
    secret = "api_key = " + '"' + A40 + '"'
    diff = f"diff --git a/old b/old\n--- a/old\n+++ /dev/null\n@@ -1 +0,0 @@\n-{secret}\n"
    assert cs.scan_diff(diff) == []


@pytest.mark.parametrize(
    ("path", "flagged"),
    [
        (".env", True),
        ("deploy/.env.production", True),
        ("keys/engine.pem", True),
        ("tls/server.key", True),
        ("credentials.json", True),
        ("config/secrets.yaml", True),
        ("service-account-prod.json", True),
        ("deploy/engine-config/tenants.json", True),
        ("jwks-private.json", True),
        ("env.example", False),
        (".env.example", False),
        ("docs/SECRETS.md", False),
        ("tests/test_credentials_api.py", False),
        ("kalos/portal/auth.py", False),
    ],
)
def test_secret_bearing_file_names(path: str, flagged: bool) -> None:
    assert cs.sensitive_name(path) is flagged


# --- against real git repositories -------------------------------------------------

def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "user.name", "t")
    _git(r, "config", "commit.gpgsign", "false")
    (r / "README.md").write_text("hello\n")
    _git(r, "add", "README.md")
    _git(r, "commit", "-q", "-m", "init")
    return r


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "-C", str(repo), *args], capture_output=True, text=True, timeout=30
    )


def test_output_never_contains_the_matched_value(repo: Path) -> None:
    value = "sk-" + "ant-" + A40
    (repo / "config.py").write_text(f'KEY = "{value}"\nother = 1\n')
    _git(repo, "add", "config.py")
    proc = _run(repo, "--staged")
    assert proc.returncode == 1
    assert proc.stdout.strip() == "config.py:1 (Anthropic API key)"
    for fragment in (value, A40, A40[:12], A40[-12:]):
        assert fragment not in proc.stdout + proc.stderr


def test_range_mode_scans_only_the_range(repo: Path) -> None:
    secret = "api_key = " + '"' + A40 + '"\n'
    (repo / "old.py").write_text(secret)
    _git(repo, "add", "old.py")
    _git(repo, "commit", "-q", "-m", "old secret")
    (repo / "new.py").write_text("x = 1\n")
    _git(repo, "add", "new.py")
    _git(repo, "commit", "-q", "-m", "clean change")
    assert _run(repo, "--range", "HEAD~1..HEAD").returncode == 0
    # Default range (@{u}..HEAD) without an upstream falls back to the last commit.
    assert _run(repo).returncode == 0
    proc = _run(repo, "--range", "HEAD~2..HEAD")
    assert (proc.returncode, proc.stdout.strip()) == (1, "old.py:1 (secret assignment)")


def test_paths_flags_a_new_env_file_with_no_secret_lines(repo: Path) -> None:
    (repo / ".env").write_text("DEBUG=1\n")
    _git(repo, "add", "-f", ".env")
    assert _run(repo, "--staged").returncode == 0
    proc = _run(repo, "--staged", "--paths")
    assert (proc.returncode, proc.stdout.strip()) == (1, ".env:0 (secret-bearing file name)")


def test_git_errors_fail_closed(repo: Path, tmp_path: Path) -> None:
    proc = _run(repo, "--range", "no-such-ref..HEAD")
    assert proc.returncode == 2
    assert "failing closed" in proc.stderr
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    assert _run(not_a_repo, "--staged").returncode == 2


def test_check_sh_fails_on_a_staged_env_file(repo: Path) -> None:
    scripts = repo / "scripts"
    scripts.mkdir()
    shutil.copy(ROOT / "scripts" / "check.sh", scripts / "check.sh")
    shutil.copy(SCRIPT, scripts / "check_secrets.py")
    (repo / ".env").write_text("BETTER_AUTH" + "_SECRET=" + A40 + "\n")
    _git(repo, "add", "-f", ".env")
    # No ruff/mypy on PATH: if the secret scan did not stop check.sh first, it
    # would fail later on "ruff not found" instead, which the assertions reject.
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp")}
    proc = subprocess.run(["bash", str(scripts / "check.sh")], capture_output=True, text=True, env=env, timeout=60)
    assert proc.returncode == 1, proc.stderr
    assert ".env:0 (secret-bearing file name)" in proc.stdout
    assert ".env:1 (named Kalos secret)" in proc.stdout
    assert "ruff" not in proc.stderr
    assert A40 not in proc.stdout + proc.stderr


def test_a_typical_push_scans_in_under_a_second(repo: Path) -> None:
    for i in range(20):
        (repo / f"m{i}.py").write_text("".join(f"value_{j} = {j}\n" for j in range(200)))
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "20 files, 4000 lines")
    t0 = time.perf_counter()
    proc = _run(repo, "--range", "HEAD~1..HEAD", "--staged", "--paths")
    assert proc.returncode == 0
    assert time.perf_counter() - t0 < 1.0
