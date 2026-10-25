#!/usr/bin/env bash
# Local replacement for CI: lint + types (fast). `--full` also runs the whole test suite (~25 min).
# Uses the tools on PATH, else the repo .venv, else uvx for ruff. Fails loudly if mypy is missing.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv/bin ] && PATH="$PWD/.venv/bin:$PATH"
if command -v ruff >/dev/null; then ruff check kalos/ tests/ examples/
elif command -v uvx >/dev/null; then uvx ruff check kalos/ tests/ examples/
else echo "ruff not found (pip install -e '.[dev]')" >&2; exit 1; fi
command -v mypy >/dev/null || { echo "mypy not found (pip install -e '.[dev]')" >&2; exit 1; }
mypy
if [ "${1:-}" = "--full" ]; then python -m pytest -q; fi
