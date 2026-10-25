#!/usr/bin/env bash
# Local replacement for CI: lint + types (fast). Add `--full` to run the whole test suite (~25 min).
set -euo pipefail
cd "$(dirname "$0")/.."
ruff check kalos/ tests/ examples/
mypy
if [ "${1:-}" = "--full" ]; then python -m pytest -q; fi
