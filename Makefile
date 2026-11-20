# Developer shortcuts. CI runs the same commands (.github/workflows/scoreboard.yml).
PY ?= .venv/bin/python
OUT ?= /tmp/kalos-scoreboard
BASE ?= origin/main
WORKERS ?= 4

.PHONY: scoreboard scoreboard-diff

## Measure this checkout and write SCOREBOARD.md (about 15 min on a 4-core CI runner).
scoreboard:
	KALOS_STATE_DIR=$$(mktemp -d) $(PY) -m kalos.bench.scoreboard run --out $(OUT)/head --workers $(WORKERS) --write SCOREBOARD.md

## Measure $(BASE) in a temporary worktree with this checkout's scoreboard and venv, then diff.
## Same-machine comparison; the venv must match $(BASE)'s uv.lock (a warning is printed if it differs).
scoreboard-diff: scoreboard
	rm -rf $(OUT)/base-src && git worktree add --detach $(OUT)/base-src $(BASE)
	@git diff --quiet $(BASE) -- uv.lock pyproject.toml || echo "WARNING: uv.lock/pyproject differ from $(BASE); the base runs with this venv"
	KALOS_STATE_DIR=$$(mktemp -d) $(PY) kalos/bench/scoreboard.py run --repo $(OUT)/base-src --out $(OUT)/base --workers $(WORKERS); \
	  $(PY) kalos/bench/scoreboard.py diff $(OUT)/base $(OUT)/head --md $(OUT)/diff.md; rc=$$?; \
	  git worktree remove --force $(OUT)/base-src; exit $$rc
