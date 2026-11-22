# Contributing to the Kalos engine

Thanks for helping. The rules every change follows are in [AGENTS.md](AGENTS.md)
(honesty, data, security, storage, tests). This page is the practical how-to.

## Set up

Use Python 3.11 (`.python-version`), the version CI runs.

```bash
uv venv -p 3.11
uv pip install -e ".[ml,portal,bench,dev]"     # the extras CI installs
uv pip install -e ".[assist]"                  # also needed for the assistant tests
pre-commit install                             # optional: ruff, secret and size checks on commit
ln -sf ../../scripts/pre-push .git/hooks/pre-push   # optional: ruff + mypy before push
```

## Check your change

```bash
scripts/check.sh            # ruff + mypy, as CI runs them
python -m pytest -q -n auto # the full suite (CI splits it across 6 shards)
```

- Route changes: regenerate the snapshots with `scripts/export_openapi.py` and
  `scripts/export_routes.py`, and give every new route a scope in
  `docs/route-scopes.json`.
- Changes to detectors, gates or interval code: run the planted-fault suite
  (`docs/PLANTED_FAULTS.md`) and put the before/after in the PR.

## Pull requests

- Fill in the template: what changed, the exact tests you ran, security impact.
- Link the Linear issue with `Fixes BRE-123` so it closes on merge.
- Add a changelog entry as a file in `changes/` (see `changes/README.md`), not in
  `CHANGELOG.md` directly. Separate files do not conflict.
- Stacked PRs: before the base PR merges, retarget the PR on top of it to `main`
  (`gh pr edit <n> --base main`). Merging deletes the base branch, and GitHub then closes
  any PR still pointing at it.
- Several PRs touching the same area: merge them together locally and run the suite
  once before merging any of them. Each PR's own CI cannot see how they interact.

## Data

Never commit data whose rights are not recorded in
[docs/DATA_RIGHTS.md](docs/DATA_RIGHTS.md). Load external data through
`KALOS_EXTERNAL_DATA` or a path argument.

## Security

Report vulnerabilities privately, as described in [SECURITY.md](SECURITY.md). Never put
secrets in code, issues, PRs or logs.

## Housekeeping

`scripts/worktrees.sh` lists your worktrees with uncommitted files, branches whose
remote is gone, and branches already merged, so work is not lost in old worktrees.

## Licence

Contributions are licensed under the Apache License 2.0 ([LICENSE](LICENSE)).
