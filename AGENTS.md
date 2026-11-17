# Notes for coding agents

Read `README.md` first. Reference documents are in `docs/`. This repo is the
Kalos backend. The UI lives in the separate `kalos-web` repo.

## Commands

Python 3.11 or later is required.

- Setup: `python -m pip install -e ".[ml,portal,bench,dev,assist]"`.
- Fast checks: `scripts/check.sh` runs ruff and mypy.
- Tests: `python -m pytest -q`. `scripts/check.sh --full` also runs them.
- Assistant checks: `python -m pytest -q tests/test_assist_*.py tests/test_web_contract.py`.
- Own test server: `KALOS_LLM_PROVIDER=none KALOS_PORT=8051 KALOS_STATE_DIR=$(mktemp -d) python -m kalos.portal`.
  Use a free port and a fresh state directory. Never reuse another person's state.

## Work lanes

- Backend lane: this repo, `kalos-engine`.
- UI lane: `kalos-web`. Do not edit its screens from a backend task.
- Different lanes coordinate only through the HTTP contract. Its guard is
  `tests/test_web_contract.py`. Change it deliberately. Note contract changes
  in the PR so the web lane can follow them.
- Give each coding agent a narrow task and clear file ownership. Do not edit
  the same file at the same time. Report changed paths and check results.

## Shared files

Re-read these just before editing. Keep edits small:
`kalos/portal/app.py`, `pyproject.toml`, `uv.lock`, `README.md`, `CHANGELOG.md`.
Never resolve a merge by taking one whole side of a shared file.

## Rules

- Never make paid model calls in tests. Use fakes. Do not use real keys for checks.
- Never send writes to someone's running engine. Do not stop or restart it.
  Start your own server as above. Tests use temporary state directories.
- Read the tenant from the authenticated principal, never the request body.
- Verdicts and checks stay in code, not in model judgement. A model may suggest
  claims or issues. It may not bypass grounding checks or choose a verdict.
- Store model replies before relying on them. A replay must not repeat a call.
- Never log prompts, replies, customer records or credentials. Log counts only.
- Keep the README capability table honest. An open PR is not implemented on main.
- Keep the diff focused. Do not commit generated state, keys or run outputs.
- Write short sentences and plain words. Verify documented commands and paths.

## Where things are

- Assistant: `kalos/assist/`. Guide, cited chat, review and SOP drafts.
- HTTP app: `kalos/portal/app.py`. Authentication: `kalos/portal/auth.py`.
- Campaign store: `kalos/portal/campaign.py`.
- Assistant docs: `docs/ASSISTANT.md`. HTTP contract: `tests/test_web_contract.py`.
