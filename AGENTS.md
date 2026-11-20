# Kalos engine: rules for agents and contributors

These rules are non-negotiable. If a task conflicts with one, stop and ask; do not work around it.

## 1. More real (honesty)
- Every number shown to a user is computed from the user's data by code in this repo. Nothing is invented, imputed, or filled with defaults.
- Measured data always outranks predicted data. Predictions carry an interval labelled with its coverage (90% or 95%) and the caveat text; never call proposal intervals "calibrated".
- Refusals are first-class: when the data cannot support an answer, return `not_judged` or a refusal with a reason and the action that would fix it. Never return a weaker answer silently.
- Never impute. Unusable rows and cells are reported with their location and reason.
- Synthetic data is only ever a labelled benchmark or test generator. It never enters a user's model as if it were measured.
- Language models never produce numbers, intervals, verdicts, proposals or units. Every LLM output is checked against engine data; anything that fails the check is refused.
- Docs, README, PRs and Linear never claim more than the code and benchmarks show. No marketing language.

## 2. Changing ML behaviour
- Any change to model outputs (predictions, intervals, proposals, verdicts) goes in its own commit behind a flag, with before/after outputs and a scoreboard diff.
- Run the scoreboard (planted-fault suite, coverage at proposals, regret, runtime, golden drift). No unexplained regression.
- Speedups must be byte-identical on the golden outputs. Approximations (e.g. optimizer iteration caps) need ML review and user approval.
- The ML benchmark PROTOCOL.md is pre-registered: never edit it to fit results. Changes go through an amendment that keeps the original.
- The planted-fault suite rules are append-only (amendment required).
- Determinism: seed everything; the same input gives the same output.

## 3. Data
- Never commit customer, partner or third-party data (including Bioqore, Cytena, LeadGene and Amii data) unless its rights are confirmed in writing in docs/DATA_RIGHTS.md. External data is loaded through KALOS_EXTERNAL_DATA or a path argument, and results that use it are labelled with its rights status.
- Never commit secrets, .env files, keys or tokens. Report secrets by file:line and type only; never print their values.

## 4. Security and tenancy
- Every route has a scope in docs/route-scopes.json (read/write/admin). There are no public writes.
- The tenant always comes from the verified token, never from the request body or query.
- Auth is EdDSA JWT verified against the JWKS. No HS256 shared secrets and no auth bypass flags in committed code.
- Responses are bounded (pagination, size caps, depth limits).

## 5. Storage
- Migrations: back up first, run in one transaction, number them linearly, document them in docs/MIGRATIONS.md, and test them on a copy of a real database.
- History tables are append-only. Corrections add a new version with a reason; never UPDATE or DELETE history.
- SQLite is the default. Postgres support stays dialect-neutral and is tested in CI.

## 6. Code structure
- New code goes in the target layers from REORG_PLAN.md: core (ML, no I/O), analysis stages, campaigns (rules), storage (SQL only), api (thin routers, no logic), jobs, assist, config.
- The import-boundary tests must pass. Do not add logic to kalos/portal/analysis.py; add a stage module instead.
- Moves use git mv with import-only edits, in separate PRs from behaviour changes.

## 7. Workflow
- Work in your own worktree and branch from origin/main. Never work in ~/kalos/kalos-engine (it serves the live engine on :8050) or in ~/.kalos. Use a temporary KALOS_STATE_DIR for tests.
- Before a PR: ruff, mypy, pytest -n 8 with KALOS_TORCH_THREADS=1, all green. Open the PR with CI green.
- Merges go through the release captain: CI green, a reviewer's SHIP verdict, a verifier's VERIFIED verdict, and no unexplained scoreboard regression.

## 8. Verification
- Every claim (numbers, "tests pass", "byte-identical", coverage, speedups, "refuses when…") must be reproducible from a command written in the PR description.
- An independent verifier (profile `verifier`) re-runs the claims and checks these rules before any merge or published doc change. A claim it cannot reproduce is UNVERIFIED and must be removed or fixed.
- Never skip, delete or xfail a test, loosen a threshold, or re-record a golden fixture without stating it and its reason in the PR.
- Post progress on the Linear issue at start, at major commits, when blocked and when done, with the branch and hash. No secrets or data in Linear.
- Ported code carries an attribution entry in NOTICE.
