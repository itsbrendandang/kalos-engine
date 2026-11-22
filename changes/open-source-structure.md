### Added - open-source project files (no engine change)

- `CONTRIBUTING.md`, `NOTICE`, `CITATION.cff`, `.python-version` (3.11, as CI), issue templates (bug, feature; security goes to private advisories), Dependabot (weekly; torch/botorch/gpytorch held), optional pre-commit hooks.
- `docs/DATA_RIGHTS.md`: the written data-rights record AGENTS.md rule 3 points to. The LeadGene/Cytena and Anagram/Bioqore datasets are owned (confirmed by the owner, 2026-11-21); results from them may be published. Raw files still load from outside the repository.
- Changelog entries are now files in `changes/`, folded into this file by `scripts/changelog.py`, so PRs stop conflicting here.
- CI: a job runs the assistant and LLM tests with the `assist` extra installed (the shards do not install it; that gap hid #202).
- `scripts/worktrees.sh` (read-only): worktrees with uncommitted files, unpushed commits, gone remotes or merged branches.
