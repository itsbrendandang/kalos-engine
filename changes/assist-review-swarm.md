### Added - assistant round review

- `POST /api/assist/review` runs a source scout, planner, isolated critic and reporter. Code checks grounding and assigns claim verdicts. No supported claims means abstaining to the guide step.
- `GET /api/assist/review/roles` lists the team and its model groups. Per-role model names are configurable. Tenant-local reply files prevent repeat model calls on replay.
- Root `AGENTS.md` describes coding lanes, HTTP contract checks and safe local test servers.

### Added - paid model call guard

- `KALOS_NO_PAID_CALLS=1` (also `true`, `yes` or `on`) refuses billable Anthropic calls. Unset keeps existing behavior. The assistant reports provider unavailable; normalization falls back offline. Local Ollama is unchanged.
- Every test starts with paid calls blocked. Only network-free fake tests may turn the guard off.
