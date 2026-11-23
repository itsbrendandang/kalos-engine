---
name: kalos-assist-review
description: Work on Kalos assistant round reviews, role isolation, grounding checks, replay and rate limits. Use for backend review changes and their HTTP contract. Not for UI implementation or scientific method execution.
---

# Kalos assistant review

Read `AGENTS.md` and `docs/ASSISTANT.md` first.
The implementation is in `kalos/assist/swarm.py`.
The routes are in `kalos/assist/routes.py`.

## Roles and verdicts

- Scout uses the local AssistIndex. It does not call a model.
- Planner sees the guide step and labelled sources. It proposes up to five claims.
- Code checks each claim with check_answer in `kalos/assist/chat.py`.
  Check citations, numbers and forbidden claims. Failed claims are rejected.
  Never show rejected claim text.
- Critic sees only surviving claims and their cited sources.
  Never give it the planner's prompt or reasoning.
- Critic returns severities, not verdicts.
  Code maps ok and minor to supported, major to contested, and blocking to rejected.
  No usable judgement means unreviewed.
- Reporter sees only supported claims and their sources.
  Code checks its answer and requires each sentence to match a supported claim.
  On failure, fall back to the supported claim texts.
  With no supported claims, abstain to the guide step.
- Keep prompts separate. Log counts only. Never log lab text or credentials.

## Paid calls

Set `KALOS_NO_PAID_CALLS=1` for agent work.
Never turn it off outside a network-free fake test.
Use the shared refuse_paid helper in `kalos/paid.py`.
It blocks Anthropic and remote OpenAI-compatible calls.
Loopback Ollama and OpenAI-compatible servers remain allowed.
Do not contact a real model for tests.

Each model role can override its model name with
`KALOS_ASSIST_MODEL_PLANNER`, `KALOS_ASSIST_MODEL_CRITIC` or
`KALOS_ASSIST_MODEL_REPORTER`. The provider stays the same.

## Replay and clearing the cache

Replies live under the configured state directory:
`assist/<tenant key>/swarm/`. This is a runtime path, not a repo folder.
Use tenant_dir in `kalos/assist/paths.py` to resolve it.
The tenant comes from the authenticated principal.

A SHA-256 key includes the role, provider, model, instructions and prompt.
Stored replies prevent repeated model calls while cached.
The cache keeps at most 128 replies per tenant. Oldest replies go first.

To clear it, stop only your own engine. Confirm the state directory and tenant.
Delete only that tenant's swarm folder. Never delete campaign state or a parent.
Do not alter another person's engine or state.
The cache contains lab text. Protect it like campaign records.
Clearing or eviction loses replay protection and can cause new paid calls.

## Rate limits

`kalos/assist/ratelimit.py` holds the shared Ask quota.
A review reserves three units, even on replay or with no model calls.
With `KALOS_ASSIST_ASK_PER_MINUTE=1` or `2`, charge that limit instead.
One review can then run per minute. A value of `0` disables the limit.
Retry-After must wait until enough quota units are free.

## Work lanes and checks

Backend work belongs in this repo. UI work belongs in the separate kalos-web repo.
Coordinate through the HTTP contract in `tests/test_web_contract.py`.
Change it deliberately and tell the web lane about contract changes.

Run from the repo root with its Python environment active:

```sh
export KALOS_NO_PAID_CALLS=1
scripts/check.sh
python -m pytest -q tests/test_assist_*.py tests/test_web_contract.py tests/test_agents_md.py tests/test_agent_skills.py
```

Use fake models. Tests must not spend or write to a running engine.
Add change notes under `changes/`, following `changes/README.md`.

## Attribution

Original guidance by the Kalos contributors, under the repo's Apache-2.0 license.
The folder and frontmatter layout follows itsbrendandang/agent-test.
No Biomni instructions or third-party skill content are included.
