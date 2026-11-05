# Skills

A skill is a small, versioned package that describes one task Kalos can do. Examples: check a run sheet, estimate replicate noise, propose the next batch.

Each skill says:

- what it is for, and what it is not for;
- what it reads and what it returns;
- which engine functions do the work;
- when it refuses;
- how each output is labelled (measured, predicted, simulated, synthetic, derived, or metadata).

The code is in `kalos/skills/`. The Kalos skill set is in `kalos/skills/builtin/`.

## Status in v1

- 19 Kalos skills ship with the engine.
- 11 skills call engine code that already exists.
- 3 skills call engine code that exists in part. Each lists its gaps in its own `SKILL.md`.
- 5 skills are `planned`. They have a skill file but no tools. Nothing runs for them.
- The HTTP routes are read-only. You can list, search, and read skills. You cannot create, edit, or run a skill over HTTP.
- No LLM is used in any skill computation.

## Install

The skills code needs the `skills` extra (pydantic and PyYAML).

```bash
uv run --extra ml --extra portal --extra skills kalos skills validate
```

Importing `kalos.skills` does not load torch. Torch loads only when a tool that fits a model runs for the first time.

## The skill folder

A skill is a folder. The folder name is the skill's slug.

```
kalos/skills/builtin/<slug>/
  SKILL.md            required: YAML frontmatter + markdown body
  references/*.md     optional: longer detail, read on demand
```

Rules for the folder:

- Text files only (`.md`, `.txt`, `.json`, `.yaml`, `.yml`, `.csv`). Scripts are not allowed.
- No symlinks.
- Each file is at most 256 KB.
- The `SKILL.md` body is at most 300 lines.

### Identity

The skill's identity is its content hash. Kalos computes SHA-256 over the sorted list of (relative path, SHA-256 of the file bytes) for the whole folder. A change to any byte of any file gives a new hash. The `version` field is a label for people. It is not the identity.

Show the hash of one skill:

```bash
kalos skills hash kalos/skills/builtin/driver-ranking
```

### Frontmatter (`kalos-skill/1`)

| Field | Rule |
|---|---|
| `schema` | `kalos-skill/1` |
| `slug` | kebab-case, at most 64 characters, equal to the folder name |
| `name` | 3 to 80 characters |
| `version` | semantic version |
| `kind` | `locked` (one engine-backed computation) or `guided` (instructions only) |
| `category` | `intake`, `data-quality`, `design`, `optimisation`, `trust`, `scale-up`, `simulation`, `analytics`, `provenance`, `governance` |
| `summary` | 60 to 700 characters; must say what the skill is not for ("Not for ...") |
| `keywords` | list of words for search |
| `starting_prompt` | an example request |
| `inputs[]` | `{name, type, required, description, label?}` |
| `parameters[]` | `{name, type, default, allowed, effect}`; `effect` is required |
| `outputs[]` | `{name, type, description, label}`; `label` is required |
| `tools[]` | names from the tool table (see below) |
| `implementation` | `{status: implemented / partial / needs-new-code, engine: [{symbol, ref}], gaps: [...]}` |
| `scope` | `read` or `write` |
| `side_effects` | `none`, `writes-campaign`, or `external`; anything other than `none` needs `scope: write` |
| `refuses_when[]` | required for locked skills |
| `verification` | `{deterministic: [...], receipt, review}` |
| `author`, `owner`, `visibility` | owner `system`, `tenant:<id>`, or `user:<id>` |
| `licence` | `{spdx, data_sources, third_party_code}` |
| `source`, `status`, `created`, `updated` | status `active`, `beta`, `planned`, or `deprecated` |

The schema rejects unknown fields. It also checks these rules:

- An enum parameter lists `allowed` values, and its default is one of them.
- A default matches the parameter type.
- `implemented` needs at least one engine reference. `partial` and `needs-new-code` need at least one gap.
- `status: planned` is used if and only if `implementation.status` is `needs-new-code`. A planned skill has no tools.
- A skill in the `simulation` category can only return `simulated` or `synthetic` outputs.
- `owner` and `visibility` agree.
- `updated` is not before `created`.

### Body

The body must have these `##` sections: Purpose, When to use, When not to use, Inputs, Procedure, Outputs, Stop conditions, Caveats, Example.

### Engine references

Each engine reference has a symbol (`kalos.core.optimize:propose_batch`) and a location (`kalos/core/optimize.py:408`). The checker parses the file with `ast`. It does not import it. The symbol must be a top-level `def` or `class` at exactly that line. When engine code moves, `kalos skills validate` fails until the skill is updated.

## Who sees which skill

| Source | Owner | Who sees it |
|---|---|---|
| Kalos | `system` | every tenant |
| Team | `tenant:<id>` | only that tenant |
| Mine | `user:<id>` | nobody in v1 |

Team skills are read from `$KALOS_SKILLS_DIR/<tenant>/<slug>/`. The owner must match the tenant folder. A Team skill cannot use the slug of a Kalos skill. There is no route to create a Team skill in v1. An operator places the folder on disk.

A skill that the caller cannot see answers 404, the same as a skill that does not exist.

## Search

Search is deterministic, so routing can be tested.

1. Kalos splits the query into words of 3 or more characters and removes common words.
2. Kalos cuts each word to its first 5 characters.
3. Each matching word adds points by field: slug and name 5, keywords 3, category 2, summary 1.5, body 0.25.
4. Results sort by score, then by slug.

By default, `beta` and `planned` skills are hidden. Use `include_beta` and `include_planned` to show them.

`tests/test_skills_routing.py` holds the routing table. Each skill has at least 2 queries that must find it first, and 1 nearby query that must find a different skill. A change to search or to a skill text must keep this table green.

## Tools

A tool is one call into the engine. The tool table is in `kalos/skills/tools.py`. It is written by hand and reviewed like other code. Nothing is generated at run time.

Each tool has:

- the engine symbol it calls;
- a pydantic model for its arguments;
- a pydantic model for its result;
- the label of its result;
- scope, side effects, whether it needs torch, and a timeout.

List the tools:

```bash
kalos skills tools
```

Inputs are given inline (`rows`, one dict per run) or as a `campaign_id` of the caller's tenant. For a campaign, the tool reads the base rows plus every measured result. Runs that are still waiting for a result are not included.

### How a tool call runs

`kalos.skills.tools.invoke(skill, tool, args, ctx)` is the only way to run a tool.

1. Kalos checks that the skill lists the tool and that the skill is not `planned`.
2. Kalos checks that the caller has the skill's scope and the tool's scope.
3. Kalos checks the arguments against the argument model. Bad arguments stop here.
4. Kalos runs the engine function with a timeout.
5. Kalos checks the result against the result model.
6. Kalos checks that the result label is one of the skill's output labels.
7. Kalos writes a receipt.

A refusal is a normal result. For example: too few rows, a campaign of another tenant, or blocking findings in strict mode. Kalos returns `outcome: refused` with the reason and writes a receipt. Do not retry a refusal with the same input.

Caution: the timeout stops waiting for the result. It does not stop the engine thread. Long model fits will move to the job runner when a run route exists.

### Labels

| Label | Meaning |
|---|---|
| `measured` | values from a real experiment |
| `predicted` | model output, such as a surrogate model mean and uncertainty |
| `simulated` | output of a simulator, such as the kinetic model |
| `synthetic` | generated test data |
| `derived` | computed from other values, such as a noise estimate or a check report |
| `metadata` | facts about data, not data |

A `predicted`, `simulated`, or `synthetic` result can never be written as measured data. `guard_measured_write` enforces this for any write path. No v1 tool writes campaign data.

## Receipts

Every tool call that passes argument checks writes one receipt. This includes refusals and errors.

A receipt holds:

- skill slug, version, and content hash;
- tool name and engine symbol;
- SHA-256 of the checked arguments;
- SHA-256 of the input table, in the same CSV form that run records use;
- the seed, when the tool takes one;
- the engine build (`KALOS_BUILD_SHA`);
- each output's name, label, and SHA-256;
- the refusal code and reason, if any;
- tenant, subject, campaign, and start and finish times.

A receipt never holds input data or output values. It holds only their hashes. For an engine error it stores the exception type, not the message.

Receipts are stored as `$KALOS_STATE_DIR/skill_receipts/<tenant>/<receipt_id>.json`. Kalos writes each file once and never changes it.

## HTTP routes

The routes are off by default. Set `KALOS_SKILLS=1` to serve them. When off, every `/api/skills` path answers 404.

All routes need the `read` scope. The tenant comes from the caller's token.

| Method and path | Returns |
|---|---|
| `GET /api/skills` | `{skills, next_cursor}`. Filters: `category`, `kind`, `status`, `implementation`, `source`, `include_beta`, `include_planned`, `limit`, `cursor` |
| `GET /api/skills/search?q=` | `{results: [{score, skill}]}`. Also `category`, `include_beta`, `include_planned`, `limit` |
| `GET /api/skills/{slug}` | summary fields, `frontmatter`, `body`, `files`. Header `ETag: "<content hash>"`; `If-None-Match` gives 304 |
| `GET /api/skills/{slug}/files` | `{files: [{path, size, type}]}` |
| `GET /api/skills/{slug}/files/{path}` | the file as text |
| `POST`, `PUT`, `PATCH`, `DELETE` under `/api/skills` | 405 with `Allow: GET` |

The file route serves only files that are part of the hashed package. It refuses `..`, absolute paths, and symlinks.

Errors use the engine's error envelope: `{"error": ..., "code": ...}`.

## Command line

```bash
kalos skills validate [DIR] [--engine-root P] [--foreign-corpus DIR ...] [--json]
kalos skills list     [DIR] [--category C] [--json]
kalos skills search   "query" [DIR] [--all] [--limit N]
kalos skills hash     DIR/slug
kalos skills tools    [--json]
```

`python -m kalos.skills` is the same. `DIR` defaults to the Kalos skill set.

`validate` exits 0 when there are no problems, 1 when there are problems, and 2 on a usage error. It checks:

1. the schema, the folder rules, and the body sections;
2. duplicate slugs;
3. each engine reference, at the exact line;
4. each tool name, and that the tool's label is one of the skill's output labels;
5. with `--foreign-corpus`, that no run of 10 words is shared with the given texts.

## Add or change a Kalos skill

1. Create or edit `kalos/skills/builtin/<slug>/SKILL.md`.
2. If the skill calls engine code, add each tool to `TOOLS` in `kalos/skills/tools.py` and its adapter and models to `kalos/skills/bindings.py`.
3. Add 2 positive queries and 1 nearby negative query to `tests/test_skills_routing.py`.
4. Add tests for the items in `verification.deterministic`. Keep test data in `tests/`, not in the skill folder.
5. Run `kalos skills validate` and `pytest tests/test_skills_*.py`.

## What a skill does not mean

- A skill describes what Kalos does. It does not show that a method fits a given process.
- A content hash shows which bytes ran. It is not a review or an approval.
- Simulated values say nothing about a real process unless the model was fitted to that process and checked against it.

## License

The Kalos skill texts are Apache-2.0, like the engine. The skills code is Kalos code. It uses no third-party code.
