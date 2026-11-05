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

A skill is a folder. The folder name is the skill's slug. `SKILL.md` is the only required file.

```
kalos/skills/builtin/<slug>/
  SKILL.md            required: YAML frontmatter + markdown body
  DATA_SOURCES.md     optional: rendered from the frontmatter, never edited by hand
  references/*.md     optional: longer detail, read on demand
  examples/*          optional: synthetic examples only
```

Rules for the folder:

- Nothing else at the top level.
- Text files only (`.md`, `.txt`, `.json`, `.yaml`, `.yml`, `.csv`). Scripts are not allowed.
- No tests, fixtures, expected answers, golden files, lockfiles, dependency pins, or runner copies. Tests and evidence live in `tests/`, outside the package.
- No symlinks.
- Each file is at most 256 KB.
- The `SKILL.md` body is at most 300 lines.
- Skills ship no lockfiles. They run in the engine's single `uv.lock`.

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
| `outputs[]` | `{name, kind, label, min_bytes, description}`; `kind` is `json`, `csv`, `table`, `image`, or `file` |
| `tools[]` | names from the tool table (see below) |
| `entrypoint` | locked skills: the one public engine callable that does the main work |
| `requires_extras` | optional-dependency groups of `pyproject.toml` the tools need |
| `implementation` | `{status: implemented / partial / needs-new-code, engine: [{symbol, ref}], gaps: [...]}` |
| `scope` | `read` or `write` |
| `side_effects` | `none`, `writes-campaign`, or `external`; anything other than `none` needs `scope: write` |
| `refuses_when[]` | required for locked skills |
| `verification` | `{deterministic: [...], receipt, review}`; locked skills set `receipt: true` |
| `author`, `owner`, `visibility` | owner `system`, `tenant:<id>`, or `user:<id>` |
| `licence` | `{spdx, data_sources: [...], third_party_code}` |
| `source`, `status`, `created`, `updated` | status `active`, `beta`, `planned`, or `deprecated` |

The schema rejects unknown fields. It also checks these rules:

- An enum parameter lists `allowed` values, and its default is one of them.
- A default matches the parameter type.
- `implemented` needs at least one engine reference. `partial` and `needs-new-code` need at least one gap.
- `status: planned` is used if and only if `implementation.status` is `needs-new-code`. A planned skill has no tools and no entrypoint.
- Output names are unique, and every output has a kind, a label, and `min_bytes`.
- A skill in the `simulation` category can only return `simulated` or `synthetic` outputs.
- `owner` and `visibility` agree.
- `updated` is not before `created`.

### Two kinds

There are two kinds of skill. There are no other package generations.

| Kind | Must have | Must not have |
|---|---|---|
| `guided` | instructions | an entrypoint; tools that write or have side effects |
| `locked` | an entrypoint, typed outputs, `refuses_when`, `receipt: true` | (none) |

A locked skill that is not `planned` must have an entrypoint. A locked skill with no entrypoint can only be `planned`.

### Entrypoint

The entrypoint is `pkg.module:function`, `pkg.module:Class`, or `pkg.module:Class.method`. It must be public (no leading underscore), and so must the method. It must be the engine symbol of one of the skill's own tools. The checker finds it with `ast`, without importing it.

Example: `identifier-anonymisation` uses `kalos.data.anonymizer:Anonymizer.anonymize_meta`.

### Data sources

Each software, dataset, API, or reference a skill depends on is one row in `licence.data_sources`:

`name`, `kind` (`software`, `dataset`, `api`, `reference`), `version`, `uri`, `licence`, `commercial_status`, `bundled`, `verified_by` (a path or URL), `checked_on` (a date).

`commercial_status` is `allowed`, `allowed-with-attribution`, `needs-agreement`, `blocked`, or `unknown`. Only the first two count as allowed. `unknown` never counts as allowed.

The frontmatter is the source of truth. `kalos skills data-sources DIR/slug` renders `DATA_SOURCES.md`. If that file is present, it must match the rendered text byte for byte. No Kalos skill has data sources today.

### Extras

A skill that needs optional packages lists the extras in `requires_extras` (for example `[ml]` for tools that fit a GP, `[portal]` for tools that read uploads or campaigns). The checker fails if an extra is not defined in `pyproject.toml`, or if a tool needs an extra that the skill does not list.

Per-skill environments (their own lockfile or image) are not used in v1. They are reserved for heavy, conflicting stacks, would be opt-in, and would run across a subprocess or job boundary.

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
- scope, side effects, whether it needs torch, whether it is long-running, a timeout (default 1800 s), and the extras it needs.

List the tools:

```bash
kalos skills tools
```

Inputs are given inline (`rows`, one dict per run; or a file as `csv_text` or `content_base64`) or as a `campaign_id` of the caller's tenant. For a campaign, the tool reads the base rows plus every measured result. Runs that are still waiting for a result are not included.

### Seeds

A tool with a random step takes a `seed`. The skill must then declare an int `seed` parameter with the same default. The seed goes to every random step and into the receipt. Same arguments, seed, engine build, and skill content hash give the same output hashes. `tests/test_skills_runner.py` runs each locked skill with tools twice and compares the output hashes.

## The runner

`kalos/skills/runner.py` is the only runner. Skills never carry a copy of it.

`invoke(skill, tool, args, ctx)` runs one tool:

1. Kalos checks that the skill lists the tool and that the skill is not `planned`.
2. Kalos checks that the caller has the skill's scope and the tool's scope.
3. Kalos checks the arguments against the argument model. Bad arguments stop here. No run directory is made.
4. Kalos makes a fresh run directory and writes a `running` receipt.
5. Kalos runs the engine function with the tool's timeout.
6. Kalos checks the result against the result model, and the result label against the skill's outputs.
7. Kalos writes the artifacts in the run directory and checks each against its `min_bytes`.
8. The receipt moves to `analysis_complete`, then to `terminal` with result `pass`.

### Status

`invoke` returns a small JSON status:

```json
{"status": "completed | blocked | refused | error",
 "summary": "...",
 "artifacts": [{"name", "path", "sha256", "bytes", "label"}],
 "blockers": [{"code", "message", "rows"}],
 "error": {"error_code", "exception_type", "message", "receipt_id"},
 "receipt_id": "skr_..."}
```

Data never travels inline. Each result goes to a JSON file. Row tables (for example the anonymised table and the normalised table) go to CSV files. Read them from the run directory.

| Status | Meaning |
|---|---|
| `completed` | the result passed every check |
| `blocked` | the data cannot support a result: too few rows, a blocking finding in strict mode, a failed model fit |
| `refused` | the request is not allowed or not well-formed for this tool: unknown campaign, a campaign of another tenant, bad column names, a policy |
| `error` | a runtime fault |

Blocked and refused runs are results, not exceptions. They are receipted and never retried automatically. Never get a run to pass by weakening a check, by changing to a weaker split, or by dropping blank or censored values.

An error has an `error_code` (`timeout`, `input_unreadable`, `invalid_result`, `output_too_small`, `label_mismatch`, `path_escape`, `internal`), the exception type, a fixed message, and the receipt id. Kalos never stores the engine's exception text, because it can quote cell values.

### Run directories

Each run writes to a fresh directory: `$KALOS_STATE_DIR/skill_runs/<tenant>/<receipt_id>/`. It holds `receipt.json` and the artifacts. Kalos resolves every output path and refuses a path that leaves the directory. Inputs are hashed, never copied into the run directory.

### Timeouts

Caution: v1 runs each tool on a worker thread in the engine process. A Python thread cannot be stopped.

- When the timeout ends, the receipt becomes `terminal`/`fail` with `error_code: timeout`.
- Kalos stops waiting and discards any late result. It never writes a late result to the receipt or the run directory.
- The computation can keep using CPU until it ends.

Tools marked `long_running` fit a GP: `engine.surrogate_fit`, `engine.propose_batch`, `engine.grouped_cv_report`, `engine.gp_shape_report`, `engine.scale_build_readout`. Gap: these should run through the engine's killable job-worker subprocess (`kalos/portal/job_runner.py`, `python -m kalos.portal.job_worker`) when a run route exists. v1 starts no child processes. A future subprocess runner must pass a minimal, explicit environment.

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

Each receipt (`kalos-skill-receipt/1`) moves forward only:

```
running -> analysis_complete -> terminal (pass | fail)
running -> terminal (fail)
```

A presentation or report step (`finalize`) can only start from `analysis_complete`. Call `invoke(..., hold=True)` to stop there. A terminal receipt is never written again.

A receipt holds:

- id, skill slug, version, and content hash;
- tool name and engine symbol;
- the checked arguments as canonical JSON, and their SHA-256;
- the SHA-256 of each input (tables in the same CSV form that run records use; files as bytes);
- the seed;
- the engine build (`KALOS_BUILD_SHA`) and library versions;
- state, result, status, and summary;
- each output's name, path, SHA-256, size, and label;
- the blockers or the error;
- tenant, subject, campaign, and timestamps.

A receipt never holds input data or output values. It holds only their hashes.

### Verify a run

`verify_receipt(run_dir, skill)` passes only if:

1. the receipt is `terminal` with result `pass`;
2. the skill's content hash today equals the one the run used;
3. every output file is present, inside the run directory, and has the recorded SHA-256 and size.

```bash
kalos skills verify-receipt $KALOS_STATE_DIR/skill_runs/<tenant>/<receipt_id>
```

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
kalos skills data-sources DIR/slug
kalos skills verify-receipt RUN_DIR [--skills-dir DIR] [--json]
```

`python -m kalos.skills` is the same. `DIR` defaults to the Kalos skill set.

`validate` exits 0 when there are no problems, 1 when there are problems, and 2 on a usage error. It checks:

1. the schema, the folder rules, and the body sections;
2. duplicate slugs;
3. each engine reference, at the exact line;
4. each tool name, the tool's label and output mapping, the entrypoint, the extras, and the seed parameter;
5. that `DATA_SOURCES.md`, if present, matches the frontmatter;
6. with `--foreign-corpus`, that no run of 10 words is shared with the given texts.

## Add or change a Kalos skill

1. Create or edit `kalos/skills/builtin/<slug>/SKILL.md`.
2. If the skill calls engine code, add each tool to `TOOLS` in `kalos/skills/tools.py` and its adapter and models to `kalos/skills/bindings.py`.
3. Add 2 positive queries and 1 nearby negative query to `tests/test_skills_routing.py`.
4. Add tests for the items in `verification.deterministic`. Keep test data in `tests/`, not in the skill folder.
5. Run `kalos skills validate` and `pytest tests/test_skills_*.py`.

## Not done yet (phase 2)

These are planned and are not in v1:

- an eval manifest per skill, outside the skill folder (`tests/skills/<slug>/eval.yaml`, schema `kalos-skill-eval/1`);
- `claims[]` in the frontmatter, each with engine references and test references;
- the maturity ladder (`draft`, `structurally_checked`, `evidence_checked`);
- the check that a locked skill has an eval manifest before it is `active`;
- the required test classes for each locked skill: one rejection test per `refuses_when` item, exact boundary tests, end-to-end on synthetic data with artifact reload, determinism, held-out label perturbation for modelling skills, and empty and minimal input;
- preflight per locked skill (`ready`, `missing_dependencies`, `unsafe_environment`);
- the held-out label-perturbation test against the engine's grouped cross-validation.

Also later: freezing skills to lock files, pipelines of skills, per-tenant enable and disable, and running long tools on the job worker.

Splitting rule for modelling skills: all splits come from `kalos/core/splits.py`. Modelling code takes indices and never builds its own splits. Group leakage raises. Distance-to-data uses training rows only.

## What a skill does not mean

- A skill describes what Kalos does. It does not show that a method fits a given process.
- A content hash shows which bytes ran. It is not a review or an approval.
- Simulated values say nothing about a real process unless the model was fitted to that process and checked against it.

## License

The Kalos skill texts are Apache-2.0, like the engine. The skills code is Kalos code. It uses no third-party code.
