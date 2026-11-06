# Changelog

Newest first.

## Unreleased

- Verdict wording: "reliable" now says "with enough runs to check the ranking" instead of "to be sure".

- Search box: a shown value within 0.0005 of the tried edge (3-decimal rounding) is no longer flagged as beyond the tried range.

- Security review S7: at most 5 `roles.specs` per upload (was 20). Each spec is one more model fit.

- Security review S4: the spread design computes nearest distances in blocks with `cdist`. At 2000 rows x 40 inputs, peak memory drops from about 2.8 GB to under 200 MB.

- Security review S3: a best bet predicted below the best run says the model does not expect any recipe in the box to beat it; "to learn, not to win" is no longer used for the best bet.

- Security review S1: `possibly_measured_after_run` is now a warning. If a flagged input is a top driver, the verdict is held at not_yet with reason `possibly_measured_driver`. New measurement words: ivcd, mab, igg, _end, dayN, peak (residual was already there). A `VCD_day14` column that tracks the target no longer gives "reliable".

- Security review S2: the best bet meets the same outcome rules as the batch: the constraint model's prediction at or above its floor, and P(feasible) at least 0.5 when the feasibility gate is on. With no qualifying candidate there is no best bet and `best_bet_notice` says why in plain words.

- Security review B1: the best bet is the highest predicted recipe among the optimizer's answer, every batch row, every training row and a Sobol sample of the box (within the linear constraints), so a local optimum can no longer leave a higher-predicted proposal labelled `learn` (pichia_HSA: 74.9 -> 110.0). `model_optimum.beyond_tried_range` added.
### Fixed - desktop-engine-2-secb (security review)

- Campaign rows: an upload row with a blank target is `_kind: no_result`, not `measured`; a sheet column named like a row label (`_origin`, `_kind`, ...) is shown as `<name> (sheet)` and listed in `renamed_columns` instead of being overwritten.

- `roles.allow_beyond` naming a column that is not in the sheet is refused with 400 `invalid_allow_beyond`, naming it, instead of being ignored.

- The audit row of a successful upload names the campaign settings it set, changed or cleared (`settings: allow_beyond 0.1 -> 0.2; features changed`): key names, with old -> new values only for small scalar settings, never column names or file contents.

- Every finding and coded refusal carries a plain `why` line from the engine itself (kalos/portal/why_lines.py), so a client never shows a bare code. A test scans the engine source for codes and fails on any without a line.

- Reanalyze refuses with 409 `roles_not_applied` when the column roles stored with the campaign cannot be read, instead of quietly fitting with inferred roles. Nothing is committed; upload the sheet again with valid roles.

### Added - desktop-engine-2-q8 (constraints and specs)

- `roles.linear_constraints`: hard linear limits on continuous inputs, including equalities for mixtures (A + B + C == 100) and budgets (Glucose + Glycerol <= 30). Every proposal and the best bet meet them; each carries `constraints_ok`, and the analysis reports `linear_constraints`. Bad or impossible constraints are refused with a 400 before any fit.

- `roles.specs`: limits on outcome columns (Purity >= 95). Each proposal and the best bet report `p_all_specs`, the chance all specs are met at once, and `p_specs` per spec; `joint_feasibility` says how (one model per spec, 4096 seeded draws each, sampled independently). Advisory: the batch does not change.

### Added - desktop-engine-2 (first-time-user study)

- `stopping.caveat` (`stopping_advice_only`) and `joint_feasibility.caveat` (`specs_treated_independently`): the two caveats in plain words in every response, not only in the docs.

- `POST /api/preflight` (and the job pre-check) refuses infeasible linear constraints with the same 400 `infeasible_linear_constraints` as `/api/run`: the preflight now stops after the search box is built, still without a fit.

- Name rule (owner decision): a column is left out by its name only on an outcome word (titer, yield, product, purity, recovery, activity, response, endpoint, harvest, viability at end, `final` without a concentration unit) or, after the target, on a measurement word. A concentration unit is not outcome evidence. Before the target, measurement words (OD, biomass, lactate, ammonium, ...) keep the column as an input with an info finding `possibly_measured_after_run` (reason `possibly_measured`). `columns[].finding_code` names the finding for every excluded or flagged column. Candida's `ammonium_g_L` and scFv's `Final Urea (M)` are inputs again; tested on all 8 sample sheets.

- `stopping`: advice on whether more rounds can still beat the best run by more than the assay noise (`keep_going`, `gains_within_noise`, or `unknown` without repeats). Advice only: it never removes proposals. `kalos/bench/stopping_sim.py` measures how often it says "within noise" too early on a synthetic case.

- `noise.floor`: the pooled assay noise with a 95% interval and its degrees of freedom. `roles.per_recipe_noise` (off by default) lets the proposal model use each recipe's own spread, shrunk toward the pooled one, when the noise rises with the recipe mean; `noise.per_recipe` says whether it was used and why.
- Exploration design (QUEUE 6): with fewer than 2 x inputs distinct recipes, or when the model cannot rank recipes yet, the batch is the best bet plus recipes spread over the untested parts of the box, labelled (`batch_strategy`, `pick: exploration_design`). `roles.exploration_design: "off"` turns it off. No detectable difference against the model's picks on two synthetic surfaces (12 seeds).
- Public benchmark (QUEUE 10): DataHow's in-silico fed-batch runs (MIT, data not in the repo, `KALOS_DATAHOW_DATA`) as an endpoint table (`kalos/bench/insilico_hybrid.py`, `examples/insilico_hybrid_benchmark.py`, `benchmarks/INSILICO_HYBRID.md`). Noisy variant, 20 seeds: ranking of the 100 test runs from 20 train runs Spearman 0.20 (-0.02 to 0.40); pool replay BO 31.4 vs random 32.8 runs to 90% of the best (difference -1.5, -11.8 to 8.0): not separated.

- `GET /api/campaigns/{id}/rows`: the campaign's rows (upload and folded rounds) with `_origin`, `_round`, `_run_id`, `_kind` (`measured`), paged.

- `POST /api/preflight`: the column reading (`columns`, `provenance`, target and inputs) and the sheet check, with the same refusals as an upload, without fitting or storing anything.

- A refused verdict also clears `model_optimum` (no best bet without proposals). `tests/test_firsttime_flow.py` runs the study path end to end with the real engine.

- Finding 2 leftovers: a row-number column is still recognised when some rows were entered twice, and a text column with no header is not an input unless declared (finding `headerless_column`). On the study's messy sheet the model no longer uses `Row` or the blank-header column; the fit went from 60 s to 8 s.

- `roles.allow_beyond` lets the search go past the tried range of an input (0.10 = 10% past each end; per input or for all; physical limits still apply). Proposals outside the tried range carry `beyond_tried_range`, the analysis reports `tried_ranges` and `allow_beyond`, and a `search_box` finding always says in plain words where Kalos searches and how to widen it (Finding 11).

- Every batch has the model's best bet as P1 (`proposals[].pick`, `model_optimum`): the highest predicted value inside the search box. It replaces the proposal with the lowest prediction, or the proposal already within 5% of it. Each proposal says how its prediction compares with the best run (`predicted_vs_best`); a run predicted below the best says it is there to learn, not to win. Before, all five proposals were predicted below the best run on 4 of 4 study sheets.

- README rewritten in plain words: what Kalos is, one round, the desktop app, the hosted mode, the evidence with the open-replay numbers verbatim (BO beats random clearly on 1 of 15 datasets), what Kalos does not do, every `KALOS_*` setting with its default and range, the branch and PR map. `tests/test_readme_env.py` fails when the settings table and the code disagree. `docs/ARCHITECTURE.md`, `deploy/RUNBOOK.md`, `deploy/compose.yaml` and `deploy/.env.example` no longer say the web app has no login or sends a static token: kalos-web `main` signs people in and mints a token per request; the desktop app uses a per-launch secret; the compose file has not been wired for sign-in yet.
- Findings about rows list every affected row as the row number in the sheet (`sheet_rows`, header = row 1), with the cells (`values`) for one-column findings. `detail.n_rows_affected` matches. Dropped and censored cells now name their rows too.

- `promotion.checks`: each promotion check says in plain words what it checks, why it matters, what to do and whether it is optional. The feasibility checks no longer read as an instruction to run failures.

- One verdict on the model: `verdict` (`reliable`, `not_yet`, `refused`) on every analysis and campaign summary. "Reliable" needs at least 20 runs and 8 groups; a refused sheet gets no proposals and a `refusal` that says what would help.
- Each campaign analysis lists the recorded runs that entered the fit (`runs_used`: measured value, prediction at start, inside the likely range) and what changed since the previous analysis (`delta`: best, verdict, driver shares, run count, inputs, and a one-line `summary`). Started runs keep their likely range (`likely_range_at_start`).

- The roles sent with an upload are kept with the campaign (`settings.roles`) and used by every reanalyze. A reanalyze keeps the campaign's name. Each analysis says which roles it used (`roles_used`), and `campaign_findings` has `inputs_changed` when the inputs differ from the analysis it replaced.

- Every analysis has an `analysis_id`, the same on `/api/run`, `/api/latest`, `/api/campaign`, `/api/campaigns/{id}/analysis` and the job view. `/api/run` returns the stored analysis. A refit that replaces a campaign's analysis says so in `supersedes` (`{analysis_id, code, why}`).

- `columns[]` lists entirely blank columns too (kind `constant`, reason `all_blank`, in sheet order).

- `proposals[].vals` holds the same rounded, box-clipped numbers as `proposals[].recipe` (they could differ by the clip before).

- `run.schema_version` is 2: `direction` is in the run record and in `config_sha256`, so new hashes do not match version-1 hashes of the same upload.

### Added - column kinds

- Every analysis carries `columns`: each uploaded column with `kind` (numeric, category, text_id, row_number, outcome_like, constant), `role`, `used_in_fit`, and for categories the levels with counts. A text column with more than 12 labels (Pichia CarbonSource = src_0..src_18) is reported as an excluded category with the finding `category_not_modeled` instead of `dropped_sparse` with no finding; categories with up to 12 labels are fitted as categorical inputs as before. Date and time columns are `kind: date`. `reason` and `finding_code` are closed, documented code lists.

### Added - desktop engine (`docs/DESKTOP.md`)

- Importing the portal no longer imports torch. The scale readout module is loaded on first use.
- Desktop mode (`python -m kalos.portal --desktop` or `KALOS_DESKTOP=1`): loopback only, a token is required, `KALOS_TORCH_THREADS` defaults to half the cores (1-4), `KALOS_JOB_WORKERS=1`, no API docs, no audit-archive timer, a Host-header allowlist, torch imported in the background after start. Refusals print one line starting `kalos engine:` and exit 2.
- Startup self-check in desktop mode: an unwritable state dir, a damaged `portal.db` (SQLite `quick_check`), a wrong table shape or a newer schema stops the engine before the port opens (exit 3, one plain line, nothing written).
- `GET /api/health` (read scope): status, mode, engine and contract versions, Python, torch and BoTorch versions (from package metadata, torch is not imported), whether torch is loaded, torch threads, analyses in flight, uptime, pid, state dir, schema version and the expected one. 503 with `code` `store_unavailable` or `schema_mismatch` and a plain `error`. `/healthz` is unchanged.
- Fit time on replicate sheets with a text recipe label: the inferred group column (for example `recipe` = R1..R10, one label per pH/temperature setting) is no longer also fitted as a categorical input (`provenance` status `dropped_recipe_label`), and leave-one-group-out is skipped when the groups are exactly the recipes. A 30-row, 10-recipe sheet went from 60-88 s to about 6 s in-process (docs/DESKTOP.md, "Fit time").
- Mixed (categorical) proposals enumerate every combination only while combinations x batch size stays within 64 solves (`MAX_MIXED_SOLVES`); larger cases use the alternating optimizer and report `proposal_optimizer: "mixed_alternating"`. The benchmark's mixed task (12 combinations, q=4) is unchanged. 20 combinations x 5 proposals went from 44 s to 10 s with the same predicted means on the shared proposals.
- The same upload is not refit: the engine keeps the last 4 analyses in memory (`KALOS_ANALYSIS_CACHE`, 0 = off) keyed on the bytes, file name, sheet, target, roles, anonymize, profile, engine version and every `KALOS_*` setting. A repeat returns the same analysis with `analysis_cache: {"hit": true}`. Off in the test suite by default.
- `scripts/build_desktop_engine.sh` builds a self-contained engine for arm64 macOS (python-build-standalone + the locked dependencies, launcher `kalos-engine`), smoke-tests it in desktop mode and writes `BUILD_INFO.txt`. 955 MB on disk, 272 MB zipped. `docs/DESKTOP.md` lists what the shell passes and what the engine answers.
- The desktop build also bundles the `assist` extra when the checkout defines it (`KALOS_BUNDLE_EXTRAS` overrides the list); `docs/DESKTOP.md` lists the assistant's Ollama defaults.
- Desktop mode sets an unset `KALOS_LLM_PROVIDER` to `ollama` when a local Ollama answers on `KALOS_OLLAMA_URL` (default 127.0.0.1:11434, 0.2 s check), else `none`.
- Graceful stop in desktop mode: on SIGTERM an in-flight analysis gets 12 s to finish; after that no store write can commit, and the process exits.
### Added - conformal caveat

- `conformal_caveat` (`code: coverage_not_guaranteed`) says the `conformal_q` band's coverage was measured near the data and is not guaranteed for proposals, especially far-from-data or box-edge ones.

### Fixed - the GP kernel is pinned, and documented as what it is

- The surrogate uses BoTorch's default kernel: an ARD RBF with a dimension-scaled LogNormal lengthscale prior and no output scale (plus a CategoricalKernel for categorical inputs). Comments and docs said Matern; they now say RBF.
- Every fit checks the kernel, lengthscale prior and constraint, noise prior and floor (`kalos/core/gp_pin.py`) and fails if a BoTorch upgrade changes them. The model is not rebuilt, so proposals are unchanged.
- `model_diagnostics.kernel` names the kernel.

### Added - applicability flags per proposal

- `proposals[].applicability` has `distance`, `threshold`, `threshold_source`, `far_from_data` and `at_box_edge` (inputs within 1% of the box width of a bound). The two flags are separate.
- Top-level `applicability` records the threshold, its source and rule. Default: `max(0.25, 3 x training_nn_distance_median)`, a product default. Labs set `roles.applicability_distance` or `KALOS_APPLICABILITY_DISTANCE`.
- Fixed (replay F9): rounded proposal values are clipped into the search box.

### Added - per-target direction (maximize or minimize)

- `roles.direction` (and a plain `direction` form field on `POST /api/run`) takes `maximize` (default) or `minimize`. Other values: `400 invalid_direction`. The response, the stored campaign and the campaign summary carry `direction`; reanalyze reuses it.
- The run record has `run.direction` (upload, job and reanalyze), and `direction` is in `CONFIG_FIELDS`, so maximize and minimize on identical bytes get different `config_sha256`. `proposal_gating.off_for_minimize` says when the gate was off for a minimize target.
- Minimize fits the proposal model on -y. Predictions, `best` (now the min), drivers and `gp_shapes` stay in original units and sign. The P(feasible) gate is off when minimizing. Maximize output is unchanged.

### Fixed - reanalyze 500 with 2 CV groups (replay F1)

- With fewer than 3 CV groups `cv_ci95` and `reliability.ci95` are now null, with `cv_ci95_reason` and a `cv_ci_unavailable` finding. They were NaN, which JSON cannot encode.
- Every analysis passes a last check that turns any NaN or infinity into null and logs the top-level keys.
- Reanalyze checks the analysis encodes before the fold is committed. On any error nothing is committed: campaign, runs and `/api/latest` stay as they were.
### Fixed - every finding has a code; promotion says why it did not pass

- `validation.findings[].code` was null everywhere. It is now a stable string: a specific code where one exists (`coded_column`, `row_number_column`, `near_constant_input`, ...), else the check name.
- `promotion.status` (`passed`, `failed`, `not_measurable`) and `promotion.unmeasured` tell a missed gate from one that could not be measured. `promotion.passed` is unchanged.

### Fixed - near-constant inputs are fixed at their median

- Inputs whose spread is within sensor precision (an inline pH of 6.98-7.03, a temperature of 36.98-37.01 C) were searched. They are now fixed at their median and left out of the model. Proposals carry the value; `fixed_inputs`, a `fixed_near_constant` provenance status and a `near_constant_input` finding say so.

### Fixed - coded columns are not read as flow rates

- A coded or pseudonymised factor with negative values (the Rudt `feed_code`) failed validation as a negative flow rate and its search range was cut to [0, 1]. Unit-based checks now skip coded columns with an `info` finding (`coded_column`), and the observed range is kept. Columns with a unit are still checked.

### Fixed - two-column sheets read as transposed

- A `condition,y` sheet (labels such as `A pH 4`) and its 3-column variant were read as transposed and refused as "1 data row". A first column with repeated labels, or a sheet too narrow to hold 3 runs, now stays a normal sheet. A transposed sheet that is still too short gets a message with its real rows and columns (`rows_as_written`).

### Fixed - column-role safety

- A headerless row index and a 0..N row-number column are no longer model inputs by default. Outcome-like columns after the target (OD readings) are left out too. A concentration input before a given target (`Carbon_conc`) is kept. Declared features are never excluded. Each `provenance` entry adds `role_inferred` and `reason`; findings add `code`.

### Changed - audit trail and recorded results (schema v5, `docs/MIGRATIONS.md`)

- **Recorded results are append-only.** A re-submitted campaign result is a new `run_results` row (value, value replaced, reason, actor, server time). A correction needs a `reason` (`400 reason_required` without one). The run's current value is the latest. Response fields are added (`result_version`, `previous_result`, `result_reason`, `result_actor`, `result_recorded_at`); none is removed.
- **`campaign.result` audit rows carry the change** (run id, old and new value, reason, actor) and are written in the same transaction as the change. A failed audit write fails the change.
- **`audit_log` rejects UPDATE and DELETE** at the database level (schema v4 triggers). The 365-day delete is gone: `KALOS_AUDIT_RETENTION_DAYS` is now an opt-in archive that copies old rows to `<state>/audit-archive/` and deletes none. Unset, it archives nothing.
- A fit-row cap refusal carries `code: too_many_fit_rows`, `max_fit_rows` and `n_rows`; the message text is unchanged.
- Migrations refuse, before any backup or rename, a database with v2 tables but no version row.

### Added - the campaign says where it is and what to do next (PR #69, re-implemented on main)

- `GET /api/campaign`, the `campaign` object from `POST /api/campaign/reanalyze`, and `summary` on `GET /api/campaigns/{id}` now carry `stage` and `next_action`, derived server-side from that campaign's own analysis, so every client shows the same round state.
- `stage` has the 1-based round (null in `setup`), the current stage (`setup`, `plan`, `run`, `record`, `learn`), the started count and five ordered steps with a state and a detail line. `next_action` is `{stage, action, reason}`.
- `POST /api/campaigns/{id}/reanalyze` and `POST /api/campaign/reanalyze` report the campaign they reanalyzed. The legacy reanalyze response now also reports `analysis_in_sync: true` for the fresh analysis (it reported `false` before).
- Each pending run reports `round`, the round it was started in (`null` for older runs).
- All existing fields are unchanged.

### Added - `feature_ranges` in the analysis output (`feat/feature-ranges`)

- `_analyze` now returns `feature_ranges`: `{feature: [lower, upper]}` for every continuous input. It is the **search box** the optimizer uses (the observed range, widened to include 0 for absent-as-zero columns), not the raw observed min/max. Clients can flag a proposal sitting on the edge of it. Categorical inputs are omitted; keys are aliased when `anonymize=True`.

### Fixed - stale torch extension lock hung multi-objective requests (`fix/fused-kernel-hang`)

- BoTorch JIT-compiles its fused qLogEHVI kernel on first use. On torch 2.13 (the version `uv.lock` and the image pin) that load waits on a plain `lock` file with no timeout, so a compile killed mid-way left a lock that blocked every later `/api/multi` call until the 120 s demo timeout. Reproduced with a stale lock and no built `.so`. torch 2.14 fixes this upstream (OS-level `FileLock`, pytorch#189245); `kalos/core/torch_ext.py` removes a lock older than 5 minutes before the first load, so the pinned torch is protected. A young lock (a real concurrent compile) is left alone. Numerics are unchanged: the fused kernel still loads.

### Security - bandit (`S`) rules enabled in ruff (`chore/bandit-triage`)

- Triaged every `ruff --select S` finding in `kalos/`. Real fixes: `normalize/llm.py` refuses non-http(s) Ollama URLs (S310); `portal/busy.py` logs a failing slot listener instead of swallowing it silently (S110).
- Suppressed with per-site reasons: constant-identifier SQL f-strings (S608), `ps`/`git`/`sysctl` fixed argv (S607), the wildcard-host refusal list (S104), an env-var name (S105). `S101` (assert) is ignored globally: asserts state invariants and narrow types, and no request validation relies on them. Tests, examples and experiments are exempt from `S`.

### Refactor - split `kalos/portal/analysis.py` (`refactor/split-analysis`)

- Behavior-preserving. `kalos/portal/analysis.py` (2,727 lines) keeps `_analyze`, `_analyze_body`
  and the seeded-RNG scope, which hold the monkeypatch points the tests use. Helpers moved to
  `kalos/portal/analysis_parts/`: `policy` (named constants, `validation_mode`), `identifiers`,
  `anonymize`, `columns`, `ranges`, `noise_blocks`, `refusals`, `diagnostics`, `proposals`.
  Every public name, and every private name a test or caller imports, is still importable from
  `kalos.portal.analysis`; the public surface used by callers and tests is unchanged, but `analysis.__all__` no longer lists every moved name (e.g. `ParsedUpload`, `build_report`). Unreferenced private helpers now live in `analysis_parts`.

### Build - lock, image pins, supply-chain gate (`chore/lock-pinned-image`)

- **`requires-python` is `>=3.11`** (was `>=3.10`). `ax-platform` 1.3.1 (the `bench` extra) needs 3.11, so the 3.10 floor made the lock unresolvable. `[tool.ruff] target-version = "py311"` is explicit; README updated.
- **`uv.lock` is current.** It pinned fastapi 0.139 while pyproject asked for `>=0.143`. `uv lock --check` now passes, and CI runs it.
- **The engine image installs the locked transitive set.** `deploy/engine/constraints.txt` is `uv export --frozen --extra ml --extra portal` minus torch; `deploy/engine/Containerfile` installs with `pip install -c constraints.txt`. Torch is pinned by `ARG TORCH_VERSION` (2.13.0, the lock's version) on the CPU index. Before this the image resolved torch 2.14.1, which the benchmark notes already flagged as giving different BO results.
- **Two CVE bumps in the lock:** filelock 3.29.7 -> 4.2.4 (CVE-2026-112940/112913/112907, fixed in 4.2.1) and setuptools 83.0.0 -> 84.2.1 (GHSA-xmj3-9gf7-h52w, fixed in 84.1.0).
- **CI:** `test` runs `uv lock --check`. New `audit` job checks that `constraints.txt` and the Containerfile torch pin match `uv.lock`, then runs `pip-audit --no-deps --disable-pip -r deploy/engine/constraints.txt`.

### Fixed - API and ops review (`fix/review-api`)

- **Unknown `KALOS_*` enum values refuse to start (H3).** `KALOS_VALIDATION_MODE=stict` used to select `warn` silently, and a `KALOS_LLM_PROVIDER` typo fell back to the live Anthropic path. One helper (`kalos/envconfig.py`) now raises; `python -m kalos.portal` checks every enum and bounded setting before it binds and exits 2 with one line. `strict` stays out of the templates (docs/HARDENING.md).
- **An over-cap CSV is refused before pandas parses it (M2).** Row counting is a streaming pass; the header row is decided from that pass; pandas reads the body once.
- **Every size and shape refusal has a code (M4).** `payload_too_large`, `too_many_rows`, `too_many_columns`, `too_many_cells` (docs/UPLOADS.md).
- **One error envelope (M3).** Every refusal is `{"error", "code"}` with the HTTP status unchanged. A 422 keeps `detail` for its field list. Unhandled errors answer a generic `internal_error`. Clients that read only a 400 as a bad upload must gate on the status (docs/HARDENING.md, "Error envelope").
- **Audit retention (L8).** `KALOS_AUDIT_RETENTION_DAYS` (default 365, 1..36500) prunes `audit_log` at startup and daily.
- **The API docs and schema are served only in `KALOS_ENV=development` (L6).** This was already true; it is now explicit and documented.
### Docs - review: known limitations (M5, M6)

- docs/HARDENING.md, "Known limitations", records two judgement calls with evidence and
  no code change: M5 (`class_weight="balanced"` understates P(feasible) on a
  mostly-feasible sheet) and M6 (the promotion gate reads the pooled Spearman, not
  `producer_spearman`; a sheet with producer Spearman 0.02 still passes). Also: the
  concurrency serialization (C3), the concentration naming gap (C1), and the
  replicate-aware fit still weighting exact copies (M1), and the M1 decision on identical
  floor (`0.0`) readings, which the scale diagnostic test now records as a strict xfail.

### Fixed - review: anonymize hides person names and their level values (H4)

- `is_identifier_name` treats `operator`, `analyst`, `technician`, `tech` and `email`
  as whole-word person tokens anywhere in a name, and `Name` after a subject
  (`Donor_Name_1`, `Lab_Tech`, `Operator`, `email` now count). Process variables
  (`Methanol`, `pH`, `sample_volume_L`) are unchanged.
- With `anonymize=true`, identifier-column categorical level values are pseudonymized
  (`level_<hash>`) in the proposal recipes and the response-surface fixed recipe.
  Identifier names are also aliased in `features`, `target`, `drivers`,
  `correlation`, `gp_shapes`, `response_surface` and the recipe keys. Before, a
  recipe carried "Jane Smith 0" verbatim. Scope: docs/HARDENING.md, "What
  `anonymize=true` covers".

### Fixed - review: a non-finite constraint floor is a ValueError (L1)

- `constraint` with a NaN or infinite `floor` (or a non-string/empty `column`)
  raises `ValueError` naming the field, before any fit. Before, botorch raised a bare
  `RuntimeError` from inside the acquisition.

### Fixed - review: exact copies are not replicates (M1)

- A row identical to another in every column (same factors AND same target) is an
  exact copy, not a replicate. Copies are removed before the replicate noise and ICC
  statistics. Before, a pasted-twice sheet gave `noise_sd: 0` and `icc: 1.0`.
  `noise.n_exact_duplicates` reports how many were removed. True replicates (same
  factors, different target) are unchanged.

### Fixed - review: proposals do not depend on concurrent analyses (C3)

- `_analyze` seeded the process-wide torch and numpy RNGs at its start. Two analyses
  on two threads (busy.py admits two) reseeded each other mid-fit, so proposals
  depended on thread timing. Every analysis now runs in `_seeded_analysis`, which
  holds a lock and seeds torch and numpy inside a fork, then restores the caller's
  numpy state. Concurrent analyses run one after the other on the RNG, with the same
  output as sequential runs. Trade-off: the seeded section of two concurrent
  analyses is serialized (docs/HARDENING.md, "Known limitations").
- `propose_batch` is called with `seed=ANALYZE_SEED`, so the run record's `seed`
  is the seed the acquisition used.

### Fixed - review: an in-flight recipe with a missing component is dropped (H2)

- `_encode_pending` zero-filled a missing continuous component. Such a recipe now
  has no coordinate and is dropped, as a missing categorical level already was.
  The count is reported as `n_pending_dropped`. Sparse / concentration components
  still read a blank as 0.

### Fixed - review: a blank cell is not 0.0 (C1, critical)

- A blank feature cell was zero-filled for every column, and 0.0 then widened the
  design box, so proposals landed at pH 0.0 (24 runs, pH 6.0-7.5, three blanks gave
  proposals down to 0.0). Now only columns declared in `roles.sparse` and
  concentration-type columns (by name) read a blank as absent (0). Any other blank
  drops its run through the row accounting (`cells_by_column: missing`).
- The design box is built from observed cells only. An absent-zero column's box
  includes 0 only when blanks were read as 0.
- `ColumnRoles` gains an optional `sparse` list.

### Fixed - review: factors are not identifiers (H1)

- `medium`, `strain`, `recipe`, `batch`, `campaign`, `group` and `lot` are no longer
  identifiers (`id_hint`). They are candidate categorical factors, reported with an
  `auto_categorical` finding. Compound IDs (`batch_id`, `lot_number`) and bare
  `batch`/`lot`/`campaign` in `_RUN_ID_RE` are still identifiers only in ID form.
- A group-named column is the CV group only when every level is confounded with a
  single recipe, or when declared in `roles.groups`. Before, `Medium in {A,B,C}` with a
  real effect was dropped as an id and became the leakage group.
- Behaviour change: sheets with such columns get new categorical features and
  therefore new proposals (mixed optimizer instead of continuous).

### Changed - no silent data loss in analysis (`feat/data-handling`, docs/UPLOADS.md)

- **Text factors are modeled (D5).** With no declared `roles`, a text column with 2-12 repeated values is a categorical feature (before: `dropped_sparse`, never varied by a proposal). Each auto-detected column gets an `auto_categorical` finding; declaring roles turns detection off. Result: `auto_categorical`.
- **Every drop is counted (D6).** Runs excluded for a missing or non-numeric target/feature cell were silent (`n_dropped_incomplete: 0`). They now carry per-column, per-reason counts (`data_drops`) and findings, and the run record echoes the counts (`run.data_drops`). Null tokens (`n/a`, `NA`, `nan`) are kept as text on read, so they are reported as missing rather than folded into blanks. Censored values (`<LOD`, `>100`, `<x`, `>x`) exclude their run from the fit, are counted separately (`n_censored_excluded`), and are never imputed. A blank feature cell still means absent (0) and is reported.
- **Merged headers (D7).** A group-label row above the column names is skipped and the names are read from the next row (`header_row: 1`). A sheet whose names cannot be identified is refused with `ambiguous_header` and the detected columns. A single unnamed data column is named by position with a warning, so pandas' `Unnamed: N` never reaches a response.
- **`sample.*` no longer swallows measurements.** `sample_volume_L` and `sample_mass_g` are features. `sample_id`, `Sample Code` and `Tank ID` are still identifiers. The test that recorded the old over-match was replaced.
- Run record: `data_drops` and `header_row` are added (`run.schema_version` stays 1: the record is unreleased).

### Added - run record and engine lock (`feat/run-record`)

- Fix (same unreleased branch): `run.n_features` is the fitted feature count
  (`d`), not the number of top drivers in `proposal_features` (at most 4).
- Identifier detection is no longer start-anchored. `Tank ID`, `Patient ID`,
  `Customer Name` (and `TankID`, `Sample ID 2`) were not recognised as
  identifiers: they were modeled as features and, under `anonymize=true`,
  published by name. A word-level rule (`is_identifier_name`) now catches them;
  real measurements (`Methanol`, `batch_titer`, `run_duration_days`) are still
  not identifiers. Note: this changes the feature set for sheets with such
  columns. (docs/HARDENING.md, "What `anonymize=true` covers")
- Documented, not changed: the campaign stores the full raw sheet in
  `campaigns.base_rows`, and `anonymize` does not cover it. (docs/HARDENING.md)
- With `anonymize=true`, workbook sheet names are aliased (`sheet_<hash>`) in
  responses, refusals (`sheet_not_found`, `no_usable_sheet`) and analysis-error
  sentences. Before, `sheet_not_found` listed the uploader's real sheet names.
- Campaign re-analysis (`POST /api/campaign/reanalyze`) attaches a `run` block
  with `kind: "reanalyze"`, the folded dataset's hash as input, and
  `pending_runs`. It also keeps the anonymize setting of the analysis it
  replaces: before, a re-analysis of an anonymized upload published the real
  identifier column names again.
- Every analysis result carries a versioned `run` block (`run.schema_version`
  1): input SHA-256 and size, sheet, target, profile, declared roles, anonymize,
  seed, torch threads, `build_sha`, library versions, validation mode, row and
  feature counts, and the keys of the gate decisions it refers to. Sync routes
  and background jobs attach the same block for the same bytes. Stored inside
  the existing analyses blob; no migration. Rows written earlier have no `run`
  key (absent, not null). `seed` and `engine_version` are unchanged.
  (docs/CAMPAIGNS_API.md)
- The record also states the fit-row cap (`max_fit_rows`), the thread count
  torch actually ran with (`torch_threads_actual`; `torch_threads` is `null`
  when unpinned, not `0`), and the analysed sheet's index and salted hash
  (`sheet_index`, `sheet_hash`).
- `KALOS_BUILD_SHA` (set by CI from `github.sha` via the image's build arg) is
  recorded as `run.build_sha`; `unknown` when not provided.
- `python -m kalos.portal` takes `<KALOS_STATE_DIR>/engine.lock` (flock) and
  refuses to start when another engine holds it (exit 2). A flock failure that
  is not contention (a filesystem without flock) is reported as such, not as
  "another engine". `uvicorn --workers N` bypasses the lock: use
  `python -m kalos.portal`. (docs/HARDENING.md, "One engine per state directory")

### Changed - anonymized uploads store no filename

- With `anonymize=true` the stored analysis carries no filename: the `run`
  block omits `filename` and `sheet`, and the stored `dataset` label is
  `anonymized upload <hash prefix>`. Before, the raw filename was stored as the
  campaign's dataset label even when anonymize was requested.
- Background jobs (`jobs.params`): under `anonymize=true` the job row stores
  the file extension, not the filename, and every terminal state clears the
  stored parameters (target, roles, sheet, filename). While a job is queued or
  running, its row still holds target, roles and sheet names in the clear, and
  SQLite may keep cleared values in free pages until vacuum. (docs/JOBS.md,
  "Job parameters")

### Added - background analysis jobs (`feat/background-jobs`, docs/JOBS.md)

- `POST /api/campaigns/{id}/jobs` queues an upload analysis and returns `202`;
  `GET /api/jobs/{id}` reports its state, stage, progress and error;
  `POST /api/jobs/{id}/cancel` cancels it; `GET /api/campaigns/{id}/jobs` lists
  a campaign's jobs. The bad-sheet refusals of `/api/run` are returned before
  queueing.
- Each job runs in its own worker process and publishes the campaign and its
  analysis in one transaction, as a compare-and-swap on the generation it was
  submitted under. A cancelled or timed-out job publishes nothing; a campaign
  that changed while the job waited fails the job instead of being overwritten.
- Per-tenant limits: concurrent analyses (`KALOS_JOB_PER_TENANT`), queue depth
  (`KALOS_JOB_QUEUE_DEPTH`, 429), upload size (`KALOS_JOB_MAX_UPLOAD_BYTES`,
  413), and a wall-clock limit per job (`KALOS_JOB_TIMEOUT_S`). Engine-wide
  concurrency is `KALOS_JOB_WORKERS`; sync routes share the same limiter and
  yield to queued work of other tenants.
- Restart recovery: jobs a previous process left running are failed
  (`interrupted`), and their workers are killed if still alive.
- Schema v3 (`jobs` table), additive, with the usual pre-migration backup
  (docs/MIGRATIONS.md).

### Changed

- The sync upload, reanalyze and demo routes now share one analysis path and
  one limiter (`kalos/portal/pipeline.py`, `kalos/portal/busy.py`). The single
  global analysis slot is now a per-tenant cap plus a global cap, so one tenant
  no longer answers every other tenant's upload with 503.
- `/api/single` and `/api/multi` (development only) run under the same limiter
  and a wall-clock timeout (`KALOS_DEMO_TIMEOUT_S`).

### Fixed - every upload failure says what is wrong; workbooks can pick a sheet

An empty file, a corrupt `.xlsx`, a `.txt` of prose, a 4-row sheet, a
one-column sheet, and a workbook whose data is on sheet 2 all used to return
"Could not parse the uploaded file". Each now returns a specific, engine-
authored message and a `code` (`empty_file`, `unsupported_file_type`,
`corrupt_excel`, `not_a_table`, `too_few_rows` with `min_rows`/`n_rows`,
`too_few_columns`, `no_numeric_inputs`, `no_process_inputs`,
`target_not_found`, `sheet_not_found`, `no_usable_sheet`, ...); no exception
text ever reaches the client. A requested `target` that is not a column is now
an error instead of silently analyzing a different column. `POST /api/run`
takes an optional `sheet` (form field or query); without it the first sheet
holding a usable table is analyzed, and Excel responses carry `sheets` and
`sheet_used`. UTF-16 "Unicode Text" exports and UTF-8 BOMs are now read.
See `docs/UPLOADS.md`.

### Fixed - a proposed batch never repeats a recipe

On a sheet whose optimum sits on the edge of the design box, the joint q=5
qLogNEI optimization could collapse two batch points onto the same corner and
the API returned two identical recipes. `kalos.core.optimize.propose_batch`
(which `propose` now wraps) keeps the jointly optimized batch when it is
already distinct, and otherwise refills each collapsed slot by sequential,
pending-aware selection: the accepted rows become `X_pending`, and the best
candidate from the optimizer's multi-start optima plus a Sobol pool over the
box (or the full enumeration of a small categorical space) that is at least
`NEAR_DUPLICATE_TOL` = 0.05 away is taken. No random jitter, every candidate
stays in the box and is scored by the same (constrained / gated) acquisition.
Distance is Chebyshev in the range-normalised space, with categorical levels
compared exactly (`proposal_separation`). If the space has fewer than q
distinct candidates the batch comes back short. The analysis response gains
`proposal_notice` (null, or `{code: "fewer_distinct_candidates", requested,
returned, min_separation, message}`) and `batch_diversity`.

### Fixed - a target that does not vary is refused, not optimized

A run sheet whose target had the same value on every row (or the same value
plus 1e-12 float noise) was accepted with a validation warning and came back
with five proposed recipes ranked by rounding error. `_analyze` now checks the
target with `kalos.validation.target.assess_target_variation` before any
surrogate is fit, and refuses with a 400 `{"error", "code": "target_constant",
"target", "target_summary", "available_targets"}` (also `target_empty` and
`target_not_numeric`). No model is fit and `propose` is never called, on the
upload, the re-upload-with-another-target, and the campaign re-analyze paths.
"Effectively constant" is `range(y) <= 1e-9 * max|y|`, justified in
`kalos/validation/target.py`. `UploadRejected` now carries an optional `code`
and `context`, returned alongside `error`.

## 2026-10-12 (Apache-2.0 and a public README)

### Added - the license

`LICENSE` is the Apache License 2.0, copyright 2026 Brendan Dang, and
`pyproject.toml` declares `license = { text = "Apache-2.0" }` (it said
"Proprietary").

### Changed - the README is rewritten for the API-only engine

The README now leads with what the engine is, who it is for, and the evidence
behind it: the CI gates and the synthetic benchmark table from `BENCHMARK.md`
(with its reproduction command), and one line that a real-data win is not yet
claimed. It adds a route table for every engine route and a per-guarantee list
that names the test or doc backing each one. The long upload-guard prose moved
to `docs/UPLOADS.md` unchanged in content. The ESM-2 feature is no longer in
the pitch. Earlier entries in this file had their former project and client
names generalized and their long dashes replaced with plain hyphens; no code
changed.

## 2026-10-10 (kalos-engine: the engine is API-only)

The repo is now `itsbrendandang/kalos-engine` (GitHub redirects the old
URL; the Python package is still `kalos`), and all UI lives in `kalos-web`.

### Removed - the built-in viewer

`GET /` (`kalos/portal/index.html`), `GET /fonts/Satoshi-Variable.woff2`, the
vendored font, and DESIGN.md (which only described that page). Nothing is
lost: kalos-web's `/analyze` covers "run your own data" (`POST /api/run`) and
`/simulate` covers both demos (`/api/single`, `/api/multi`). In the deployed
stack the viewer was never reachable anyway - only kalos-web's port is
published. The Scale-Up Readout template stays: it is what
`POST /api/scale/readout` returns, not a UI.

## 2026-10-10 (one request can no longer stall the engine)

### Fixed - the demo routes did unbounded work for anyone

`GET /api/single` and `/api/multi` took `rounds` and `q` straight from the URL
with no limit and no token, outside the one-analysis slot - and kalos-web's
proxy forwards them with the query string. `?rounds=100000&q=64` queued hours
of GP fits; `rounds=0` crashed with a 500. Now `rounds` is 1-12 and `q` is 1-4
(422 otherwise, defaults unchanged), and both need the `read` scope like every
other data route (open mode still serves them on loopback).

### Fixed - uploads were read whole before the size check

`/api/run` and `/api/scale/readout` read the entire body into memory, then
rejected anything over the cap (`KALOS_MAX_UPLOAD_MB`, 25 MB by default). A multi-GB upload could
exhaust the single engine process for every tenant. They now read at most one
byte past the cap, which is all the check needs.

## 2026-09-24 (proper container files)

The deploy pack is now organized the way container tooling expects, and it
builds the same under Docker and Podman.

### Changed - one directory per image

`deploy/engine/Containerfile` and `deploy/backup/Containerfile` replace
`Dockerfile.engine` and `backup/Dockerfile`; the web image's Containerfile
moved into the kalos-web repo itself (it builds that repo), so the kalos
checkout no longer has to be named `kalos`. `deploy/compose.yaml` replaces
`docker-compose.yaml`. Image contents are unchanged. Update any local
scripts: `docker compose -f deploy/compose.yaml ...`.

### Fixed - the engine build no longer ships the whole checkout to the builder

There was no ignore file, so every engine build sent `.venv`, `.git`,
agent worktrees and `deploy/backups/` (real database backups) as build
context. A repo-root `.containerignore` allowlist (`.dockerignore` links to
it) now sends only `pyproject.toml`, `README.md` and `kalos/`: about 0.8 MB,
verified under both Docker and Podman.

### Docs

`deploy/RUNBOOK.md` gains a "Container files" section and Podman
instructions (build with `--format docker`, or Podman silently drops the
`HEALTHCHECK` the stack depends on). The web Containerfile's header now
describes the same-origin engine proxy kalos-web actually uses.

## 2026-09-23 (the Scale-Up Readout)

You can now hand a prospect one page that answers "what will this process do at the next scale,
and should I believe the number?" - and that says no when the data cannot support an answer.

### Added - `POST /api/scale/readout`

Upload a multi-scale run sheet plus a target scale and get a self-contained, print-ready HTML
readout (page 1 is the decision; a provenance appendix follows). Kalos first backtests itself on
the sheet's own scale ladder against two naive baselines, then issues a prediction with an
approximate-coverage interval, a prediction with named warnings, or a refusal with every reason.
The license to extrapolate comes only from backtest steps the model actually won. The route is
stateless: nothing from the upload is written to disk or the database. See `docs/SCALE_READOUT.md`.
The gate no longer requires a minimum run count per scale, so a sheet with many bench runs and only
one or two runs at each large scale is read honestly instead of rejected outright. When the evidence
is too thin (too few pooled residuals, or no rung with enough runs to license a reference step
ratio), the readout still returns a 200 page showing everything it could compute, plus a "What it
would take" section stating in plain terms how many more runs, and at what scale, would change that.
`agitation_rpm`/`airflow_L_per_min` are now optional on the sheet and target: when a sheet does not
record one or both, the readout falls back to a scale-only feature set instead of a 422, and the
page names exactly what is not modeled as a result.

The extrapolation license comes only from backtest steps near the scale being extrapolated from (within one decade of the largest trained scale), so a step won at bench scale cannot license a plant-scale step.

### Added - a synthetic multi-scale demo sheet

`examples/synthetic_scaleup/`: 9 scales from 1 to 5000 L, 10 runs each, seeded and pinned by a test
so the committed CSV cannot drift. On it, a 1.5x step (7500 L) issues a prediction whose backtest
MAE is 0.263 against 0.699 and 0.729 for the naive baselines.

### Changed - the scale-model evidence is now reconciled

The scale evaluation's docstring said v0 did not clearly beat the naive baselines on upward
extrapolation; rerunning it on the synthetic mab-scaleup dataset shows it does on MAE (0.533 vs
1.510 and 1.382, n=5), while its ranking of recipes at the larger scale stays unmeasured at that n.
Docstring and `docs/SCALE_EVIDENCE.md` now say the same thing, with the command to reproduce it.

### For contributors

`leave_one_scale_out_report` gains `include_oof` (per-row out-of-fold arrays) and `held_out_scales`
(evaluate only the named folds, skipping the pooled LOGO pass). Both default off, and the default
output is pinned byte-identical by a regression test. `deploy/backups/` is now gitignored.

## 2026-09-16 (production test cycle finds a pandas 3.x regression)

The first full production test cycle against the merged deploy pack - a cold
`docker compose` rebuild, then a real upload/campaign-start/log-result/
reanalyze loop through the one published front door - found a fourth
container-only-in-spirit defect: real, but invisible to the 791+ tests that
existed before it because none of them exercised a dataset with a declared
group column through the campaign loop.

### Fixed - reanalyze crashed on any dataset with a declared group column

`kalos/portal/analysis.py`'s LOGO grouping built `groups` from
`df.loc[keep_index, gcol].astype(str)`. A folded campaign run has no value
for a declared group column (e.g. `medium_base`) - it is not a proposable
numeric feature - so that column always carries a missing value after a
fold. Every other `.astype(str)` cast in that file already guards with
`.fillna("")` for exactly this reason; this one call site did not, and
pandas 3.x's default string-backed dtype stopped silently stringifying NaN
to "nan" the way older pandas did - the gap used to be masked, now it raises
`TypeError: '<' not supported between instances of 'float' and 'str'` inside
`np.unique`, which the reanalyze route's catch-all turned into a generic
400. Fixed with the same `.fillna("")` convention already used nearby, plus
a regression test that seeds a dataset with a declared group column and
reanalyzes after logging a result.

## 2026-09-10 (the deploy pack meets reality)

The wave-2 deploy pack was written and config-tested but its images were never
built (disk-deferred). Building and actually running them surfaced three real
production defects - each invisible to every unit and CI gate, each now fixed
with a regression test.

### Fixed - set-but-empty env crashed the boot

Compose/k8s templates pass `KALOS_MAX_UPLOAD_MB=${KALOS_MAX_UPLOAD_MB:-}`,
which arrives SET BUT EMPTY - `os.environ.get`'s default never applies and
`float("")` killed the engine at import, on the very first container start.
The sizing knobs (upload MB, fit rows, torch threads) now treat empty as
unset, the same convention kalos/providers/ always followed.

### Fixed - one analysis at a time, honestly (kalos/portal/busy.py)

`_analyze` is CPU-bound for tens of seconds and `run_in_threadpool` work
cannot be cancelled: a client that gives up (a proxy timeout, a closed tab)
orphans a thread that keeps computing. With no admission control, every retry
contended with the ghosts of its predecessors - measured as 598% engine CPU
with ZERO connected clients. A single analysis slot now guards /api/run and
campaign reanalyze: the second request gets HTTP 503 + Retry-After (answered
in 0.1s in the live test), and the slot is released by the WORKER THREAD when
the fit truly finishes, so an orphan keeps holding it and the 503 stays a
true statement about the machine.

### Fixed - the torch thread pin is pathological in the container

The bare-metal default (pin intra-op threads to 4) took the demo analyze from
47s to over 600s on the image's linux-aarch64 torch/OpenBLAS build - a >12x
slowdown from the pin itself, isolated by timing the identical computation
pinned vs unpinned in the same container. `KALOS_TORCH_THREADS=0` now means
"do not pin" and the compose passes it by default; bare metal keeps the tuned
4. (The old code floored 0 to a 1-thread pin - the worst possible reading.)

### Changed - Dockerfile layering, and the verified numbers

Torch (the 700-900 MB layer) now installs before any source COPY, so a
one-line engine change rebuilds in ~40s instead of re-downloading torch -
found the first time the image needed a real fix. Verified footprint: engine
1.82 GB (inside its own honest 1.3-1.8 GB estimate), web 314 MB, first
authed proxied analyze ~19s warm. kalos-web's proxy timeout became
env-tunable (`KALOS_ENGINE_TIMEOUT_MS`, default 120s) after the 30s guess
502'd a cold container fit mid-computation.

## 2026-08-25 (wave 2: acted-on diagnostics, one front door, and a vindication)

Five agents plus orchestrator integration; every number below is from a
reported run or a locked regression test.

### The finding of the wave - ScaleBridge v0 was vindicated, not fixed

Wave 1 reported v0 ranking upward-extrapolated runs backwards (Spearman -0.7)
and framed it as "regressing toward a scale-adjusted mean". The mandatory
rank-crossing check told a different story: the real dataset shows NO
detectable recipe-x-scale interaction (F(3,47)=0.917, p=0.44), the exact
permutation p-value for rho=-0.7 at n=5 is 0.233 (noise), and on a harder
synthetic with a PLANTED rank crossing, all three candidates - including
unmodified v0 - recover it (Spearman ~0.88, and v0 with the best MAE). The
precise conclusion: v0's extrapolation ranking is UNMEASURED at this power,
not demonstrated-bad. No candidate promoted; `physics_mean` (physics-informed
mean function) and `multi_fidelity` (scale as a fidelity dimension,
SingleTaskMultiFidelityGP verified against installed botorch) ship as
documented opt-in alternates on ScaleUpTransferModel, evidence attached.

### Added - the heteroscedasticity diagnosis is finally acted on

`alternative_scale`: when (and only when) the noise report's
`suggests_transform` fires, a second labeled evaluation runs on
log(y + offset) - the exact transform icc_log already uses, NOT literal
log1p, which the report's own history records as wrong at titer scale - and
reports log-scale CV, calibration, and ICC beside the raw numbers.
Proposals and every other number stay on the raw scale; the block is proven
purely additive by a deep-equality test, and it is computed LAST so its GP
fit cannot advance the shared seeded RNG before the proposal path consumes
it (running it earlier would have silently changed proposed batches whenever
the diagnostic fired). Measured cost ~0.12s, only when triggered.

### Fixed - state-path split-brain and the missing health surface

SqliteStore and the runner lock now honor KALOS_STATE_DIR (call-time reads,
matching campaign.py's convention; unset env is byte-identical to before).
/healthz returns exactly {"status": "ok"} unauthenticated - verified the
version string is NOT otherwise public, so it is not leaked here - and
/readyz does a read-only store probe, 503-with-reason on failure. Deploy
healthchecks now hit /healthz; the stale "engine ignores KALOS_STATE_DIR"
comments are gone.

### Changed - one front door

With kalos-web's server-side proxy (kalos-web #26) attaching the bearer
token, the compose no longer publishes the engine's port at all: the browser
calls same-origin /api/engine, the web SERVER forwards over the internal
network. KALOS_ENGINE_URL/KALOS_ENGINE_TOKEN are runtime env on the web
service (never in the client bundle); the runbook's known-gap bullet is
replaced with setup instructions. Direct-browser mode survives as an
explicit dev escape hatch.

### Evidence recorded - MAP-SAAS: a clean negative result

The trial the research memo motivated ran in full
(kalos/bench/saas_experiment.py, 8 (n, d) cells x 5 seeds, both models fed
the same known noise): SAAS never beats the plain SingleTaskGP outside its
own noise floor, costs 1.9-19x the fit time, and proposes worse in 6 of 8
cells - including at kalos's actual low-d operating point. NOT wired;
surrogate.py untouched by construction. Banked for any future revisit:
gp_shape's lengthscale reader would silently read 1 of SAAS's 4 additive
kernels (wrong relevance ranking), and SAAS's constructor draws from the
global RNG outside propose()'s seed fork.

## 2026-08-25 (wave 1: the engine practices what its benchmark proved)

Six parallel agents; every measured claim below is from a locked regression
test or a reported run.

### Added - production proposals are finally feasibility-gated

BENCHMARK.md proved plain BO loses to random search on zero-inflated titers
and that feasibility-gated EI fixes it - and the fix lived only in the bench
harness while production proposed ungated. Now the acquisition itself is
gated: log(EI x p_feasible) = logEI + log p_feasible, with the fitted
StandardScaler+LogisticRegression rebuilt exactly as a differentiable torch
module (gradients verified end to end), q-batch feasibility composed as a
product (conservative under zero-inflation, documented as an approximation).

Gate POLICY, all three conditions reported and none silent: the classifier
actually fit (not its cold-start fallback), n_infeasible >= 5, and CV
feasibility AUC >= the SAME 0.65 floor the promotion gate already uses - one
number, one meaning. Any condition failing reproduces the ungated path
byte-identically (asserted at the propose() call level, not assumed).
Measured: on a feature-dependent zero-inflated sheet, gated batches score
mean p_feasible 0.953 vs 0.389 ungated on the identical classifier; through
the production propose() path on an 85%-infeasible pool, gated best-found
9.084 vs ungated 8.884 over 8 seeds. The response reports `proposal_gating`
and every proposal row carries `p_feasible`.

### Added - the decided constrained behavior exists: maximize titer s.t. a floor

`_analyze(..., constraint={"column": "purity_pct", "floor": 95.0})` fits a
second GP on the constraint outcome and threads it through BoTorch's native
constrained qLogNEI (ModelListGP + objective index + less-than-zero callable;
verified against the installed 0.18.1 source, including that prune_baseline
respects the constraints). Composes with the feasibility gate. Falls back to
unconstrained - reported, never crashing - on a missing/sparse column or fit
failure. The constraint column is stripped from features unconditionally,
including against a declared-roles bypass; both leakage paths are tested.
Measured: on an anti-correlated titer/purity design, constrained batches
predict purity 94.5 vs 89.6 unconstrained.

### Added - ScaleBridge v0 (kalos/scale/): physics-informed scale transfer

Geometry/power/gas-velocity/kLa/hydrostatic proxies with every constant
exposed as a fittable parameter and cited (van't Riet 1979, Rushton 1950,
Doran); transfer as feature engineering over the proven Surrogate;
leave-one-scale-out evaluation with extrapolation direction reported
separately. Measured on the 55-run synthetic scale-up set: pooled MAE 0.66 vs
naive 1.77/1.42, Spearman 0.68 vs ~0. Stated plainly: on upward extrapolation
to 2000 L the model wins on MAE (0.53 vs 1.4-1.5) but has NOT yet
demonstrated correct ranking (Spearman -0.7 on an n=5 coarse grid) - v0
regresses toward a scale-adjusted mean; the v1 path (physics-informed mean
function, more scales) is documented in the module.

### Added - ingestion grows two tiers

- Orientation pre-pass (kalos/normalize/orientation.py): transposed run
  sheets (rows = parameters) are detected and normalized before anything else
  sees them, traced via frame metadata and surfaced as `orientation` in the
  response. Hardened beyond the ported heuristic: a first column that is
  itself numeric can never be read as parameter labels, and shape signals
  alone never suffice - the port's raw heuristic false-positived on kalos's
  own standard tall/narrow fixtures, caught by the existing hardening tests.
- Open-source LLM provider seam (KALOS_LLM_PROVIDER: anthropic | ollama |
  none): the Ollama path speaks the real API (health via /api/tags,
  generate with format json, temperature 0), validates against the identical
  schema the Anthropic path uses, and falls back deterministically on any
  failure - the LLM proposes, deterministic code validates, unchanged.
  Stdlib HTTP only; "none" never touches the network. Units gained rpm, bar,
  g/kg, L/min + agitation/pressure synonyms; bare L/mL stays deliberately
  excluded (vessel-size labels, not measurements - the existing test that
  guards this exclusion caught the naive addition).

### Added - deploy/ - the stack becomes deployable

Multi-stage non-root images (CPU-only torch), compose with health checks and
env-driven secrets satisfying the bind-safety guard by construction, and a
SQLite backup sidecar using the Online Backup API (a cp of a live rollback-
journal database can silently lose the in-flight transaction), retention
windowed, backups outside the compose volumes. RUNBOOK.md carries deploy /
upgrade / rollback / restore-drill / security checklist. Found and
documented: SqliteStore and the runner lock hardcode ~/.kalos and ignore
KALOS_STATE_DIR (compose works around it; engine fix queued), and kalos-web
sends no Authorization header yet (both options documented in the runbook).

### Fixed / smaller

- FeasibilityClassifier gained a public `fitted` accessor and
  `torch_gate_params()` so callers never reach into sklearn internals.
- The research memo (scratchpad, feeding wave 2) verified kalos is clean of
  the upstream-DELETED HeteroskedasticSingleTaskGP and already on the
  recommended train_Yvar pattern.

## 2026-08-23 (the review's remainder, built by a Sonnet fleet)

Five agents implemented the rest of the 2026-08-22 engine review plus portable
findings mined from the owner's earlier engine. Each
change below carries its own tests; the full suite, ruff, and mypy are green on
the combined result.

### Added - the CV confidence interval now covers partition variance

`make_splits` takes the unshuffled GroupKFold branch for a continuous target, so
the group-bootstrap CI on the headline `cv_spearman` was conditional on ONE fixed
partition - and the splits.py docstring admitted the damage (0.44 to 0.71 across
n_splits 3 to 8) while nothing computed the sweep it recommended.

`grouped_cv_report` now supports repeated grouped CV (`n_repeats`): repeat 0 is
the exact historical unshuffled partition, later repeats shuffle the group
assignment deterministically, the bootstrap runs per repeat, and `ci95` pools all
draws. Every point estimate and OOF-derived quantity (`cv_spearman`,
`conformal_q`, calibration, the `oof` scatter) stays anchored to repeat 0, so
nothing moved except the interval - the pinned-number tests pass unmodified. The
portal runs `CV_N_REPEATS = 2` (3 measured over the runtime budget) and reports
`cv_n_repeats` + `cv_spearman_per_repeat` so the spread is visible. On an
engineered partition-sensitive sheet the CI width grows 0.769 -> 0.833; leave-
one-group-out sheets collapse to 1 effective repeat because only one partition
exists.

### Added - the promotion gate finally has real numbers to read

`check_gates` had zero callers on the analysis path and `feasibility_cv_auc`
zero callers outside tests, so the fail-closed gate guarded nothing.
`feasibility_cv_report` now returns AUC, Brier, and the classifier's 10-bin ECE
from the same pooled-OOF CV (`feasibility_cv_auc` is a thin wrapper over it), and
`_analyze` assembles all four gate stats and reports `check_gates` as a new
top-level `promotion` verdict. Reported, never enforced: the verdict cannot
reject an upload, because changing what the API accepts is a product decision.
On an all-producer sheet the feasibility metrics are unmeasurable and the verdict
fails closed with "unmeasured" failures - correct semantics, and the block
carries a `meaning` string so that reads as "not yet shown fit to promote", not
as a rejection.

Named distinction, because two metrics share a name: the gate's `ece` is the
FEASIBILITY CLASSIFIER's calibration (is P(feasible) honest); the
`reliability.calibration` ece is the regression interval calibration (are the
titer error bars honest). Different questions, same scale, both reported.

### Added - three evaluation-hygiene anchors from an earlier engine

All three earned their place on the owner's real study data before porting:

- **`cv_logo`** (leave-one-group-out): on the real clone funnel, LOSO Spearman
  was -0.12 where shuffled 5-fold read 0.91 - the gap IS the cross-campaign
  generalization claim. Computed when a declared group column exists and the
  group count is within `LOGO_MAX_GROUPS = 12` (measured cost ~0.5s); skipped
  with a stated reason otherwise, never silently.
- **`cv_topk`**: top-5 overlap between true and predicted ranking on the pooled
  OOF - the direct form of "which N do I advance", which Spearman can obscure on
  zero-inflated targets. Ties resolve by stable sort, documented.
- **`cv_group_mean_baseline`**: predict each row by the mean of the OTHER rows in
  its recipe group and report that Spearman. On the owner's real data a
  campaign-mean-only anchor captured ~0.75 of a model's ~0.86 headline; a model
  that does not clearly beat this floor may be recognizing recipes, not modeling
  the process. The interpretation ships in the payload.

### Added - two validation checks the real data demanded

- **Informative missingness (MNAR).** On the real plate-reader funnel, per-row
  missingness vs titer ran Spearman -0.835: "no measurement" was the culling
  decision, not benign absence - and kalos zero-fills blanks before the GP fit.
  `check_informative_missingness` correlates row-level missing fraction (and
  per-sparse-column missingness indicators) with the target and warns when |rho|
  clears 0.35. Warning severity only; strict mode cannot start rejecting sheets
  over it. The engine's own `experiments/missingness_indicator/RESULTS.md` posed
  exactly this question and could not answer it; the check now answers it per
  upload. Running it against the real media DoE still needs a `KALOS_MEDIA_DATA`
  checkout, which this machine does not have.
- **Constant-within-group.** 4 of 6 clone features on the same data had exactly
  one unique value within every campaign - group identity in disguise, inflating
  within-population scores to ~0.86 when the campaign-mean anchor alone scored
  ~0.75. `check_constant_within_group` flags features that vary globally but are
  frozen within every group, which the global constant-column check cannot see.

### Added - a non-proprietary benchmark fixture

`examples/synthetic_bioprocess/`: 160 seeded rows, 10 continuous bioprocess
inputs in real units, a known optimum, a feasibility gate zeroing ~44% of runs.
Ported from an earlier engine with its generator (verified to reproduce the
CSV byte-for-byte). Fills the gap between `bench/`'s abstract math surfaces and
the uncommittable client data: a domain-realistic regression fixture that is safe
to commit.

### Fixed - four smaller defects

- `propose()` and `propose_multiobjective()` accept an optional `seed`. Applied
  inside `torch.random.fork_rng()`, so a seeded call is reproducible WITHOUT
  clobbering the caller's global RNG state. Caught during testing: qLogNEI with
  `prune_baseline=True` draws posterior samples at construction time, so the
  acquisition must be built inside the fork too - the first attempt leaked.
  Default `None` is byte-identical to the historical caller-seeds-globally
  contract.
- `_sanitize_pending`'s wholesale width-mismatch rejection is logged at warning
  level instead of silent. Without it, a caller bug that disables the in-flight
  guard is indistinguishable from "nothing running" - both read
  `n_pending_considered == 0`.
- `bootstrap_spearman` recorded a degenerate resample draw as rho = 0.0, pulling
  the bootstrap distribution toward zero for exactly the noisiest features. NaN
  draws are now excluded via nan-aware reductions; a fully degenerate feature
  falls back to the old all-zeros output so the JSON response stays finite.
- Each objective in `MultiObjectiveSurrogate` gets its own `Normalize` instance
  instead of sharing one stateful module across the `ModelListGP` - numerically
  identical today, and no longer one `learn_bounds=True` away from cross-coupling
  the objectives.

### Recorded - validated negatives from an earlier engine, so they are not re-attempted

- A scalar per-campaign offset multi-task GP is rank-frozen for held-out
  campaigns by construction; a real transfer test needs a coregionalization
  kernel, and probably more than 3 campaigns.
- `averageTotalTiter` is the row-mean of its per-stage components; any
  titer-derived feature "predicts" it mechanically. Check derived targets for
  circularity before trusting a high score.
- Hardcoded scale-up projectors and a DirichletCalibrator on a ~20-row tail were
  both rejected there for inflation reasons that still apply here.

## 2026-08-22 (the unit of analysis is the recipe)

### Fixed - the driver panel counted replicates as independent evidence

Every other statistic in the engine knows that three wells of one recipe are not
three independent runs. The CV groups by recipe, the noise floor is estimated
within recipe, the proposal is fit on the replicate-averaged objective. The
driver panel was the one place that still counted rows.

That is not a cosmetic inconsistency. Both tests behind `significant` - the
bootstrap CI and the Benjamini-Hochberg correction - ask how surprising an
association is given how much independent evidence stands behind it. Testing
rows on a replicated sheet computes every p-value against an `n` the sheet does
not have, so BH stops correcting anything, and BH is what the 2026-08 multiplicity
fix rests on.

Re-running that same simulation (30 pure-noise features, target independent of
all of them, 300 reports) at the replication depth a media DoE actually ships
with:

| sheet | reports containing a "significant" driver |
| --- | --- |
| 60 independent rows | **7.3%** |
| 20 recipes x 3 replicates | **87.0%** |
| 20 recipes x 3 replicates, averaged first | **8.0%** |

A cluster bootstrap alone does not fix it (82.5%): the row-level p-values are
what BH reads, so the aggregation has to happen before the test rather than
around it. The panel now runs on replicate-averaged rows keyed by the same
`recipe_key` the CV grouping and the noise floor already use, and
`driver_selection` states its unit (`unit`, `n_units`, `n_rows`) instead of
leaving a client to infer it from `n_tested`.

End to end, on a 60-row sheet whose target is independent of every column, the
engine reported three significant drivers at p=0.0006, p=0.0034 and p=0.0036. It
now reports none. On a sheet with no replicates every group is a singleton and
the aggregation is an exact no-op, so unreplicated uploads are unchanged.

### Fixed - `feasibility_cv_auc` raised on the data it exists to score

The guard capped `n_splits` by the minority-class count but not by the number of
groups, and those are different numbers: a replicated sheet can hold six
non-producing rows across only four recipes. sklearn refuses to make more folds
than there are groups, so the function raised `ValueError` against a docstring
that promised it never raises.

The value feeds the fail-closed promotion gate, and an exception is not a closed
gate - it is a crash that skips the verdict. Unmeasurable now returns `nan`,
which the gate blocks on.

### Fixed - the shape report was anchored to the luckiest single well

`gp_shape_report` sweeps each feature through the incumbent, the row with the
best measured target, and states that its `X`/`y` are the rows the surrogate was
fit on. On the replicate-aware path they were not: the surrogate is fit on
replicate-averaged rows and the raw sheet was passed, so every sweep was anchored
at `best_single` - a row that GP never saw, selected by taking a max over assay
noise - while the proposals in the same response were scored against
`best_reproducible`. Two incumbents, one response.

Stated plainly, because it was measured rather than assumed: this is a
consistency fix and not an accuracy win. Against a known interior optimum over 40
seeds, mean absolute error in the reported optimum moved 0.177 -> 0.188 at an
assay sd of 1.5 and 0.283 -> 0.270 at 3.5. The case for it is that the report now
describes the model that produced the proposals, and no longer selects its anchor
by maximizing noise.

### Added - calibration is measured instead of disclaimed

A model can rank held-out runs correctly and still quote every interval at half
its true width, and the scientist reading "predicted 4.2 +/- 0.3" is acting on the
0.3. `reliability.unmodeled` listed "calibration (ECE)" for a mechanical reason:
the grouped CV computed a held-out posterior sd on every fold and then discarded
it, keeping only the mean.

It is kept now, and `interval_calibration` scores it: `ece` is the mean gap
between nominal and empirical coverage across four central intervals, on the same
0-to-1 scale `GatesConfig.max_ece` is written against, and `z_std` says which way
a miscalibrated model errs. It is reported as `reliability.calibration` beside the
verdict, never folded into `clears_floor` - tightening the verdict changes which
uploads the API accepts, which is a product decision.

WHICH band is scored turned out to matter more than the metric. `Surrogate.
posterior` returns the LATENT band by default: uncertainty about the response
surface, which is what the acquisition reasons over. A held-out value is a
measurement and carries assay noise on top of that, so scoring observations
against the latent band under-covers by construction. On a replicated sheet with
an assay sd of 1.0 that read as z_std=3.20, ece=0.48 - a catastrophically
overconfident model that was nothing of the kind. Against the predictive band
(`Surrogate.posterior(..., observation_noise=True)`, new keyword, default
unchanged) the same fit measures z_std=1.32, ece=0.09: mildly overconfident,
which is true and useful. Assay noise is zero-mean, so no existing number moved -
`cv_spearman` and `conformal_q` are asserted unchanged.

### Fixed - the deferred-torch design had been quietly defeated

`kalos/portal/analysis.py` documents that `kalos.core.evaluation`, `kalos.core.
optimize` and torch are imported lazily, so the idle `--watch` poller and the
portal do not pay a ~220 MB import for an analysis they may never run. A single
top-level `from kalos.core.evaluation import producer_only_spearman`, added for
one call site inside `_analyze`, had broken it: `evaluation` imports `surrogate`,
so `import kalos.portal.analysis` cost 1.2s and pulled the whole stack. The
intent lived only in a comment; `tests/test_portal.py` now asserts it in a
subprocess.

## 2026-08-19 (in-flight runs are not proposed again)

### Fixed - the acquisition now knows what is already running

A campaign round is not atomic. Five recipes are proposed, the scientist starts them, some assays come back before the others, and the loop is re-analyzed on what has landed so far. The runs still incubating have no outcome, so they cannot join the fit - and the acquisition was never told about them separately, so it treated their region of the design space as unexplored and proposed them again.

That is budget spent twice for one point of information, and it is invisible in a regret curve, because regret only counts what was measured.

`kalos/bench/pending.py` measures it (`python -m kalos.bench --pending`). On a 4-factor design at `q=5` over 10 seeds:

| re-proposal | duplicated recipes per round (of 5) | worst seed |
| --- | --- | --- |
| blind | **2.10** | 4 of 5 |
| pending-aware | **0.00** | 0 of 5 |

Roughly 40% of a mid-round batch was repeat work. The surface is a smooth single optimum on purpose: that is the case where a blind re-proposal is *least* likely to collide, because the acquisition's own q-batch diversity already spreads one batch out.

**The plumbing.**

- `kalos/core/optimize.py`: `propose(..., pending=)` feeds BoTorch's `X_pending`, which integrates over the unknown outcomes of started runs. Pending points get the same discipline the returned proposals already get - clamped into the design box, categorical coordinates snapped to integer level codes - so an in-flight recipe recorded slightly outside the box still marks the right neighborhood as taken. A malformed row is dropped individually rather than raising: failing to propose anything at all is a worse outcome than proposing without the pending penalty.
- `kalos/portal/analysis.py`: `_analyze(..., pending=)` encodes in-flight recipes into the same design as the fitted rows (continuous components zero-filled like `Xc_zf`, categoricals through the fitted level codes). A recipe naming a level that does not exist in this design is dropped rather than placed at a guessed coordinate, and `n_pending_considered` reports how many actually reached the optimizer - what was used, not what was offered.
- `kalos/portal/campaign.py`: `awaiting_recipes(tenant)` returns the started-but-unmeasured runs. Read outside the plan/commit transaction on purpose: a run started in that gap is simply absent from the list, which costs the acquisition one in-flight point of information and nothing else.
- `kalos/portal/campaign_routes.py`: `reanalyze` passes them to `_analyze`.

**The honesty constraint holds.** An awaiting run informs the *acquisition*, as a point already taken. It never informs the *fit* - it has no measured outcome, and a run without an outcome must never become a data point.

**Not to be confused with the feasibility result.** `BENCHMARK.md` already tested gating acquisition by a feasibility classifier on the real media DoE and found it changed nothing (`bo_feas` was identical to `bo`). This is not a claim about finding better recipes. It is a claim about not paying twice for the same experiment, and it is the kind of waste that only appears once the loop is run in rounds rather than benchmarked in one shot.

## 2026-08-09 (GP-native response shapes)

### Added - `kalos/core/gp_shape.py`: interior optima and feature relevance from the GP that proposes

`kalos.core.drivers` reports a signed Spearman rho, which is univariate and monotonic, so it is structurally blind to the shape bioprocess responses usually have. A simulated titer peaking at pH 7.0 gives rho = +0.164 - below the trust floor - so the driver panel reports `significant: false` for the single most important variable on the sheet, and the sign of rho points the process the wrong way.

The same sheet now also returns:

```
GP SHAPES  swept_at=incumbent
  pH        rel=0.663 rho=+0.164 interior_optimum  opt=7.000  missed_by_spearman=True
  Methanol  rel=0.299 rho=+0.768 monotonic_up      opt=-      missed_by_spearman=False
  noise     rel=0.038 rho=+0.002 flat              opt=-      missed_by_spearman=False
```

The optimum is recovered at exactly 7.000, and both views ship side by side so a scientist sees the rank correlation and the shape together rather than having to trust one.

**Read off the surrogate the engine already fits, not a second model.** Three things follow, and each was a problem with the gradient-boosted-tree prototype this replaces (parked on `feat/xgboost-drivers`, which segfaults: xgboost and torch each carry an OpenMP runtime and the second into a parallel region crashes the process, uncatchably):

- **One authority.** The shapes describe the same posterior that produced `proposals`, so the report cannot rank features differently from the model being optimized.
- **Uncertainty is free.** `Surrogate.posterior` returns a mean AND a standard deviation, so an interior optimum is claimed only when the peak clears the better endpoint by at least `PEAK_SD_MULTIPLE` combined posterior sds. `peak_gain` and `peak_separation_sd` are both reported so a stricter bar can be applied without re-running. A point-predicting tree cannot make this check at all.
- **Relevance is free.** The kernel (an ARD RBF; this entry first said Matern, which was wrong) is fitted with ARD, one lengthscale per dimension. A short lengthscale means the response moves fast along that axis, which is per-feature relevance already paid for during the fit.

**Swept at the incumbent, not the medians.** Sweeping one feature means fixing the rest, and the usual choice - column medians - is a recipe that may never have been run, putting the whole profile where the GP has no data and reporting the shape of its prior. The sweep is taken around the best observed row instead: a real recipe, near data, and the question a scientist is actually asking. `swept_at` names this in the response rather than leaving it implied.

**Gated on out-of-fold skill, which the negative control forced.** Posterior separation alone is NOT sufficient: on a pure-noise target the GP fits small wiggles, and near the training data its posterior sd is tiny, so a meaningless bend scores a large separation ratio. The first version confidently claimed interior optima on noise. The shapes are now conditioned on `cv_spearman`, the same out-of-fold measurement and `RELIABILITY_SPEARMAN_FLOOR` the reliability verdict already uses, so a model that has not shown it can predict held-out runs reports no shapes at all. When no skill measure is supplied the report says so in `unmodeled` rather than implying the shapes were validated.

Also: the 0.20 floor was a bare literal in two places in `analysis.py` and is now the single `RELIABILITY_SPEARMAN_FLOOR`, so the verdict and the shapes can never be held to different bars.

`tests/test_gp_shape.py` (+15): the headline interior optimum where Spearman is blind, ARD ranking the real driver above noise, monotonic features not flagged, an ignored feature reported `flat` rather than as a trend, no interior claim on a monotonic response, the noise-target refusal, refusing an unfitted surrogate (it must describe the deployed model, never fit its own), mismatched names, a constant column, determinism, and strict-JSON safety.

Suite: 402 passed, 1 skipped. ruff and mypy clean.
## 2026-08-09 (driver multiplicity + stated coverage)

### Fixed - the driver panel manufactured false process insights

The panel tests every continuous feature against the target, then ships the strongest by `|rho|`, and until now applied no multiplicity correction at all. Measured over 400 simulated reports on 30 pure-noise features against an independent target:

```
reports containing >=1 false "significant" driver
  uncorrected per-feature 95% CI:   78.8%
  Benjamini-Hochberg at q = 0.05:    3.5%
```

Nearly four in five reports would show a fabricated driver. Selection makes it worse than the raw rate implies, because ranking by `|rho|` preferentially surfaces exactly the flukes, and the frontend then labelled them "confirmed driver" with causal verbs ("raises", "lowers"). This is the same failure class as the `run_number` rho = 1.0 incident: a statistic that is technically computed correctly and scientifically meaningless.

Demonstrated on one realistic sheet (1 real driver, 14 noise columns, n=60): `noise_1` (p=0.032) and `noise_10` (p=0.047) both had bootstrap CIs excluding zero and would both have shipped as significant. Both are now correctly rejected, leaving only the real driver.

- `kalos/core/drivers.py`: `benjamini_hochberg(pvals, q)`, a step-up FDR procedure. BH rather than Bonferroni because it controls the expected PROPORTION of false findings, which is the right error rate when a scientist acts on several drivers; Bonferroni at 30 collinear media-DoE features would suppress the real drivers too. The docstring states the positive-dependence assumption and that surviving does not make a driver causal.
- `kalos/portal/analysis.py`: `significant` now requires BOTH the bootstrap CI excluding zero AND surviving BH. Applied over ALL tested features before the top-k cut, never after - correcting for the 8 shipped when 30 were tested would understate the multiplicity it exists to control. Each driver also carries `p`, `ci_excludes_zero` and `survives_fdr` so a reviewer can see which test a borderline feature failed.
- New `driver_selection` block reports `n_tested`, `n_reported`, `top_k`, `fdr_method`, `fdr_q`, `n_bootstrap` and `ranked_by`. Selecting the strongest of many tested features is itself a statistical act, and the client needs to see that it happened rather than being shown eight drivers as though eight were tested.

Known limitation, stated rather than hidden: `n_bootstrap` stays at 200, so the 2.5% CI percentile rests on roughly the 5th order statistic and is coarse. Raising it to 2000 was measured at 5.1s (n=55) and 15.8s (n=2000), which would triple analyze latency, and BH on exact p-values is now the primary gate with the CI as a secondary check. The count is reported so the coarseness is visible.

### Fixed - the response now states the coverage it actually computed

The band is computed at `alpha=0.1`, a 90% band. The frontend had drifted to labelling that same band "95%" on `/decide` and "90%" on the older surface. A stated coverage number that is wrong on screen is an overclaim, not a hedge.

- `kalos/portal/analysis.py`: `CONFORMAL_ALPHA` is now a named constant and the response carries `conformal_coverage` (0.9), so no consumer has to hardcode a percentage. Coverage is a property of the method, not the data, so it is reported even when the band itself is `None` for want of out-of-fold residuals.

`tests/test_driver_multiplicity.py` (+15): the BH primitive (step-up behaviour, carry-along, order alignment, empty/NaN), only-the-real-driver-is-significant, the specific rescue of a noise feature whose CI excludes zero by chance, `significant` as a strict conjunction, selection disclosure, FDR applied over all tested features, and coverage reported with and without a band.

Suite: 400 passed, 1 skipped. ruff and mypy clean.

## 2026-08-08 (identifier columns: privacy + spurious drivers)

### Fixed - a numeric identifier column was being modeled as a process input

Given a sheet carrying `run_number` and `batch_id`, the engine fit on both, reported both as significant drivers at rho = 1.0, and proposed a recipe instructing the scientist to **"set batch_id = 100.037"**. A run index rises monotonically with time, so it correlates with any drift or learning trend across a campaign and will almost always rank as a top driver. That is a spurious correlation presented as process insight, which is the failure mode this codebase works hardest everywhere else to prevent.

### Fixed - identifier column names survived `anonymize=True`

The same root cause: `DomainProfile.id_hint` is anchored to whole tokens (`^(id|name|run|batch|campaign|lot|...)$`), which matches a column called exactly `run` but not `run_number`, `batch_id`, `campaign_id` or `lot_number` - and compound names are what real run sheets use. So `anonymize=True` published those names verbatim while claiming to pseudonymize identifier columns. The codebase already contradicted itself here: the metadata scrubber's `HASH_EXACT` does list `campaign_id`, so one id was hashed as metadata and published as a column name in the same response.

- `kalos/portal/analysis.py`: `identifier_pattern` unions a profile's own `id_hint` with the compound-identifier pattern, and feature selection, provenance, and anonymization all now resolve through it, so they cannot disagree about what an identifier is. Union, never intersection - it can only classify more names as identifiers, never fewer. `id_hint` itself is untouched, because it also drives role classification and narrowing it would discard real measurements like `batch_titer`.
- `_anonymize_result` now covers the `validation` block and `design_box_exclusions`, not just `group_col` and `provenance`. A finding names its column twice - a `column` field and the sentence built around it - so both are rewritten together. Aliasing the structured field while leaving the name in the prose beside it would have been anonymization worth nothing.
- Precision holds in the other direction: `Methanol`, `pH`, `scale_L`, `lipase_titer`, `batch_titer` and `run_duration_days` are all still treated as real columns.

`tests/test_identifier_privacy.py` (+29): the pattern in both directions, identifiers never becoming features / drivers / recipe entries, provenance agreeing, no identifier surviving `anonymize=True` (including inside message prose), real feature names preserved, pseudonyms stable, and opt-in behavior unchanged when `anonymize=False`.

Also recorded as a test rather than silently changed: `id_hint` carries a `sample.*` glob, so a genuine measurement named `sample_volume_L` is classified as an identifier and dropped. That is pre-existing behavior and narrowing it is a scientific decision, not a drive-by edit inside a privacy fix.

Suite: 385 passed, 1 skipped. ruff and mypy clean.

## 2026-08-08 (cleanup: remove the barcode registry)

### Removed - `kalos.data.barcode_registry` and its demo

Barcoding is explicitly off the roadmap (2026-07-14: keep the data model simple), and the registry had no caller in the product path - only a re-export, one test, and a demo script entirely about it. Removed rather than left to rot:

- `kalos/data/barcode_registry.py` (145 lines) and `examples/organize_data.py`.
- `kalos/data/__init__.py`: dropped the `BarcodeRegistry` / `RunRecord` re-exports.
- `kalos/data/anonymizer.py`: `run_barcode`, `dataset_barcode`, and the `_stable_payload` helper went with it - the registry was their only caller, so keeping them would have left dead code behind the thing that was removed to avoid dead code. The `Anonymizer` class itself stays; it is used across `kalos.normalize`, the `kit` facade, and the analyze path.

**Coverage was preserved, not dropped.** `anonymize_meta` had no test of its own - its only exercise rode along inside the barcode-registry test. Since it is the function standing between a client's identity and anything the engine persists, it now has a direct test asserting the full contract: identity keys (client/strain/operator) dropped entirely, grouping ids (campaign/lot) kept but irreversibly hashed so CV can still group by them, ordinary metadata passed through, hashing deterministic under a fixed salt and salt-dependent across tenants.

Audited the other dead-code candidates before touching anything and kept them all: `core/gates.py` (public API with tests, though still not wired into the portal - a real follow-up), `core/feasibility.py` (used by the benchmark sweep), `features/protein.py` (ESM-2, on hold but planned), `core/multiobjective.py` (used by the portal).

Suite: 356 passed, 1 skipped. ruff and mypy clean (65 source files, down from 66).

## 2026-08-08 (campaign store schema repair)

### Fixed - an incompatible `campaigns` table silently discarded every campaign write

`CREATE TABLE IF NOT EXISTS` is not a migration. It matches on table NAME and ignores shape, so a `campaigns` table with the wrong columns or key was accepted at startup, and the failure only surfaced per-write, as a `sqlite3.OperationalError` ("ON CONFLICT clause does not match any PRIMARY KEY or UNIQUE constraint") deep inside `_write_locked`. `/api/run` catches and logs that while continuing, so the portal looked healthy while persisting nothing - indefinitely, because nothing ever repaired the table.

Observed, not hypothetical: a development database carried `PRIMARY KEY (tenant, campaign_id)` from an unmerged multi-campaign branch. `tenant` is *in* that key but is not the sole key, and SQLite's `ON CONFLICT(tenant)` requires exactly the latter. Every campaign write on `main` had been failing. The table was found empty, which is the proof - nothing had ever been persisted.

- `kalos/portal/campaign.py`: `_ensure_campaigns_schema` checks the real shape via `PRAGMA table_info` (every required column present, and `tenant` the sole primary key) and repairs a mismatch. Repair is **non-destructive**: the old table is renamed to `campaigns_backup_N` and left in place, a correct table is created, and salvageable state is copied across. Where the old shape held several campaigns per tenant, the most recently updated one wins, since that is the campaign the single-campaign API would have been serving. Logged at WARNING, because an operator needs to know a table was renamed under them and where the rows went.
- The backup name is the first unused `campaigns_backup_N`, so an upgrade/downgrade ping-pong never clobbers an earlier backup.
- `tests/test_campaign_schema_migration.py` (+12): the exact composite-key shape found in the wild, newest-per-tenant selection, non-destructiveness, repeated migration, an unsalvageable same-name table, and both happy paths (a fresh database and reopening a correct one must not migrate or churn backups).

Audited the sibling `kalos/store/sqlite_store.py` for the same hazard: it is sound, its `ALTER TABLE` tenant backfill having applied correctly.

## 2026-08-08 (bioprocess data validation gate)

### Fixed - the optimizer could propose physically impossible recipes

Reproduced end-to-end through the real `_analyze` path, not hypothesized.
A 12-row sheet with one `-999` "sensor offline" sentinel in its temperature column returned these recommendations:

```
recipe: {'feed_mL_h': -0.6,   'temp_C': -422.948, 'pH': 40.0}
recipe: {'feed_mL_h': 0.75,   'temp_C': -350.659, 'pH': 36.759}
```

Bioreactor temperatures below absolute zero, negative pump rates, and an impossible pH, each with a confidence interval attached.
The cause is short: the design box handed to the acquisition optimizer was the raw observed `[min, max]` of every feature, so a single sentinel widened the search space to `[-999, 37]`.
Per-column provenance made it worse by reporting `coerced_cells=0` on all four columns - an active clean bill of health.

- `kalos/portal/analysis.py`: `_physical_range` builds each continuous dimension's bounds from physically valid observations only, using the hard bounds in `kalos/validation/bounds.py`. The same sheet now proposes `temp_C` 37.3, `pH` 6.9, `feed_mL_h` 0.636. Narrowing is never silent: `design_box_exclusions` reports every affected column and how many cells were excluded.
- This safety property holds in every validation mode. It does not depend on a client reading the report.

### Added - `kalos/validation`: a nine-check ingest gate wired into `/api/run`

`kalos/normalize/` (a complete, tested unit registry) had **zero imports from the product path** - so a titer column mixing g/L and mg/mL across rows passed ingest silently, as did a pH of 40. The registry is now live.

- `kalos/validation/{report,bounds,checks,runner}.py` (new): units consistency, physical bounds, duplicates vs replicates, missingness, MAD-based outliers, provenance metadata, replicate adequacy, controls presence, constant columns. Errors are physics violations; warnings are operationally suspect but possible.
- `kalos/portal/analysis.py`: the gate runs on the sheet as uploaded, before any column is typed or dropped, so it can still see `"34.6 C"` and a mixed-unit column. The report is served as `validation` in the `/api/run` response.
- `KALOS_VALIDATION_MODE` = `warn` (default) or `strict`. Default is `warn` deliberately: adding a gate must not start rejecting data clients push through today. `strict` refuses an upload whose status is `fail`.
- Unit-tagged columns are now **converted rather than discarded**. `"34.6 C"` parses at 0% as a bare number, so a real temperature input used to fail the >=80% numeric test and come back `dropped_sparse`. It is now a usable feature in Celsius, with the conversion reported explicitly.
- Only units the registry can actually convert are rewritten. A vessel column of `"5L"`/`"500L"` uses an unrecognized token, and converting it would turn an identifier into a measurement; it is left untouched and the client is told. Backed by `kalos/normalize/units.py::is_known_unit`, which exists because `canonical_suffix` returns `""` for both an unknown unit and a genuinely dimensionless one (pH, OD).

### Fixed - three defects found by running the gate against a real 55-batch mAb scale-up dataset

- **Outliers on a designed scale ladder.** MAD z-scores assume a unimodal linear distribution. On a bioreactor ladder (0.01 to 2000 L) the median is 10 and the MAD about 10, so every run at 200 L or above scored `|z| > 5`: 20 of 55 rows flagged, plus every scale-proportional column. A strictly-positive column spanning >= 2 orders of magnitude is now assessed on log10, which is how a multiplicative quantity is actually distributed. False positives went from 11 warnings to 6, and all 6 survivors are correct - the 4 remaining flagged batches are the intentionally degraded runs, coherent across viability, monomer, aggregate and endotoxin at once.
- **Provenance false alarm.** The check reused `DomainProfile.id_hint`, which is anchored (`^(...|batch|lot)$`) because `_analyze` uses it to decide whether a column is *entirely* an identifier. Asking "does an identifier exist" needs to match compound headers, so a sheet whose first two columns were `batch_id` and `run_number` was told it had no traceability. `_RUN_ID_RE` now handles compound names while still rejecting measurements like `batch_titer`.
- **`inf` broke the response.** Unbounded dimensions carry `hard_hi = inf`, which reached `detail["hard_range"]`; Starlette's `JSONResponse` correctly refuses non-finite floats, so the gate's own finding turned a good analysis into a 400. `report_dict` sanitizes at the serialization boundary, mapping non-finite floats to `null` ("no bound").
- `infer_dimension` coverage went from 7/25 to 22/25 against the real column vocabulary (glucose, lactate, glutamine, ammonium, sodium, osmolality, pCO2, VCD/TCD), and the pH hint no longer matches `phosphate`, `morphology`, `sulphate` or `phase` - a `phosphate_g_L` feed column was being given pH's 0-14 range and reported as impossible.

### Added - `kalos/providers`: credential slots with keyless fallbacks

Every provider is unavailable and every feature works without a key. No speculative client code for unwired services.

- Anthropic (LLM column/unit normalization; falls back to the deterministic offline mapper), NVIDIA BioNeMo (a documented SLOT - there is no BioNeMo client in this codebase; the public non-gated ESM-2 checkpoint needs no token), an ELN (a SLOT; the honest source of the run-id/operator/date provenance the gate can currently only warn about).
- `GET /api/providers` reports status. `ProviderStatus` carries env var NAMES only, never values.
- `.env.example` documents every environment variable the codebase reads.

### Fixed - reproducibility and CI

- `kalos/features/protein.py`: `ESM2Embedder` pinned to an immutable commit (`DEFAULT_ESM2_REVISION`). `from_pretrained` with no `revision` resolves to a mutable branch, so upstream could change every embedding with no diff in our code - breaking reproducibility in the worst way, silently, under a fixed seed.
- `pyproject.toml`: `anthropic` added to the `dev` extra. CI installs `.[ml,portal,dev]`, so the two mocked-live tests in `tests/test_normalize_llm.py` failed on a fresh checkout. Chosen over a skip guard so CI keeps genuinely covering that path.

### Tests

`tests/test_validation.py` (+25, checks in isolation) and `tests/test_validation_gate.py` (+14, the live path, led by the absolute-zero regression). Suite: 344 passed, 1 skipped. ruff and mypy clean.

## 2026-07-22 (P1: experiments tenancy)

### Fixed - production hardening P1: the experiments store is now tenant-scoped and auth-gated

A progress review found that campaign + latest were tenant-scoped (Phase 1b) but the **experiments store was still global and its endpoints entirely ungated** - a live cross-tenant leak once multiple tenants exist. Closed:

- `kalos/store/sqlite_store.py`: `experiments` gains a `tenant` column (with an index and a one-time `ALTER TABLE ... DEFAULT 'default'` backfill migration for pre-existing databases); every `get`/`list`/`set_status`/`save_result` filters by tenant and `create` records it. A cross-tenant id reads as not-found, so tenants can't probe each other's ids.
- `kalos/portal/experiments.py`: every route now requires the `read` or `write` scope and passes `Principal.tenant` to the store.
- `kalos/runner/adapter.py`: `LocalStoreAdapter` is bound to a tenant, so the Singleton runner only ever sees and mutates the calling tenant's experiments. (The remote-runner `/api/experiments/{id}/result` channel still operates on the `default` tenant - a documented follow-up.)
- `tests/test_tenant_isolation.py`: +3 tests - experiments isolated per tenant (list/get/mutate), the legacy backfill to `default`, and HTTP proof that the endpoints require auth and one tenant never sees another's experiments. Existing M2/experiments suites green (default tenant unchanged).

## 2026-07-22 (even later still)

### Added - production hardening Phase 1c: CORS allowlist + startup security posture (`kalos/portal/config.py`)

- `kalos/portal/config.py` (new): centralizes the portal's deployment security config. CORS is an explicit allowlist when `KALOS_CORS_ORIGINS` is set (the production posture) and the permissive localhost regex otherwise (dev/pilot, unchanged). `log_security_posture` logs auth+CORS state once at startup and warns when the portal is not locked down.
- `kalos/portal/app.py`: the CORS middleware now uses `cors_config()`; the posture is logged at startup.
- `docs/HARDENING.md`: `KALOS_CORS_ORIGINS` config reference and a TLS-via-reverse-proxy deployment note.
- `tests/test_config.py` (new): 9 tests - allowlist parsing, dev-default regex vs configured allowlist, and the posture warning.

## 2026-07-22 (later)

### Changed - production hardening Phase 1b: per-tenant persistence (`docs/HARDENING.md`)

Every portal read/write is now keyed by the caller's tenant (`Principal.tenant` from the auth layer), so two tenants can never see or overwrite each other's campaign or analysis. Backward compatible: in open mode everything maps to the `default` tenant, so the dev/pilot loop is unchanged.

- `kalos/portal/campaign.py`: the `CampaignStore` is now backed by **SQLite** - one `campaigns(tenant, state, updated_at)` row per tenant in `<state_dir>/portal.db`, replacing the single `~/.kalos/campaign.json`. Each method takes a `tenant` (default `"default"`); each write is one transaction. The transactional generation-token logic is unchanged (it lives inside the per-tenant `state` blob).
- `kalos/portal/app.py`: `_LATEST` is now a per-tenant map plus a per-tenant best-effort cache file under `<state_dir>/latest/<tenant>.json` (tenant sanitized for the filename), replacing the single global. `/api/latest` requires the `read` scope and returns the caller's tenant's analysis; `/api/run` seeds the campaign and saves latest under `Principal.tenant`.
- `kalos/portal/campaign_routes.py`: `GET /api/campaign` requires `read`; every store call (`summary`/`start`/`set_result`/`plan_fold`/`commit_fold`/`get`) and the reanalyze `_save_latest`/`_load_latest` pass `principal.tenant`.
- `tests/test_tenant_isolation.py` (new): 4 tests - campaigns and `/api/latest` isolated per tenant at the store level, reseeding one tenant never touches another, and an HTTP end-to-end proof that one tenant's campaign is invisible and untouchable by another through the authenticated API. Existing portal/campaign fixtures updated for the SQLite + per-tenant-latest shape.

## 2026-07-22

### Added - production hardening Phase 1a: in-house API authentication (`kalos/portal/auth.py`)

First step of the behind-the-scenes hardening track (`docs/HARDENING.md`) toward a multi-tenant, operable service. Compliance/certification work is handled offline and is out of scope here.

- `kalos/portal/auth.py` (new): a self-hosted token layer. A bearer token resolves to a `Principal` (subject, tenant, scopes); tokens are provisioned as SHA-256 hashes (never plaintext, never logged) and constant-time compared. Config is read at request time from `KALOS_AUTH_TOKENS_FILE` (preferred) or `KALOS_AUTH_TOKENS`, so rotation needs no restart.
  FastAPI dependencies `require_principal` / `require_scope(scope)` gate endpoints.
- **Backward compatible.** With no tokens configured the API runs in *open mode* - every request gets an anonymous `default`-tenant principal with read+write (today's behavior) and a startup warning is logged; `admin` is never granted without a real token. The moment tokens are provisioned, enforcement turns on.
- Gated the mutating endpoints on the `write` scope: `POST /api/run` (`kalos/portal/app.py`) and `POST /api/campaign/{start,result,reanalyze}` (`kalos/portal/campaign_routes.py`). `GET` reads stay open for now; per-tenant data isolation is the next slice.
- `docs/HARDENING.md` (new): the in-house hardening plan and configuration reference (auth -> tenant-scoped persistence -> reliability -> operability).
- `tests/test_auth.py` (new): 27 tests - open mode, valid/invalid/missing/malformed tokens, tenant isolation, file-over-env precedence, scope gating, and HTTP end-to-end that `/api/campaign/start` is 401 unauthenticated, 403 for a read-only token, and clears the gate with a valid write token. Full portal suite green (open mode unchanged).

## 2026-07-21 (even later)

### Changed - the SNR lever ships in production: replicate-aware proposals (`kalos/portal/analysis.py`)

BENCHMARK.md established that BO only beats random on the real zero-inflated media data when it optimizes the replicate-averaged (reproducible) titer with the measured assay noise floor fed to the GP - single measurements reward lucky noise spikes (ICC ~0.26, roughly three quarters of titer variance is assay noise).
That fix lived only in the benchmark harness; the production analysis path still fit the GP on raw single measurements.

- `kalos/portal/analysis.py` (`_analyze`): when the fitted rows have replicated recipes (>= 2 replicated, >= 6 distinct recipes, positive noise floor), the proposal surrogate is now fit on `aggregate_replicates()` means with per-recipe fixed observation variance `sigma^2 / n_reps` (via `Surrogate.fit(noise=...)`), and the proposed batch optimizes that reproducible objective; the shown incumbent is the reproducible best.
  Non-replicated sheets fall through to the unchanged raw fit.
- Every analysis result now carries a `noise` block: `n_recipes`, `n_replicated`, `replicate_aware`, `icc`, `noise_sd`, `signal_sd`, `best_single`, `best_reproducible` - the honest signal-to-noise picture and the reproducible ceiling, not just the lucky spike.
  Diagnostics (grouped-CV reliability, drivers) stay on the raw rows - they are already replicate-grouped for leakage and describe the as-measured signal.
- `examples/benchmark_media_pool.py` (new): a committed, runnable reproduction of the real-data pool retrospective (`KALOS_MEDIA_DATA=/path python examples/benchmark_media_pool.py`), racing BO / feasibility-gated BO / random on the reproducible objective (BO leads) and the single-measurement objective (the artifact). No client data is committed.
- `tests/test_replicate_aware_analysis.py` (new): replicate-aware fit triggers + honest noise report; non-replicated unchanged; deterministic.

## 2026-07-21 (later)

### Added - campaign loop: the closed optimization loop (`kalos/portal/campaign.py`, `/api/campaign*`)

Until now the portal was a one-shot analysis viewer: upload a run sheet, get a batch, done.
This adds the loop that makes it a product - propose, run, log the measured outcome, re-propose - reusing the existing `_analyze` path with no new engine capability.
Design and full contract in `docs/CAMPAIGN_LOOP.md`.

- `kalos/portal/campaign.py` (new, torch-free): `CampaignStore` persists one campaign (a target plus a growing `base_rows` dataset and a list of started `pending` runs) to `~/.kalos/campaign.json`, lock-guarded and written atomically (temp file + `os.replace`), mirroring the `_LATEST` state pattern.
  `best` is the max MEASURED target over `base_rows`, never a prediction; a non-finite result is rejected; only runs with a real measured outcome are ever folded into the dataset.
- `kalos/portal/campaign_routes.py` (new): `GET /api/campaign` (summary for the `/decide` view), `POST /api/campaign/start` (append proposed recipes as awaiting runs), `POST /api/campaign/result` (log a measured outcome), `POST /api/campaign/reanalyze` (fold measured runs into the dataset, re-run `_analyze` via a worker thread, persist as the new `/api/latest`, increment the round).
  `reanalyze` returns the same shape `GET /api/latest` does (including `dataset` and `updated`), so the frontend can swap it straight into its `PopulatedResult` state.
- `kalos/portal/app.py`: a fresh `/api/run` upload now seeds a fresh campaign from `(df, target, proposal_features)` (best-effort - a seeding failure never breaks the upload); the campaign router is mounted beside the experiments router.
- Honest by construction each round: re-analyze routes through the same leakage-controlled grouped-CV `_analyze`, so reliability, conformal bands, and the "not modeled" callouts stay first-class every cycle.
- Progress trajectory: the campaign records a `history` of `{round, best, n_base}` points (round 0 at seed, one per re-analyze) so the frontend can plot best-so-far converging. `best` is measured and base rows only grow, so the trajectory is non-decreasing.

`tests/test_campaign.py`: 12 tests (seed, summary states, start, result validation, and the fold-and-re-analyze round-trip that grows the base, increments the round, and extends the history trajectory).

### Fixed - campaign re-analyze is now transactional (no data loss, no phantom rounds)

Review of the closed loop surfaced a persist-then-validate ordering defect: `fold_and_snapshot` committed the fold (round++, measured runs merged into `base_rows`) to disk *before* `_analyze` ran, so an analysis failure permanently advanced the round with no rollback, and a concurrent `/api/run` upload during the multi-second analysis could silently destroy the just-folded measured data and clobber `/api/latest` with stale numbers.

- `kalos/portal/campaign.py`: `fold_and_snapshot` is split into `plan_fold()` (computes the folded dataset in memory, mutating nothing) and `commit_fold(generation)` (persists only if the campaign is unchanged).
  Every write now stamps a fresh `generation` token, so a re-analysis that was planned against one campaign refuses to commit if a `seed()`/`set_result`/`start` landed underneath it.
- `kalos/portal/campaign_routes.py`: `reanalyze` now plans the fold, runs `_analyze`, and only then commits - so a failed analysis leaves the campaign untouched (a retry is meaningful), and a campaign reseeded mid-analysis returns `409` without ever calling `_save_latest`, so `campaign.json` is never overwritten with stale results.
  Re-analyzing with no newly measured run is rejected up front (it would only inflate the round and the progress trajectory).
- `/api/latest` is a separate resource (own lock, taken in the opposite order by the upload path), so its final write is guarded best-effort: the route re-checks the campaign generation immediately before `_save_latest` and skips the stale write if a fresh upload reseeded in between.
  This eliminates the multi-second race across `_analyze` and the upload-lands-after-commit case; a sub-millisecond residual window is documented in `docs/CAMPAIGN_LOOP.md` (fully sealing it needs an ordered stamp on `_LATEST`).
- `kalos/portal/campaign.py`: `plan_fold` reads `generation` with `.get()` so a pre-token `campaign.json` re-analyzes without crashing (`KeyError` -> `500`); the migration is self-healing (the commit re-stamps a real token).
- `kalos/portal/campaign.py`: `start()` now validates each `recipe` is a non-empty mapping and raises `CampaignError` at the point of the bad input, instead of surfacing as an unhandled `500` rounds later inside the fold.
- `kalos/portal/campaign_routes.py`: documented the deliberate unauthenticated auth posture for `/api/campaign*` (browser-facing, localhost-bound, same as `/api/run`).
- `docs/CAMPAIGN_LOOP.md`: added Mermaid diagrams (the closed loop, the pending-run lifecycle, the transactional re-analyze sequence) and documented the transactional design and the residual `/api/latest` window.
- `pyproject.toml`: added `httpx>=0.27,<1` to the `dev` extra. `starlette.testclient.TestClient` (used by every portal test) needs `httpx`, but it is not pulled in transitively, so CI's `pip install -e ".[ml,portal,dev]"` left it absent and the whole portal suite errored at import (`RuntimeError: ... requires the httpx2 package`). This was red on `feat/campaign-loop` before this branch; the fix lands with the merge.

`tests/test_campaign.py`: +8 tests - malformed-recipe rejection (×4), no-measured-runs rejection, `_analyze`-failure leaves round/base untouched then a retry folds normally, `commit_fold` aborting when the campaign is reseeded underneath it, and re-analyze on a legacy (pre-`generation`) `campaign.json` not crashing.

## 2026-07-21

### Changed - CI now gates ruff + mypy + the portal tests, and the type layer is clean

CI ran `pytest` only, and it installed `.[ml,dev]` without the `portal` extra, so the portal tests (which `importorskip("fastapi")`) were silently skipped on every run.
Lint and type checking were never enforced at all.
This wires the quality gates that were only ever run by hand.

- `.github/workflows/ci.yml`: installs `.[ml,portal,dev]` and runs `ruff check kalos/`, `mypy`, then `pytest`.
  The portal is a shipped surface, so its tests now actually execute in CI instead of skipping.
- `pyproject.toml`: `dev` extra gains `ruff`, `mypy`, `pandas-stubs`, and `scipy-stubs`, so the checks are reproducible from a clean `.[dev]` install rather than depending on a globally installed tool.
- `pyproject.toml`: added a `[tool.mypy]` block targeting Python 3.12 (the CI and dev interpreter) over the `kalos` package.
  `torch` / `botorch` / `gpytorch` / `linear_operator` are set to `follow_imports = skip`; following their full typed surface pushed a cold-cache run into minutes, and we typecheck kalos's own code, not theirs.

### Fixed - 12 real mypy errors + a `rounds=0` edge case in `/api/multi`

With the type stubs installed, mypy surfaced twelve genuine typing gaps.
None changed runtime behavior except the last.

- `kalos/core/gates.py`: `_finite` now returns `TypeGuard[float]`, so mypy narrows `stats.get(key)` from `Any | None` to `float` inside the guarded branch (the `float(v)` / `v < lo` comparisons were untyped before).
- `kalos/core/splits.py`, `kalos/core/drivers.py`: array-like parameters (`y`, `groups`, `signal`) are typed `numpy.typing.ArrayLike` instead of `Sequence`, since they are called with numpy arrays and immediately go through `np.asarray`.
- `kalos/portal/serialization.py`, `kalos/portal/analysis.py`, `kalos/core/drivers.py`: narrowed a few `object`-typed values (pandas `to_dict` records, driver dicts, `feature_names`) with `cast` / explicit annotations so the downstream `float(...)` / indexing typechecks.
- `kalos/portal/app.py`: `run_multi` now clamps `rounds`/`q` to at least 1 and pre-binds `last_batch`.
  A `?rounds=0` request previously hit `last_batch` unbound and raised `NameError`; it now returns one honest round.
  Added an `assert s.model is not None` after `fit()` to document the invariant mypy could not otherwise see.

## 2026-07-20

### Added - domain-neutral core: declared column roles + mixed continuous/categorical design spaces (`kalos/domains/`)
Kalos was branded bioprocess-only, but the engine (`core/`), store, runner, and upload pipeline
operate on numeric arrays and are domain-agnostic. This change makes that reusable without touching
the engine, and adds categorical-parameter support so the platform fits industries with discrete
process choices (which catalyst, which resin), not just continuous recipes.

- `kalos/domains/` (new, torch-free): `ColumnRoles` (an explicit target/features/groups/ids/
  categoricals schema), `DomainProfile` (fallback role-hint regexes as data, not engine code),
  `DesignSpace` + `Dimension` (per-dimension continuous/categorical spec with integer encoding and
  label decoding), and `build_design_space`. Ships `BIOPROCESS_PROFILE` (the legacy hints, so the
  default path is unchanged) and `GENERIC_PROFILE` (domain-neutral). Importing `kalos.domains` never
  loads torch (`tests/test_domains.py`).
- `kalos/core/surrogate.py`: `Surrogate.fit(..., cat_dims=)` fits a BoTorch `MixedSingleTaskGP`
  (CategoricalKernel on the categorical dims, RBF on the continuous (this entry first said Matern), continuous dims normalized to
  the box) when categoricals are present; the continuous `SingleTaskGP` path is unchanged.
- `kalos/core/optimize.py`: `propose(..., cat_dims=, cat_cardinalities=)` uses `optimize_acqf_mixed`
  (enumerating categorical assignments, exact) for small categorical spaces and falls back to
  `optimize_acqf_mixed_alternating` past `MAX_MIXED_COMBOS`. Continuous path unchanged.
- `kalos/core/evaluation.py`: `grouped_cv_report(..., cat_dims=)` threads the mixed GP through
  leakage-controlled CV so the reported number matches the deployed model.
- `kalos/portal/analysis.py`: `_analyze(..., roles=, profile=)`. With a declared `ColumnRoles` the
  roles are used directly (generic profile); with none, the bioprocess profile infers them exactly as
  before. Drivers are computed over continuous features only (a Spearman "driver" for a nominal
  category is not meaningful). Proposals carry a decoded `recipe` (`{feature: value}`), and the
  response adds `categorical_features`.
- `kalos/portal/app.py`: `POST /api/run` accepts an optional `roles` JSON form field; when present the
  upload is analyzed domain-neutrally.
- `kalos/portal/validate.py`: `column_provenance(..., declared=)` records each column's role `source`
  (`declared` vs `inferred`) so the audit trail is honest about who decided the role.
- `kalos/bench/`: `MixedObjective` + `mixed_bump` + `run_mixed_one` exercise the mixed loop; a test
  confirms mixed BO reaches far lower simple regret than random choice.
- No behavior change on the bioprocess path: the default profile and continuous engine branch are
  byte-for-byte the prior code paths (verified field-by-field against `main`); the analyze response
  only gains additive fields (`categorical_features`, `proposal_optimizer`, per-proposal `recipe`,
  per-column `source`). Guarded by the existing portal/hardening/bench tests.

### Fixed - provenance honesty for declared roles + surfaced mixed optimizer (review follow-ups)
An adversarial review of the change above found the modeling core correct but flagged honesty gaps in
the new declared-roles/provenance layer (no correctness bugs). Addressed:
- `kalos/portal/analysis.py`: a declared feature / categorical / group / id name that does not match a
  sheet header now raises a clear `ValueError` instead of being silently dropped (a typo previously
  vanished with no signal, so a client believed a column was honored when it was not; a typo'd id in
  particular used to leave the real id column in as a feature).
- `kalos/portal/validate.py`: `column_provenance` gained `declared_ids` and `declared_features`. A
  declared id now reports as `dropped_id` with `source="declared"` (not the misleading
  `dropped_sparse`/`inferred`), and a column declared as a continuous feature but holding text now
  reports as the new `dropped_non_numeric` status - an honest "you likely meant to mark this
  categorical" - instead of `dropped_sparse`.
- `kalos/portal/analysis.py`: the analyze response now carries `proposal_optimizer`
  (`continuous` | `mixed_exact` | `mixed_alternating`) so the client can tell when a large categorical
  space fell back from exact enumeration to the alternating heuristic, alongside the existing
  seed/timestamp/engine_version audit fields.

A second review round found and closed further honesty gaps:
- `kalos/portal/analysis.py`: a self-contradictory schema (the target also declared an id or
  categorical) now raises instead of silently dropping one role.
- `kalos/portal/analysis.py`: declared continuous features must clear the same >=80% numeric-parse
  gate inference mode uses; a mostly-text column declared as a feature is now reported
  `dropped_non_numeric` rather than silently zero-filled into the model as a `kept_feature`.
- `kalos/portal/analysis.py`: blank categorical cells are no longer an ordinary proposable level -
  they are excluded from the levels and their rows dropped from the fit (an unknown categorical can
  neither be modeled nor recommended). The response reports `n_dropped_incomplete`, and `n` is the
  row count actually fit.
- Test coverage added for all of the above plus previously-untested paths: the alternating optimizer
  (large cardinality), all-categorical design spaces, single-level categorical drop, declared group
  columns, and mixed-BO run reproducibility.
- Known follow-up (not yet done): there is no per-level replicate-count warning for a categorical
  level too sparse to identify - the recommended materials guardrail before running on real small-n
  formulation data.

## 2026-07-18

### Changed - torch/botorch/gpytorch are now optional (`kalos[ml]`); new `kalos.kit` torch-free facade
Phase 1 of engine consolidation: a sibling repo (an earlier engine, deliberately torch-free)
is meant to import Kalos's leakage-controlled splits, driver analysis, conformal intervals,
promotion gates, and anonymizer instead of keeping its own copies. That only works if installing
`kalos` does not drag in a ~220 MB torch/botorch/gpytorch stack.

- `pyproject.toml`: `torch`, `botorch`, `gpytorch` moved out of core `dependencies` into a new
  `ml` extra. Core install (`pip install kalos`) is now torch-free: `numpy` / `pandas` /
  `scikit-learn` / `scipy` only. The `portal` extra still needs the live GP, so it is installed as
  `kalos[ml,portal]`.
- `kalos/kit/__init__.py` (new): a thin re-export facade over the already-torch-free
  `kalos.core.splits`, `kalos.core.drivers`, `kalos.core.conformal`, `kalos.core.gates`, and
  `kalos.data.anonymizer`. Nothing moved - existing imports like
  `from kalos.core.drivers import ...` are unchanged. `import kalos.kit` is guaranteed to never
  load torch (`tests/test_kit_torch_free.py`).
- No behavior changes: `kalos/__init__.py` and `kalos/core/__init__.py` were already lazy-loading
  the torch-dependent surrogate/optimize/evaluation exports (PEP 562 `__getattr__`) from a prior
  commit; this change only reorganizes the install metadata and adds the `kit` facade on top of
  that existing lazy-load boundary.

## 2026-07-06

### Added - Replicate-aware aggregation + assay noise floor + fixed-noise GP (`kalos/core/replicates.py`)
`BENCHMARK.md`'s SNR write-up found the real media DoE is heavily replicated (96 rows over 27
distinct recipes, up to 14 reps per recipe) with an ICC of ~0.26 - roughly 74% of titer variance
is assay noise, not recipe-to-recipe signal. This adds the tooling to act on that: aggregate
replicates into a reproducible per-recipe objective, estimate the assay noise floor from the
replicate spread, and optionally hand that noise estimate to the GP directly instead of making it
re-infer noise from a handful of points.

- `kalos/core/replicates.py` (new): `aggregate_replicates(X, y)` groups rows by identical
  rounded feature vectors and returns `(X_unique, y_mean, y_var, n_reps)` in deterministic
  first-occurrence order. `estimate_noise_floor(X, y)` pools the within-group sample variance
  over replicated groups into a single assay noise variance (`nan` if nothing is replicated).
  `noise_report(X, y)` adds `n_rows` / `n_recipes` / `n_replicated` / `signal_var` / `icc` on top,
  for a one-call summary of how much of the variance is real signal.
- `kalos/core/surrogate.py`: `Surrogate.fit(X, y, bounds, *, noise=None)` gains an optional fixed
  observation-noise variance (`None` default, a scalar, or a per-point array, all in the target's
  original units). When given, the GP is built with `train_Yvar` alongside
  `outcome_transform=Standardize` - BoTorch scales `Yvar` through the standardization internally
  and `SingleTaskGP` auto-selects a `FixedNoiseGaussianLikelihood`, so no extra likelihood
  plumbing was needed. Confirmed working on the installed BoTorch 0.18.1 / GPyTorch 1.15.2 before
  wiring it in. `noise=None` is byte-for-byte the previous inferred-noise behavior.
- `kalos/bench/pool.py`: `run_pool_one` / `run_pool` take a `noise` parameter, forwarded to every
  `Surrogate.fit(...)` call in the BO branches (default `None`, bo/random unchanged).
  `pool_from_frame(df, target, *, aggregate=False)` can collapse replicate rows to per-recipe
  means before returning `(X, y, feats)`; `feats` is unaffected, default `False` is unchanged.
- +10 tests (`tests/test_replicates.py`): known-duplicate aggregation, rounding-based near-duplicate
  merging, noise-floor recovery against an injected variance, ICC ballpark on synthetic
  signal/noise, fixed-noise `Surrogate.fit` (scalar and per-point array) end to end, `run_pool`
  with fixed noise producing a finite monotone trajectory, and `pool_from_frame(..., aggregate=True)`.

### Added - Feasibility classifier + gated acquisition (`kalos/core/feasibility.py`)
BENCHMARK.md's finding was that BO loses to random on the real media DoE because titer is
zero-inflated (~21% non-producers) and the GP over-exploits a noisy incumbent on a spiky
feasible/infeasible surface. Rather than change the acquisition function, feasibility (producer
vs non-producer) is now modeled as a separate binary classifier that gates EI, so the fix is
composable with the existing GP.

- `FeasibilityClassifier`: `StandardScaler` + `LogisticRegression(class_weight="balanced")` on
  binary labels. Cold-start safe: fewer than 2 classes or fewer than 3 minority-class examples in
  `fit` skips sklearn entirely and `predict_proba` returns all-ones (no gating, defers to EI) -
  this is what keeps the pool loop from crashing on sklearn's single-class fit error before enough
  non-producers have been observed.
- `feasible_labels(y, threshold=0.0)`: feasible iff `y > threshold` (strict).
- `feasibility_cv_auc(...)`: pooled out-of-fold CV-AUC for the feasibility classifier, using
  `StratifiedGroupKFold` when `groups` is given and `StratifiedKFold` otherwise; reduces
  `n_splits` to the minority class count and returns `nan` (never raises) when a stratified split
  or a well-defined AUC isn't possible.
- `kalos/bench/pool.py`: two new pool strategies, `bo_feas` (GP fit on all evaluated points, EI
  gated by predicted P(feasible)) and `bo_feas_clean` (GP fit only on feasible evaluated points,
  falling back to all points below 3 feasible examples, EI gated the same way). `bo` and `random`
  are unchanged. +6 tests (`tests/test_feasibility.py`, `tests/test_pool.py`) including a
  zero-inflated synthetic pool comparison.

### Update - Honest benchmark result (`BENCHMARK.md`)
Feasibility is highly learnable on the real media DoE (grouped-CV AUC 0.891), but gating EI
with it does not rescue BO (`bo_feas` matches plain `bo`; `bo_feas_clean` still loses to
random, 0.046 vs 0.082). The lever remains replicates / signal-to-noise, not the classifier.

## 2026-07-05

### Added - Closed-loop benchmark (`kalos/bench/`, `BENCHMARK.md`)
An honest answer to "does the optimizer beat a space-filling design?". `python -m kalos.bench`
races the BO loop against Latin Hypercube and random on synthetic surfaces with known optima,
sweeping observation noise. Finding: BO dominates when the signal is clean (reaches LHS's
end-value ~11-14 experiments sooner, near-zero regret) but its edge shrinks to marginal-or-nil
at 15% measurement noise - which is why the real, noisy media data shows only a weak ~0.37-0.52
signal. The lever is data quality (replicates, signal-to-noise), not the algorithm. Full write-up
and reproduction steps in `BENCHMARK.md`; +4 tests.

### Fixed - unify the anonymization scrub lists
`kalos/ingest/feed.py` kept its own second copy of the identity-scrub rules, which had drifted from
the canonical lists in `kalos/data/anonymizer.py`: the feed copy was missing the `subject` and `mrn`
drop-substrings and only hashed `campaign_id` / `campaign`, so `lot`, `batch_id`, `experiment`, and
`batch` columns in an uploaded run sheet passed through un-hashed on the live ingestion path
(`FileDataFeed.read` -> `anonymize_frame`, reached from `propose_next_batch`).

- `anonymize_frame` now imports `DROP_EXACT`, `DROP_SUBSTR`, `HASH_EXACT`, `HASH_SUBSTR` from
  `kalos.data.anonymizer` (single source of truth) and checks both the exact and substring hash sets,
  matching `Anonymizer.anonymize_meta`.
- `anonymize_frame`'s `salt` now defaults to `None` and resolves via `default_salt()`, so it honors
  `KALOS_ANON_SALT` instead of silently hardcoding the dev salt literal.

### Changed - Wave A1.2: production hardening
Go-live hardening from a memory / data-volume / BoTorch review. Response contract preserved.

- **Concurrency** (`kalos/portal/app.py`): the CPU-bound `/api/run` body (parse + GP fit + save)
  now runs via `run_in_threadpool` instead of blocking the single event loop, so concurrent
  uploads no longer hang the whole service. Added `torch.set_num_threads` (`KALOS_TORCH_THREADS`)
  to avoid CPU oversubscription, and a `threading.Lock` around the `_LATEST` read-modify-write.
- **GP-training-row cap** (`kalos/portal/app.py`): a `MAX_FIT_ROWS` guard (`KALOS_MAX_FIT_ROWS`,
  default 2000) rejects oversized fits before building the O(n^2) exact-GP kernel, separately from
  the raw-upload row cap. Rejects rather than silently subsamples.
- **Anonymization salt** (`kalos/data/anonymizer.py`): the salt now reads from `KALOS_ANON_SALT`
  and warns loudly on the dev fallback, so barcodes are not a fixed, dictionary-attackable pseudonym.
- **BoTorch robustness** (`kalos/core/surrogate.py`, `optimize.py`): `fit_gpytorch_mll` is retried
  with escalating Cholesky jitter and raises a distinct `FitError` on persistent failure, mapped to
  its own honest 400 (not the generic parse message). The single-objective acquisition moved from
  `qLogExpectedImprovement` to `qLogNoisyExpectedImprovement` (titer is noisy; matches the
  multi-objective side). `bounds` is now a required argument on the surrogate fits.
- Tests: `tests/test_production_hardening.py` covers the row cap, the env salt, and the FitError 400.

### Fixed - Wave A1.1: review fixes for the upload + optimization path
Follow-up fixes to Wave A1 from a code review of the FastAPI Bayesian-optimization engine.
No architecture change; the `/api/run` and `/api/latest` response contract is preserved and only
extended (a new `dropped_constant_on_fitted_rows` provenance status).

- **Multi-objective bounds-sanity** (`kalos/core/multiobjective.py`): the multi-objective path now
  mirrors the single-objective degenerate-bounds guard.
  `MultiObjectiveSurrogate.fit` sanitizes bounds (via `sanitize_bounds`) before building the
  `Normalize` box, and `propose_multiobjective` sanitizes before `optimize_acqf` and clips returned
  proposals into the observed box.
  This closes the constant-feature (e.g. fixed `Culture_Volume`) NaN / out-of-range blow-up class
  on qLogNEHVI proposals, the same ~33,000,000 regression already fixed on the single-objective side.
- **Error hygiene: fit-time failures stay in the JSON envelope** (`kalos/portal/app.py`, `/api/run`):
  a catch-all `except Exception` was added after the narrow parser catch.
  A `torch.linalg.LinAlgError` from a GP fit or an `AssertionError` from the leakage guard is not a
  plain `ValueError`, so it previously escaped to FastAPI's default text/plain HTTP 500 and broke the
  `{error}` JSON contract the frontend parses.
  It now returns the same generic 400 envelope; the full traceback is logged server-side and no
  parser text, exception message, or stack trace reaches the client.
- **Zip-bomb: xlsx cell cap enforced before materialization** (`kalos/portal/app.py`,
  `_reject_oversized_xlsx`): the xlsx cell-count ceiling now runs on the workbook's declared
  dimensions (opened read-only with openpyxl) BEFORE `pd.read_excel` materializes the frame, so a
  zip-bomb is rejected without the memory spike.
  The post-read shape check is kept as a belt-and-suspenders re-check.
- **CSV row cap fails closed** (`kalos/portal/app.py`, `_parse_upload`): an over-cap CSV is now
  REJECTED with a 400 (`MAX_CSV_ROWS`), matching the xlsx cell-cap behavior, instead of silently
  truncating to the first 100k rows.
  Silent data loss contradicted the safe-errors contract.
- **Honest constant-on-fitted-rows check** (`kalos/portal/app.py` `_analyze`,
  `kalos/portal/validate.py`): the varying-feature filter is recomputed on the target-present rows
  the GP actually fits, not the full column.
  A feature that varies over the whole sheet but is constant on the fitted rows would collapse its
  design box to zero width; it is now dropped and flagged in provenance as
  `dropped_constant_on_fitted_rows`, never silently pinned.

## 2026-07-02 (later)

### Added - Wave A1: safety + product-readiness hardening for the upload path
The client-facing `/api/run` upload path was hardened for external, untrusted run sheets.
No architecture change (still no auth/tenancy); the `/api/run` and `/api/latest` response
contract is preserved and only extended.

- **Upload guards** (`kalos/portal/app.py`, `_parse_upload`): a byte-size cap on the raw upload
  (default 25 MB, env `KALOS_MAX_UPLOAD_MB`), a filetype sniff by MAGIC BYTES (`PK\x03\x04` zip
  header -> xlsx/xls, otherwise UTF-8 text/CSV), a column ceiling (`MAX_COLUMNS=512`), a CSV row
  cap (`MAX_CSV_ROWS=100000`), and an xlsx cell-count ceiling (`MAX_XLSX_CELLS=2,000,000`, a
  zip-bomb guard). Any guard trip returns HTTP 400 with a generic message.
- **Error hygiene**: the catch-all `except Exception: return {"error": str(exc)}` was replaced by
  a narrow catch (`pandas.errors.*`, `ValueError`, `UnicodeError`) that returns a single generic
  message ("Could not parse the uploaded file. Check it is a CSV or Excel run-sheet.") and logs
  the full traceback server-side via the `logging` module. No parser text, column name, cell
  value, path, or stack trace ever reaches the client.
- **Ingestion provenance** (`kalos/portal/validate.py`): a new typed `column_provenance` returns a
  per-column status (`kept_feature`, `target`, `dropped_id`, `dropped_output`, `dropped_constant`,
  `dropped_sparse`, `dropped_all_blank`) plus a non-numeric `coerced_cells` count, surfaced as a
  new `provenance` field on the analyze result. This fixes the silent-column-drop problem: clients
  now see exactly what was used and what was dropped and why. Duplicate column labels are
  de-duplicated (`X`, `X.1`) so both stay visible.
- **Privacy**: raw column names and cell values never appear in logs or client error messages.
  Feature/target names are NOT force-anonymized (the owner UI legitimately shows drivers like
  "Methanol"); instead `/api/run` gains an opt-in `anonymize: bool = False` form field that
  pseudonymizes identifier-type columns only (stable, irreversible hash via `data/anonymizer`).
- **Reproducibility + audit**: the analyze path seeds `torch.manual_seed` + `np.random.seed`, so
  the same upload yields identical proposals, and the result now carries `seed`, `timestamp`
  (unix int), and `engine_version` (from `kalos.__version__`).
- **Bounds-sanity** (`kalos/core/surrogate.py` `sanitize_bounds`, applied in `Surrogate.fit` and
  `core/optimize.propose`): non-finite bounds are repaired and zero-width (constant-feature)
  intervals are widened, and every proposed coordinate is clamped into the observed
  `[min, max]` box. This closes the historical out-of-range blow-up (a constant `Culture_Volume`
  proposing ~= 33,000,000) at its root: a degenerate normalization box no longer NaN-poisons the
  GP fit, and no proposal can escape the observed range. Locked with a regression test.
- +14 tests (`tests/test_hardening.py`): oversized/wrong-magic-bytes/too-many-columns rejections,
  provenance on a messy sheet (units-in-cells, %-strings, a duplicate column, a constant column),
  error-hygiene (malformed bytes -> generic 400, no stack trace/path in the body), bounds-sanity,
  and seed reproducibility. 32 passing, 1 skipped (the ESM-2 test stays behind its flag).

## 2026-07-02

### Added - conformal band + honest reliability in the analyze output
`_analyze` (and therefore `/api/run` and `/api/latest`) now also returns:
- `conformal_q`: a distribution-free +/- half-width from the pooled out-of-fold residuals
  (`core/conformal.q_from_residuals`, alpha=0.1). An honest, often-wider alternative to the
  surrogate's own posterior std, which tends to be overconfident on small bioprocess datasets.
  Coverage is exact for iid split-conformal; treat it as approximate under grouped CV.
- `reliability`: `{spearman, ci95, spearman_floor: 0.20, clears_floor, ci_excludes_zero,
  unmodeled}`. Only what this path can actually assess; the spearman floor mirrors
  `GatesConfig.min_spearman`. It deliberately does NOT assert feasibility or calibration gates
  (listed under `unmodeled`), which this path does not measure. Consumed by the kalos-web
  The older "Phase 2 (live)" view so the UI shows a truthful uncertainty band and trust signal
  instead of a fabricated success probability or scale-up curve.

### Added - /api/latest: the Overview runs on real data, not the demo
The portal persists the most recent uploaded analysis (in-memory, plus best-effort JSON at
`$KALOS_STATE_DIR/latest_analysis.json`, default `~/.kalos`) on every `/api/run`, and serves it
at `GET /api/latest` (`{has_data, dataset, updated, ...analysis}`). This lets the kalos-web
Overview reflect the last dataset a user actually uploaded - real reliability, drivers, and
proposed experiments - instead of the synthetic `_titer`/`_purity` demo objective. `has_data`
is `false` until the first upload, so the home shows an upload prompt rather than pretending
there is data. A serialization or disk error can never fail an upload (persistence is
best-effort). +1 test.

### Fixed - portal upload path routed through the honest CV kit
The `/api/run` analyzer had its own inline CV loop and grouping. Rewired it to the one
leakage-checked path from `core`:
- Grouping now uses `row_hash_groups` on the RAW (NaN-preserving) feature values, so rows
  missing different components are not merged into one replicate group by the zero-fill.
- CV now uses `grouped_cv_report`, so the portal returns a pooled out-of-fold Spearman with a
  group-bootstrap 95% CI (`cv_ci95`) and the effective group count (`cv_n_groups`) - the honest
  signal, not a bare point estimate.
- Added the portal's first test (`tests/test_portal.py`): other measured outputs stay excluded
  from features (anti-leakage) and the honest CV fields are returned. On the raw client upload
  the honest number is -0.23 [-0.42, 0.10] (23 features / 21 groups) - no reliable ranking at
  that feature/group ratio, the truthful "reduce features / collect more data" message rather
  than a fake positive.

### Fixed - leakage + honesty in grouped cross-validation (independent Codex + multi-agent review)
An independent review (OpenAI Codex plus a multi-agent modeling audit) found the reported CV
skill could be both contaminated and overstated at small n. Fixed the confirmed items:
- **NaN grouping leak** (`core/splits.py`): `row_hash_groups` collapsed missing values to 0.0,
  so rows missing different features hashed into one replicate group and leaked across every
  fold. NaN now gets a distinct token, so a real 0.0 and a missing value are different groups.
- **Degenerate split removed** (`core/splits.py`): `make_splits` returned a train==validation
  dummy when too few groups remained. It now returns no split (CV unavailable), so a model is
  never scored against itself.
- **One splitter** (`core/evaluation.py`): deleted the weaker round-robin `grouped_folds` /
  `_row_groups`; all grouped CV routes through the single leakage-checked `splits.make_splits`
  (GroupKFold + a no-overlap assertion). `grouped_folds` stays as a thin compatible wrapper.
- **Train/serve normalization consistency**: the CV loop fits every fold under one fixed
  normalization box (design bounds, else the observed range), matching the deployed model
  instead of each fold's own training envelope.
- **Honest reporting** (`core/evaluation.py`): new `grouped_cv_report` pools out-of-fold
  predictions and returns a group-level bootstrap 95% CI plus `n_oof` / `n_groups` / `n_folds`.
  `grouped_cv_spearman` now pools OOF too (was: a mean of tiny per-fold rhos that can only be
  +/-1 at this n).
- 16 tests pass (3 new regression tests: NaN-distinct grouping, categorical grouping, no dummy
  split); the ESM-2 model test stays skipped behind its flag.

## Older entries

Entries dated before 2026-07-01 are in [`docs/CHANGELOG_ARCHIVE.md`](docs/CHANGELOG_ARCHIVE.md), moved there unchanged and in the same order.
