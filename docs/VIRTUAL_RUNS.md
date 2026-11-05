# Virtual runs: simulated and synthetic data, kept apart

Kalos can generate runs that were never done in the lab. They are useful for
demos, for testing the optimizer against a surface whose best point is known, and
(later) for comparing a process model with what the lab measures. They are not
evidence about a real process. This page says how Kalos keeps them apart from lab
measurements.

This is phase 1 of the plan in the research notes. There is no calibrated process
model (twin) yet. Every generator that ships today produces **synthetic** data.

## Three origins

Every row of data has one origin. It is a fixed type, not a free-text tag
(`kalos/twin/origin.py`).

| Origin | What it is | Where it lives |
|---|---|---|
| `measured` | Entered or uploaded from the lab by a named user. | campaigns, `run_results` |
| `simulated` | Output of a calibrated process model (twin) version for a design. No generator makes this yet. A simulated row must name its twin version. | `virtual_runs` |
| `synthetic` | Output of a generator that is not tied to the lab's process (benchmark surfaces, demo data, test fixtures). A synthetic row never names a twin version. | `virtual_runs` |

When data of several origins is combined, the result takes the "highest" one:
synthetic, then simulated, then measured.

## Generators

A generator is a seeded function that returns a table of runs. Each one is
registered with an id, a version, its origin, its default parameters, and the
source files that make up its code hash (`kalos/twin/generators.py`).

Built in (`kalos/twin/builtin.py`, all synthetic; the wrapped files are not changed):

| Id | Wraps | Design | Outcome |
|---|---|---|---|
| `bench.gaussian_bump` | `kalos/bench/objectives.py` | `x1..x<dim>` in [0, 1] | `y` (with noise), `y_true` |
| `bench.ackley` | `kalos/bench/objectives.py` | `x1..x<dim>` in [-2, 2] | `y`, `y_true` |
| `bench.mixed_bump` | `kalos/bench/objectives.py` | numeric `x*`, one `level1` | `y`, `y_true` |
| `bench.bioprocess_like` | `kalos/bench/objectives.py` | 3 numeric, strain, inducer | `y`, `y_true` |
| `bench.fed_batch` | `kalos/bench/mechanistic_benchmark.py` (`fed_batch`) | `S0_g_per_L`, `feed_L_per_h` | `product_g`, `product_g_true` |
| `examples.synthetic_scaleup` | `examples/synthetic_scaleup/make_synthetic.py` | scale, agitation, airflow, pH, temperature | `titer_g_per_L` |

`examples.synthetic_scaleup` exists only in a source checkout, because the
`examples/` folder is not packaged. `GET /api/virtual-runs/generators` reports
`available: false` elsewhere.

### Adding a generator

A module adds itself to the registry. No change to the registry file is needed:

```python
from kalos.twin.generators import register_generator

@register_generator("bench.my_surface", version="1", origin="synthetic",
                    description="...", defaults={"noise": 0.1},
                    outcome_columns=("y",))
def _generate(params, rng, n):
    ...  # return a DataFrame with n rows; use only `rng` for randomness
```

Discovery imports every module in `kalos/bench/` whose source contains the text
`register_generator`. Modules that do not register are never imported, so listing
generators does not load torch. Parameters a caller sends are checked against
`defaults`: unknown names are refused, and each value must have the type of its
default.

## Seeds and replay

Each request makes one batch with a new id (`vjb_` + 32 hex). The seed is
`numpy.random.SeedSequence([tenant, campaign, twin version, batch id])`. Text
parts are turned into numbers with SHA-256; a missing twin version is 0. The
generator gets one PCG64 stream from that seed. Row `k` is the k-th row of the
batch.

Every stored row keeps what is needed to make it again:

- generator id and version (`<version>+<first 12 hex of the code hash>`);
- the parameters as JSON and their SHA-256;
- the seed entropy and the initial PCG64 state;
- the solver settings (for the ODE: LSODA, scipy's default tolerances, 50 output
  points, 48 h);
- the Python, numpy, scipy, pandas and Kalos versions;
- the design, its SHA-256, and the outcome;
- who asked (`requested_by`; `anonymous` in open mode), how they signed in, and
  when.

The same inputs on the same build give the same rows
(`tests/test_twin_generators.py`, `tests/test_virtual_runs_api.py`). A different
build of numpy or scipy can give different numbers; the stored versions say which
build made a row.

## Storage (schema v6)

`virtual_runs` is a separate table. Measured data never goes in it, and virtual
data never goes in the campaign tables.

- A CHECK allows `origin` to be only `simulated` or `synthetic`.
- A CHECK requires a twin version on simulated rows and forbids it on synthetic rows.
- `(campaign_id, tenant)` is a foreign key to `campaigns`, so a row can only point
  at a campaign of its own tenant.
- Rows are never changed or removed. Two triggers reject UPDATE and DELETE, the
  same guard as `audit_log` and `run_results`.
- Creating a batch writes a `campaign.virtual_runs.create` audit row in the same
  transaction. If the audit row cannot be written, no rows are stored. The audit
  detail has the batch id, origin, generator, version, row count and parameter
  hash, never the generated values.

Migration and rollback: see docs/MIGRATIONS.md, "Schema v6".

## The analysis refuses virtual rows

Virtual data is marked in two ways: a column named `origin` whose cells say
`simulated` or `synthetic` (case and spaces ignored), and
`DataFrame.attrs["kalos_origin"]`. Arrays from a generator batch (`batch.X`,
`batch.y`) carry a `kalos_origin` attribute.

These entry points refuse marked data unless the caller passes
`allow_origins=[...]` naming the origin:

- the campaign analysis (`kalos.portal.analysis._analyze`), which every upload,
  background job and re-analysis goes through;
- `Surrogate.fit`, `HybridSurrogate.fit`, `MultiObjectiveSurrogate.fit`,
  `FeasibilityClassifier.fit`;
- the scale-up model (`ScaleUpTransferModel.fit`, `PhysicsMeanSurrogate.fit`,
  `MultiFidelitySurrogate.fit`) and the scale-up readout (`build_readout`).

No HTTP route passes `allow_origins`. Over HTTP a refusal is `400` with
`code: virtual_data_refused` (`422` on `/api/scale/readout`, like its other
refusals). When a caller does allow virtual data in Python, the result carries
`data_origin` (the highest origin it read).

What this does not stop: code inside the Python process that turns a marked
table into a plain numpy array, or renames the `origin` column, removes the mark.
A person who deletes the `origin` column from an exported file and uploads it
again also removes the mark. The guard stops accidents and every path the service
offers. It cannot stop someone who removes the label on purpose.

`tests/test_virtual_guard.py` checks the entry points, the HTTP paths, and a
seeded property test: random generators and chains of pandas operations (subsets,
reordering, mixing with measured rows, CSV round trips, case changes) are always
refused.

## HTTP API

All routes act on the caller's own tenant. A malformed, missing or other-tenant
campaign id gets the same `404` as every campaign route.

### `GET /api/virtual-runs/generators` (scope `read`)

The registered generators: id, version, origin, description, defaults, design and
outcome columns, solver settings, and whether it is available in this install.

### `POST /api/campaigns/{id}/virtual-runs` (scope `write`)

```json
{"generator": "bench.fed_batch", "params": {"noise_sd_g": 1.0}, "n": 20}
```

`n` is 1 to 500. Generation runs in the request; the built-in generators take well
under a second for 500 rows. `201` returns the batch: `batch_id`, `origin`,
`label` (for example `SYNTHETIC - not measured`), generator id and version,
parameters and their hash, seed entropy (as strings), PCG64 state, solver,
library versions, `requested_by`, `created_at`, and `runs` (each with `k`,
`origin`, `design`, `outcome`).

Refusals: `400 invalid_generator_request` (unknown generator, bad parameters),
`400` for a bad `n` or body, `409 campaign_archived`. Nothing is stored.

### `GET /api/campaigns/{id}/virtual-runs` (scope `read`)

Query: `origin` (`simulated` or `synthetic`), `batch_id`, `after` (row id),
`limit` (1 to 1000, default 200). Oldest first. Each row has the full record plus
`origin` and `label`. `next_after` is set when there may be more rows.

### `GET /api/campaigns/{id}/virtual-runs/export` (scope `read`)

A CSV file with one origin:

- The file name carries the origin: `kalos-synthetic-virtual-runs-<campaign>.csv`.
- The first lines start with `#` and say the origin, that the rows are not lab
  measurements, and, for each batch, the generator, version, parameter hash, seed
  entropy and twin version.
- The first column is `origin` on every row. Then batch id, `k`, generator id and
  version, twin version, the design columns and the outcome columns.
- The `X-Kalos-Origin` response header repeats the origin.
- A cell that starts with `=`, `+`, `-` or `@` as text gets a leading `'`, so a
  spreadsheet does not run it as a formula.

Refused with `409 mixed_export_refused`: an export that would hold both simulated
and synthetic rows (pass `origin=`), and any request with `include_measured=true`.
Measured data has no export route that could include virtual rows. `404
no_virtual_runs` when nothing matches. Exports and export refusals are written to
the audit log.

An exported file uploaded back to Kalos is refused: as exported, the `#` lines
make the parser refuse it; with those lines removed, the origin guard refuses it.

## Not in phase 1

- No calibrated process model (twin), so no simulated rows from any generator.
- No measured-vs-predicted ledger, no drift view, no recalibration.
- No web screen yet. When it comes, virtual data gets its own tab, a distinct
  look (hatched points or dashed lines) and a text label, never color alone.
- Virtual rows never feed the campaign's optimizer.
- The PostgreSQL backend (branch `postgres-backend`) needs a matching v6 step;
  see docs/MIGRATIONS.md.
