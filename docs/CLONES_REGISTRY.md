# Clone registry (labels only)

> Draft section for the lead to merge into `docs/CLONES.md` and
> `docs/CAMPAIGNS_API.md`. Code: `kalos/portal/clone_registry.py` (storage),
> `kalos/portal/clones_api.py` (routes). Schema: v7 (`docs/MIGRATIONS.md`).
> Tests: `tests/test_clones_api.py`, `tests/test_schema_v7.py`.

A campaign can keep a small registry of the production clones (or strains) it
runs: a status and a few text labels per clone. Kalos stores, shows and filters
these values. It never parses them, never computes from them except counts and
equality, and never sends them anywhere. **Kalos stores labels only**: no
sequence, plasmid map, structure or database accession is accepted.

Registry edits live outside the campaign state blob, so they never mint a new
campaign generation and never put an analysis out of sync.

## Storage (schema v7)

- `campaign_clones(tenant, campaign_id, label, fields, status, updated_at,
  updated_by)`, primary key `(tenant, campaign_id, label)`. `label` is 1..64
  characters and must equal the run-sheet value exactly (it is not trimmed).
  `status` is one of `lead`, `backup`, `retired`, `candidate`. `fields` is a JSON
  object of the allowlisted keys below.
- `clone_titer_history(id, tenant, campaign_id, label, recorded_at, measured_on,
  basis, value, unit, scale_label, age_kind, age_value, source_note, actor,
  previous_id, reason)`: append-only (UPDATE and DELETE are rejected by triggers).
  A correction is a new row that names the row it supersedes in `previous_id` and
  carries a non-blank `reason`.
- Limits: 200 clones per campaign (`too_many_clones`), 20,000 titer rows per
  campaign (`titer_history_limit`).

## Fields (all optional except `status`)

| Key | Type | Limit | Example |
|---|---|---|---|
| `status` | `lead` \| `backup` \| `retired` \| `candidate` | required | `lead` |
| `host` | string | 64 | `CHO-K1 GS-KO`, `K. phaffii X-33` |
| `parent_cell_bank` | string | 64 | `RCB-2024-03` |
| `construct_id` | string (opaque) | 64 | `pK-mAb7-v3` |
| `phenotype_label` | string | 64 | `MutS`, `MTX 50 nM` |
| `product` | string | 64 | `mAb7 (IgG1)` |
| `passage_at_banking` | finite number or string | 64 | `12`, `P12` |
| `note` | string (newlines allowed) | 500 | free text, shown escaped |

Strings are trimmed; a blank or `null` value means "not set". Control
characters are refused (newline and tab are allowed in `note`).

## Routes

All routes follow the campaign API rules: the scope is checked before the id is
looked at; the tenant is always the caller's; a malformed, missing or another
tenant's campaign id gets the byte-identical `404 {"error": "campaign not
found", "code": "not_found"}`; errors never echo a submitted label, key or value;
an archived campaign answers `409 campaign_archived` on writes; the JSON body cap
is `KALOS_MAX_REQUEST_BYTES` (default 1 MiB, `413 payload_too_large`). Every
write records an audit row (`campaign.clone.put`,
`campaign.clone.titer_append`) with the label, status / basis, the keys sent and
the row id, never field values.

The `{label}` path segment is the URL-encoded clone label: 1..64 characters, not
blank, no control characters (`400 invalid_label` otherwise). A label that
contains `/` cannot be addressed.

### `GET /api/campaigns/{id}/clones` (read)

```jsonc
{
  "campaign_id": "cmp_…",
  "clones": [{"label": "CHO-B07", "status": "lead",
              "fields": {"host": "CHO-K1 GS-KO"}, "updated_at": 1730000000.0,
              "updated_by": "bob@acme"}],
  "seen_in_data": ["CHO-A12", "CHO-B07"],     // strain_summary.levels[].label of the latest analysis
  "unregistered_labels": ["CHO-A12"],         // seen_in_data minus the registry, by name
  "max_clones": 200,
  "seen_in_data_reason": "…"                  // only when seen_in_data is empty because there is
                                              // no analysis, no/unavailable strain_summary, or the
                                              // analysis was anonymized (labels are aliases)
}
```

### `PUT /api/campaigns/{id}/clones/{label}` (write)

Creates or replaces the whole row (omitted keys are cleared). Body: the fields
table above, `extra="forbid"`. `sequence`, `fasta`, `smiles`, `pdb`, `uniprot` and
any other key are refused with `400 invalid_body` ("Kalos stores clone labels
only; …"). Over 200 clones: `409 too_many_clones` (with `max_clones` and
`what_would_fix_it`); editing an existing clone is always allowed.

Response: `{"campaign_id", "clone": {"label", "status", "fields", "updated_at",
"updated_by", "created": bool}}`.

### `POST /api/campaigns/{id}/clones/{label}/titer-history` (write)

Appends one MEASURED value to a registered clone (`404 clone_not_registered`
otherwise).

| Key | Type | Notes |
|---|---|---|
| `basis` | `harvest_titer` \| `qP` \| `space_time_yield` | required |
| `value` | finite number | required |
| `unit` | `g/L` \| `mg/L` \| `pg/cell/day` \| `g/L/day` | required; must match the basis: `harvest_titer` g/L or mg/L, `qP` pg/cell/day, `space_time_yield` g/L/day (`unit_basis_mismatch`) |
| `measured_on` | `YYYY-MM-DD` | optional (`invalid_date`) |
| `scale_label` | string ≤ 64 | optional, e.g. `ambr15` |
| `age_kind` | `passage` \| `generation` \| `pdl` | optional |
| `age_value` | finite number | optional |
| `source_note` | string ≤ 200 | optional |
| `previous_id` | integer ≥ 1 | correction only: an existing row of the same campaign and label (`previous_not_found`) |
| `reason` | string ≤ 500 | required with `previous_id` (`reason_required`); refused without it (`previous_id_required`) |

The actor and `recorded_at` come from the token and the server clock. Response:
`{"campaign_id", "row": {"id", "label", "recorded_at", …the fields…, "actor",
"basis_kind": "measured", "is_correction": bool}}`.

### `GET /api/campaigns/{id}/clones/{label}/titer-history` (read)

Query: `limit` (1..500, default 100), `before_id` (cursor). Newest first, the
same page shape as `/results/history`:

```jsonc
{"campaign_id": "cmp_…", "label": "CHO-B07", "limit": 100,
 "history": {"items": [{"id": 7, "label": "CHO-B07", "recorded_at": …, "measured_on": "2025-03-01",
                        "basis": "harvest_titer", "value": 3.2, "unit": "g/L", "scale_label": "ambr15",
                        "age_kind": "passage", "age_value": 14.0, "source_note": null,
                        "actor": "bob@acme", "previous_id": null, "reason": null,
                        "basis_kind": "measured", "is_correction": false}],
             "has_more": false, "next_before_id": null},
 "notes": "…"}
```

### `GET /api/campaigns/{id}/clones/summary` (read)

The latest stored analysis's `strain_summary`, joined with the registry.

- No analysis: `409 {"code": "no_analysis", "what_would_fix_it": …}`.
- No `strain_summary`, or `available: false`: `200 {"campaign_id", "available":
  false, "reason"}`.
- Anonymized analysis (`run.anonymize` true): the block is returned unchanged
  plus `registry_join: {"applied": false, "reason": "anonymized analysis: labels
  are aliases"}`. No level gets a `registry` key and there is no
  `registry_order` (aliases never match real labels).
- Otherwise: the block plus
  - `levels[i].registry`: `{"status", "fields"}` or `null` (unregistered);
  - `unregistered_labels`: level labels without a registry row, by name;
  - `registry_order`: the level labels ordered lead, backup, candidate, retired
    (by name within a status), then unregistered by name. Never ordered by a
    computed score;
  - `registry_join: {"applied": true, "reason": null}`.

The route is registered before the `/clones/{label}` routes. Only GET exists on
`/clones/summary` and no GET exists on `/clones/{label}`, so it is never read as
a label; a `PUT /clones/summary` registers a clone literally named `summary`.

## Python helpers (`kalos.portal.clone_registry`)

- `list_clones(store, *, tenant, campaign_id) -> list[dict]`
- `get_clone(store, *, tenant, campaign_id, label) -> dict | None`
- `put_clone(store, *, tenant, campaign_id, label, status, fields, updated_by, auth_method=None) -> dict`
- `append_titer(store, *, tenant, campaign_id, label, row, actor, auth_method=None) -> dict`
- `list_titer_history(store, *, tenant, campaign_id, label, before_id=None, limit=100) -> dict`
- `lead_clone_label(store, *, tenant, campaign_id) -> str | None`: the single
  `status='lead'` clone; `None` when there is none or more than one.

All raise `CampaignNotFound` for a missing or another tenant's campaign.
