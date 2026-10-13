# Uploading a run sheet (`POST /api/run`)

The engine accepts an uploaded CSV / TSV / Excel run sheet and returns the analysis: auto-detected target, leakage-safe features, honest grouped-CV, signed drivers, and a proposed next batch.
Because the sheet comes from an external client, the upload path is guarded:

- **Size + shape caps.**
  A raw-byte cap (default 25 MB, override with `KALOS_MAX_UPLOAD_MB`), a 512-column ceiling, a 100k-row CSV cap, and a 2M-cell xlsx cap (a zip-bomb guard).
  Every cap fails closed: an over-cap upload is REJECTED with a 400, never silently truncated.
  The xlsx cell cap is enforced from the workbook's declared dimensions BEFORE the frame is materialized, so a zip-bomb is rejected without the memory spike.
  Filetype is sniffed by magic bytes (`PK\x03\x04` -> Excel, else text/CSV), not by extension.
- **Safe errors.**
  A bad upload returns a generic HTTP 400 ("Could not parse the uploaded file. Check it is a CSV or Excel run-sheet.").
  This holds even when the failure is a GP fit error (`torch.linalg.LinAlgError`) or a leakage-guard assertion, not just a parser error: a catch-all normalizes any such failure to the same `{error}` JSON envelope.
  Parser details, column names, cell values, paths, and stack traces are logged server-side and never returned to the caller.
- **Provenance.**
  The response includes a `provenance` list: for every column, its status (`kept_feature`, `target`, `dropped_id`, `dropped_output`, `dropped_constant`, `dropped_constant_on_fitted_rows`, `dropped_sparse`, `dropped_all_blank`) and how many cells had to be coerced from non-numeric text (units like "34.6 C").
  A feature that varies over the full sheet but is constant on the target-present rows the GP actually fits is dropped and flagged (`dropped_constant_on_fitted_rows`), never silently pinned to a zero-width bound.
- **Data validation gate.**
  Before any column is typed or dropped, the sheet runs through `kalos.validation`: eleven checks covering unit consistency, physical bounds, duplicates vs replicates, missingness, informative missingness, outliers, provenance metadata, replicate adequacy, controls presence, constant columns, and columns constant within a group.
  The report is returned as `validation`, with `status` one of `pass`, `pass_with_warnings`, or `fail`.
  An **error** is a physics violation (pH 40, a negative titer, one column mixing g/L and mg/mL); a **warning** is possible but operationally suspect.
  Row lists are capped at 20 with the true total in `detail.n_rows_affected`, so a finding never implies its list is complete.
  `KALOS_VALIDATION_MODE=strict` refuses a `fail` upload; the default `warn` analyzes it anyway and returns the findings.
- **Units are converted, not discarded.**
  Single-unit columns such as `"34.6 C"` are converted to their base unit before feature selection and reported in `validation.conversions` with a human label (`"Celsius"`), instead of failing the numeric test and coming back `dropped_sparse`.
  Only units the registry can actually convert are rewritten: a vessel column of `"5L"`/`"500L"` uses an unrecognized token, and rewriting it would turn an identifier into a measurement, so it is left alone and the client is told.
- **Proposals stay physically possible.**
  The design box for each continuous feature is built from physically valid observations only, so one `-999` sensor sentinel cannot widen the search space into impossible recipes (it once produced proposals at -422 C, below absolute zero).
  Any narrowing is reported in `design_box_exclusions`, never silent, in every validation mode.
- **Reproducibility.**
  The analyze path is seeded, so the same upload yields identical proposals; the response carries `seed`, `timestamp`, and `engine_version`.
  Protein embeddings pin the ESM-2 checkpoint to an immutable commit, so upstream cannot change model features without a diff here.
- **Opt-in anonymization.**
  Pass the form field `anonymize=true` to strip client/strain identity and hash grouping keys (`kalos/data/anonymizer.py`), so a run stays groupable for leakage-safe CV without carrying who it belongs to.
  Feature and target names are kept as-is (the owner UI legitimately shows drivers like "Methanol").

See also [`docs/CAMPAIGN_LOOP.md`](CAMPAIGN_LOOP.md) and [`docs/SCALE_READOUT.md`](SCALE_READOUT.md).
