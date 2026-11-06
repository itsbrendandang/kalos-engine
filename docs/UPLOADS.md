# Uploading a run sheet (`POST /api/run`)

The engine accepts an uploaded CSV / TSV / Excel run sheet and returns the analysis: auto-detected target, leakage-safe features, honest grouped-CV, signed drivers, and a proposed next batch.
Because the sheet comes from an external client, the upload path is guarded:

- **Size + shape caps.**
  A raw-byte cap (default 25 MB, override with `KALOS_MAX_UPLOAD_MB`), a 512-column ceiling, a 100k-row CSV cap, a 2M-cell xlsx cap, and a 64 MiB cap on an xlsx's total uncompressed size (zip-bomb guards).
  Every cap fails closed: an over-cap upload is REJECTED with a 400, never silently truncated.
  The xlsx cell cap is enforced from the workbook's declared dimensions BEFORE the frame is materialized, so a zip-bomb is rejected without the memory spike.
  Because a sheet can under-declare its dimensions, the total uncompressed size of every part of the workbook (from the zip directory; Python's zipfile never yields more than an entry declares) is also checked before anything is parsed. That bounds the bytes parsed, but not the cells: a sparse sheet (one cell far to the right on every row) is small on disk and expands to rows x columns in pandas. So every sheet part (found by its root element, whatever it is named) is also streamed once through expat before pandas sees the workbook, tracking the largest row and column any `<row>`/`<c>` actually uses; a sheet whose real extent passes 512 columns or 2M cells is refused at that point, after reading only as much XML as it took to cross the cap. The materialized frame is re-checked against the cell cap after parsing too.
  When the engine searches a multi-sheet workbook for its first usable sheet, every sheet it tries is charged against one shared 2M-cell budget: a parsed sheet by its real size, a sheet that fails mid-parse by the most cells its XML could hold. At most 20 sheets are tried.
  Column names are always strings (an Excel header cell holding the number `1` is the column `"1"`, so `target=1` finds it).
  Filetype is sniffed by magic bytes first (`PK\x03\x04` -> Excel, known binary signatures -> refused, else UTF-8 or BOM-marked UTF-16 text/CSV); an xlsx is read as Excel whatever its name, and a few clearly non-tabular extensions (`.pdf`, `.json`, `.docx`, ...) are refused by name.
- **Specific, safe errors.**
  A bad upload returns HTTP 400 with `{"error": "<what is wrong and what to do>", "code": "<machine code>", ...context}`; `error` is always present, so a client that only shows `error` keeps working.
  Messages are engine-authored: parser text, exception text, paths, cell values and stack traces are never returned. Server-side, an unexpected failure is logged as its exception type and the innermost source locations plus an error code, never the exception's message (pandas and openpyxl messages quote cell values and column names). The only user content a message may name is a column or sheet name the uploader supplied; such names are clipped to 100 characters, and name lists (in the message and in `available_targets`, `sheets`) to 20 entries, with `<key>_total` (e.g. `sheets_total`) carrying the full count when a list was cut. Server logs record a rejection's `code` and counts only, never the names. With `anonymize=true`, identifier-type column names in an error (`target`, `available_targets` and the message) get the same `col_<hash>` pseudonyms a successful response uses; sheet names are not pseudonymized (a successful response returns them too).

  | `code` | When | Extra fields |
  | --- | --- | --- |
  | `empty_file` | zero bytes, or only whitespace / a BOM | |
  | `unsupported_file_type` | PDF, image, archive, Word/PowerPoint zip, legacy `.xls` or password-protected (OLE2) workbook, binary bytes, or a non-tabular extension (`.json`, `.pdf`, ...) | |
  | `corrupt_excel` | a zip/xlsx that cannot be opened, or a sheet that cannot be read | `sheets`, `sheet` when known |
  | `not_a_table` | text whose lines have inconsistent numbers of comma/tab fields | |
  | `too_few_columns` | fewer than 2 columns (e.g. prose in a `.txt`, a one-column CSV) | `n_columns` |
  | `too_few_rows` | fewer than 6 data rows, or fewer than 6 rows with a numeric target (even if those few values are identical) | `min_rows`, `n_rows`, `target`; `rows_as_written` when the sheet was read as transposed |
  | `no_numeric_inputs` | no numeric column at all | |
  | `no_process_inputs` | numeric columns exist, but all are the target, ids, other outputs, or constant | |
  | `target_not_found` | the requested `target` is not a column | `target`, `available_targets` |
  | `target_empty` / `target_not_numeric` / `target_constant` | the target has no values / no numbers / does not vary | `target`, `available_targets` (+ `target_summary`) |
  | `sheet_not_found` | the requested `sheet` does not exist | `sheet`, `sheets`, `sheet_used: null` |
  | `no_usable_sheet` | no sheet of a multi-sheet workbook holds a usable table | `sheets`, `sheet_used: null` |
  | `ambiguous_header` | the first two rows are mostly blank or group labels (merged cells), so the column names cannot be identified. Refused rather than guessed. | `columns`: the columns detected (positional names such as `column 3` for unnamed cells; never pandas' `Unnamed: N`) |
  | `too_few_complete_rows` | at least 6 numeric targets, but fewer than 6 runs remain after the runs with a missing, non-numeric or censored cell are excluded | `min_rows`, `n_rows`, `n_dropped_incomplete`, `n_censored_excluded` |
  | `payload_too_large` | the file is over `KALOS_MAX_UPLOAD_MB` (25 MB by default), or an xlsx's parts add up to more than the uncompressed-size cap | |
  | `too_many_rows` | a CSV/TSV has more than 100,000 data rows (blank lines are not rows; a quoted newline is one row). Refused before the file is parsed. | `max_rows` |
  | `too_many_columns` | more than 512 columns | |
  | `too_many_cells` | an xlsx sheet has more than 2,000,000 cells (rows x columns) | |
  | `too_many_fit_rows` | more complete runs than the exact-GP fit cap (`KALOS_MAX_FIT_ROWS`, default 2000) after incomplete and censored rows are dropped. Refused, not subsampled. | `max_fit_rows`, `n_rows` |
  | `parse_error` / `fit_failed` | anything else (generic message), or a numerically intractable fit | `sheets`/`sheet_used` for Excel |

  CSV/TSV rows are counted by a streaming pass before pandas parses the body, and the header row is decided from that pass, so an over-cap file is refused without the memory cost of a full parse. Measured on a 4.2 M-row, 16 MB body: the refusal raised resident memory by about 230 MB before this change and by none after it (the process high-water mark does not move).

  The size and shape codes above are stable: a client can branch on them. Their messages are engine-authored and may change. The engine's proxy maps `payload_too_large` to a plain upload-limit message; the other codes show the engine's own sentence.
- **Header row.**
  The column names are read from row 1. When row 1 is mostly blank, or is a merged-cell group label ("Process", "Analytics" above the real headers), the engine reads the names from row 2 instead and records `header_row: 1`, with an `info` finding. A row of text labels over a complete header row is treated the same way.
  If the sheet is ambiguous (a group row with a few gaps, a units row under a partly blank header, or no usable names), the upload is refused with `ambiguous_header`, which lists the detected columns. A single data column with no header name is named by its position (`column 4`) and flagged with a `warning` finding (`unnamed_column`), so `Unnamed: N` never reaches a client.
- **Orientation.**
  A sheet is read as transposed (one run per column) only when its first column reads like parameter names. A first column whose labels repeat (a factor such as `condition`) is never read as transposed, and neither is a sheet with fewer than 3 value columns and at least 6 rows. If a transposed sheet still has too few runs, the `too_few_rows` message gives the run count and the sheet's rows and columns as written.
- **Workbooks and sheets.**
  `POST /api/run` accepts an optional `sheet` (multipart form field, or `?sheet=` query parameter; the form field wins) naming the sheet to analyze; an exact name, else a unique case- and whitespace-insensitive match.
  Without it, the engine analyzes the FIRST sheet in workbook order that holds a usable table: at least 2 columns, at least 6 data rows, and a numeric column. A workbook whose first sheet is the data therefore behaves exactly as before; a workbook with a notes sheet in front of the data now finds the data.
  A single-sheet workbook is always analyzed, so its error is the specific one; a multi-sheet workbook with no usable sheet is refused with `no_usable_sheet`.
  Every Excel response, success or error, carries `sheets` (all names, workbook order) and `sheet_used` (the analyzed sheet, or `null`). When an analyzed sheet of a multi-sheet workbook is unusable, the message names it and lists the others. CSV/TSV responses carry neither key, and `sheet` is ignored for them.
- **Provenance.**
  The response includes a `provenance` list: for every column, its status (`kept_feature`, `target`, `dropped_id`, `dropped_output`, `dropped_constant`, `dropped_constant_on_fitted_rows`, `dropped_sparse`, `dropped_all_blank`) and how many cells had to be coerced from non-numeric text (units like "34.6 C").
  A feature that varies over the full sheet but is constant on the target-present rows the GP actually fits is dropped and flagged (`dropped_constant_on_fitted_rows`), never silently pinned to a zero-width bound.
  Each entry also has `role_inferred` (`target`, `input`, `identifier`, `outcome`, `group`, `fixed_input`, `unused`) and a short `reason` code (`declared`, `requested_target`, `numeric_input`, `headerless`, `monotone_row_number`, `after_target_in_sheet`, `outcome_like_name`, `name_matches_target_candidate`, `concentration_before_target`, `recipe_label`, `not_declared`, ...). The rules are in `kalos/portal/analysis_parts/roles.py`.
- **Column-role safety.**
  These rules apply unless the column is declared in `roles.features`.
  - A column with no header that holds unique whole numbers, one per row, is a saved row index. It is not an input (`dropped_id`, reason `headerless`).
  - A named column of 8 or more rows of unique whole numbers that run up or down in sheet order from any start (0, 1, 2, ...; 1, 2, 3, ...; Excel's 2, 3, 4, ...; n, n-1, ...), with at most a few numbers missing (at least 90% of the range present), is a row number. It is not an input (`dropped_id`, reason `monotone_row_number`).
  - A column named with an outcome word (titer, yield, product, purity, recovery, activity, response, endpoint, harvest, viability at end, or `final` without a concentration unit) is not an input, wherever it sits (`dropped_output`, reason `outcome_like_name`).
  - A column after the target whose name reads like a measurement taken during the run (OD, optical density, biomass, VCD, viability, DCW, lactate, ammonia, residual, measured, readout) is not an input (reason `after_target_in_sheet`).
  - Before the target, those measurement words alone do not exclude a column. A concentration unit (`_g_L`, `_mM`, `_mg_mL`, `conc`, `(M)`) is not evidence of an outcome: Candida's `ammonium_g_L` (the nitrogen source) and scFv's `Final Urea (M)` stay inputs. Such a column keeps `reason` `possibly_measured` with an `info` finding `possibly_measured_after_run`: check it, and untick it if it is a result.
  - A column before the target whose only outcome-like part is a concentration word (`Carbon_conc`) is kept as an input (reason `concentration_before_target`, `info` finding), whether or not the target was named.
  - Inputs with plain names (pH, temperature) stay inputs wherever they sit. To use an excluded column as an input, declare it in `roles.features`: a declared role always wins. Every excluded or checked column is listed in `columns[]` with its `reason` and `finding_code`.
  - Each exclusion has a `warning` finding (`row_number_column`, `after_target_in_sheet`, `outcome_like_name`). A declared input whose name reads like an outcome gets `feature_looks_outcome_like`. An input after the target with |Spearman| >= 0.9 to it gets `after_target_high_correlation`. Those two only warn; the input stays.
- **Every drop is counted and reported.**
  Each cell the model reads is classified: a number; blank; a null token (`n/a`, `NA`, `nan`, `null`, `-`, `?`); a censored value (`<LOD`, `>100`, `<x`, `>x`, `≤x`, `BLQ`, `LOQ`); or any other non-number (`ERR`).
  - A **missing or non-numeric** cell in the target or in a fitted feature (or a categorical with no level) **excludes its run** from the fit. Nothing is imputed.
  - A **censored** value **excludes its run** from the fit and is counted separately. Nothing is imputed.
  - A **blank** feature cell is UNKNOWN, not 0. The run is excluded (`cells_by_column` counts it as `missing`) and nothing is imputed. Only two kinds of column read a blank as absent (0): a column declared in `roles.sparse`, and a concentration-type column by name (`Glucose_g_L`, `Methanol`, a salt or titer-like analyte). For those the run stays in the fit and the count is reported under `blank_cells_as_absent`. The design box is built from observed cells only; for an absent-zero column it is widened to include 0 when blanks were read as 0. (Review C1: before, every blank became 0.0 and the box widened to include it, so pH proposals landed at 0.0.)
  The result carries `n_dropped_incomplete` (runs excluded for missing or non-numeric cells), `n_censored_excluded` (runs excluded for censoring), and `data_drops` with `cells_by_reason`, `cells_by_column` (per column, per reason), and `blank_cells_as_absent`. Each is also a finding: `dropped_rows`, `dropped_cells` (per column), `censored_rows`, `censored_values` (per column, "excluded, not imputed"), and `blank_as_absent`. The run record carries the same counts (`run.data_drops`).
  A numeric column still needs at least 80% of its non-blank cells to parse, and null tokens count against that gate. A column that fails it is not a feature (`dropped_non_numeric` when declared, otherwise `dropped_sparse`).
- **Text factors are categorical features (no declared roles).**
  Without `roles`, a text column with 2 to 12 distinct values, each repeated (at least two runs per value on average), that holds no numeric-looking cells, is modeled as a categorical feature. It is listed in `auto_categorical` and `categorical_features`, and proposals vary it. Each one has an `auto_categorical` finding that says how to confirm or override it: declare it in `roles.categoricals`. Declaring any `roles` turns auto-detection off, so declare every column the model should use. Identifiers and outcome-like names are never auto-categorical, and neither is a high-cardinality text column such as a per-run note.
- **Direction: maximize or minimize.**
  The target is maximized by default. To minimize it (an impurity, a cost), send `"direction": "minimize"` in `roles`, or the plain form field `direction=minimize` on `POST /api/run`. Any other value is a `400` with code `invalid_direction`. Two different stated directions (roles and form field) are a `400` with code `direction_conflict`.
  When minimizing, the proposals look for low values and `best` is the lowest measured value. Predictions, drivers and `gp_shapes` stay in the original units and sign. The P(feasible) gate is not applied, because it scores y > 0; `proposal_gating.off_for_minimize` is true and `proposal_gating.reason` says why. `cv_topk` compares the true and predicted lowest k runs. In `promotion`, the `feasibility_auc`, `ece` and `brier` gates are listed in `not_judged` as "not applicable when minimizing".
  The response carries `direction`. The campaign stores it, so a reanalyze uses the same direction, and the campaign summary reports it. The run record has `run.direction`, and it is part of `proposal_provenance.config_sha256`.
- **Data validation gate.**
  Before any column is typed or dropped, the sheet runs through `kalos.validation`: eleven checks covering unit consistency, physical bounds, duplicates vs replicates, missingness, informative missingness, outliers, provenance metadata, replicate adequacy, controls presence, constant columns, and columns constant within a group.
  The report is returned as `validation`, with `status` one of `pass`, `pass_with_warnings`, or `fail`.
  An **error** is a physics violation (pH 40, a negative titer, one column mixing g/L and mg/mL); a **warning** is possible but operationally suspect.
  Row lists are capped at 20 with the true total in `detail.n_rows_affected`, so a finding never implies its list is complete.
  `KALOS_VALIDATION_MODE=strict` refuses a `fail` upload; the default `warn` analyzes it anyway and returns the findings.
- **A target that does not vary is refused.**
  If the target column has one value on every row, or values that differ only by floating-point rounding (`range <= 1e-9 * max|value|`, see `kalos/validation/target.py`), no model can learn from it.
  The upload (and a campaign re-analyze) is refused with a 400 before any model is fit, and no proposals are generated: `{"error": "...", "code": "target_constant", "target": "...", "target_summary": {...}, "available_targets": [...]}`, where `available_targets` lists other outcome columns that do vary.
  A target with no values or no numeric values gets `target_empty` / `target_not_numeric`; a date or duration column is `target_not_numeric`. A target with fewer than 6 numeric values is `too_few_rows`, not `target_constant`.
- **Units are converted, not discarded.**
  Single-unit columns such as `"34.6 C"` are converted to their base unit before feature selection and reported in `validation.conversions` with a human label (`"Celsius"`), instead of failing the numeric test and coming back `dropped_sparse`.
  Only units the registry can actually convert are rewritten: a vessel column of `"5L"`/`"500L"` uses an unrecognized token, and rewriting it would turn an identifier into a measurement, so it is left alone and the client is told.
- **Proposals stay physically possible.**
  The design box for each continuous feature is built from physically valid observations only, so one `-999` sensor sentinel cannot widen the search space into impossible recipes (it once produced proposals at -422 C, below absolute zero).
  Any narrowing is reported in `design_box_exclusions`, never silent, in every validation mode.
  A coded column is not checked against a physical range. It is coded when its name says so (`code`, `coded`, `level`, `scaled`, `normalised`, `pseudonymised`, `z-score`), or when its name matched only a weak hint (`feed`, `flow`, `rate`, `time`, `day`), carries no unit in the name or the cells, and has a value below the physical minimum (a negative feed). It gets an `info` finding with code `coded_column`, and its observed range is its search range. A column with a unit (`feed_mL_h`, cells such as "0.45 mL/h") is always checked. The rule is `kalos.validation.bounds.is_coded_column`.
- **Near-constant inputs are fixed, not searched.**
  An input whose spread on the fitted runs is within sensor precision is fixed at its median. With a known precision (pH 0.05 units, temperature 0.1 C, percent 0.5 points) the range must be at most that and at most 1% of the median. Any other input needs a range below 0.1% of its median. A median of 0 never counts. The input leaves the model and the search box. Each proposal's `recipe` has the fixed value, `fixed_inputs` lists them, provenance says `fixed_near_constant`, and an `info` finding has code `near_constant_input`. The rule is in `kalos/portal/analysis_parts/near_constant.py`.
- **A batch never repeats a recipe.**
  No two proposals in one batch are within `NEAR_DUPLICATE_TOL` (5%) of each other, nor of any run already in flight (a started, unmeasured campaign run): same level of every categorical factor and every continuous factor within 5% of its design range (`kalos.core.optimize.proposal_separation`). A jointly optimized row that repeats an in-flight run is replaced like a collapsed one.
  A jointly optimized batch that collapsed (typically two points on the same box corner when the optimum is on the edge of the design) is refilled by sequential, pending-aware selection from a candidate pool, not by random jitter.
  When the design space has fewer distinct candidates than requested (e.g. a small all-categorical design), the batch comes back short and `proposal_notice` says so: `{"code": "fewer_distinct_candidates", "requested": 5, "returned": 4, "min_separation": 0.05, "message": "..."}`; otherwise it is `null`.
  `batch_diversity` reports `requested`, `returned`, `near_duplicates_replaced`, `min_separation`, and `metric`.
- **Reproducibility.**
  The analyze path is seeded, so the same upload yields identical proposals; the response carries `seed`, `timestamp`, and `engine_version`.
  Protein embeddings pin the ESM-2 checkpoint to an immutable commit, so upstream cannot change model features without a diff here.
- **Identifiers versus measurements.**
  A `sample` column is an identifier only when it is a sample ID, code, name or number (`sample_id`, `Sample Code`), or a bare `sample`. Measured quantities that begin with the word, such as `sample_volume_L` or `sample_mass_g`, are process features and are modeled. `Tank ID`, `Patient ID` and similar compound identifiers remain identifiers.
- **Opt-in anonymization.**
  Pass the form field `anonymize=true` to strip client/strain identity and hash grouping keys (`kalos/data/anonymizer.py`), so a run stays groupable for leakage-safe CV without carrying who it belongs to.
  Feature and target names are kept as-is (the owner UI legitimately shows drivers like "Methanol").
  Identifier column names (`Tank ID`, `Customer Name`, `run_number`) and sheet names are aliased; the stored campaign still holds the raw sheet. Full scope: docs/HARDENING.md, "What `anonymize=true` covers".

See also [`docs/CAMPAIGN_LOOP.md`](CAMPAIGN_LOOP.md) and [`docs/SCALE_READOUT.md`](SCALE_READOUT.md).
