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
  Messages are engine-authored: parser text, exception text, paths, cell values and stack traces are logged server-side and never returned. The only user content a message may name is a column or sheet name the uploader supplied; such names are clipped to 100 characters, and name lists (in the message and in `available_targets`, `sheets`) to 20 entries, with `<key>_total` (e.g. `sheets_total`) carrying the full count when a list was cut. Server logs record a rejection's `code` and counts only, never the names. With `anonymize=true`, identifier-type column names in an error (`target`, `available_targets` and the message) get the same `col_<hash>` pseudonyms a successful response uses; sheet names are not pseudonymized (a successful response returns them too).

  | `code` | When | Extra fields |
  | --- | --- | --- |
  | `empty_file` | zero bytes, or only whitespace / a BOM | |
  | `unsupported_file_type` | PDF, image, archive, Word/PowerPoint zip, legacy `.xls` or password-protected (OLE2) workbook, binary bytes, or a non-tabular extension (`.json`, `.pdf`, ...) | |
  | `corrupt_excel` | a zip/xlsx that cannot be opened, or a sheet that cannot be read | `sheets`, `sheet` when known |
  | `not_a_table` | text whose lines have inconsistent numbers of comma/tab fields | |
  | `too_few_columns` | fewer than 2 columns (e.g. prose in a `.txt`, a one-column CSV) | `n_columns` |
  | `too_few_rows` | fewer than 6 data rows, or fewer than 6 rows with a numeric target (even if those few values are identical) | `min_rows`, `n_rows`, `target` |
  | `no_numeric_inputs` | no numeric column at all | |
  | `no_process_inputs` | numeric columns exist, but all are the target, ids, other outputs, or constant | |
  | `target_not_found` | the requested `target` is not a column | `target`, `available_targets` |
  | `target_empty` / `target_not_numeric` / `target_constant` | the target has no values / no numbers / does not vary | `target`, `available_targets` (+ `target_summary`) |
  | `sheet_not_found` | the requested `sheet` does not exist | `sheet`, `sheets`, `sheet_used: null` |
  | `no_usable_sheet` | no sheet of a multi-sheet workbook holds a usable table | `sheets`, `sheet_used: null` |
  | `parse_error` / `fit_failed` | anything else (generic message), or a numerically intractable fit | `sheets`/`sheet_used` for Excel |

  A size or shape cap (`too large`, `too many columns`) keeps its existing message.
- **Workbooks and sheets.**
  `POST /api/run` accepts an optional `sheet` (multipart form field, or `?sheet=` query parameter; the form field wins) naming the sheet to analyze; an exact name, else a unique case- and whitespace-insensitive match.
  Without it, the engine analyzes the FIRST sheet in workbook order that holds a usable table: at least 2 columns, at least 6 data rows, and a numeric column. A workbook whose first sheet is the data therefore behaves exactly as before; a workbook with a notes sheet in front of the data now finds the data.
  A single-sheet workbook is always analyzed, so its error is the specific one; a multi-sheet workbook with no usable sheet is refused with `no_usable_sheet`.
  Every Excel response, success or error, carries `sheets` (all names, workbook order) and `sheet_used` (the analyzed sheet, or `null`). When an analyzed sheet of a multi-sheet workbook is unusable, the message names it and lists the others. CSV/TSV responses carry neither key, and `sheet` is ignored for them.
- **Provenance.**
  The response includes a `provenance` list: for every column, its status (`kept_feature`, `target`, `dropped_id`, `dropped_output`, `dropped_constant`, `dropped_constant_on_fitted_rows`, `dropped_sparse`, `dropped_all_blank`) and how many cells had to be coerced from non-numeric text (units like "34.6 C").
  A feature that varies over the full sheet but is constant on the target-present rows the GP actually fits is dropped and flagged (`dropped_constant_on_fitted_rows`), never silently pinned to a zero-width bound.
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
- **A batch never repeats a recipe.**
  No two proposals in one batch are within `NEAR_DUPLICATE_TOL` (5%) of each other, nor of any run already in flight (a started, unmeasured campaign run): same level of every categorical factor and every continuous factor within 5% of its design range (`kalos.core.optimize.proposal_separation`). A jointly optimized row that repeats an in-flight run is replaced like a collapsed one.
  A jointly optimized batch that collapsed (typically two points on the same box corner when the optimum is on the edge of the design) is refilled by sequential, pending-aware selection from a candidate pool, not by random jitter.
  When the design space has fewer distinct candidates than requested (e.g. a small all-categorical design), the batch comes back short and `proposal_notice` says so: `{"code": "fewer_distinct_candidates", "requested": 5, "returned": 4, "min_separation": 0.05, "message": "..."}`; otherwise it is `null`.
  `batch_diversity` reports `requested`, `returned`, `near_duplicates_replaced`, `min_separation`, and `metric`.
- **Reproducibility.**
  The analyze path is seeded, so the same upload yields identical proposals; the response carries `seed`, `timestamp`, and `engine_version`.
  Protein embeddings pin the ESM-2 checkpoint to an immutable commit, so upstream cannot change model features without a diff here.
- **Opt-in anonymization.**
  Pass the form field `anonymize=true` to strip client/strain identity and hash grouping keys (`kalos/data/anonymizer.py`), so a run stays groupable for leakage-safe CV without carrying who it belongs to.
  Feature and target names are kept as-is (the owner UI legitimately shows drivers like "Methanol").

See also [`docs/CAMPAIGN_LOOP.md`](CAMPAIGN_LOOP.md) and [`docs/SCALE_READOUT.md`](SCALE_READOUT.md).
