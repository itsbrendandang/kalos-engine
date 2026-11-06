"""One plain why-line per code: what it means for the lab's work.

Mirrors the why-lines kalos-web shows (the `words` glossary,
/tmp/redesign/words/GLOSSARY.md while it is being written), so the engine and
the web say the same thing. Rules for every line: no statistics or software
words, one idea per sentence, no claim stronger than the record.

`tests/test_assist_why.py` fails when the engine gains a refusal code, a
finding check, or a model check that has no line here.
"""
from __future__ import annotations

# Refusals: the engine did not fit a model to the sheet.
REFUSALS: dict[str, str] = {
    "too_many_fit_rows": "Kalos fits its model to a limited number of runs, so a larger sheet has to be split or trimmed first.",
    "too_few_rows": "With this few runs there is not enough to learn from, so no batch can be proposed yet.",
    "too_few_complete_rows": "Runs with blank or unreadable cells are left out, and too few complete runs were left to learn from.",
    "too_few_columns": "Kalos needs at least one setting you changed and one result you measured.",
    "target_empty": "The result column has no numbers in it, so there is nothing to learn from.",
    "target_not_found": "Kalos could not find the result column you named, so it does not know what to improve.",
    "target_not_numeric": "The result column holds text, and Kalos can only learn from measured numbers.",
    "target_constant": "Every run has the same result, so there is no difference between runs to learn from.",
    "no_numeric_inputs": "No setting column holds numbers, so Kalos cannot tell which settings matter.",
    "no_process_inputs": "No column looks like a setting you control, so Kalos has nothing to change in the next batch.",
    "fit_failed": "The numbers in the sheet did not let Kalos build a model; checking for repeated or extreme rows usually helps.",
    "parse_error": "Kalos could not read the file as a table of runs.",
    "empty_file": "The file has no rows of data.",
    "not_a_table": "The file does not look like a table with one run per row.",
    "too_many_rows": "The file has more rows than Kalos reads in one upload.",
    "too_many_columns": "The file has more columns than Kalos reads in one upload.",
    "too_many_cells": "The file is larger than Kalos reads in one upload.",
    "payload_too_large": "The file is larger than Kalos accepts.",
    "unsupported_file_type": "Kalos reads CSV and Excel run sheets only.",
    "corrupt_excel": "The Excel file could not be opened; saving it again from Excel usually fixes this.",
    "ambiguous_header": "Kalos could not tell which row holds the column names.",
    "sheet_not_found": "The workbook has no sheet with the name you chose.",
    "no_usable_sheet": "No sheet in the workbook holds a table of runs.",
    "interrupted": "The analysis stopped before it finished, so nothing was saved; running it again is safe.",
    "engine_stopped": "Kalos was closed while the analysis ran, so nothing was saved; running it again is safe.",
    "read_timeout": "The file took too long to arrive, so it was not read.",
    "invalid_content_length": "The upload arrived incomplete, so it was not read.",
}

# Findings on a sheet that was analysed. Keyed by the finding's `code` (or its
# `check` name on engines that do not send a code yet).
FINDINGS: dict[str, str] = {
    "constant_columns": "A column with the same value in every run cannot show any effect, so it is left out.",
    "constant_within_group": "A setting that never changes inside a group cannot be told apart from the group itself.",
    "controls_present": "Control runs help you see day-to-day drift between batches.",
    "duplicate_rows": "Rows that are exact copies can make one result count twice.",
    "informative_missingness": "Blank cells that follow a pattern can hide which runs failed.",
    "missingness": "Runs with blank cells are left out of the model, so they do not count.",
    "outliers": "One unusual result can pull the model toward it; check it was entered right.",
    "physical_bounds": "A value outside what is physically possible is usually a typing or unit error.",
    "provenance_metadata": "Knowing where each run came from makes the results easier to trace later.",
    "replicate_adequacy": "Repeat runs show how much results vary on their own, which tells you how big a real change must be.",
    "units_consistency": "Mixed units in one column make some runs look much larger or smaller than they are.",
    "auto_categorical": "A text column was treated as a set of labels, so each label is compared with the others.",
    "blank_as_absent": "Blank cells in this column were read as none added, so check that is what you meant.",
    "censored_rows": "Runs with results like below detection are left out, because their exact value is unknown.",
    "censored_values": "Values like below detection have no exact number, so they are not used as numbers.",
    "dropped_cells": "Some cells could not be read as numbers and were left out.",
    "dropped_rows": "Some runs were left out of the model because of blank or unreadable cells.",
    "header_row": "Kalos picked the row it thinks holds the column names; check it is the right one.",
    "unnamed_column": "A column with no name is hard to trace back to a setting or a result.",
    "near_constant_input": "A setting that barely changed was held fixed, because the sheet cannot show its effect.",
    "cv_ci_unavailable": "There were too few groups of runs to say how sure the check of the model is.",
    "coverage_not_guaranteed": "The range around each predicted value is a guide, not a promise about the next run.",
    "row_number_like": "A column that counts rows is not a setting, so it is left out.",
    "outcome_like": "A column measured after the run is not a setting you choose, so it is left out.",
    "category_not_modeled": "A text column with many different labels is left out, because there are too few runs per label.",
}

# Model checks (the `promotion` block). Keyed by the check name.
MODEL_CHECKS: dict[str, str] = {
    "surrogate_spearman": "If the model cannot put runs it was not shown in the right order, its proposals are closer to guesses.",
    "feasibility_auc": "If the model cannot tell failing recipes from working ones, it can propose recipes that fail.",
    "ece": "If the model's chance of failure does not match what happens, its warnings about failing recipes are off.",
    "brier": "If the model's guesses about failing recipes are often wrong, its warnings are not reliable.",
    "group_mean_margin": "If the model does no better than averaging repeat runs, it adds little beyond the runs themselves.",
}

# Guide steps that are not tied to one code.
STEPS: dict[str, str] = {
    "runs_awaiting": "A run without a recorded result cannot teach the model anything.",
    "proposal_notice": "The batch is shorter than asked for, because there were not enough different recipes left to try.",
    "analysis_out_of_sync": "The analysis was made before the latest change to the campaign, so it may not include it.",
}

WHY: dict[str, str] = {**REFUSALS, **FINDINGS, **MODEL_CHECKS, **STEPS}

_FALLBACK = "Kalos recorded this on the campaign; the details are on the page this step links to."


def why_for(code: str | None) -> str:
    """The why-line for `code`, or a plain fallback for a code with none."""
    if code and code in WHY:
        return WHY[code]
    return _FALLBACK


__all__ = ["FINDINGS", "MODEL_CHECKS", "REFUSALS", "STEPS", "WHY", "why_for"]
