"""One plain why-line per finding code and refusal code the engine emits.

Each line says what the code means for the lab's work, in glossary style: no
statistics words and no software words, one idea per sentence. Most lines are
the plain-words glossary's own sentences (the same ones `kalos/assist/why.py`
carries on the assistant branch), copied verbatim so the engine, the
assistant and the web say the same thing. `ENGINE_LINES` are written here for
codes the glossary does not have yet; they were sent to the assistant owner to
mirror.

Every finding in `validation.findings` and `campaign_findings` carries its line
as `why`, and coded 400/409 refusals carry it next to `code`.
`tests/test_why_lines.py` fails when the engine gains a code without a line.
"""
from __future__ import annotations

from typing import Any

GLOSSARY_LINES: dict[str, str] = {
    # Reused verbatim from the glossary lines in kalos/assist/why.py (branch assistant).
    'reliable': 'If the model puts your past runs in the right order, its suggestions are worth lab time; still measure each one, since a prediction is not a result.',
    'not-yet': 'If the model cannot yet rank your past runs, its suggestions are closer to guesses, and you would spend runs on them.',
    'not_yet': 'If the model cannot yet rank your past runs, its suggestions are closer to guesses, and you would spend runs on them.',
    'insufficient': 'With too few measured runs there is no way to tell whether the model is any good, so treat its suggestions as ideas to try, not predictions.',
    'refused': 'When Kalos cannot check the model, it proposes no recipes rather than send you guesses; adding runs of different recipes is what helps.',
    'no_ordering': 'If the model does no better than chance at putting your past runs in order, its suggestions would be guesses, so Kalos proposes none until the sheet changes.',
    'too_few_groups': 'With too few groups of runs there is no way to check the model on runs it has not seen, so its order cannot be trusted yet.',
    'model_refused': 'When Kalos cannot check the model, it proposes no recipes rather than send you guesses; adding runs of different recipes is what helps.',
    'surrogate_spearman': 'If the model cannot rank the runs it was tested on, it cannot tell a good recipe from a poor one, and its suggestions are guesses.',
    'group_mean_margin': 'If the model does no better than the average of each group of runs, it has learned the groups, not the recipes, and suggests more of the same.',
    'feasibility_auc': 'If the model cannot tell which recipes fail outright, it can suggest runs that produce nothing.',
    'ece': "If the model's chance that a recipe works does not match how often it does, you cannot tell a safe bet from a long shot.",
    'brier': "If the model's confidence that a recipe works is unreliable, runs get spent on recipes that fail.",
    'unmeasured': 'A check that could not be run on this sheet is a check the model has not passed, so its suggestions have not been tested that way.',
    'bands_narrow': 'If fewer results land inside the likely range than it promises, the range is too narrow and a "safe" recipe can miss by more than shown.',
    'bands_wide': 'If far more results land inside the likely range than it promises, the range is wider than it needs to be and good recipes look riskier than they are.',
    'bands_ok': 'When the likely range is about right, you can read it as how far a measured result usually lands from its prediction.',
    'bands_unchecked': 'Until the likely range has been checked on measured runs, treat its width as a rough guide.',
    'fewer_distinct_candidates': 'Fewer recipes than you asked for means the rest would have repeated these; running near-duplicates would teach you nothing new.',
    'coverage_not_guaranteed': 'The likely range was checked on past runs, not on new recipes, so read it as a guide and not a promise about the next run.',
    'units_consistency': 'If one column mixes units, the model treats 37 °C and 98 °F as different settings, and every pattern built on that column is wrong.',
    'physical_bounds': 'A value that is physically impossible is usually a typo or a unit mix-up, and the model would learn from it as if it were real.',
    'duplicate_rows': 'A run entered twice counts double, so one lucky run can pull the model toward its recipe.',
    'missingness': 'Blank cells shrink what the model can learn from, and a column that is mostly blank teaches it almost nothing.',
    'outliers': 'A value far from the rest is either a real extreme run or a data-entry mistake; the model cannot tell which, and one mistake can bend its suggestions.',
    'provenance_metadata': 'Without who ran it, when, and with which materials, a surprising result cannot be traced back to its source.',
    'replicate_adequacy': 'Runs that were never repeated hide how much your results vary, so a "best" run may just be a lucky one.',
    'controls_present': 'Without a control run there is no baseline, so a change in the result cannot be told apart from ordinary run-to-run variation.',
    'constant_columns': 'An input that never changes cannot show any effect, so the model learns nothing from it and cannot suggest changing it.',
    'informative_missingness': 'If blanks line up with poor results, filling them in or dropping them would quietly teach the model the answer and make it look better than it is.',
    'constant_within_group': 'An input that only changes between groups stands in for the group itself, so a pattern on it may be the group, not the setting.',
    'auto_categorical': 'A text column was read as a set of labels, so Kalos compares each label with the others instead of treating it as a number.',
    'blank_as_absent': 'Blank cells in this column were read as "none added"; if a blank meant "not recorded", the model learns from a value the lab never set.',
    'censored_rows': 'Runs whose result is only "below detection" have no exact value, so they are left out and teach the model nothing.',
    'censored_values': 'A value like "below detection" has no exact number, so it is not used as a number and that run teaches the model less.',
    'dropped_cells': 'Cells that could not be read as numbers were left out, so those runs teach the model less than the sheet holds.',
    'dropped_rows': 'Runs with blank or unreadable cells were left out, so fewer runs teach the model than the sheet holds.',
    'header_row': 'If Kalos picked the wrong row as the column names, every column is misnamed and settings and results get mixed up.',
    'unnamed_column': 'A column with no name cannot be traced back to a setting or a result, so Kalos cannot use it.',
    'near_constant_input': 'A setting that barely changed was held fixed: the sheet cannot show its effect, so any suggestion about it would be a guess.',
    'cv_ci_unavailable': 'With too few groups of runs there is no way to say how sure the check of the model is, so treat its score as rough.',
    'category_not_modeled': 'A text column with many different labels has too few runs per label to learn from, so it was left out.',
    'headerless_column': 'A column with no header cannot be named in the model, so Kalos cannot use it.',
    'search_box': 'Kalos searched only inside the range of each setting you have tried, so a recipe outside it was not considered.',
    'inputs_changed': "The settings in this sheet differ from the campaign's earlier rounds, so results from before and after may not compare.",
    'coded_column': 'A column of codes was kept as it is and not checked against physical limits, since its numbers stand for labels.',
    'outcome_like_name': 'A column whose name reads like a result is not used as a setting, so no suggestion changes it.',
    'stopping_advice_only': 'Kalos never ends a campaign; this advice has been tried only on one small simulated case, not on real assays, so whether another round is worth it is your call.',
    'specs_treated_independently': 'Kalos judges each spec on its own, so when two results rise and fall together the chance of meeting all of them at once can be too high or too low; read it as a rough guide.',
    'possibly_measured_driver': 'The model leans most on a column that might be measured during the run; until you confirm it is a setting, the model is not called reliable.',
    'possibly_measured_after_run': 'A column that might be measured during the run, used as a setting, would let the model look better than it is; untick it if it is a result.',
    'payload_too_large': 'A sheet over the size limit cannot be read, so nothing from it reaches the model until it is trimmed.',
    'too_many_fit_rows': 'The model takes a fixed number of complete runs, and a campaign that outgrows it stops updating until older runs are removed.',
    'too_many_rows': 'Over the row limit, the sheet cannot be read at all, so no run in it is used.',
    'too_many_columns': 'Over the column limit, the sheet cannot be read at all, so no run in it is used.',
    'too_many_cells': 'Over the size limit, the sheet cannot be read at all, so no run in it is used.',
    'too_few_rows': 'With too few runs the model cannot tell a pattern from chance, so any suggestion would be a guess.',
    'too_few_complete_rows': 'Only runs with a value for the target and every input can teach the model; with too few of them it cannot learn.',
    'too_few_columns': 'With no input columns to change, there is nothing for the model to suggest.',
    'target_empty': 'A target column with no values gives the model nothing to aim at.',
    'target_not_found': 'Without a target column Kalos does not know what to raise.',
    'target_not_numeric': 'A target that is not a number cannot be raised or lowered, so the model cannot learn from it.',
    'target_constant': 'If the target never changes, the model cannot learn what raises it and proposes nothing.',
    'no_numeric_inputs': 'Without a number column to change, the model has nothing to vary and nothing to suggest.',
    'no_process_inputs': 'Without a column Kalos can treat as a setting, the model has nothing to vary and nothing to suggest.',
    'fit_failed': 'When the model cannot be built, there are no suggestions; the sheet or the target usually needs a change.',
    'parse_error': 'A sheet that cannot be read gives the model nothing; the first row should hold the column names and each run one row.',
    'empty_file': 'An empty file has no runs to learn from.',
    'not_a_table': 'A file that is not a table of runs (one row per run, one column per setting) cannot be read.',
    'unsupported_file_type': 'Only CSV, TSV and Excel files can be read; anything else gives the model nothing.',
    'corrupt_excel': 'An Excel file that cannot be opened gives the model nothing; re-saving it from Excel usually fixes it.',
    'ambiguous_header': 'If Kalos cannot tell which row holds the column names, it cannot tell settings from results.',
    'sheet_not_found': 'The worksheet named is not in this file, so nothing was read.',
    'no_usable_sheet': 'None of the worksheets holds a table of runs, so nothing was read.',
    'campaign_has_runs': 'Replacing the sheet discards the runs recorded in this campaign, and they cannot be recovered here.',
    'unauthenticated': 'Until you sign in again, nothing you enter is saved.',
    'rate_limited': 'Too many requests in a short time are refused, so wait a minute before trying again.',
    'engine_busy': 'Kalos is busy with other work; nothing is lost, and trying again in a moment usually works.',
    'engine_timeout': 'The work took too long and was stopped; a smaller sheet usually finishes.',
    'interrupted': 'The work stopped before it finished, so nothing was saved; running it again is safe.',
    'read_timeout': 'The sheet took too long to arrive, so it was not read and nothing from it reaches the model.',
    'invalid_content_length': 'The upload arrived incomplete, so it was not read and nothing from it reaches the model.',
    'row_number_column': 'A column that counts rows is not a setting, so it was left out; used as an input, the model memorises runs instead of learning recipes.',
}

ENGINE_LINES: dict[str, str] = {
    # Column choices (findings)
    "after_target_in_sheet": (
        "A column after the result whose name reads like a measurement was probably measured "
        "during or after the run, so it is not used as a setting."
    ),
    "after_target_outcome": (
        "A column after the result whose name reads like a measurement was probably measured "
        "during or after the run, so it is not used as a setting."
    ),
    "after_target_high_correlation": (
        "An input that sits after the result and rises and falls with it may have been measured "
        "during the run; used as a setting, it would let the model look better than it is."
    ),
    "feature_looks_outcome_like": (
        "An input whose name reads like a result may be measured during or after the run; used "
        "as a setting, it would let the model look better than it is."
    ),
    "concentration_before_target": (
        "A concentration column before the result was kept as a setting; if it was measured "
        "during the run rather than set, untick it, or the model will look better than it is."
    ),
    # Refused column choices (400 before any work)
    "invalid_linear_constraint": (
        "A limit on the inputs that Kalos cannot read, or one that names something other than a "
        "number setting in the sheet, cannot be applied, so nothing was analyzed until it is fixed."
    ),
    "infeasible_linear_constraints": (
        "No recipe within the tried ranges meets all your limits on the inputs at once, so there "
        "is nothing to suggest until a limit or a range changes."
    ),
    "linear_constraints_with_categorical": (
        "Limits that add inputs together cannot be applied while one setting is a set of labels, "
        "so nothing was analyzed; drop the limits or that setting."
    ),
    "invalid_specs": (
        "A requirement on a result that Kalos cannot read, or one placed on a setting instead of a "
        "result, cannot be checked, so nothing was analyzed until it is fixed."
    ),
    "invalid_allow_beyond": (
        "How far past the tried range to search must be a share from 0 to 1, for settings in the "
        "sheet; otherwise Kalos cannot tell where to look, so nothing was analyzed."
    ),
    "invalid_exploration_design": (
        "Spreading runs over untested recipes is either automatic or off; with any other choice "
        "Kalos cannot tell what you want, so nothing was analyzed."
    ),
    "invalid_per_recipe_noise": (
        "Using each recipe's own repeat spread is either on or off; with any other choice Kalos "
        "cannot tell what you want, so nothing was analyzed."
    ),
    "invalid_applicability_distance": (
        "How far from your runs a recipe may be before it is flagged must be a positive number; "
        "otherwise Kalos cannot flag far recipes, so nothing was analyzed."
    ),
    "invalid_direction": (
        "Kalos needs to know whether to raise or lower the result; without that, nothing was analyzed."
    ),
    "direction_conflict": (
        "The upload asked both to raise and to lower the result, so Kalos analyzed nothing "
        "rather than pick one for you."
    ),
    "roles_not_applied": (
        "The column choices saved with this campaign could not be read, so Kalos did not analyze "
        "it again rather than guess them; upload the sheet again with your column choices."
    ),
    # Campaign and request refusals
    "upload_pending": (
        "Another sheet for this campaign is still being analyzed; wait for it to finish so the "
        "two do not replace each other."
    ),
    "analysis_failed": (
        "Kalos could not analyze the campaign's runs, so nothing changed; your sheet and recorded "
        "results are as they were."
    ),
    "campaign_archived": (
        "An archived campaign is kept as a record and cannot change, so nothing was saved."
    ),
    "conflict": (
        "The campaign changed while this was in progress, so nothing was saved; trying again "
        "works from the latest state."
    ),
    "no_data": "The campaign holds no runs yet, so there is nothing to analyze.",
    "not_found": "Kalos found nothing by that name in your workspace, so nothing was changed.",
    "reason_required": (
        "Changing a result already recorded needs a reason, so the record shows why it changed; "
        "nothing was saved without one."
    ),
    "invalid_result": "A result must be a number, so nothing was saved for this run.",
    "invalid_body": "Kalos could not read what was sent, so nothing was saved.",
    "invalid_request": "Kalos could not use what was sent, so nothing was saved.",
    "validation_error": "Kalos could not read what was sent, so nothing was saved.",
    # Kalos itself is not ready (sign-in, background work, storage)
    "auth_not_configured": (
        "Sign-in is not set up, so Kalos refuses all work from other computers; nothing is lost."
    ),
    "auth_misconfigured": (
        "Sign-in is set up wrongly, so Kalos cannot tell who you are and refuses the work; "
        "nothing is lost."
    ),
    "auth_unavailable": (
        "Kalos cannot check who you are right now, so it refuses the work; nothing is lost, and "
        "trying again later usually works."
    ),
    "jobs_misconfigured": (
        "Background work is set up wrongly, so the sheet was not taken; nothing is lost."
    ),
    "service_unavailable": (
        "Kalos cannot take this work right now; nothing is lost, and trying again later usually works."
    ),
    "store_unavailable": (
        "Kalos cannot reach where campaigns are kept, so nothing was read or saved; nothing is lost."
    ),
    "schema_mismatch": (
        "Kalos and its saved campaigns are from different versions, so it reads and saves nothing "
        "until they match; nothing is lost."
    ),
    # The generic envelope codes for a refusal with no more specific code
    # (kalos/portal/app.py `_STATUS_CODES`).
    "bad_request": "Kalos could not use what was sent, so nothing was saved.",
    "forbidden": (
        "Your sign-in does not allow this action, so nothing was changed; ask whoever runs "
        "Kalos for access."
    ),
    "method_not_allowed": "Kalos does not do that here, so nothing was changed.",
    "timeout": "The work took too long and was stopped; nothing was saved, and trying again is safe.",
    # Scale-up readout refusals (kalos/scale/readout.py gate)
    "invalid_target": (
        "The planned scale-up is missing a value or has one Kalos cannot read, so no readout "
        "was made."
    ),
    "required_columns": (
        "The sheet lacks a column the scale-up readout needs, so no readout was made."
    ),
    "min_distinct_scales": (
        "Runs at too few different vessel sizes cannot show how results change with size, so "
        "no readout was made."
    ),
    "non_finite_dropped_fraction": (
        "Too many runs have a blank or unreadable value in a needed column, so too few are left "
        "to judge the scale-up and no readout was made."
    ),
    "target_not_interpolation": (
        "The planned vessel is no bigger than ones already run, so the readout has nothing to "
        "project; compare those runs directly."
    ),
    "target_ratio_hard_cap": (
        "The planned vessel is so much bigger than any run that a projection would be a guess, "
        "so no readout was made."
    ),
    "internal_error": (
        "Kalos hit an error, so the result did not come back; your sheet and recorded results are "
        "unaffected."
    ),
}

WHY: dict[str, str] = {**GLOSSARY_LINES, **ENGINE_LINES}


def why_for(code: object) -> str | None:
    """The why-line for `code`, or None for a code with no line (the test
    forbids that for every code the engine emits)."""
    return WHY.get(code) if isinstance(code, str) else None


def coded_body(error: str, code: str) -> dict[str, Any]:
    """The error envelope `{error, code}` of a coded refusal, plus `why` when
    the code has a line (additive)."""
    body: dict[str, Any] = {"error": error, "code": code}
    why = why_for(code)
    if why:
        body["why"] = why
    return body


def with_why(validation: dict[str, Any]) -> dict[str, Any]:
    """A `report_dict` with `why` on each finding (additive). The finding's
    `code` picks the line; the check name when the code has none."""
    findings = validation.get("findings")
    if isinstance(findings, list):
        validation["findings"] = [finding_with_why(f) for f in findings]
    return validation


def finding_with_why(finding: Any) -> Any:
    """`finding` with `why` set from its code (or check), unchanged if not a dict."""
    if not isinstance(finding, dict):
        return finding
    why = why_for(finding.get("code")) or why_for(finding.get("check"))
    return {**finding, "why": why}


__all__ = ["ENGINE_LINES", "GLOSSARY_LINES", "WHY", "coded_body", "finding_with_why", "why_for", "with_why"]
