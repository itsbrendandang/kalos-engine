"""One plain why-line per code: what it means for the lab's work.

`GLOSSARY` mirrors, verbatim, the "Why-lines by code" section of the
plain-words glossary kalos-web uses (kalos-web `docs/PLAIN-WORDS.md`,
`lib/why.ts` `WHY[code]`), so the engine and the web say the same sentence for
the same code. Do not edit those lines here: change them in the glossary and
copy them over, so the keys and sentences stay identical.

`ENGINE_ONLY` holds lines for codes the engine emits that the glossary does
not have yet, written in the same style (no statistics or software words, one
idea per sentence). Each is sent to the glossary owner to adopt; once the
glossary has a code, its line moves to `GLOSSARY` and leaves `ENGINE_ONLY`.

`tests/test_assist_why.py` fails when the engine gains a refusal code, a
finding check, or a model check that has no line here.
"""
from __future__ import annotations

GLOSSARY: dict[str, str] = {
    # Model verdict (reliability state)
    'reliable': 'If the model puts your past runs in the right order, its suggestions are worth lab time; still measure each one, since a prediction is not a result.',
    'not-yet': 'If the model cannot yet rank your past runs, its suggestions are closer to guesses, and you would spend runs on them.',
    'insufficient': 'With too few measured runs there is no way to tell whether the model is any good, so treat its suggestions as ideas to try, not predictions.',
    # Model checks (promotion gates)
    'surrogate_spearman': 'If the model cannot rank the runs it was tested on, it cannot tell a good recipe from a poor one, and its suggestions are guesses.',
    'group_mean_margin': 'If the model does no better than the average of each group of runs, it has learned the groups, not the recipes, and suggests more of the same.',
    'feasibility_auc': 'If the model cannot tell which recipes fail outright, it can suggest runs that produce nothing.',
    'ece': 'If the likely range is wrong about how often it is right, you cannot tell a safe bet from a long shot.',
    'brier': "If the model's confidence that a recipe works is unreliable, runs get spent on recipes that fail.",
    'unmeasured': 'A check that could not be run on this sheet is a check the model has not passed, so its suggestions have not been tested that way.',
    # Likely-range check (calibration)
    'bands_narrow': 'If fewer results land inside the likely range than it promises, the range is too narrow and a "safe" recipe can miss by more than shown.',
    'bands_wide': 'If far more results land inside the likely range than it promises, the range is wider than it needs to be and good recipes look riskier than they are.',
    'bands_ok': 'When the likely range is about right, you can read it as how far a measured result usually lands from its prediction.',
    'bands_unchecked': 'Until the likely range has been checked on measured runs, treat its width as a rough guide.',
    # Proposed-recipe flags (Plan)
    'negative': 'A predicted value below zero for something that cannot be negative means the model is guessing outside what it knows, so this recipe is a long shot.',
    'low-feasibility': 'A recipe the model thinks is unlikely to work can still be worth one run to learn from, but do not count on a result.',
    'constraint': 'A recipe predicted below the limit you asked for would not meet your own requirement even if the prediction holds.',
    'outside-range': 'An input outside what you have tried so far means the model has no runs to go on there, so the prediction is a guess and the run may behave unexpectedly.',
    'at-edge': 'An input at the edge of what you have tried means the model has runs on one side only, so the prediction is less sure than it looks.',
    'gain-within-band': 'A predicted gain smaller than the likely range may be no gain at all; the run can land below your best.',
    'fewer_distinct_candidates': 'Fewer recipes than you asked for means the rest would have repeated these; running near-duplicates would teach you nothing new.',
    'empty_batch': 'With no proposed recipes there is nothing to run this round; the sheet gives the model nothing to go on.',
    'target_constant_upload': 'If the target never changes, the model cannot learn what raises it and proposes nothing.',
    # Sheet check (validation findings)
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
    'mixed_row_types': 'If simulated or synthetic rows sit among measured ones, the model learns from numbers the lab never saw and can look better than it is.',
    'no_data_check': 'Without a sheet check, nothing here has been confirmed, so a mistake in the sheet would pass straight into the model.',
    'sheet_has_errors': 'A sheet with errors cannot be trusted as it is: a wrong unit, an impossible value or a doubled run would pass straight into the model and bend its suggestions.',
    'sheet_usable_with_warnings': 'The model can learn from this sheet, but each warning is a place where a mistake in the sheet could pass into the model, so read them before you trust its suggestions.',
    'sheet_passed': 'A sheet that passes every check gives the model clean runs to learn from, so a surprise later is more likely a real result than a sheet error.',
    # Sheet facts (Your data flags)
    'column_empty': 'A column with no values adds nothing and may mean a column header was shifted.',
    'column_missing': 'Runs with a blank in this column may be left out of the model, so fewer runs teach it.',
    'column_constant': 'A column with one value everywhere cannot tell runs apart, so the model ignores it.',
    'column_headerless': 'A column with no header cannot be named in the model, so Kalos cannot use it.',
    'column_unique': 'A column that is different in every run looks like a run label; used as an input, the model memorises runs instead of learning recipes.',
    'column_mixed': 'A number column with text entries is read as text, so its numbers are not used as numbers.',
    'rows_duplicate': 'A run that repeats an earlier one exactly counts twice, so one lucky run can pull the model toward its recipe.',
    'columns_duplicate': 'Two identical columns are one setting entered twice; the model cannot tell them apart.',
    'no_repeats': 'Without any repeated recipe there is no way to see how much the same recipe varies run to run, so a "best" run may just be a lucky one.',
    'nearly_constant': 'An input that barely changes across the sheet cannot show a pattern, so no suggestion about it can be trusted.',
    'left_out_rows': 'Rows marked simulated or synthetic are kept out of every chart so what you see is only what the lab measured.',
    # Column use (how each column was read)
    'kept_feature': 'This column is a setting the model can change, so a pattern in it can turn into a suggestion.',
    'target': 'This is the number the model is trying to raise; everything else is judged against it.',
    'dropped_id': 'A label or ID that changes every run would let the model memorise runs instead of learning recipes, so it was left out.',
    'dropped_output': 'A column that looks like a result cannot be set in the lab, so suggesting a value for it would be meaningless.',
    'dropped_constant': 'A column with one value cannot show any effect, so the model learns nothing from it.',
    'dropped_constant_on_fitted_rows': 'A column that only changes in rows the model could not use cannot show any effect in the rows it learned from.',
    'dropped_all_blank': 'A column with no values cannot teach the model anything.',
    'dropped_non_numeric': 'A column of mostly text cannot be placed on a scale, so the model cannot learn from it as a setting.',
    'dropped_sparse': 'A column that is mostly blank would leave out most runs if it were used, so it was left out instead.',
    'user_excluded': 'You chose to leave this column out, so no suggestion changes it.',
    'former_target': 'The old target is a result, not a setting, so it is not used as an input.',
    'unknown_role': 'Kalos did not use this column; the reason is one click down, as Kalos wrote it.',
    # Next steps (campaign stages)
    'setup_no_sheet': 'Until you upload a run sheet, Kalos has nothing to learn from and cannot suggest anything.',
    'setup_done': 'Kalos has read your runs and built a first model; the first suggestions rest on this sheet alone, so a mistake in it carries through.',
    'plan_ready': 'Each proposed recipe costs a run; untick the ones you would not run so the lab time goes where it counts.',
    'run_waiting': 'Until these results are entered, the model cannot learn from this round and the next suggestions are no better.',
    'record_partial': 'Every result you enter sharpens the next suggestions; a round with missing results teaches the model less.',
    'learn_ready': "Updating the model with this round's results is how the next round gets better suggestions; skipping it means planning on stale information.",
    # Kalos refusals and errors
    'payload_too_large': 'A sheet over the size limit cannot be read, so nothing from it reaches the model until it is trimmed.',
    'too_many_fit_rows': 'The model takes a fixed number of complete runs, and a campaign that outgrows it stops updating until older runs are removed.',
    'fit_rows_over_cap': 'The model takes a fixed number of complete runs, and a campaign that outgrows it stops updating until older runs are removed.',
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
    'engine_error': 'Kalos hit an error, so the result did not come back; your sheet and recorded results are unaffected.',
    'engine_unreachable': 'Kalos could not be reached, so this is a connection problem, not a result; nothing you have entered is lost.',
    'engine_timeout': 'The work took too long and was stopped; a smaller sheet usually finishes.',
    'engine_auth_failed': 'Kalos could not confirm who you are, so nothing was saved; signing in again usually fixes it.',
    'start_unconfirmed': 'If the start was not confirmed, some runs may have started and some not, so check Run before starting again.',
    'record_partial_save': 'A result that did not save is not in the model; re-enter it so the round is complete.',
    'generic': 'Kalos could not use this sheet, so nothing from it reaches the model until the problem is fixed.',
    # Kalos refusals and errors (more codes)
    'interrupted': 'The work stopped before it finished, so nothing was saved; running it again is safe.',
    'engine_stopped': 'Kalos was closed while the work ran, so nothing was saved; running it again is safe.',
    'read_timeout': 'The sheet took too long to arrive, so it was not read and nothing from it reaches the model.',
    'invalid_content_length': 'The upload arrived incomplete, so it was not read and nothing from it reaches the model.',
    'analysis_out_of_sync': 'The model was built before the latest change to the campaign, so its suggestions may not include it.',
    'model_check': 'A check the model has not passed means its suggestions have not been tested that way.',
    # Sheet facts from the upload (how Kalos read the sheet)
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
    'coverage_not_guaranteed': 'The likely range was checked on past runs, not on new recipes, so read it as a guide and not a promise about the next run.',
    'row_number_like': 'A column that counts rows is not a setting, so it was left out; used as an input, the model memorises runs instead of learning recipes.',
    'outcome_like': 'A column measured after the run is a result, not a setting you choose, so suggesting a value for it would be meaningless.',
    'category_not_modeled': 'A text column with many different labels has too few runs per label to learn from, so it was left out.',
    # Review tool findings
    'divergence_handling': 'If repeated wells disagree more than they should, handling was uneven and the difference may not be the recipe.',
    'divergence_unknown': 'Without repeated wells and a record of what was done, a difference cannot be pinned on handling or on the recipe.',
}

ENGINE_ONLY: dict[str, str] = {
    # Model verdict codes from the desktop engine (kalos/portal/verdict.py and
    # gate_text.py on desktop-engine-2); the engine texts match word for word.
    "no_ordering": "If the model does no better than chance at putting your past runs in order, its suggestions would be guesses, so Kalos proposes none until the sheet changes.",
    "too_few_groups": "With too few groups of runs there is no way to check the model on runs it has not seen, so its order cannot be trusted yet.",
    "model_refused": "When Kalos cannot check the model, it proposes no recipes rather than send you guesses; adding runs of different recipes is what helps.",
    # Column and campaign finding codes from the same round, not in the glossary yet.
    "headerless_column": "A column with no header cannot be named in the model, so Kalos cannot use it.",
    "search_box": "Kalos searched only inside the range of each setting you have tried, so a recipe outside it was not considered.",
    "inputs_changed": "The settings in this sheet differ from the campaign's earlier rounds, so results from before and after may not compare.",
    "coded_column": "A column of codes was kept as it is and not checked against physical limits, since its numbers stand for labels.",
    "outcome_like_name": "A column whose name reads like a result is not used as a setting, so no suggestion changes it.",
}

# The desktop engine names the verdict states reliable / not_yet / refused; the
# glossary key for the middle one is "not-yet".
VERDICT_STATE_KEYS = {"reliable": "reliable", "not_yet": "not-yet", "refused": "model_refused"}

WHY: dict[str, str] = {**ENGINE_ONLY, **GLOSSARY}

# Which group each code belongs to, for the coverage test.
MODEL_CHECKS = ("surrogate_spearman", "group_mean_margin", "feasibility_auc", "ece", "brier", "unmeasured")

_FALLBACK = GLOSSARY.get("generic", "Kalos recorded this on the campaign; the details are on the linked page.")


def why_for(code: str | None) -> str:
    """The why-line for `code`, or the glossary's generic line for a code with none."""
    if code and code in WHY:
        return WHY[code]
    return _FALLBACK


__all__ = ["ENGINE_ONLY", "GLOSSARY", "MODEL_CHECKS", "VERDICT_STATE_KEYS", "WHY", "why_for"]
