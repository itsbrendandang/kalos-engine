# Blood-reactor profile (schema only)

**This folder is a schema template. It contains no wet-lab data.** The source
project this layout was modelled on (a cytokine-cocktail Bayesian-optimization
study on leukemic-cell expansion) keeps its own measured dataset; that data is
**not included here and must not be added** to this repository, in any form
(raw, summarized, anonymized, or used to set the ranges below).

It shows how a second, non-fermentation domain is described to Kalos with an
explicit column-roles schema instead of a new domain profile: eight cytokine
concentrations are the process inputs and a cell fold change is the measured
outcome.

## Files

| File | What it is |
|---|---|
| `roles.json` | A `ColumnRoles` object (`kalos.domains.ColumnRoles.from_dict`), the same JSON the portal accepts as its `roles` field. |
| `bounds.json` | **Placeholder** design bounds in ng/mL. Invented generic ranges, not from any dataset or protocol. Replace them with your own. |
| `make_template.py` | Writes an empty, header-only CSV run sheet with the columns from `roles.json`. |

## Columns

- `run_id`: row identifier, ignored by the model (`ids`).
- `replicate_group`: groups replicate wells or runs of one recipe so that cross-validation
  never splits a group across train and test (`groups`).
- `SCF_ng_mL`, `TPO_ng_mL`, `IL6_ng_mL`, `IL3_ng_mL`, `FLT3L_ng_mL`, `GMCSF_ng_mL`,
  `IL9_ng_mL`, `IL1b_ng_mL`: cytokine concentrations (`features`). They are also
  declared `sparse`, so a blank cell means "not added" (0) rather than a missing value.
- `leukemic_fold_change`: the measured outcome (`target`). Kalos **maximizes** the
  target. If your goal is to suppress expansion, add a column with the negated or
  inverted value and declare that column as the target.

## Usage

```bash
python examples/blood_reactor_profile/make_template.py /tmp/blood_reactor_runs.csv
# fill in your own measured runs, then upload the sheet with roles.json as the roles schema
```

Do not commit the generated CSV or any filled-in copy. The test
`tests/test_blood_reactor_profile.py` checks that this folder contains no CSV with data
rows. It also runs the schema end to end on a frame of random numbers generated inside
the test, so no real measurements are involved.
