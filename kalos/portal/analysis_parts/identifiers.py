"""Identifier detection for column NAMES: the profile's anchored id hint unioned
with the run-id pattern, plus the word-level compound rule. Feature selection,
provenance and anonymization all resolve through `is_identifier_name`, so the
three cannot disagree about what an identifier is.
"""
from __future__ import annotations

import functools
import re

from kalos.portal.analysis_parts.policy import _ID_HINT


@functools.lru_cache(maxsize=8)
def identifier_pattern(id_hint_pattern: str) -> re.Pattern[str]:
    """A profile's id hint UNIONED with the compound-identifier pattern.

    A `DomainProfile.id_hint` is anchored to whole tokens
    (`^(id|name|run|batch|campaign|lot|...)$`), which matches a column called
    exactly `run` or `batch` but not `run_number`, `batch_id`, `campaign_id` or
    `lot_number` - and compound names are what real run sheets actually use.

    Two things went wrong because of that gap, and both are worse than a naming
    nit:

    1. A NUMERIC identifier column became a model FEATURE. Given `run_number`
       and `batch_id`, the engine fit on them and reported both as significant
       drivers at rho = 1.0, then proposed a recipe instructing the scientist to
       "set batch_id = 100.037". A run index rises monotonically with time, so it
       correlates with any drift or learning trend in the campaign and will
       almost always look like a top driver. That is a spurious correlation
       presented as a process insight, which is the failure mode this codebase
       works hardest everywhere else to prevent.
    2. `anonymize=True` published those same names verbatim while claiming to
       pseudonymize identifier columns. The codebase already contradicted itself
       here: the metadata scrubber's `HASH_EXACT` does list `campaign_id`, so one
       id was hashed as metadata and published as a column name.

    Both now resolve through this single pattern, so feature selection,
    provenance, and anonymization cannot disagree about what an identifier is.

    Union, never intersection - it can only ever classify MORE names as
    identifiers, never fewer. Precision still matters in the other direction,
    since dropping or aliasing a real process input would be its own defect:
    both halves reject `Methanol`, `pH`, `scale_L`, `lipase_titer`,
    `batch_titer` and `run_duration_days`.

    Cached because it is called per column per analysis; keyed on the pattern
    string so a different domain profile gets its own union.
    """
    return re.compile(
        f"(?:{id_hint_pattern})|(?:{_RUN_ID_COMPOUND_PATTERN})", re.IGNORECASE
    )


# The run-id pattern from `kalos.validation.checks`, with the BARE words `batch`,
# `lot` and `campaign` removed: on their own they name a process factor (a medium
# lot, a batch of media), not an identifier. Only their ID forms (`batch_id`,
# `lot_number`, `campaign_id`) remain identifiers. See the H1 note in
# `kalos.domains.bioprocess`.
_RUN_ID_COMPOUND_PATTERN = (
    r"^(run|experiment|expt|sample|culture|vessel|reactor|bioreactor)"
    r"([ _\-.]?(id|ids|no|num|number|code|label|name|key|ref))?$"
    r"|^(batch|lot|campaign)([ _\-.]?(id|ids|no|num|number|code|label|name|key|ref))$"
    r"|^(id|uuid|index|well|barcode)$"
)


# Token rule for compound identifier names (see `is_identifier_name`). The last
# word of the name must be an identifier noun ("ID", "Name", "number"); a
# preceding word must name the thing being identified ("Tank", "Patient",
# "run"). So "Tank ID" and "Customer Name" are identifiers, while "batch_titer"
# (last word "titer") and "run_duration_days" (last word "days") are not.
_ID_NOUNS = frozenset({
    "id", "ids", "uuid", "guid", "name", "number", "num", "no", "code", "label",
    "key", "ref", "barcode", "identifier",
})
_ID_SUBJECTS = frozenset({
    "run", "batch", "lot", "campaign", "experiment", "expt", "sample", "culture",
    "vessel", "reactor", "bioreactor", "patient", "customer", "tank", "subject",
    "donor", "animal", "plate", "well",
})
_ID_ALONE = frozenset({"id", "ids", "uuid", "guid", "name", "index", "barcode", "identifier", "well", "mrn"})

# Words that name a PERSON wherever they appear in a column name (review H4). They
# are matched as whole words anywhere in the name, so "Operator", "Lab_Tech" and
# "Donor_Name_1" are identifiers, while "Methanol" and "pH" are not (no substring
# match: "technology" is not "tech").
_PERSON_WORDS = frozenset({"operator", "analyst", "technician", "tech", "email"})


def _token_identifier(name: str) -> bool:
    """The compound-name rule. Words are split on any non-alphanumeric run and
    at camelCase boundaries ("TankID" -> "tank id"); trailing numbers are
    ignored ("Sample ID 2"). This is a search over the whole name, not a match
    from its start, so a leading word like "Tank" or "Patient" is no obstacle."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name.strip())
    words = [w for w in re.split(r"[^a-z0-9]+", spaced.lower()) if w]
    while words and words[-1].isdigit():
        words.pop()
    if not words:
        return False
    if any(w in _PERSON_WORDS for w in words):
        return True
    *rest, last = words
    if not rest:
        return last in _ID_ALONE
    return last in _ID_NOUNS and any(w in _ID_SUBJECTS for w in rest)


def is_identifier_name(name: object, id_hint: re.Pattern[str] | None = None) -> bool:
    """Whether a column NAME is an identifier: the profile's anchored `id_hint`
    (plus the run-id pattern, `identifier_pattern`), or the word-level compound
    rule above. Used for feature selection, provenance and anonymization alike,
    so the three cannot disagree."""
    text = str(name).strip()
    hint = _ID_HINT.pattern if id_hint is None else id_hint.pattern
    return bool(identifier_pattern(hint).match(text)) or _token_identifier(text)


def _is_identifier_name(name: str) -> bool:
    """Whether a column NAME should be pseudonymized when `anonymize=True`."""
    return is_identifier_name(name)
