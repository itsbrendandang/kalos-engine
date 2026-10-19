"""The bioprocess domain profile: the column-role hint regexes Kalos shipped
with, kept here as data so the engine stays domain-neutral.

`BIOPROCESS_PROFILE` reproduces the exact inference the analyze path used before
the domain abstraction, so an upload with no declared roles behaves identically
to before. It is the default profile in `kalos.portal.analysis`.
"""
from __future__ import annotations

import re

from .profile import DomainProfile

BIOPROCESS_PROFILE = DomainProfile(
    name="bioprocess",
    outcome_hint=re.compile(
        r"titer|titre|yield|conc|purity|lipase|biomass|od\d|product|response|output|score|kda|activity",
        re.I,
    ),
    target_pref=re.compile(r"titer|titre|lipase|yield", re.I),
    id_hint=re.compile(
        # `sample` is an identifier only as a bare name or as a sample ID/code/name
        # (`sample_id`, `Sample Code`). The old `sample.*` glob also swallowed
        # measured quantities such as `sample_volume_L`, which were then silently
        # dropped from the model. Compound forms are caught by the word rule in
        # `kalos.portal.analysis.is_identifier_name` either way.
        r"^(id|name|sample(?:[ _.-]?(?:ids?|codes?|names?|no|num|number|ref|label|uuid))?"
        # Medium, strain, recipe, batch, campaign, group and lot are NOT here:
        # they are process FACTORS in most sheets (a medium lot with a real effect
        # on titer), so they are modeled as categoricals. Only the compound ID
        # forms (`batch_id`, `lot_number`, `campaign_id`) are identifiers, via
        # `kalos.portal.analysis.is_identifier_name`.
        r"|well|index|run|experiment|round|date|time|notes?)$",
        re.I,
    ),
    # Names that MAY be a replicate/leakage grouping. Used as the CV group only
    # when it is declared in roles, or when each of its levels is confounded with
    # a single recipe (see `kalos.portal.analysis._auto_group_column`).
    group_hint=re.compile(r"medium|strain|recipe|batch|campaign|group|lot", re.I),
)

__all__ = ["BIOPROCESS_PROFILE"]
