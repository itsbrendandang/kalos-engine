"""The grounded assistant (docs/ASSISTANT.md).

- `guide`: the one next step for a campaign, from its record alone. No model.
- `index`: a local search index over the tenant's own data and the docs.
- `chat`: answers a question from retrieved chunks only, with sources.
- `sop`: SOP drafts filled from the campaign record, with named placeholders.

Nothing in this package writes campaign state, approves anything, or states a
fact the engine did not record. Importing it loads no torch and no model.
"""
from __future__ import annotations
