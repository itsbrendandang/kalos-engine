"""Connectors to outside systems. Each one is off unless `KALOS_INTEGRATIONS`
names it (see `kalos.integrations.enabled`)."""
from __future__ import annotations

import os

INTEGRATIONS_VAR = "KALOS_INTEGRATIONS"


def enabled(name: str) -> bool:
    """Whether the integration `name` is switched on. Read per call, so a test
    or an operator can flip it without a restart. The value is a comma list,
    e.g. `KALOS_INTEGRATIONS=benchling`."""
    raw = os.environ.get(INTEGRATIONS_VAR, "")
    return name in {part.strip().lower() for part in raw.split(",") if part.strip()}
