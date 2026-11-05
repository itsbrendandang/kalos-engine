"""Virtual runs: simulated and synthetic rows kept apart from lab measurements.

`kalos.twin.origin` defines the origin type and the guard every fit and analysis
entry point calls; `kalos.twin.generators` is the registry of seeded generators.
See docs/VIRTUAL_RUNS.md. Torch-free.
"""
from kalos.twin.origin import (
    ORIGIN_COLUMN,
    Origin,
    OriginArray,
    VirtualDataRefused,
    check_origin,
    highest_origin,
)

__all__ = [
    "ORIGIN_COLUMN",
    "Origin",
    "OriginArray",
    "VirtualDataRefused",
    "check_origin",
    "highest_origin",
]
