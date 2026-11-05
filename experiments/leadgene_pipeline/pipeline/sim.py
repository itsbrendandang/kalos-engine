"""Re-export shim: the simulator now lives in `kalos.bench.cho_sim`.

Kept so the experiment's examples and tests keep importing `pipeline.sim`.
"""
from kalos.bench.cho_sim import (  # noqa: F401
    CULTURE_HOURS,
    DESIGN_SPACE,
    ProcessParams,
    endpoint_titer,
    replicate_titers,
    sample_doe,
    simulate,
    to_featurized_rows,
    true_optimum,
)
