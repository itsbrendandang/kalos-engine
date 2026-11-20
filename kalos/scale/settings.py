"""`KALOS_SCALE_READOUT_CHECKS`: the switch for the scale readout's measured-data
honesty checks (fix/scale-readout-honesty).

`on` (default; unset or empty means `on`):
  * the assay-method confound check (`kalos.scale.assay_confound`) runs and may
    downgrade the readout (NUMBER -> NUMBER_WITH_WARNING, or REFUSAL);
  * plate/flask labels in `scale_L` become labelled working-volume assumptions
    (`kalos.scale.working_volume`); unknown labels refuse by name;
  * `leave_one_scale_out_report` raises, naming the columns, instead of
    returning a report over zero rows.

`off` restores the decisions, reasons and numbers from before these checks
existed: no confound finding, `scale_L` labels are left to the normalize step
(which drops the text column), and LOSO returns its empty report. The additive
fields stay in the output and say the check did not run
(`assay_confound.skipped`, `provenance.readout_checks: "off"`). It exists so the change is
reversible and testable both ways (AGENTS.md section 2); turning it off removes
honesty checks and is not recommended.

Fail-closed like every enum setting (`kalos.envconfig.env_enum`): any other
value raises `ConfigError`. Read at call time, so a test can flip it.
"""
from __future__ import annotations

from kalos.envconfig import env_enum

READOUT_CHECKS_VAR = "KALOS_SCALE_READOUT_CHECKS"
READOUT_CHECKS_VALUES = ("on", "off")


def readout_checks_enabled() -> bool:
    """Whether `KALOS_SCALE_READOUT_CHECKS` is `on` (the default)."""
    return env_enum(READOUT_CHECKS_VAR, READOUT_CHECKS_VALUES, "on") == "on"


__all__ = ["READOUT_CHECKS_VALUES", "READOUT_CHECKS_VAR", "readout_checks_enabled"]
