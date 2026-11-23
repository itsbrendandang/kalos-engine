"""Opt-in guard for calls that can spend API credits."""
from __future__ import annotations

import os


class PaidCallsOff(RuntimeError):
    """A billable call was refused by the operator's switch."""


def refuse_paid(what: str) -> None:
    """Refuse a paid step when KALOS_NO_PAID_CALLS is true; unset stays allowed."""
    if os.environ.get("KALOS_NO_PAID_CALLS", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise PaidCallsOff(f"{what}: refused. Paid calls are off (KALOS_NO_PAID_CALLS).")
