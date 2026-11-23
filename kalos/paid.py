"""Call-time policy for potentially billable model requests."""
from __future__ import annotations

import os


def no_paid_calls() -> bool:
    """Whether hosted model calls are disabled by operator configuration."""
    return os.environ.get("KALOS_NO_PAID_CALLS", "").strip().lower() in {"1", "true", "yes", "on"}
