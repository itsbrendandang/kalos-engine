"""Reply cleanup shared by local model callers."""
from __future__ import annotations

import re


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
