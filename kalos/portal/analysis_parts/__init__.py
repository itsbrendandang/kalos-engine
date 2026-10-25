"""Pieces of kalos.portal.analysis, split out by concern.

Callers keep importing from `kalos.portal.analysis`, which re-exports every name
it used to define. Nothing in this package may import `kalos.portal.analysis`
(that would be a cycle), and nothing here may import torch at module level: the
portal imports `kalos.portal.analysis` at boot (see tests/test_portal.py).
"""
