"""Every refusal the portal sends carries the one error envelope: `{"error": ...,
"code": ...}` (docs/HARDENING.md, "Error envelope", #98). A static guard, so a new
route (the job routes, the run record, anything later) cannot quietly answer
`{"error": ...}` without a stable `code` that a client can branch on.

It reads the source rather than exercising each route: the envelope is a
property of every literal refusal body, and a route-by-route list would drift.
"""
from __future__ import annotations

import ast
from pathlib import Path

PORTAL = Path(__file__).resolve().parents[1] / "kalos" / "portal"


def _json_response_bodies():
    for path in sorted(PORTAL.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", getattr(func, "attr", ""))
            if name == "JSONResponse" and node.args:
                yield path.name, node.lineno, node.args[0]


def test_every_literal_error_body_carries_a_code():
    offenders = []
    for name, lineno, body in _json_response_bodies():
        if not isinstance(body, ast.Dict):
            continue  # a variable or a call: checked where it is built (see below)
        keys = {k.value for k in body.keys if isinstance(k, ast.Constant)}
        if "error" in keys and "code" not in keys:
            offenders.append(f"{name}:{lineno}")
    assert not offenders, "error bodies without a code: " + ", ".join(offenders)


def test_module_level_error_bodies_carry_a_code():
    """The shared not-found bodies are module constants; check their literals too."""
    for path in sorted(PORTAL.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
                continue
            keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
            if "error" in keys:
                assert "code" in keys, f"{path.name}: module-level error body without a code"
