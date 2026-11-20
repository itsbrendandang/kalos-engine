"""Setup suggestions: the model helps name column roles; the user decides.

Given an uploaded sheet, suggest for each column a role and a plausible unit,
each with a short reason and a confidence, all marked as suggestions the user
must confirm. Nothing here is applied: no route writes setup, roles or the
campaign, and nothing here is read by the model fit (suggestions are never
priors). `tests/test_setup_suggestions.py` checks both.

Roles map one-to-one onto the analyze path's `ColumnRoles` keys
(`kalos/domains/profile.py`; the data-checks work adds `outputs`), so a
confirmed suggestion is posted as a normal roles object:

| role                  | ColumnRoles key            |
|-----------------------|----------------------------|
| `setting`             | `features` (+ `categoricals` for a text column) |
| `outcome`             | `target` (one), else `outputs` |
| `covariate`           | `covariates`               |
| `batch_group`         | `groups`                   |
| `clone`               | `strain`                   |
| `id`                  | `ids`                      |
| `derived_from_target` | `outputs` (reason `target_leak`) |

What a model sees: headers, the unit parsed from each header, and summary
statistics (count, missing, distinct values, min/median/max, up to five level
labels of a short text column). Never a row, never an identity column (the
normalize payload's header screen), never a free-text value.

What a model may say: a role from the table, a unit from the known-unit list
or none, and a reason of at most 200 characters that names no column the sheet
does not have and states no number the statistics do not contain. Anything else
fails the check (`kalos.providers.grounding`) and goes to the quality model,
then to the engine's own offline suggestion (`provider_path`).

The model never gives the confidence. The engine sets it from its own evidence
(the header unit, the identifier rule, the clone and covariate name rules, and
a target-leak check on the data): `high` when the evidence agrees with the
suggested role, `low` when it disagrees (the evidence is listed), `medium`
when there is none.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from kalos.assist.words import banned_in, claims_in
from kalos.providers.cascade import Cascade, CascadeResult, ValidationFailed
from kalos.providers.grounding import (
    identifiers_in,
    number_keys,
    numbers_from,
    parse_json_object,
)

ROLES: dict[str, dict[str, str]] = {
    "setting": {"roles_key": "features", "meaning": "a process input you choose before the run"},
    "outcome": {"roles_key": "target", "meaning": "a result measured after the run (one is the target)"},
    "covariate": {"roles_key": "covariates", "meaning": "recorded about the run but not chosen (passage, thaw)"},
    "batch_group": {"roles_key": "groups", "meaning": "a batch, lot, plate or replicate group"},
    "clone": {"roles_key": "strain", "meaning": "the production clone or strain label"},
    "id": {"roles_key": "ids", "meaning": "an identifier or free text: not used by the model"},
    "derived_from_target": {"roles_key": "outputs",
                            "meaning": "computed from the result (a grade, a tier): never an input"},
}
MAX_COLUMNS = 200
MAX_REASON_CHARS = 200
MAX_LEVELS_SHOWN = 5
NOTICE = ("Suggestions only. Nothing has been applied: confirm or change each role in the campaign "
          "setup. Suggestions are never used by the model fit.")
# Units a suggestion may name. Header units are parsed by the engine itself.
KNOWN_UNITS = frozenset({
    "g/L", "mg/L", "mg/mL", "µg/mL", "g", "mg", "kg", "L", "mL", "µL", "%", "mM", "M", "µM", "nM",
    "mmol", "mol", "mOsm", "mOsm/kg", "°C", "K", "h", "min", "s", "d", "rpm", "vvm", "L/min", "mL/h",
    "mL/min", "L/h", "bar", "mbar", "Pa", "kPa", "U", "IU", "U/mL", "cells/mL", "10^6 cells/mL",
    "1e6 cells/mL", "OD", "OD600",
})

_SYSTEM = """You help a bioprocess scientist set up a run sheet. For each column you get the
header, the unit the engine read from the header (or null) and summary statistics. Suggest:
- role: one of setting, outcome, covariate, batch_group, clone, id, derived_from_target.
  setting = a process input chosen before the run. outcome = a measured result.
  covariate = recorded but not chosen (passage, generation, thaw). batch_group = batch, lot, plate.
  clone = production clone or strain label. id = identifier or free text.
  derived_from_target = computed from the result (a quality grade, a yield tier, pass/fail).
- unit: the unit of the values, or null if unknown or unitless. Prefer the header unit.
- reason: one short sentence (under 25 words) using only the header and the statistics.
  Do not invent numbers. Do not give a confidence.
Reply with ONE JSON object: {"columns": [{"column": "...", "role": "...", "unit": "..." or null,
"reason": "..."}]} with one entry per column, using the headers exactly as given."""


# --- engine evidence ------------------------------------------------------------- #


def _header_unit(header: str) -> str | None:
    try:  # data-checks moves this to kalos.normalize.header_units (re-exported here)
        from kalos.portal.analysis_parts.units import parse_header_unit
    except ImportError:  # pragma: no cover - the portal extra is part of every install that serves this
        return None
    return parse_header_unit(header)


def _is_identity(header: str) -> bool:
    from kalos.normalize.payload import _is_identity_header

    return _is_identity_header(header)


def _is_identifier(header: str) -> bool:
    from kalos.domains.bioprocess import BIOPROCESS_PROFILE
    from kalos.portal.analysis_parts.identifiers import is_identifier_name

    return bool(is_identifier_name(header, BIOPROCESS_PROFILE.id_hint))


def _name_role(header: str) -> str | None:
    """A role the engine's own name rules give (clone, covariate, batch)."""
    from kalos.portal.analysis_parts.strain_roles import (
        BLOCK_NAME_HINT,
        COVARIATE_NAME_HINT,
        STRAIN_NAME_HINT,
    )

    if STRAIN_NAME_HINT.search(header):
        return "clone"
    if COVARIATE_NAME_HINT.search(header):
        return "covariate"
    if BLOCK_NAME_HINT.search(header) and not re.search(r"round", header, re.I):
        return "batch_group"
    return None


def _round(value: float) -> float:
    return float(f"{value:.4g}")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def leak_evidence(df: pd.DataFrame, target: str | None) -> dict[str, dict[str, Any]]:
    """Columns that track the target too closely to be an input. Uses the
    data-checks target-leak check when it is installed, else a small local
    version: a numeric column with |Spearman| >= 0.95 against the target, or a
    text column whose levels explain >= 90% of the target's variance, over at
    least 8 rows."""
    if not target or target not in df.columns:
        return {}
    import importlib

    from kalos.domains.bioprocess import BIOPROCESS_PROFILE

    checker = getattr(importlib.import_module("kalos.validation.checks"), "target_leak_evidence", None)
    if callable(checker):
        found = checker(df, target, factor_hint=BIOPROCESS_PROFILE.group_hint)
        return {str(k): dict(v) for k, v in (found or {}).items()}
    y = _numeric(df[target])
    out: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        if col == target:
            continue
        x = df[col]
        mask = y.notna() & x.notna()
        if int(mask.sum()) < 8:
            continue
        xn = _numeric(x)
        if xn[mask].notna().mean() >= 0.9:
            rho = xn[mask].rank().corr(y[mask].rank())
            if rho is not None and np.isfinite(rho) and abs(rho) >= 0.95:
                out[str(col)] = {"kind": "numeric", "spearman": _round(float(rho)), "n": int(mask.sum())}
        else:
            levels = x[mask].astype(str)
            if 2 <= levels.nunique() <= 10:
                yy = y[mask]
                total = float(((yy - yy.mean()) ** 2).sum())
                within = float(sum(((g - g.mean()) ** 2).sum() for _, g in yy.groupby(levels)))
                if total > 0 and 1 - within / total >= 0.9:
                    out[str(col)] = {"kind": "levels", "share_explained": _round(1 - within / total),
                                     "n": int(mask.sum())}
    return out


def column_summaries(df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[str]]:
    """Per-column summaries a model may see, and the identity columns left out."""
    from kalos.normalize.payload import build_payload

    payload, dropped = build_payload(df, max_sample=1)
    dtype = {c["header"]: c["dtype"] for c in payload["columns"]}
    out: list[dict[str, Any]] = []
    for col in [c for c in df.columns if str(c) not in dropped][:MAX_COLUMNS]:
        s = df[col]
        name = str(col)
        entry: dict[str, Any] = {"column": name, "header_unit": _header_unit(name),
                                 "n": int(s.notna().sum()), "missing": int(s.isna().sum()),
                                 "distinct": int(s.nunique(dropna=True))}
        kind = dtype.get(name, "categorical")
        xn = _numeric(s)
        if kind in ("numeric", "numeric+unit") and xn.notna().any():
            entry.update(kind="number", min=_round(float(xn.min())), median=_round(float(xn.median())),
                         max=_round(float(xn.max())))
        elif kind == "free-text":
            entry.update(kind="free text")
        else:
            entry["kind"] = "label"
            if entry["distinct"] <= 20:
                entry["levels"] = sorted(map(str, s.dropna().unique()))[:MAX_LEVELS_SHOWN]
        out.append(entry)
    return out, dropped


def engine_role(entry: dict[str, Any], *, target: str | None, leaks: dict[str, Any],
                outcome_hint: re.Pattern[str]) -> tuple[str, str]:
    """The engine's own role and reason for one column (the offline answer)."""
    name = entry["column"]
    if name == target:
        return "outcome", "This is the target you chose."
    if name in leaks:
        return "derived_from_target", "Its values follow the target almost exactly, so it reads as computed from the result."
    if entry["kind"] == "free text" or _is_identifier(name) or (
        entry["kind"] == "label" and entry["distinct"] == entry["n"] and entry["n"] > 1
    ):
        return "id", "The name or the values read as an identifier or free text."
    named = _name_role(name)
    if named == "clone":
        return "clone", "The name reads as a clone or strain label."
    if named == "covariate":
        return "covariate", "The name reads as something recorded about the cells, not chosen."
    if named == "batch_group":
        return "batch_group", "The name reads as a batch, lot, plate or day."
    if entry["kind"] == "number" and outcome_hint.search(name):
        return "outcome", "The name reads as a measured result."
    return "setting", "A column with values you could choose before a run."


def _evidence(entry: dict[str, Any], leaks: dict[str, Any], target: str | None) -> dict[str, str]:
    """Role -> the engine's evidence for it, for one column."""
    name = entry["column"]
    ev: dict[str, str] = {}
    if name == target:
        ev["outcome"] = "you chose this column as the target"
    if name in leaks:
        ev["derived_from_target"] = "target_leak: its values track the target almost exactly"
    if _is_identifier(name):
        ev["id"] = "the name matches the engine's identifier rule"
    named = _name_role(name)
    if named:
        ev[named] = f"the name matches the engine's {named.replace('_', ' ')} rule"
    return ev


# --- the model reply ------------------------------------------------------------- #


def _validator(summaries: list[dict[str, Any]]) -> Any:
    by_name = {s["column"]: s for s in summaries}

    def validate(reply: str) -> dict[str, dict[str, Any]]:
        obj = parse_json_object(reply)
        items = obj.get("columns")
        if not isinstance(items, list) or not items:
            raise ValidationFailed("schema", "no columns list")
        out: dict[str, dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("column"), str):
                raise ValidationFailed("schema", "an entry has no column")
            col = item["column"]
            if col not in by_name:
                raise ValidationFailed("unknown_column", "a suggestion names a column the sheet does not have")
            role = str(item.get("role") or "").strip().lower()
            if role not in ROLES:
                raise ValidationFailed("schema", "a role is not one of the roles")
            unit = item.get("unit")
            if isinstance(unit, str) and unit.strip().lower() in {"", "null", "none", "n/a", "unitless"}:
                unit = None
            if unit is not None and (not isinstance(unit, str) or unit.strip() not in KNOWN_UNITS):
                raise ValidationFailed("schema", "a unit is not one of the known units")
            reason = " ".join(str(item.get("reason") or "").split())
            if not reason or len(reason) > MAX_REASON_CHARS:
                raise ValidationFailed("schema", "a reason is missing or too long")
            summary = by_name[col]
            allowed = numbers_from(v for v in summary.values() if not isinstance(v, list)) | numbers_from(
                summary.get("levels") or []) | number_keys(col)
            if number_keys(reason) - allowed:
                raise ValidationFailed("mismatched_number", "a reason states a number the statistics do not hold")
            if identifiers_in(reason) - set(by_name):
                raise ValidationFailed("unknown_column", "a reason names a column the sheet does not have")
            if banned_in(reason) or claims_in(reason):
                raise ValidationFailed("unsupported_claim", "a reason makes a claim")
            if col in out:
                raise ValidationFailed("schema", "a column is listed twice")
            out[col] = {"role": role, "unit": unit.strip() if isinstance(unit, str) else None, "reason": reason}
        return out

    return validate


# --- the result ------------------------------------------------------------------ #


def suggest(df: pd.DataFrame, *, target: str | None, cascade: Cascade | None) -> dict[str, Any]:
    """Suggestions for every non-identity column of `df`. `target` is the
    user's chosen target, if any. Applies nothing."""
    import json

    from kalos.domains.bioprocess import BIOPROCESS_PROFILE

    if target is not None and target not in df.columns:
        target = None
    summaries, dropped = column_summaries(df)
    leaks = leak_evidence(df, target)
    validate = _validator(summaries)
    prompt = "Columns:\n" + json.dumps(summaries, ensure_ascii=False, default=str)
    if target:
        prompt += f"\n\nThe user chose {target!r} as the target (role outcome)."
    result: CascadeResult[dict[str, dict[str, Any]]]
    if cascade is None:
        result = CascadeResult({}, "offline", [], offline=True)
    else:
        result = cascade.run("setup_suggestions", _SYSTEM, prompt, validate, lambda: {}, json_mode=True)
    model_out = result.value

    suggestions: list[dict[str, Any]] = []
    for entry in summaries:
        name = entry["column"]
        e_role, e_reason = engine_role(entry, target=target, leaks=leaks, outcome_hint=BIOPROCESS_PROFILE.outcome_hint)
        evidence = _evidence(entry, leaks, target)
        m = model_out.get(name)
        role, reason, source = (m["role"], m["reason"], "model") if m else (e_role, e_reason, "engine")
        if name == target and role != "outcome":
            role, reason, source = "outcome", "This is the target you chose.", "engine"
        if evidence and role in evidence:
            confidence = "high"
        elif evidence:
            confidence = "low"
        else:
            confidence = "medium" if source == "model" else "low"
        unit_header = entry["header_unit"]
        unit = unit_header or (m["unit"] if m else None)
        unit_source = "header" if unit_header else ("model" if unit else None)
        if unit_header and m and m["unit"] and m["unit"] != unit_header:
            confidence = "low" if confidence == "medium" else confidence
        suggestions.append({
            "column": name,
            "role": role,
            "roles_key": _roles_key(role, name, target, entry),
            "unit": unit,
            "unit_source": unit_source,
            "reason": reason,
            "reason_source": source,
            "confidence": confidence,
            "evidence": [{"role": r, "note": n} for r, n in sorted(evidence.items())],
            "engine_role": e_role,
            "requires_confirmation": True,
        })
    return {
        "status": "suggestions",
        "applied": False,
        "notice": NOTICE,
        "target": target,
        "role_vocabulary": [{"role": r, **v} for r, v in ROLES.items()],
        "suggestions": suggestions,
        "left_out_identity_columns": len(dropped),
        "roles_draft": _roles_draft(suggestions, target),
        "confidence_basis": ("Set by the engine, never by the model: high when the engine's own "
                             "evidence agrees with the role, low when it disagrees or the evidence "
                             "is missing for an engine-only suggestion, medium otherwise."),
        **result.report(),
    }


def _roles_key(role: str, name: str, target: str | None, entry: dict[str, Any]) -> str:
    if role == "outcome":
        return "target" if name == target else "outputs"
    if role == "setting" and entry.get("kind") == "label":
        return "categoricals"
    return ROLES[role]["roles_key"]


def _roles_draft(suggestions: list[dict[str, Any]], target: str | None) -> dict[str, Any]:
    """The suggestions as a `ColumnRoles`-shaped object, for the user to review
    and post themselves. Not saved anywhere."""
    draft: dict[str, Any] = {"target": target, "features": [], "categoricals": [], "groups": None,
                             "ids": [], "strain": None, "covariates": [], "outputs": [],
                             "unplaced": []}
    for s in suggestions:
        key, col = s["roles_key"], s["column"]
        if key == "categoricals":
            draft["features"].append(col)
            draft["categoricals"].append(col)
        elif key in ("groups", "strain"):
            if draft[key] is None:
                draft[key] = col
            else:  # one group and one clone column at most: the user picks
                draft["unplaced"].append({"column": col, "role": s["role"]})
        elif key == "target":
            continue
        else:
            draft[key].append(col)
    return draft


__all__ = ["KNOWN_UNITS", "NOTICE", "ROLES", "column_summaries", "leak_evidence", "suggest"]
