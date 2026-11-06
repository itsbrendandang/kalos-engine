"""`anonymize=True` shaping: pseudonymize identifier-type column names, categorical
level values and sheet names in responses and refusals. Pseudonyms are stable
(salted hash), so the report stays internally consistent across fields.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence

from kalos.data.anonymizer import _hash
from kalos.portal.analysis_parts.identifiers import _is_identifier_name
from kalos.portal.uploads import ParsedUpload, UploadRejected, clip_name


def _anonymize_result(
    result: dict, *, categorical_levels: Mapping[str, Sequence[str]] | None = None
) -> dict:
    """Replace identifier-type column names and values in the response with stable
    pseudonyms (`anonymize=True`).

    Identifier-like columns (`is_identifier_name`: ids, samples, operators,
    analysts, technicians, names, emails) are renamed everywhere a name appears:
    provenance, features, drivers, correlation labels, the response-surface axes
    and fixed recipe, the proposal recipes, and the validation prose. Their
    CATEGORICAL LEVEL VALUES are pseudonymized too (`categorical_levels`, the levels
    of each kept categorical feature): a level such as "Jane Smith 0" is a person's
    name, and leaked verbatim in the recipes before (review H4). Process variables
    (Methanol, pH, sample_volume_L) and their values keep their real names because
    the owner UI shows them. The mapping is stable (same name -> same pseudonym)
    via the anonymizer's irreversible hash, so the report stays internally
    consistent across fields.
    """
    def alias(name: str) -> str:
        return f"col_{_hash(name)[:8]}" if _is_identifier_name(name) else name

    id_levels: dict[str, set[str]] = {
        str(c): {str(v) for v in levels}
        for c, levels in (categorical_levels or {}).items()
        if _is_identifier_name(str(c))
    }

    def alias_level(column: str, value: object) -> object:
        if isinstance(value, str) and value in id_levels.get(column, ()):
            return f"level_{_hash(column + chr(31) + value)[:8]}"
        return value

    def alias_recipe(recipe: object) -> object:
        if not isinstance(recipe, dict):
            return recipe
        return {alias(str(k)): alias_level(str(k), v) for k, v in recipe.items()}

    out = dict(result)
    if out.get("group_col"):
        out["group_col"] = alias(out["group_col"])
    if isinstance(out.get("target"), str):
        out["target"] = alias(out["target"])
    for key in ("targets", "features", "categorical_features", "proposal_features"):
        if isinstance(out.get(key), list):
            out[key] = [alias(str(c)) for c in out[key]]
    if isinstance(out.get("feature_ranges"), dict):
        out["feature_ranges"] = {alias(str(k)): v for k, v in out["feature_ranges"].items()}
    if isinstance(out.get("fixed_inputs"), dict):
        out["fixed_inputs"] = {alias(str(k)): v for k, v in out["fixed_inputs"].items()}
    shapes = out.get("gp_shapes")
    if isinstance(shapes, dict) and isinstance(shapes.get("features"), list):
        out["gp_shapes"] = {
            **shapes,
            "features": [
                {**f, "name": alias(str(f["name"]))}
                if isinstance(f, dict) and "name" in f else f
                for f in shapes["features"]
            ],
        }
    if isinstance(out.get("drivers"), list):
        out["drivers"] = [
            {**d, "name": alias(str(d["name"]))} if isinstance(d, dict) and "name" in d else d
            for d in out["drivers"]
        ]
    if isinstance(out.get("proposals"), list):
        out["proposals"] = [
            {**p, "recipe": alias_recipe(p["recipe"])} if isinstance(p, dict) and "recipe" in p else p
            for p in out["proposals"]
        ]
        for p in out["proposals"]:
            app = p.get("applicability") if isinstance(p, dict) else None
            if isinstance(app, dict) and isinstance(app.get("at_box_edge"), list):
                p["applicability"] = {**app, "at_box_edge": [
                    {**e, "feature": alias(str(e["feature"]))} if isinstance(e, dict) else e
                    for e in app["at_box_edge"]
                ]}
    opt = out.get("model_optimum")
    if isinstance(opt, dict) and isinstance(opt.get("recipe"), dict):
        out["model_optimum"] = {**opt, "recipe": alias_recipe(opt["recipe"])}
    corr = out.get("correlation")
    if isinstance(corr, dict) and isinstance(corr.get("labels"), list):
        out["correlation"] = {**corr, "labels": [alias(str(c)) for c in corr["labels"]]}
    surf = out.get("response_surface")
    if isinstance(surf, dict):
        surf_out = dict(surf)
        for key in ("x_feature", "y_feature"):
            if isinstance(surf_out.get(key), str):
                surf_out[key] = alias(surf_out[key])
        if isinstance(surf_out.get("fixed_recipe"), dict):
            surf_out["fixed_recipe"] = alias_recipe(surf_out["fixed_recipe"])
        out["response_surface"] = surf_out
    out["provenance"] = [
        {**row, "name": alias(row["name"])} for row in out.get("provenance", [])
    ]
    if isinstance(out.get("columns"), list):
        from kalos.portal.analysis_parts.column_kinds import anonymize_columns

        out["columns"] = anonymize_columns(
            out["columns"], alias,
            lambda column, value: f"level_{_hash(column + chr(31) + value)[:8]}",
        )

    # The validation report names columns too, in two places: a `column` field and
    # the human-readable `message` built around it ("column 'Sample Name' carries
    # ..."). Aliasing only the field would leak the real name through the prose, so
    # both are rewritten together. Anonymization is worth nothing if it covers the
    # structured copy of a name and not the sentence next to it.
    validation = out.get("validation")
    if isinstance(validation, dict):
        out["validation"] = {
            **validation,
            "findings": [_alias_named_row(f, alias) for f in validation.get("findings", [])],
            "conversions": [
                _alias_named_row(c, alias) for c in validation.get("conversions", [])
            ],
        }
    out["design_box_exclusions"] = [
        _alias_named_row(e, alias) for e in out.get("design_box_exclusions", [])
    ]
    if isinstance(out.get("auto_categorical"), list):
        out["auto_categorical"] = [alias(str(c)) for c in out["auto_categorical"]]
    drops = out.get("data_drops")
    if isinstance(drops, dict):
        out["data_drops"] = {
            **drops,
            "cells_by_column": {
                alias(str(c)): dict(r) for c, r in drops.get("cells_by_column", {}).items()
            },
            "blank_cells_as_absent": {
                alias(str(c)): n for c, n in drops.get("blank_cells_as_absent", {}).items()
            },
        }
    return out


def sheet_alias(name: str) -> str:
    """The `anonymize=True` name of a workbook sheet. Sheet names are not column
    names, but they are the uploader's words ("Tank 3 raw", "Patient 12"), so
    under anonymize every response, error and stored field carries this instead.
    Same salted-hash scheme as column aliases (`col_...`)."""
    return f"sheet_{_hash(name)[:8]}"


def anonymize_sheet_rejection(rej: UploadRejected) -> UploadRejected:
    """`rej` with every sheet name in its context and message replaced by
    `sheet_alias`: the requested `sheet`, the `sheets` list and `sheet_used`."""
    context = dict(rej.context)
    message = rej.message
    names: list[str] = []
    for key in ("sheet", "sheet_used"):
        if isinstance(context.get(key), str):
            names.append(context[key])
            context[key] = sheet_alias(context[key])
    if isinstance(context.get("sheets"), list):
        names.extend(n for n in context["sheets"] if isinstance(n, str))
        context["sheets"] = [sheet_alias(n) if isinstance(n, str) else n for n in context["sheets"]]
    for name in names:
        message = message.replace(repr(clip_name(name)), repr(sheet_alias(name)))
    return UploadRejected(message, code=rej.code, context=context)


def anonymize_parsed(parsed: ParsedUpload) -> ParsedUpload:
    """The same workbook with its sheet names aliased (see `sheet_alias`), so the
    response context, the error annotations and the stored fields all agree."""
    # `replace` keeps sheet_index and sheet_hash (computed from the real names).
    return dataclasses.replace(
        parsed,
        sheets=[sheet_alias(n) for n in parsed.sheets],
        sheet_used=sheet_alias(parsed.sheet_used) if parsed.sheet_used is not None else None,
    )


def anonymize_rejection(rej: UploadRejected) -> UploadRejected:
    """The `anonymize=True` counterpart of `_anonymize_result` for a refused
    upload: identifier-type column names in the error context (`target`,
    `available_targets`) and in its message are replaced by the same stable
    pseudonyms a successful response would use. Sheet names are not column
    names and are left as they are (a successful response returns them too)."""
    def alias(name: str) -> str:
        return f"col_{_hash(name)[:8]}" if _is_identifier_name(name) else name

    context = dict(rej.context)
    message = rej.message
    names: list[str] = []
    if isinstance(context.get("target"), str):
        names.append(context["target"])
        context["target"] = alias(context["target"])
    if isinstance(context.get("available_targets"), list):
        names.extend(n for n in context["available_targets"] if isinstance(n, str))
        context["available_targets"] = [
            alias(n) if isinstance(n, str) else n for n in context["available_targets"]
        ]
    for name in names:
        aliased = alias(name)
        if aliased != name:
            message = message.replace(repr(clip_name(name)), repr(aliased))
    return UploadRejected(message, code=rej.code, context=context)


def _alias_named_row(row: dict, alias: Callable[[str], str]) -> dict:
    """Alias `row["column"]` and scrub the real name out of `row["message"]`.

    Only rewrites the message when the alias actually differs, so a non-identifier
    column (a process feature like "Methanol", which the owner UI legitimately
    shows) is left completely untouched rather than being needlessly rewritten.
    """
    name = row.get("column")
    if not isinstance(name, str) or not name:
        return dict(row)
    aliased = alias(name)
    if aliased == name:
        return dict(row)
    out = {**row, "column": aliased}
    message = out.get("message")
    if isinstance(message, str):
        out["message"] = message.replace(name, aliased)
    return out
