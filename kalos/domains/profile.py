"""Domain abstraction: declared column roles and mixed design-space specs.

The engine (`kalos.core`) is domain-agnostic: it optimizes numeric arrays and
knows nothing about biology. This module holds the thin, torch-free layer that
maps a tabular problem in any domain onto that engine.

`DomainProfile` carries the fallback hint patterns used to INFER column roles
when the caller does not declare them. A domain's vocabulary (biology's
"titer"/"strain", say) lives in a profile such as
`kalos.domains.bioprocess.BIOPROCESS_PROFILE`, never in the engine.

`ColumnRoles` is an explicit, caller-declared schema. When supplied it skips
inference entirely, so a non-bio user declares what their columns mean instead
of renaming them to match a regex.

`DesignSpace` describes each optimized dimension as continuous (numeric bounds)
or categorical (ordered levels), and encodes/decodes between human labels and
the integer-coded numeric matrix the GP consumes.

Torch-free by contract (numpy and pandas only, both core dependencies) so it
stays importable from the lean `kalos.kit` facade without loading the
torch/botorch/gpytorch stack.
"""
from __future__ import annotations

from dataclasses import dataclass
from re import Pattern
from typing import Literal

import numpy as np
import pandas as pd

DimensionKind = Literal["continuous", "categorical"]

# Optimisation direction of the target. "maximize" is the default everywhere.
DIRECTIONS: tuple[str, ...] = ("maximize", "minimize")


class InvalidDirection(ValueError):
    """A direction other than "maximize" or "minimize" (HTTP 400 `invalid_direction`)."""

    code = "invalid_direction"

    def __init__(self) -> None:
        super().__init__('direction must be "maximize" or "minimize"')


class InvalidApplicabilityDistance(ValueError):
    """`roles.applicability_distance` that is not a positive finite number
    (HTTP 400 `invalid_applicability_distance`)."""

    code = "invalid_applicability_distance"

    def __init__(self) -> None:
        super().__init__("roles.applicability_distance must be a positive number")


def parse_applicability_distance(value: object) -> float | None:
    """None when not given, else a positive finite float. Raises
    `InvalidApplicabilityDistance` for anything else (bools and strings too)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidApplicabilityDistance()
    v = float(value)
    if not np.isfinite(v) or v <= 0:
        raise InvalidApplicabilityDistance()
    return v


class InvalidExplorationDesign(ValueError):
    """`roles.exploration_design` other than "auto" or "off" (HTTP 400
    `invalid_exploration_design`)."""

    code = "invalid_exploration_design"

    def __init__(self) -> None:
        super().__init__('roles.exploration_design must be "auto" or "off"')


def parse_exploration_design(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"auto", "off"}:
        return value.strip().lower()
    raise InvalidExplorationDesign()


class InvalidAllowBeyond(ValueError):
    """`roles.allow_beyond` that is not a fraction in [0, 1], or a mapping of
    input name to such a fraction (HTTP 400 `invalid_allow_beyond`)."""

    code = "invalid_allow_beyond"

    def __init__(self) -> None:
        super().__init__(
            "roles.allow_beyond must be a number from 0 to 1 (0.10 = 10% past the tried "
            'range), or an object of input name to such a number ("*" for every other input)'
        )


def parse_allow_beyond(value: object) -> tuple[tuple[str, float], ...]:
    """`allow_beyond` as sorted (input, fraction) pairs; "*" is the default for
    every input not named. A plain number means "*". Empty when not given."""
    if value is None:
        return ()

    def frac(v: object) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise InvalidAllowBeyond()
        f = float(v)
        if not np.isfinite(f) or f < 0 or f > 1:
            raise InvalidAllowBeyond()
        return f

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (("*", frac(value)),)
    if isinstance(value, dict):
        return tuple(sorted((str(k), frac(v)) for k, v in value.items()))
    raise InvalidAllowBeyond()


class InvalidPerRecipeNoise(ValueError):
    """`roles.per_recipe_noise` that is not true or false
    (HTTP 400 `invalid_per_recipe_noise`)."""

    code = "invalid_per_recipe_noise"

    def __init__(self) -> None:
        super().__init__("roles.per_recipe_noise must be true or false")


def parse_per_recipe_noise(value: object) -> bool:
    """False when not given, else the boolean. Raises `InvalidPerRecipeNoise`
    for anything that is not a JSON boolean (strings and numbers too)."""
    if value is None:
        return False
    if not isinstance(value, bool):
        raise InvalidPerRecipeNoise()
    return value
# At most this many `roles.linear_constraints` per upload.
MAX_LINEAR_CONSTRAINTS = 20
LINEAR_OPS: tuple[str, ...] = ("<=", ">=", "==")


class InvalidLinearConstraint(ValueError):
    """A malformed `roles.linear_constraints` entry, or one that names an input
    that is not a continuous input of the design (HTTP 400 `invalid_linear_constraint`)."""

    code = "invalid_linear_constraint"


@dataclass(frozen=True)
class LinearConstraint:
    """`sum(coefficient * input) op rhs` on continuous inputs, in the inputs' own
    units (`terms` keeps the caller's order)."""

    terms: tuple[tuple[str, float], ...]
    op: str
    rhs: float
    label: str | None = None

    def to_dict(self) -> dict:
        return {
            "terms": {k: v for k, v in self.terms}, "op": self.op, "rhs": self.rhs,
            "label": self.label,
        }

    def text(self) -> str:
        """Plain form, e.g. "Glucose + Glycerol <= 30"."""
        parts: list[str] = []
        for i, (name, c) in enumerate(self.terms):
            mag = abs(c)
            coef = "" if mag == 1 else f"{mag:g} * "
            sign = ("- " if c < 0 else "") if i == 0 else ("- " if c < 0 else "+ ")
            parts.append(f"{sign}{coef}{name}")
        return f"{' '.join(parts)} {self.op} {self.rhs:g}"


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if np.isfinite(v) else None


def parse_linear_constraints(value: object) -> tuple[LinearConstraint, ...]:
    """`roles.linear_constraints`: a list of `{terms: {input: coefficient}, op:
    "<=" | ">=" | "==", rhs: number, label?: string}`. Empty when not given.
    Raises `InvalidLinearConstraint` with a plain message for anything else.
    Whether each input is a continuous input is checked against the sheet later."""
    if value is None:
        return ()
    if not isinstance(value, list):
        raise InvalidLinearConstraint("roles.linear_constraints must be a list")
    if len(value) > MAX_LINEAR_CONSTRAINTS:
        raise InvalidLinearConstraint(
            f"roles.linear_constraints has {len(value)} entries; the most allowed is "
            f"{MAX_LINEAR_CONSTRAINTS}"
        )
    out: list[LinearConstraint] = []
    for k, item in enumerate(value, start=1):
        where = f"Linear constraint {k}"
        if not isinstance(item, dict):
            raise InvalidLinearConstraint(f"{where} must be an object with terms, op and rhs")
        terms = item.get("terms")
        if not isinstance(terms, dict) or not terms:
            raise InvalidLinearConstraint(f"{where} needs at least one input in terms")
        pairs: list[tuple[str, float]] = []
        for name, coef in terms.items():
            c = _finite_number(coef)
            if not str(name).strip() or c is None:
                raise InvalidLinearConstraint(
                    f"{where}: each term needs an input name and a finite number"
                )
            pairs.append((str(name), c))
        if not any(c for _, c in pairs):
            raise InvalidLinearConstraint(f"{where}: every coefficient is zero")
        op = item.get("op")
        if op not in LINEAR_OPS:
            raise InvalidLinearConstraint(f'{where}: op must be "<=", ">=" or "=="')
        rhs = _finite_number(item.get("rhs"))
        if rhs is None:
            raise InvalidLinearConstraint(f"{where}: rhs must be a finite number")
        label = item.get("label")
        if label is not None and not isinstance(label, str):
            raise InvalidLinearConstraint(f"{where}: label must be text")
        out.append(LinearConstraint(tuple(pairs), str(op), rhs, (label or None)))
    return tuple(out)


# At most this many `roles.specs` per upload.
MAX_SPECS = 20


class InvalidSpecs(ValueError):
    """A malformed `roles.specs` entry, or a spec on a model input
    (HTTP 400 `invalid_specs`)."""

    code = "invalid_specs"


@dataclass(frozen=True)
class Spec:
    """A specification on an outcome column: met when `min <= value <= max`
    (either side may be open)."""

    column: str
    min: float | None = None
    max: float | None = None

    def to_dict(self) -> dict:
        return {"column": self.column, "min": self.min, "max": self.max}


def parse_specs(value: object) -> tuple[Spec, ...]:
    """`roles.specs`: a list of `{column, min?, max?}` on outcome columns, at
    least one limit each, `min <= max`, one entry per column. Empty when not
    given. Raises `InvalidSpecs` with a plain message for anything else."""
    if value is None:
        return ()
    if not isinstance(value, list):
        raise InvalidSpecs("roles.specs must be a list")
    if len(value) > MAX_SPECS:
        raise InvalidSpecs(f"roles.specs has {len(value)} entries; the most allowed is {MAX_SPECS}")
    out: list[Spec] = []
    for k, item in enumerate(value, start=1):
        where = f"Spec {k}"
        if not isinstance(item, dict):
            raise InvalidSpecs(f"{where} must be an object with column, min and/or max")
        column = item.get("column")
        if not isinstance(column, str) or not column.strip():
            raise InvalidSpecs(f"{where} needs a column name")
        limits: list[float | None] = []
        for key in ("min", "max"):
            raw = item.get(key)
            if raw is None:
                limits.append(None)
                continue
            v = _finite_number(raw)
            if v is None:
                raise InvalidSpecs(f"{where}: {key} must be a finite number")
            limits.append(v)
        lo, hi = limits
        if lo is None and hi is None:
            raise InvalidSpecs(f"{where} needs a min, a max, or both")
        if lo is not None and hi is not None and lo > hi:
            raise InvalidSpecs(f"{where}: min is above max")
        if any(sp.column == column for sp in out):
            raise InvalidSpecs(f"{where}: {column!r} already has a spec; give min and max in one entry")
        out.append(Spec(column, lo, hi))
    return tuple(out)


def parse_direction(value: object) -> str | None:
    """None for no stated direction, else "maximize"/"minimize". Raises
    `InvalidDirection` for anything else (case and surrounding spaces are
    ignored)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str) and value.strip().lower() in DIRECTIONS:
        return value.strip().lower()
    raise InvalidDirection()


@dataclass(frozen=True)
class DomainProfile:
    """Fallback regexes for inferring column roles when none are declared.

    Kept as data (not code in the engine) so a new domain is a new profile, not
    an engine edit. `outcome_hint` matches measured-output columns (candidate
    targets, and excluded from features to avoid leakage); `target_pref` picks a
    preferred target among the outcomes; `id_hint` matches identifier columns to
    drop; `group_hint` matches a replicate-grouping column.
    """

    name: str
    outcome_hint: Pattern[str]
    target_pref: Pattern[str]
    id_hint: Pattern[str]
    group_hint: Pattern[str]
    # Direction used when neither roles nor the request states one.
    direction: str = "maximize"


@dataclass(frozen=True)
class ColumnRoles:
    """An explicit, caller-declared column schema for the analyze path.

    `target` is the column to optimize (required). `features` are the process
    inputs to model; empty means "infer the features from the profile as
    usual". `groups` is the optional replicate-grouping column. `ids` are
    columns to ignore. `categoricals` names the feature columns whose values are
    unordered labels rather than numbers; their levels are derived from the data.
    `sparse` names continuous feature columns where a BLANK cell means "absent"
    (a component not added, so 0). Only these, and concentration-type columns by
    name, take a blank as 0; every other blank feature cell drops its run.
    `direction` is "maximize" or "minimize"; None means not stated (the
    request or the profile decides; default "maximize").
    `applicability_distance` is a lab-set threshold for
    `proposals[].applicability.far_from_data` (None: the product default).
    `allow_beyond` lets the search go past the tried range of an input, as a
    fraction of that range at each end (("*", 0.1) = 10% for every input).
    Empty: the search box is the tried range.
    `per_recipe_noise` (default False) lets the proposal model use each
    recipe's own replicate spread, shrunk toward the shared one, when the
    noise is uneven across recipes (`noise.per_recipe` in the analysis).
    `linear_constraints` are hard linear limits on continuous inputs (for
    example a mixture that sums to 100); every proposal meets them.
    `specs` are limits on outcome columns; each proposal reports the chance
    that all of them are met at once (advisory, the batch is unchanged).
    """

    target: str
    features: tuple[str, ...] = ()
    groups: str | None = None
    ids: tuple[str, ...] = ()
    categoricals: tuple[str, ...] = ()
    sparse: tuple[str, ...] = ()
    direction: str | None = None
    applicability_distance: float | None = None
    allow_beyond: tuple[tuple[str, float], ...] = ()
    per_recipe_noise: bool = False
    exploration_design: str | None = None
    linear_constraints: tuple[LinearConstraint, ...] = ()
    specs: tuple[Spec, ...] = ()

    @classmethod
    def from_dict(cls, data: dict) -> ColumnRoles:
        """Parse a roles object (e.g. from a JSON request body).

        `target` is required and must be a non-empty string. List fields accept
        a list of strings (or a single string); missing fields default to empty.
        """
        target = data.get("target")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("roles.target must be a non-empty column name")

        def as_tuple(value: object) -> tuple[str, ...]:
            if value is None:
                return ()
            if isinstance(value, str):
                return (value,) if value.strip() else ()
            if isinstance(value, (list, tuple)):
                return tuple(str(v) for v in value if str(v).strip())
            raise ValueError("roles list fields must be a string or list of strings")

        groups = data.get("groups")
        if groups is not None and not isinstance(groups, str):
            raise ValueError("roles.groups must be a column name or null")
        return cls(
            target=target,
            features=as_tuple(data.get("features")),
            groups=(groups or None),
            ids=as_tuple(data.get("ids")),
            categoricals=as_tuple(data.get("categoricals")),
            sparse=as_tuple(data.get("sparse")),
            direction=parse_direction(data.get("direction")),
            applicability_distance=parse_applicability_distance(
                data.get("applicability_distance")
            ),
            allow_beyond=parse_allow_beyond(data.get("allow_beyond")),
            per_recipe_noise=parse_per_recipe_noise(data.get("per_recipe_noise")),
            exploration_design=parse_exploration_design(data.get("exploration_design")),
            linear_constraints=parse_linear_constraints(data.get("linear_constraints")),
            specs=parse_specs(data.get("specs")),
        )


@dataclass(frozen=True)
class Dimension:
    """One optimized dimension: continuous with `[lower, upper]`, or categorical
    with an ordered tuple of `levels` (its integer code is the level index)."""

    name: str
    kind: DimensionKind
    lower: float = 0.0
    upper: float = 1.0
    levels: tuple[str, ...] = ()

    @property
    def is_categorical(self) -> bool:
        return self.kind == "categorical"


@dataclass(frozen=True)
class DesignSpace:
    """A mixed continuous/categorical design space over an ordered set of dims.

    The engine sees an integer-coded numeric matrix (categorical label -> level
    index) and a `(2, d)` bounds box; this class owns the mapping in both
    directions so labels never leak into the GP and integer proposals decode
    back to labels for the client.
    """

    dims: tuple[Dimension, ...]

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.dims]

    @property
    def cat_dims(self) -> list[int]:
        """Indices of the categorical dimensions (BoTorch `cat_dims`)."""
        return [i for i, d in enumerate(self.dims) if d.is_categorical]

    @property
    def cont_indices(self) -> list[int]:
        return [i for i, d in enumerate(self.dims) if not d.is_categorical]

    @property
    def has_categoricals(self) -> bool:
        return bool(self.cat_dims)

    @property
    def cat_cardinalities(self) -> list[int]:
        """Level count per categorical dim, aligned to `cat_dims`."""
        return [len(self.dims[i].levels) for i in self.cat_dims]

    def bounds(self) -> np.ndarray:
        """Full `(2, d)` box: continuous dims use their range, categorical dims
        span `[0, n_levels - 1]` over their integer codes."""
        lo: list[float] = []
        hi: list[float] = []
        for d in self.dims:
            if d.is_categorical:
                lo.append(0.0)
                hi.append(float(max(len(d.levels) - 1, 0)))
            else:
                lo.append(float(d.lower))
                hi.append(float(d.upper))
        return np.vstack([lo, hi])

    def cont_bounds(self) -> np.ndarray:
        """`(2, n_cont)` box over just the continuous dims, for input
        normalization (`Normalize(indices=cont_indices)`)."""
        idx = self.cont_indices
        if not idx:
            return np.empty((2, 0), float)
        lo = [float(self.dims[i].lower) for i in idx]
        hi = [float(self.dims[i].upper) for i in idx]
        return np.vstack([lo, hi])

    def encode_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Encode a dataframe (columns named as in `dims`) to the numeric matrix
        the GP consumes: continuous columns coerced to float, categorical
        columns mapped to their integer level codes."""
        cols: list[np.ndarray] = []
        for d in self.dims:
            if d.is_categorical:
                code = {lvl: i for i, lvl in enumerate(d.levels)}
                mapped = df[d.name].astype(str).map(code)
                cols.append(mapped.to_numpy(float))
            else:
                cols.append(pd.to_numeric(df[d.name], errors="coerce").to_numpy(float))
        if not cols:
            return np.empty((len(df), 0), float)
        return np.column_stack(cols)

    def decode_row(self, x) -> list:
        """Decode one proposal row back to labels: categorical codes rounded and
        clamped to a valid level, continuous coordinates rounded to 3 places."""
        vals = np.asarray(x, float).reshape(-1)
        out: list = []
        for i, d in enumerate(self.dims):
            if d.is_categorical:
                if not d.levels:
                    out.append(None)
                    continue
                k = int(round(float(vals[i])))
                k = max(0, min(k, len(d.levels) - 1))
                out.append(d.levels[k])
            else:
                # Rounded to 3 places, then clipped into the box (replay F9:
                # rounding put values up to 5e-4 outside it). The clip bounds
                # sit inside both the box and its 6-place `feature_ranges`.
                lo = max(float(d.lower), round(float(d.lower), 6))
                hi = min(float(d.upper), round(float(d.upper), 6))
                v = round(float(vals[i]), 3)
                out.append(min(max(v, lo), hi) if lo <= hi else v)
        return out


def build_design_space(
    df: pd.DataFrame,
    features: list[str],
    categoricals: tuple[str, ...] = (),
    *,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> DesignSpace:
    """Build a `DesignSpace` from a run sheet and a feature list.

    `features` is the ordered set of feature columns. `categoricals` is the
    subset to treat as unordered labels; their levels are the sorted unique
    non-blank values (sorted for determinism). Continuous bounds come from
    `bounds[name]` when provided, else from the column's observed min/max.
    """
    cat = set(categoricals)
    dims: list[Dimension] = []
    for name in features:
        if name in cat:
            values = {
                str(v)
                for v in df[name].dropna().tolist()
                if str(v).strip() != ""
            }
            dims.append(
                Dimension(name, "categorical", levels=tuple(sorted(values)))
            )
        else:
            if bounds is not None and name in bounds:
                lo, hi = bounds[name]
            else:
                col = pd.to_numeric(df[name], errors="coerce")
                lo, hi = float(col.min()), float(col.max())
            dims.append(Dimension(name, "continuous", lower=lo, upper=hi))
    return DesignSpace(tuple(dims))


__all__ = [
    "ColumnRoles",
    "DesignSpace",
    "Dimension",
    "DimensionKind",
    "DomainProfile",
    "build_design_space",
]
