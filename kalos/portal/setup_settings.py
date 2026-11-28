"""Campaign setup: goal direction, operating boundaries and target unit.

The setup document is stored per campaign in `campaign_settings` (kind
``setup``, `kalos.portal.campaign_settings`) and edited over HTTP by
`kalos.portal.setup_api`. This module holds everything that is not HTTP:

- `normalize_setup`: structural validation of a document (types, caps, finite
  numbers, ``lo < hi``, no zero coefficients, the unit's characters). Also run on
  every STORED document before it is used, so a hand-edited row fails closed.
- `validate_against_dataset`: the document checked against the campaign's
  latest analysis (features must be continuous features of that analysis, the
  target is not a feature, no categoricals in bounds or constraints, bounds must
  overlap the observed range) plus the LP feasibility check
  (`check_feasible`). A document that admits no recipe is refused, never
  stored with a constraint quietly ignored.
- `setup_hash`: a stable hash of the normalized document. The analysis records
  the hash it was computed with (`analysis.setup.hash`) so the GET route can say
  whether the stored setup is the one the current analysis used.
- `plan_setup`: the analysis-time conversion into optimizer arguments
  (intersected proposal box, ``(indices, coefficients, rhs)`` constraints, the
  goal), the post-check every proposal must pass, and the analysis `setup` block.

POLICY: user bounds beyond the observed data range are ALLOWED (stored as typed)
but never widen the search: proposals stay inside the observed design box (the
engine does not extrapolate the surrogate), so the bound applied is the
intersection, and a bound reaching past the data is flagged ``extrapolates:
true`` in the GET response and in the analysis. A bound that does not overlap
the observed range at all is refused (no proposal could satisfy it).

Torch-free (numpy + scipy only), so the portal can import it at start-up.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("kalos.portal")

GOALS = ("maximize", "minimize")
OPS = ("<=", ">=")
MAX_BOUND_FEATURES = 64
MAX_LINEAR_CONSTRAINTS = 16
MAX_CONSTRAINT_TERMS = 64
MAX_TARGET_UNIT_CHARS = 32
MAX_FEATURE_NAME_CHARS = 256
# Numbers beyond this magnitude are refused: they are never a real setpoint and
# make the feasibility LP numerically meaningless.
MAX_ABS_VALUE = 1e12
# A constraint set whose feasible region has (relatively) no interior is refused:
# the optimizer samples starting points inside the region and cannot find any.
MIN_INTERIOR_SLACK = 1e-6
# Same relative slack the optimizer's own constraint check uses
# (`kalos.core.optimize._LIN_TOL`), plus a relative box tolerance.
LIN_TOL = 1e-6
BOX_TOL = 1e-9

DEFAULT_SETUP: dict[str, Any] = {
    "goal": "maximize", "bounds": {}, "linear_constraints": [], "target_unit": None,
}
# Clone / strain keys (docs/CLONES.md). Optional: a document that does not set
# them normalizes WITHOUT them, so every existing setup keeps its hash.
STRAIN_KEYS = ("strain_column", "strain_policy", "strain_level", "covariates")
STRAIN_POLICIES = ("fixed", "factor")
MAX_CLONE_LABEL_CHARS = 64
MAX_COVARIATES = 32

SIGN_CONVENTION = (
    "drivers[].rho, cv_spearman and gp_shapes describe the target as measured "
    "(a positive rho means the target rises with the feature) whatever the goal; "
    "with goal=minimize a NEGATIVE driver is the one that helps."
)


class SetupInvalid(ValueError):
    """A setup document that cannot be stored or applied. `code` is machine
    readable; `reason` and `what_would_fix_it` never quote caller input (feature
    names are referred to by position); `context` carries only engine-known data
    (e.g. the campaign's own continuous feature names)."""

    def __init__(
        self, code: str, reason: str, what_would_fix_it: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.what_would_fix_it = what_would_fix_it
        self.context = dict(context or {})

    def to_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {"error": self.reason, "code": self.code}
        if self.what_would_fix_it:
            body["what_would_fix_it"] = self.what_would_fix_it
        body.update(self.context)
        return body


# --- structural normalization -------------------------------------------------- #


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SetupInvalid("invalid_setup", f"{where} must be a number")
    v = float(value)
    if not math.isfinite(v):
        raise SetupInvalid("invalid_setup", f"{where} must be finite")
    if abs(v) > MAX_ABS_VALUE:
        raise SetupInvalid(
            "invalid_setup", f"{where} is out of range (|value| must be at most {MAX_ABS_VALUE:g})"
        )
    return v


def _feature_name(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise SetupInvalid("invalid_setup", f"{where} must be a feature name (string)")
    if not 1 <= len(value) <= MAX_FEATURE_NAME_CHARS:
        raise SetupInvalid(
            "invalid_setup", f"{where} must be 1-{MAX_FEATURE_NAME_CHARS} characters"
        )
    if any(unicodedata.category(ch).startswith("C") for ch in value):
        raise SetupInvalid("invalid_setup", f"{where} must not contain control characters")
    return value


def clean_target_unit(value: Any) -> str | None:
    """The stored unit: stripped, None when empty. Refuses non-strings, more than
    `MAX_TARGET_UNIT_CHARS` characters, and any control / format / unassigned
    character (Unicode category C*), the same rule the analysis applies."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise SetupInvalid("invalid_setup", "target_unit must be a string or null")
    text = value.strip()
    if not text:
        return None
    if len(text) > MAX_TARGET_UNIT_CHARS:
        raise SetupInvalid(
            "invalid_setup", f"target_unit must be at most {MAX_TARGET_UNIT_CHARS} characters"
        )
    # Category C* covers Cc controls AND Cf format characters (bidi overrides
    # such as U+202E, zero-width joiners): text that would display differently
    # from what it contains.
    if any(unicodedata.category(ch).startswith("C") for ch in text):
        raise SetupInvalid(
            "invalid_setup", "target_unit must not contain control or format characters"
        )
    return text


def normalize_setup(raw: Mapping[str, Any]) -> dict[str, Any]:
    """A canonical setup document, or `SetupInvalid`. Missing keys take their
    defaults (goal maximize, no bounds, no constraints, no unit). Bounds are
    sorted by name; constraint terms by name; constraint order is kept."""
    if not isinstance(raw, Mapping):
        raise SetupInvalid("invalid_setup", "the setup must be a JSON object")
    unknown = set(raw) - set(DEFAULT_SETUP) - set(STRAIN_KEYS)
    if unknown:
        raise SetupInvalid("invalid_setup", "the setup has unknown keys")
    goal = raw.get("goal", "maximize")
    if goal not in GOALS:
        raise SetupInvalid("invalid_setup", "goal must be 'maximize' or 'minimize'")

    bounds_raw = raw.get("bounds") or {}
    if not isinstance(bounds_raw, Mapping):
        raise SetupInvalid("invalid_setup", "bounds must be an object of {feature: {lo, hi}}")
    if len(bounds_raw) > MAX_BOUND_FEATURES:
        raise SetupInvalid("invalid_setup", f"bounds may name at most {MAX_BOUND_FEATURES} features")
    bounds: dict[str, dict[str, float]] = {}
    for k, (name, spec) in enumerate(sorted(bounds_raw.items(), key=lambda kv: str(kv[0]))):
        where = f"bounds entry {k + 1}"
        name = _feature_name(name, where)
        if not isinstance(spec, Mapping) or set(spec) != {"lo", "hi"}:
            raise SetupInvalid("invalid_setup", f"{where} must be exactly {{lo, hi}}")
        lo = _number(spec["lo"], f"{where} lo")
        hi = _number(spec["hi"], f"{where} hi")
        if not lo < hi:
            raise SetupInvalid("invalid_bounds", f"{where}: lo must be strictly less than hi")
        bounds[name] = {"lo": lo, "hi": hi}

    cons_raw = raw.get("linear_constraints") or []
    if not isinstance(cons_raw, Sequence) or isinstance(cons_raw, (str, bytes)):
        raise SetupInvalid("invalid_setup", "linear_constraints must be a list")
    if len(cons_raw) > MAX_LINEAR_CONSTRAINTS:
        raise SetupInvalid(
            "invalid_setup", f"at most {MAX_LINEAR_CONSTRAINTS} linear constraints are allowed"
        )
    constraints: list[dict[str, Any]] = []
    for k, c in enumerate(cons_raw):
        where = f"linear_constraints[{k}]"
        if not isinstance(c, Mapping) or set(c) != {"coefficients", "op", "rhs"}:
            raise SetupInvalid(
                "invalid_setup", f"{where} must be exactly {{coefficients, op, rhs}}"
            )
        coefs = c["coefficients"]
        if not isinstance(coefs, Mapping) or not coefs:
            raise SetupInvalid("invalid_constraint", f"{where}: coefficients must be a non-empty object")
        if len(coefs) > MAX_CONSTRAINT_TERMS:
            raise SetupInvalid(
                "invalid_constraint", f"{where}: at most {MAX_CONSTRAINT_TERMS} terms are allowed"
            )
        terms: dict[str, float] = {}
        for t, (name, coef) in enumerate(sorted(coefs.items(), key=lambda kv: str(kv[0]))):
            name = _feature_name(name, f"{where} term {t + 1}")
            v = _number(coef, f"{where} term {t + 1} coefficient")
            if v == 0.0:
                raise SetupInvalid(
                    "invalid_constraint", f"{where} term {t + 1}: a coefficient must be non-zero"
                )
            terms[name] = v
        if c["op"] not in OPS:
            raise SetupInvalid("invalid_constraint", f"{where}: op must be '<=' or '>='")
        constraints.append(
            {"coefficients": terms, "op": c["op"], "rhs": _number(c["rhs"], f"{where} rhs")}
        )

    doc: dict[str, Any] = {
        "goal": goal, "bounds": bounds, "linear_constraints": constraints,
        "target_unit": clean_target_unit(raw.get("target_unit")),
    }
    doc.update(_normalize_strain(raw))
    return doc


def _normalize_strain(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The clone keys of a setup document, only those that are set. A policy or
    level without a clone column is refused; `fixed` is the default policy once a
    clone column is declared; a level applies to `fixed` only."""
    out: dict[str, Any] = {}
    col = raw.get("strain_column")
    if col is not None:
        out["strain_column"] = _feature_name(col, "strain_column")
    policy = raw.get("strain_policy")
    if policy is not None:
        if policy not in STRAIN_POLICIES:
            raise SetupInvalid("invalid_setup", "strain_policy must be 'fixed' or 'factor'")
        out["strain_policy"] = policy
    level = raw.get("strain_level")
    if level is not None:
        if not isinstance(level, str):
            raise SetupInvalid("invalid_setup", "strain_level must be a clone label (string)")
        text = level.strip()
        if not 1 <= len(text) <= MAX_CLONE_LABEL_CHARS:
            raise SetupInvalid(
                "invalid_setup", f"strain_level must be 1-{MAX_CLONE_LABEL_CHARS} characters"
            )
        if any(unicodedata.category(ch).startswith("C") for ch in text):
            raise SetupInvalid("invalid_setup", "strain_level must not contain control characters")
        out["strain_level"] = text
    covs = raw.get("covariates")
    if covs is not None:
        if not isinstance(covs, Sequence) or isinstance(covs, (str, bytes)):
            raise SetupInvalid("invalid_setup", "covariates must be a list of column names")
        if len(covs) > MAX_COVARIATES:
            raise SetupInvalid("invalid_setup", f"at most {MAX_COVARIATES} covariates are allowed")
        names = sorted({_feature_name(c, f"covariates[{k}]") for k, c in enumerate(covs)})
        if names:
            out["covariates"] = names
    if ("strain_policy" in out or "strain_level" in out) and "strain_column" not in out:
        raise SetupInvalid(
            "invalid_setup", "strain_policy / strain_level need strain_column",
            "name the clone column in strain_column",
        )
    if out.get("strain_policy") == "factor" and "strain_level" in out:
        raise SetupInvalid(
            "invalid_setup", "strain_level applies to strain_policy 'fixed' only",
            "remove strain_level, or use strain_policy 'fixed'",
        )
    if "strain_column" in out and out["strain_column"] in out.get("covariates", []):
        raise SetupInvalid("invalid_setup", "the clone column cannot also be a covariate")
    return out


def setup_hash(doc: Mapping[str, Any]) -> str:
    """sha256 (hex) of the canonical JSON of a NORMALIZED document."""
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


DEFAULT_SETUP_HASH = setup_hash(DEFAULT_SETUP)


def is_default(doc: Mapping[str, Any]) -> bool:
    return setup_hash(doc) == DEFAULT_SETUP_HASH


# --- the campaign's dataset ---------------------------------------------------- #


@dataclass(frozen=True)
class DatasetContext:
    """What the latest analysis says the campaign's design is."""

    target: str | None
    continuous: tuple[str, ...]
    categorical: tuple[str, ...]
    ranges: dict[str, tuple[float, float]]  # observed design box per continuous feature
    # Clone / covariate context (docs/CLONES.md); empty for an analysis without them.
    columns: tuple[str, ...] = ()
    strain_column: str | None = None
    strain_levels: tuple[str, ...] | None = None  # None: unknown (no block, or anonymized)
    covariates: tuple[str, ...] = ()


def dataset_context(analysis: Mapping[str, Any] | None) -> DatasetContext | None:
    """From a stored analysis blob, or None when the campaign has no usable one."""
    if not analysis or not isinstance(analysis.get("features"), list):
        return None
    cats = tuple(str(c) for c in analysis.get("categorical_features") or [])
    cont = tuple(str(f) for f in analysis["features"] if str(f) not in set(cats))
    ranges: dict[str, tuple[float, float]] = {}
    fr = analysis.get("feature_ranges")
    if isinstance(fr, Mapping):
        for name, pair in fr.items():
            if (
                isinstance(pair, Sequence) and len(pair) == 2
                and all(isinstance(v, (int, float)) and math.isfinite(v) for v in pair)
            ):
                ranges[str(name)] = (float(pair[0]), float(pair[1]))
    target = analysis.get("target")
    columns = tuple(
        str(r["name"]) for r in analysis.get("provenance") or []
        if isinstance(r, Mapping) and isinstance(r.get("name"), str)
    )
    strain = analysis.get("strain_summary")
    strain_column: str | None = None
    strain_levels: tuple[str, ...] | None = None
    if isinstance(strain, Mapping) and isinstance(strain.get("column"), str):
        strain_column = strain["column"]
        run = analysis.get("run")
        anonymized = isinstance(run, Mapping) and bool(run.get("anonymize"))
        if not anonymized and isinstance(strain.get("levels"), list):
            strain_levels = tuple(
                str(lv["label"]) for lv in strain["levels"]
                if isinstance(lv, Mapping) and lv.get("label") is not None
            )
    cov = analysis.get("covariates")
    covariates = tuple(
        str(c) for c in (cov.get("columns") or [])
    ) if isinstance(cov, Mapping) else ()
    return DatasetContext(
        target=str(target) if isinstance(target, str) else None,
        continuous=cont, categorical=cats, ranges=ranges,
        columns=columns, strain_column=strain_column, strain_levels=strain_levels,
        covariates=covariates,
    )


def _intersect(lo: float, hi: float, observed: tuple[float, float] | None) -> tuple[float, float]:
    if observed is None:
        return lo, hi
    return max(lo, observed[0]), min(hi, observed[1])


def bounds_report(
    doc: Mapping[str, Any], ctx: DatasetContext | None
) -> dict[str, dict[str, Any]]:
    """Per stored bound: what was typed, the observed design range, the range
    proposals are actually confined to (the intersection), and `extrapolates`
    (the typed bound reaches past the observed data). `observed`, `effective`
    and `extrapolates` are null when the campaign has no analysis to compare to."""
    out: dict[str, dict[str, Any]] = {}
    for name, b in doc.get("bounds", {}).items():
        observed = ctx.ranges.get(name) if ctx is not None else None
        if observed is None:
            out[name] = {"lo": b["lo"], "hi": b["hi"], "observed": None,
                         "effective": None, "extrapolates": None}
            continue
        lo, hi = _intersect(b["lo"], b["hi"], observed)
        out[name] = {
            "lo": b["lo"], "hi": b["hi"], "observed": [observed[0], observed[1]],
            "effective": [lo, hi] if hi > lo else None,
            "extrapolates": bool(b["lo"] < observed[0] or b["hi"] > observed[1]),
        }
    return out


def check_feasible(
    box: Mapping[str, tuple[float | None, float | None]],
    constraints: Sequence[Mapping[str, Any]],
) -> None:
    """Raise `SetupInvalid` unless the box + constraints admit a point with some
    room to move. `box` maps every feature in play to its (lo, hi) (None = free).
    Two LPs (scipy `linprog`, HiGHS): feasibility, then the largest uniform slack
    every constraint can keep (relative to the constraint's spread over the box),
    which refuses equality-like pairs the optimizer cannot sample inside."""
    if not constraints:
        return
    from scipy.optimize import linprog

    names = sorted(box)
    pos = {n: i for i, n in enumerate(names)}
    d = len(names)
    # Every row as  a @ x >= b.
    a = np.zeros((len(constraints), d))
    b = np.zeros(len(constraints))
    for r, c in enumerate(constraints):
        sign = 1.0 if c["op"] == ">=" else -1.0
        for name, coef in c["coefficients"].items():
            a[r, pos[name]] = sign * float(coef)
        b[r] = sign * float(c["rhs"])
    lp_bounds = [box[n] for n in names]
    res = linprog(np.zeros(d), A_ub=-a, b_ub=-b, bounds=lp_bounds, method="highs")
    if res.status == 2:
        raise SetupInvalid(
            "infeasible_constraints",
            "the bounds and linear constraints admit no recipe: no point satisfies all of them "
            "inside the search range",
            "loosen a constraint's right-hand side or widen the bounds; every constraint is "
            "checked together with the bounds and the observed data range",
        )
    if res.status not in (0, 3):
        raise SetupInvalid(
            "feasibility_unverified",
            "the feasibility of the bounds and linear constraints could not be verified "
            "(the LP solver did not converge)",
            "simplify the constraints or rescale coefficients to similar magnitudes",
        )
    # Interior: maximize t s.t. a@x - b >= t * spread, 0 <= t <= 1. `spread` is the
    # constraint's range over the box (1 per unbounded feature).
    width = np.array([
        (hi - lo) if lo is not None and hi is not None else 1.0 for lo, hi in lp_bounds
    ])
    spread = np.maximum(np.abs(a) @ width, 1e-12)
    c_obj = np.zeros(d + 1)
    c_obj[-1] = -1.0
    a_ub = np.hstack([-a, spread[:, None]])
    res2 = linprog(c_obj, A_ub=a_ub, b_ub=-b, bounds=[*lp_bounds, (0.0, 1.0)], method="highs")
    if res2.status == 0 and res2.fun is not None and -float(res2.fun) < MIN_INTERIOR_SLACK:
        raise SetupInvalid(
            "degenerate_constraints",
            "the bounds and linear constraints leave only a boundary (zero-volume) region, "
            "e.g. a '<=' and a '>=' with the same right-hand side; the optimizer cannot place "
            "proposals inside it",
            "give equality-like constraints a small tolerance (for example <= 1.01 and >= 0.99)",
        )


def validate_against_dataset(doc: Mapping[str, Any], ctx: DatasetContext | None) -> None:
    """Refuse a normalized document that cannot apply to the campaign's current
    analysis (`ctx`; None = no dataset yet: only feasibility over the typed
    bounds is checked). Errors refer to bounds / constraints by position;
    `context.continuous_features` lists the campaign's own valid names."""
    bounds = doc["bounds"]
    cons = doc["linear_constraints"]
    if ctx is not None:
        fix = {"continuous_features": list(ctx.continuous)}
        cont, cats = set(ctx.continuous), set(ctx.categorical)
        strain_col = doc.get("strain_column") or ctx.strain_column
        covs = set(doc.get("covariates") or ctx.covariates)
        _validate_strain_against_dataset(doc, ctx)
        for name in [*bounds, *(n for c in cons for n in c["coefficients"])]:
            if name in covs:
                raise SetupInvalid(
                    "covariate_not_proposable", "a bound or linear constraint names a covariate; "
                    "covariates (passage, generation, PDL, thaw) are recorded run properties, "
                    "held in proposals and never proposed", "remove it from the bounds and "
                    "constraints", fix,
                )
        for k, name in enumerate(bounds):
            where = f"bounds entry {k + 1} (sorted by name)"
            if ctx.target is not None and name == ctx.target:
                raise SetupInvalid(
                    "target_not_a_feature", f"{where} names the target; the target is "
                    "optimized, not bounded", "bound process inputs only", fix,
                )
            if name in cats:
                raise SetupInvalid(
                    "categorical_feature", f"{where} names a categorical feature; bounds apply "
                    "to continuous features only", None, fix,
                )
            if name not in cont:
                raise SetupInvalid(
                    "unknown_feature", f"{where} is not a continuous feature of this campaign's "
                    "current analysis", "use one of `continuous_features`", fix,
                )
            observed = ctx.ranges.get(name)
            lo, hi = _intersect(bounds[name]["lo"], bounds[name]["hi"], observed)
            if not hi > lo:
                raise SetupInvalid(
                    "bounds_outside_data", f"{where} does not overlap the observed range of "
                    "that feature; proposals stay inside the observed data range, so none "
                    "could satisfy it", "set a range that overlaps the observed range "
                    "(analysis `feature_ranges`)", fix,
                )
        for k, c in enumerate(cons):
            where = f"linear_constraints[{k}]"
            for name in c["coefficients"]:
                if ctx.target is not None and name == ctx.target:
                    raise SetupInvalid(
                        "target_not_a_feature", f"{where} names the target", None, fix,
                    )
                if name in cats:
                    raise SetupInvalid(
                        "categorical_feature", f"{where} names a categorical feature; linear "
                        "constraints apply to continuous features only", None, fix,
                    )
                if name not in cont:
                    raise SetupInvalid(
                        "unknown_feature", f"{where} names a feature that is not a continuous "
                        "feature of this campaign's current analysis",
                        "use one of `continuous_features`", fix,
                    )
        if cons and ctx.categorical and strain_col is None:
            raise SetupInvalid(
                "constraints_with_categoricals",
                "this campaign's design has categorical features, and the optimizer does not "
                "support linear constraints together with categorical dimensions",
                "remove the linear constraints (bounds are still supported)", fix,
            )
    box: dict[str, tuple[float | None, float | None]] = {}
    for c in cons:
        for name in c["coefficients"]:
            observed = ctx.ranges.get(name) if ctx is not None else None
            b = bounds.get(name)
            if b is not None:
                box[name] = _intersect(b["lo"], b["hi"], observed)
            elif observed is not None:
                box[name] = observed
            else:
                box[name] = (None, None)
    check_feasible(box, cons)


def _validate_strain_against_dataset(doc: Mapping[str, Any], ctx: DatasetContext) -> None:
    """The clone keys against the campaign's latest analysis: the clone column
    and covariates must be columns of the sheet, and a fixed level must be a clone
    that has runs (`clone_not_in_training`: Kalos never predicts an unseen clone)."""
    columns = set(ctx.columns)
    col = doc.get("strain_column")
    if col is not None and columns and col not in columns:
        raise SetupInvalid(
            "unknown_column", "strain_column is not a column of this campaign's sheet",
            "name the clone column exactly as in the sheet header",
        )
    if col is not None and ctx.target is not None and col == ctx.target:
        raise SetupInvalid("invalid_setup", "strain_column cannot be the target")
    for k, c in enumerate(doc.get("covariates") or []):
        if columns and c not in columns:
            raise SetupInvalid(
                "unknown_column", f"covariates[{k}] (sorted) is not a column of this campaign's sheet",
                "name covariate columns exactly as in the sheet header",
            )
        if ctx.target is not None and c == ctx.target:
            raise SetupInvalid("invalid_setup", "the target cannot be a covariate")
    level = doc.get("strain_level")
    if (
        level is not None and ctx.strain_levels is not None
        and (col == ctx.strain_column) and level not in set(ctx.strain_levels)
    ):
        raise SetupInvalid(
            "clone_not_in_training",
            "strain_level names a clone with no runs in this campaign's data; Kalos never "
            "predicts an unseen clone (the categorical kernel has nothing to say about a new label)",
            "choose a clone that has runs, or run the new clone first",
        )


# --- loading ------------------------------------------------------------------- #


def analysis_setup(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The `setup=` mapping `kalos.portal.analysis._analyze` takes."""
    out: dict[str, Any] = {
        "goal": doc["goal"],
        "bounds": {k: dict(v) for k, v in doc["bounds"].items()},
        "linear_constraints": [
            {**c, "coefficients": dict(c["coefficients"])} for c in doc["linear_constraints"]
        ],
        "user_target_unit": doc.get("target_unit"),
        "hash": setup_hash(doc),
    }
    for key in STRAIN_KEYS:
        if key in doc:
            out[key] = list(doc[key]) if key == "covariates" else doc[key]
    return out


# --- analysis-time plan -------------------------------------------------------- #


def _warning(code: str, message: str, feature: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"code": code, "message": message}
    if feature is not None:
        out["feature"] = feature
    return out


@dataclass
class SetupPlan:
    """How one analysis applies the campaign setup.

    `bounds` is the proposal box (the design box, narrowed by the user bounds);
    `linear_constraints` the optimizer's `(indices, coefficients, rhs)` triples
    (``sum(coef * x) >= rhs`` in raw design units: the design applies no scaling,
    the surrogate normalizes internally); `refusal` is set when the setup cannot
    be applied to this sheet, in which case proposals are withheld."""

    goal: str
    hash: str | None
    bounds: np.ndarray
    linear_constraints: list[tuple[list[int], list[float], float]]
    bounds_applied: dict[str, dict[str, Any]] = field(default_factory=dict)
    constraints_applied: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    refusal: dict[str, Any] | None = None
    n_dropped: int = 0

    @property
    def minimize(self) -> bool:
        return self.goal == "minimize"

    @property
    def refused(self) -> bool:
        return self.refusal is not None

    def propose_kwargs(self) -> dict[str, Any]:
        """Extra `propose_batch` keywords. Empty for maximize with no constraints,
        so the call is the historical one."""
        kw: dict[str, Any] = {}
        if self.linear_constraints:
            kw["linear_constraints"] = self.linear_constraints
        if self.minimize:
            kw["maximize"] = False
        return kw

    def _refuse(self, code: str, reason: str, fix: str | None = None) -> None:
        if self.refusal is None:
            self.refusal = {"code": code, "reason": reason, "what_would_fix_it": fix}

    def satisfied(self, batch: np.ndarray) -> np.ndarray:
        """Row mask: inside the proposal box and every linear constraint holds."""
        X = np.asarray(batch, float)
        if X.size == 0:
            return np.zeros(X.shape[0], dtype=bool)
        lo, hi = self.bounds
        tol = BOX_TOL * np.maximum(1.0, np.maximum(np.abs(lo), np.abs(hi)))
        ok = np.all(np.greater_equal(X, lo - tol) & np.less_equal(X, hi + tol), axis=1)
        for idx, coef, rhs in self.linear_constraints:
            ok &= X[:, idx] @ np.asarray(coef) >= rhs - LIN_TOL * max(1.0, abs(rhs))
        return ok & np.isfinite(X).all(axis=1)

    def post_check(self, batch: np.ndarray) -> np.ndarray:
        """`batch` minus any row that breaks a bound or constraint (counted)."""
        ok = self.satisfied(batch)
        self.n_dropped = int((~ok).sum())
        if self.n_dropped:
            log.warning("setup post-check dropped %d proposal(s)", self.n_dropped)
            self.warnings.append(_warning(
                "proposals_dropped",
                f"{self.n_dropped} proposal(s) broke a bound or linear constraint in the final "
                "check and were removed",
            ))
        return np.asarray(batch)[ok]

    def block(self) -> dict[str, Any]:
        return {
            "status": "refused" if self.refused else "applied",
            "goal": self.goal,
            "hash": self.hash,
            "bounds_applied": self.bounds_applied if not self.refused else {},
            "constraints_applied": self.constraints_applied if not self.refused else [],
            "warnings": list(self.warnings),
            "refusal": self.refusal,
            "n_proposals_dropped": int(self.n_dropped),
            "sign_convention": SIGN_CONVENTION,
        }

    def result_fields(self) -> dict[str, Any]:
        return {"goal": self.goal, "setup": self.block()}


def plan_setup(
    setup: Mapping[str, Any],
    *,
    names: Sequence[str],
    categorical: Sequence[str],
    bounds: np.ndarray,
    target: str,
    converted: Sequence[str] = (),
    held: Sequence[str] = (),
    mixed_constraints_ok: bool = False,
) -> SetupPlan:
    """Turn the `_analyze(setup=...)` mapping into a `SetupPlan` for one sheet.

    `names` are the design's dimension names in X order, `categorical` the
    categorical ones, `bounds` the design box (2, d), `converted` the columns the
    validation gate unit-converted (their bounds are read in the converted unit).
    Never raises: anything that cannot be applied becomes a `refusal` (a
    constraint is never dropped) or, for a bound on a feature this sheet does not
    model (the proposals do not set it at all), a warning.

    `held` names dims every proposal holds at one value (a fixed clone level,
    covariates); a bound or constraint on one is refused (`covariate_not_proposable`).
    Linear constraints with categorical features are applied when every
    categorical dim is held, or when `mixed_constraints_ok` (a clone campaign
    whose free categorical combinations the exact mixed optimizer enumerates,
    ML_AUDIT N2); otherwise refused as before."""
    box = np.array(bounds, dtype=float, copy=True)
    goal = setup.get("goal") if setup.get("goal") in GOALS else "maximize"
    plan = SetupPlan(goal=str(goal), hash=setup.get("hash"), bounds=box, linear_constraints=[])
    if setup.get("unreadable"):
        plan._refuse(
            "setup_unreadable", "the campaign's stored setup could not be read, so it cannot "
            "be applied; proposals are withheld rather than ignoring it",
            "save the setup again (PUT /api/campaigns/{id}/setup)",
        )
        return plan
    pos = {str(n): i for i, n in enumerate(names)}
    cats = {str(c) for c in categorical}
    conv = {str(c) for c in converted}
    held_set = {str(c) for c in held}
    for name in [*(setup.get("bounds") or {}), *(n for c in setup.get("linear_constraints") or []
                                                 for n in c["coefficients"])]:
        if name in held_set and name not in cats:
            plan._refuse(
                "covariate_not_proposable", "a bound or linear constraint names a covariate; "
                "covariates are held at a reference value in every proposal and never proposed",
                "remove the covariate from the campaign setup's bounds and constraints",
            )

    if plan.minimize:
        plan.warnings.append(_warning(
            "gp_shapes_swept_at_highest_run",
            "gp_shapes sweeps each feature through the run with the HIGHEST measured target; "
            "with goal=minimize, read its interior_minimum entries as the candidate optima",
        ))

    for name, b in (setup.get("bounds") or {}).items():
        lo_u, hi_u = float(b["lo"]), float(b["hi"])
        if name not in pos:
            plan.warnings.append(_warning(
                "bound_feature_not_modeled",
                "this bound names a column that is not a feature of this analysis; proposals "
                "do not set it, so the bound has nothing to constrain", name,
            ))
            continue
        if name in cats:
            plan._refuse(
                "bound_on_categorical", "a bound names a feature that is categorical in this "
                "sheet; bounds apply to continuous features only",
                "remove that bound from the campaign setup",
            )
            continue
        j = pos[name]
        olo, ohi = float(bounds[0, j]), float(bounds[1, j])
        lo, hi = max(lo_u, olo), min(hi_u, ohi)
        if not hi > lo:
            plan._refuse(
                "bounds_outside_data", "a bound does not overlap the observed range of its "
                "feature in this sheet; proposals stay inside the observed range, so none could "
                "satisfy it", "widen the bound to overlap `feature_ranges`",
            )
            continue
        box[0, j], box[1, j] = lo, hi
        plan.bounds_applied[name] = {
            "requested": [lo_u, hi_u], "observed": [olo, ohi], "applied": [lo, hi],
            "extrapolates": bool(lo_u < olo or hi_u > ohi),
        }
        if lo_u < olo or hi_u > ohi:
            plan.warnings.append(_warning(
                "bound_extrapolates",
                "this bound reaches past the observed data range; proposals stay inside the "
                "observed range (the model is not extrapolated)", name,
            ))
        if name in conv:
            plan.warnings.append(_warning(
                "bound_on_converted_column",
                "this column's values were converted to a base unit before modeling "
                "(validation.conversions); the bound is read in that converted unit", name,
            ))

    cons = list(setup.get("linear_constraints") or [])
    if cons and (cats - held_set) and not mixed_constraints_ok:
        plan._refuse(
            "constraints_with_categoricals", "this sheet has categorical features, and the "
            "optimizer does not support linear constraints together with categorical "
            "dimensions; proposals are withheld rather than ignoring the constraints",
            "remove the linear constraints from the campaign setup, or the categorical column "
            "from the sheet",
        )
    for c in cons:
        terms = c["coefficients"]
        missing = [n for n in terms if n not in pos]
        if missing or any(n in cats for n in terms):
            plan._refuse(
                "constraint_feature_not_modeled", "a linear constraint names a column that is "
                "not a continuous feature of this sheet, so it cannot be enforced; proposals are "
                "withheld rather than dropping it",
                "edit the constraint to use this sheet's continuous features",
            )
            continue
        sign = 1.0 if c["op"] == ">=" else -1.0
        items = sorted(((pos[n], sign * float(v)) for n, v in terms.items()))
        plan.linear_constraints.append(
            ([i for i, _ in items], [v for _, v in items], sign * float(c["rhs"]))
        )
        plan.constraints_applied.append(
            {"coefficients": dict(terms), "op": c["op"], "rhs": float(c["rhs"])}
        )
    if plan.refused:
        plan.linear_constraints = []
        return plan

    if cons:
        lp_box: dict[str, tuple[float | None, float | None]] = {
            n: (float(box[0, pos[n]]), float(box[1, pos[n]]))
            for c in cons for n in c["coefficients"]
        }
        try:
            check_feasible(lp_box, cons)
        except SetupInvalid as exc:
            plan._refuse(exc.code, exc.reason + " (on this sheet's observed ranges)",
                         exc.what_would_fix_it)
            plan.linear_constraints = []
    return plan


def withheld_notice(q: int, reason: str) -> dict[str, Any]:
    """`proposal_notice` when the setup could not be applied."""
    return {
        "code": "setup_not_applied",
        "requested": int(q),
        "returned": 0,
        "min_separation": None,
        "message": (
            "No proposals: the campaign setup (goal, bounds, linear constraints) could not be "
            f"applied to this sheet ({reason}). See `setup.refusal`."
        ),
    }


def anonymize_setup_block(block: Mapping[str, Any], alias: Any) -> dict[str, Any]:
    """`block` with every feature name passed through `alias` (anonymize=True)."""
    out = dict(block)
    if isinstance(out.get("bounds_applied"), Mapping):
        out["bounds_applied"] = {alias(str(k)): v for k, v in out["bounds_applied"].items()}
    if isinstance(out.get("constraints_applied"), list):
        out["constraints_applied"] = [
            {**c, "coefficients": {alias(str(k)): v for k, v in c["coefficients"].items()}}
            if isinstance(c, Mapping) and isinstance(c.get("coefficients"), Mapping) else c
            for c in out["constraints_applied"]
        ]
    if isinstance(out.get("warnings"), list):
        out["warnings"] = [
            {**w, "feature": alias(str(w["feature"]))}
            if isinstance(w, Mapping) and w.get("feature") is not None else w
            for w in out["warnings"]
        ]
    return out


__all__ = [
    "DEFAULT_SETUP", "DEFAULT_SETUP_HASH", "DatasetContext", "SetupInvalid", "SetupPlan",
    "analysis_setup", "anonymize_setup_block", "bounds_report", "check_feasible",
    "clean_target_unit", "dataset_context", "is_default", "normalize_setup",
    "plan_setup", "setup_hash", "validate_against_dataset",
    "withheld_notice",
]
