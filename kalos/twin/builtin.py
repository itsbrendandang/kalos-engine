"""Built-in generators: thin wrappers around seeded code that already exists.

- `bench.gaussian_bump`, `bench.ackley`, `bench.mixed_bump`, `bench.bioprocess_like`:
  the analytic surfaces with known optima in `kalos/bench/objectives.py`. Designs
  are drawn uniformly in the surface's box (categorical levels uniformly); the
  outcome is the noisy observation `y` plus the noiseless `y_true`.
- `bench.fed_batch`: the Monod/Haldane fed-batch ODE of
  `kalos/bench/mechanistic_benchmark.py` (`fed_batch(...)`), with its TRUTH
  parameters as defaults. Design = [S0, F] uniform in that module's BOUNDS.
- `examples.synthetic_scaleup`: the planted pH-by-scale generator in
  `examples/synthetic_scaleup/make_synthetic.py`, loaded from that file so the
  demo dataset, the test fixture and this generator stay one piece of code. It is
  only available in a source checkout (the `examples/` folder is not packaged).

Every output here is `synthetic`: none of it is tied to a lab's process model.
The wrapped modules are not edited; their own docstrings describe the surfaces.
"""
from __future__ import annotations

import importlib
import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kalos.twin.generators import GeneratorError, GeneratorSpec, register
from kalos.twin.origin import Origin

_BENCH = Path(__file__).resolve().parents[1] / "bench"
_OBJECTIVES = _BENCH / "objectives.py"
_FED_BATCH = _BENCH / "mechanistic_benchmark.py"
_MECHANISTIC = Path(__file__).resolve().parents[1] / "core" / "mechanistic.py"
_SCALEUP = Path(__file__).resolve().parents[2] / "examples" / "synthetic_scaleup" / "make_synthetic.py"
_THIS = Path(__file__).resolve()

# ODEModel.evaluate uses scipy solve_ivp(method="LSODA") with scipy's default
# tolerances and 50 output points; recorded as-is on every row.
_FED_BATCH_SOLVER = {
    "integrator": "scipy.integrate.solve_ivp",
    "method": "LSODA",
    "rtol": 1e-3,
    "atol": 1e-6,
    "t_eval_points": 50,
    "t_end_h": 48.0,
    "failure": "row outcome is null (the ODE contract returns NaN)",
}


def _check_noise(p: dict[str, Any]) -> None:
    if not 0.0 <= p["noise"] <= 1.0:
        raise GeneratorError("noise must be a fraction from 0 to 1 of the surface's scale")


def _uniform(rng: np.random.Generator, lo: np.ndarray, hi: np.ndarray, n: int) -> np.ndarray:
    return lo + (hi - lo) * rng.random((n, lo.shape[0]))


# --- objectives.py ------------------------------------------------------------- #


def _objective(name: str) -> Any:
    """`kalos.bench.objectives.<name>`, imported at generation time: importing
    `kalos.bench` loads torch (its `__init__` imports the benchmark loop), and
    listing generators must not pay for that."""
    return getattr(importlib.import_module("kalos.bench.objectives"), name)


def _continuous(make: str, gen_id: str, description: str) -> None:
    def fn(params: dict[str, Any], rng: np.random.Generator, n: int) -> pd.DataFrame:
        obj = _objective(make)(params["dim"])
        X = _uniform(rng, obj.bounds[0], obj.bounds[1], n)
        y_true, y_obs = obj.evaluate(X, noise=params["noise"], rng=rng)
        df = pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(obj.dim)])
        df["y"] = y_obs
        df["y_true"] = y_true
        return df

    def validate(p: dict[str, Any]) -> None:
        _check_noise(p)
        if not 1 <= p["dim"] <= 12:
            raise GeneratorError("dim must be from 1 to 12")

    fn.__qualname__ = f"builtin.{gen_id}"
    register(GeneratorSpec(
        id=gen_id, version="1", origin=Origin.SYNTHETIC, description=description, fn=fn,
        defaults={"dim": 4, "noise": 0.1}, source_files=(_OBJECTIVES, _THIS),
        solver={"kind": "closed-form surface"}, outcome_columns=("y", "y_true"),
        validate=validate,
    ))


def _mixed(make: str, gen_id: str, description: str, defaults: dict[str, Any]) -> None:
    def fn(params: dict[str, Any], rng: np.random.Generator, n: int) -> pd.DataFrame:
        kwargs = {k: v for k, v in params.items() if k != "noise"}
        obj = _objective(make)(**kwargs)
        Xc = _uniform(rng, obj.cont_bounds[0], obj.cont_bounds[1], n)
        cats = np.column_stack([rng.integers(0, c, size=n) for c in obj.cat_cardinalities])
        X = np.hstack([Xc, cats.astype(float)])
        y_true, y_obs = obj.evaluate(X, noise=params["noise"], rng=rng)
        cols = [f"x{i + 1}" for i in range(obj.n_cont)] + [
            f"level{j + 1}" for j in range(len(obj.cat_cardinalities))
        ]
        df = pd.DataFrame(X, columns=cols)
        for j in range(len(obj.cat_cardinalities)):
            df[f"level{j + 1}"] = df[f"level{j + 1}"].astype(int)
        df["y"] = y_obs
        df["y_true"] = y_true
        return df

    def validate(p: dict[str, Any]) -> None:
        _check_noise(p)
        if "n_cont" in p and not 1 <= p["n_cont"] <= 12:
            raise GeneratorError("n_cont must be from 1 to 12")
        if "n_levels" in p and not 2 <= p["n_levels"] <= 12:
            raise GeneratorError("n_levels must be from 2 to 12")

    fn.__qualname__ = f"builtin.{gen_id}"
    register(GeneratorSpec(
        id=gen_id, version="1", origin=Origin.SYNTHETIC, description=description, fn=fn,
        defaults={**defaults, "noise": 0.1}, source_files=(_OBJECTIVES, _THIS),
        solver={"kind": "closed-form surface"}, outcome_columns=("y", "y_true"),
        validate=validate,
    ))


def _register_objectives() -> None:
    _continuous("gaussian_bump", "bench.gaussian_bump",
                "One smooth Gaussian peak on [0, 1]^dim (max 10). kalos/bench/objectives.py.")
    _continuous("ackley", "bench.ackley",
                "Negated Ackley on [-2, 2]^dim, rugged, max 0. kalos/bench/objectives.py.")
    _mixed("mixed_bump", "bench.mixed_bump",
           "Gaussian bump plus one categorical factor with a best level (max 13). "
           "kalos/bench/objectives.py.", {"n_cont": 2, "n_levels": 3})
    _mixed("bioprocess_like", "bench.bioprocess_like",
           "Made-up titer surface: 3 numeric factors, strain (4 levels), inducer (3 levels); "
           "max 10.5. Not calibrated to any process. kalos/bench/objectives.py.", {})


# --- mechanistic_benchmark.fed_batch --------------------------------------------- #


def _fed_batch_fn(params: dict[str, Any], rng: np.random.Generator, n: int) -> pd.DataFrame:
    from kalos.bench.mechanistic_benchmark import BOUNDS, fed_batch

    model = fed_batch(params["mu_max"], params["ks"], params["ki"], params["y_xs"], params["qp"])
    X = _uniform(rng, BOUNDS[0], BOUNDS[1], n)
    y_true = model.evaluate(X)
    sd = params["noise_sd_g"]
    y_obs = y_true + (rng.normal(0.0, sd, size=n) if sd > 0 else 0.0)
    return pd.DataFrame({
        "S0_g_per_L": X[:, 0], "feed_L_per_h": X[:, 1],
        "product_g": y_obs, "product_g_true": y_true,
    })


def _fed_batch_validate(p: dict[str, Any]) -> None:
    for key in ("mu_max", "ks", "y_xs", "qp"):
        if not p[key] > 0:
            raise GeneratorError(f"{key} must be positive")
    if p["ki"] is not None and not p["ki"] > 0:
        raise GeneratorError("ki must be positive, or null for plain Monod (no inhibition)")
    if not 0.0 <= p["noise_sd_g"] <= 1000.0 or not math.isfinite(p["noise_sd_g"]):
        raise GeneratorError("noise_sd_g must be from 0 to 1000")


_fed_batch_fn.__qualname__ = "builtin.bench.fed_batch"


def _register_fed_batch() -> None:
    register(GeneratorSpec(
        id="bench.fed_batch", version="1", origin=Origin.SYNTHETIC,
        description=(
            "Monod/Haldane fed-batch ODE (kalos/bench/mechanistic_benchmark.py). Defaults are "
            "that module's TRUTH parameters; ki=null is plain Monod. Outcome: final product "
            "mass in g. A simulation of the benchmark, not of any lab's culture."
        ),
        fn=_fed_batch_fn,
        defaults={"mu_max": 0.35, "ks": 4.0, "ki": 45.0, "y_xs": 0.45, "qp": 0.25,
                  "noise_sd_g": 0.0},
        source_files=(_FED_BATCH, _MECHANISTIC, _THIS), solver=_FED_BATCH_SOLVER,
        design_columns=("S0_g_per_L", "feed_L_per_h"),
        outcome_columns=("product_g", "product_g_true"), validate=_fed_batch_validate,
        nullable=("ki",),
    ))


# --- examples/synthetic_scaleup/make_synthetic.py --------------------------------- #

_SCALEUP_MODULE: Any = None


def _load_scaleup() -> Any:
    global _SCALEUP_MODULE
    if _SCALEUP_MODULE is None:
        spec = importlib.util.spec_from_file_location("kalos_twin_synthetic_scaleup", _SCALEUP)
        if spec is None or spec.loader is None:  # pragma: no cover - checked by available()
            raise GeneratorError("examples.synthetic_scaleup is not available in this install")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SCALEUP_MODULE = mod
    return _SCALEUP_MODULE


def _scaleup_fn(params: dict[str, Any], rng: np.random.Generator, n: int) -> pd.DataFrame:
    mod = _load_scaleup()
    scales = tuple(mod.SCALES)
    per_scale = max(1, math.ceil(n / len(scales)))
    seed = int(rng.integers(0, 2**32))
    df = mod.make_synthetic_scaleup(scales=scales, n_per_scale=per_scale, seed=seed)
    # Interleave the scales (row 0 of every scale, then row 1, ...) so a short
    # batch still spans the scales instead of the smallest ones only.
    df = df.assign(_i=np.tile(np.arange(per_scale), len(scales)),
                   _s=np.repeat(np.arange(len(scales)), per_scale))
    df = df.sort_values(["_i", "_s"], kind="stable").head(n).drop(columns=["_i", "_s"])
    return df.reset_index(drop=True)


_scaleup_fn.__qualname__ = "builtin.examples.synthetic_scaleup"


def _register_scaleup() -> None:
    register(GeneratorSpec(
        id="examples.synthetic_scaleup", version="1", origin=Origin.SYNTHETIC,
        description=(
            "Planted pH-by-scale titer data across 1 L to 5000 L "
            "(examples/synthetic_scaleup/make_synthetic.py). The seed passed to that "
            "function is drawn from this batch's seed stream and stored. Source checkout only."
        ),
        fn=_scaleup_fn, defaults={}, source_files=(_SCALEUP, _THIS),
        solver={"kind": "closed-form with Gaussian noise (sd 0.15)"},
        design_columns=("scale_L", "agitation_rpm", "airflow_L_per_min", "ph_setpoint",
                        "temperature_C"),
        outcome_columns=("titer_g_per_L",), available=_SCALEUP.is_file,
    ))


_register_objectives()
_register_fed_batch()
_register_scaleup()
