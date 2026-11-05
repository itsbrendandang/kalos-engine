"""GP surrogate on BoTorch.

A `SingleTaskGP` with input normalization and output standardization — the
production-grade replacement for the lean hand-rolled GP. Fits by exact marginal
likelihood and returns a posterior mean + standard deviation in the target's
original units.

Device note: BoTorch GPs run in float64 for numerical stability, and Apple's MPS
backend does not support float64, so the GP runs on CPU. This is the right call
for the small-sample regime BO targets anyway (GP fits on ~100 rows are fast on
CPU). The deep-learning protein embedder runs on MPS separately.
"""
from __future__ import annotations

import logging
import re
import warnings

import linear_operator
import numpy as np
import torch
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.utils.errors import NotPSDError
from linear_operator.utils.warnings import NumericalWarning

log = logging.getLogger("kalos.core.surrogate")

DTYPE = torch.double
DEVICE = torch.device("cpu")

# Escalating Cholesky jitter for the fit-retry ladder. A near-duplicate or
# otherwise ill-conditioned design makes the kernel matrix numerically
# non-positive-definite, so the first Cholesky can fail (LinAlgError / NotPSDError
# / ModelFittingError). We retry with progressively larger jitter before giving
# up, so a numerically-hard-but-valid dataset still fits instead of 500-ing.
#
# Units: added to the diagonal of the kernel matrix in the model's standardized
# output space (the outcome is standardized, so the prior variance is ~1).
#
# History (found by the compmodel review, verified against linear_operator
# 0.6.1): this ladder used to be passed as `cholesky_jitter(j)`, which sets only
# the FLOAT32 value. The GP runs in float64, so the rungs never applied; every
# rung used linear_operator's float64 default (1e-8, retried x10 up to 3 times,
# so at most 1e-6). Now the rung is set for float64 and `cholesky_max_tries` is
# 1, so the diagonal addition in any one Cholesky during the fit is at most the
# current rung, and never more than `JITTER_CAP`.
_FIT_JITTERS: tuple[float, ...] = (1e-8, 1e-6, 1e-4)
JITTER_CAP: float = _FIT_JITTERS[-1]

# Conditioning of the matrix the fit factors, K + noise*I (model space:
# normalized inputs, standardized outputs). Above WARN the fit proceeds and the
# value is reported; above REFUSE the fit is refused with FitError (HTTP 400).
# Thresholds are the owner's defaults (compmodel PLAN E2), not validated numbers.
# Note: the noise floors (inferred noise >= 1e-4, fixed noise >= 1e-6 in model
# units) bound the condition number by about n * outputscale / floor, so REFUSE
# is reached only for very large or very tightly clustered designs.
COND_WARN: float = 1e8
COND_REFUSE: float = 1e12

_JITTER_MSG = re.compile(r"added jitter of ([0-9.eE+-]+) to the diagonal")


class FitError(RuntimeError):
    """The GP could not be fit even after escalating the Cholesky jitter.

    Raised when every retry in the jitter ladder fails on a numerical error
    (`torch.linalg.LinAlgError`, GPyTorch `NotPSDError`, or a botorch
    `ModelFittingError`). Carries an honest, caller-safe message; the portal maps
    it to a distinct 400 so an ill-conditioned file is not confused with a parse
    failure.
    """


# Numerical fit failures we retry with more jitter. A LinAlgError from the
# Cholesky, GPyTorch's NotPSDError, and botorch's ModelFittingError all mean the
# same thing here: the kernel matrix was not positive-definite at this jitter.
_FIT_NUMERICAL_ERRORS = (torch.linalg.LinAlgError, NotPSDError, ModelFittingError)


def _fit_mll_with_retry(mll) -> dict:
    """Fit `mll` in place, retrying on a numerical failure with escalating jitter.

    Tries `fit_gpytorch_mll` under each jitter in `_FIT_JITTERS` (float64 and
    float32, one Cholesky try each); the first success wins. Returns the jitter
    record: `jitter_rung` (the rung the fit succeeded under), `jitter_rung_index`
    (0 = first), `jitter_added_max` (the largest diagonal addition linear_operator
    actually made during the successful fit, read from its NumericalWarning; 0.0
    when every Cholesky succeeded without jitter), `jitter_additions` (how many
    times it added jitter) and `jitter_cap`. If every rung fails, raise
    `FitError`. Any non-numerical exception propagates unchanged. Works for both
    `ExactMarginalLogLikelihood` and `SumMarginalLogLikelihood`.
    """
    last: Exception | None = None
    for i, jitter in enumerate(_FIT_JITTERS):
        try:
            with warnings.catch_warnings(record=True) as caught, \
                    linear_operator.settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
                    linear_operator.settings.cholesky_max_tries(1):
                warnings.simplefilter("always", NumericalWarning)
                fit_gpytorch_mll(mll)
        except _FIT_NUMERICAL_ERRORS as exc:
            last = exc
            continue
        added = [
            float(m.group(1))
            for w in caught
            if issubclass(w.category, NumericalWarning)
            and (m := _JITTER_MSG.search(str(w.message))) is not None
        ]
        # Re-emit anything that is not a jitter notice, so other warnings keep
        # their normal handling.
        for w in caught:
            if not (issubclass(w.category, NumericalWarning) and _JITTER_MSG.search(str(w.message))):
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        added_max = max(added, default=0.0)
        if added_max > JITTER_CAP:
            # Cannot happen with max_tries=1; checked so a library change that
            # adds more than the cap fails loudly instead of fitting quietly.
            raise FitError(
                f"the model fit needed a diagonal jitter of {added_max:.1e}, above the "
                f"cap of {JITTER_CAP:.1e}; the rows are too close to duplicates to fit"
            )
        if i > 0:
            log.warning("GP fit needed Cholesky jitter rung %.0e (cap %.0e)", jitter, JITTER_CAP)
        return {
            "jitter_rung": jitter,
            "jitter_rung_index": i,
            "jitter_added_max": added_max,
            "jitter_additions": len(added),
            "jitter_cap": JITTER_CAP,
        }
    raise FitError(
        "the model could not be fit on this data - likely near-duplicate or "
        f"ill-conditioned rows (Cholesky failed with a diagonal jitter up to the cap "
        f"of {JITTER_CAP:.0e})"
    ) from last


def kernel_condition_number(model) -> float:
    """2-norm condition number of the fitted training covariance K + noise*I,
    from its eigenvalues (eigvalsh; ~1 s at the 2,000-row cap). `inf` when the
    smallest eigenvalue is not positive."""
    model.eval()
    with torch.no_grad():
        X0 = model.train_inputs[0]
        cov = model.likelihood(model.forward(X0), [X0]).covariance_matrix
        ev = torch.linalg.eigvalsh(cov)
    lo, hi = float(ev.min()), float(ev.max())
    return float("inf") if lo <= 0.0 else hi / lo


def condition_report(cond: float) -> dict:
    """The conditioning fields recorded with a fit. Raises FitError above
    COND_REFUSE, with a plain-language reason."""
    if not (cond <= COND_REFUSE):
        raise FitError(
            "some rows are so close to each other that the model cannot tell them "
            f"apart numerically (condition number {cond:.1e}, limit {COND_REFUSE:.0e}). "
            "Remove or merge near-duplicate rows, or record them as replicates of one recipe."
        )
    warn = cond > COND_WARN
    return {
        "condition_number": cond,
        "condition_warning": warn,
        "condition_notice": (
            f"Some rows are very close together (condition number {cond:.1e}, above "
            f"{COND_WARN:.0e}); predictions near them are less stable."
            if warn else None
        ),
        "condition_warn_above": COND_WARN,
        "condition_refuse_above": COND_REFUSE,
    }


def sanitize_bounds(bounds) -> tuple[np.ndarray, np.ndarray]:
    """Split `bounds` (2 x d) into finite `(lower, upper)` rows with `lower <= upper`.

    A degenerate or near-constant feature (lower == upper, or a NaN/inf bound)
    would let input normalization divide by a zero-width range and poison the fit
    (and let the acquisition optimizer wander to an absurd value). We order the
    rows, replace any non-finite bound with the finite partner (or 0.0 if both are
    non-finite), and widen a zero-width interval by a tiny epsilon so both the GP
    normalization and `optimize_acqf` see a valid, non-inverted box. This is the
    guard behind the historical out-of-range blow-up (e.g. a constant
    `Culture_Volume` proposing ~= 33,000,000).
    """
    arr = np.asarray(bounds, float)
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError("bounds must have shape (2, d) = [lower_row, upper_row]")
    lower = arr[0].astype(float).copy()
    upper = arr[1].astype(float).copy()
    for row, partner in ((lower, upper), (upper, lower)):
        bad = ~np.isfinite(row)
        if bad.any():
            fill = np.where(np.isfinite(partner), partner, 0.0)
            row[bad] = fill[bad]
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
    degenerate = upper - lower <= 0.0
    if degenerate.any():
        pad = 1e-6 * np.maximum(np.abs(lower), 1.0)
        upper = np.where(degenerate, lower + pad, upper)
    return lower, upper


def _build_mixed_gp(
    Xt: torch.Tensor,
    yt: torch.Tensor,
    box: torch.Tensor,
    cat_dims: list[int],
    d: int,
    yvar: torch.Tensor | None,
) -> MixedSingleTaskGP:
    """A `MixedSingleTaskGP` with the continuous dims input-normalized to the
    design box and the categorical dims (integer codes) passed through.

    `box` is the sanitized full `(2, d)` box. Only the continuous columns are
    normalized (via `Normalize(indices=...)`); normalizing the categorical codes
    would corrupt the CategoricalKernel's equality test. When every dim is
    categorical there is nothing to normalize, so no input transform is applied.
    """
    cat_set = set(cat_dims)
    cont_dims = [i for i in range(d) if i not in cat_set]
    input_transform = (
        Normalize(d=d, indices=cont_dims, bounds=box[:, cont_dims])
        if cont_dims
        else None
    )
    kwargs: dict = {
        "cat_dims": cat_dims,
        "input_transform": input_transform,
        "outcome_transform": Standardize(m=1),
    }
    if yvar is not None:
        kwargs["train_Yvar"] = yvar
    return MixedSingleTaskGP(Xt, yt, **kwargs)


class Surrogate:
    """SingleTaskGP over a continuous design space, or a MixedSingleTaskGP when
    some dimensions are categorical (`cat_dims`)."""

    def __init__(self) -> None:
        self.model: SingleTaskGP | MixedSingleTaskGP | None = None
        self._y: np.ndarray | None = None
        self._X: torch.Tensor | None = None
        # Integer indices of categorical columns in X (None/empty = all continuous).
        # Recorded so the acquisition path knows which dims to enumerate.
        self._cat_dims: list[int] | None = None
        # What the last fit did numerically (jitter rung, ...). Empty before fit.
        self.fit_diagnostics: dict = {}

    def fit(
        self, X, y, bounds, *, noise=None, cat_dims=None, check_conditioning: bool = False,
    ) -> Surrogate:
        """Fit on (X, y). `bounds` (2 x d) ties input normalization to the fixed
        design box rather than the training-data envelope; it is required so a
        caller can never fall into the training-envelope normalization footgun.

        `noise` optionally fixes the observation noise variance instead of
        inferring it from the data:
          - `None` (default): infer noise as before (unchanged behavior).
          - a scalar: the same noise variance sigma^2, in the target's original
            units, applied to every training point.
          - an array of length `n`: a per-point noise variance sigma^2_i, in
            the target's original units.
        This is for a known assay noise floor (e.g. from `estimate_noise_floor`
        on replicated recipes), so the GP stops re-inferring noise it already
        has an external estimate for and stops chasing single-replicate noise
        spikes as if they were signal.

        `cat_dims` optionally marks a subset of columns as categorical (their
        values are integer level codes, not measurements). When given, the GP is
        a `MixedSingleTaskGP` with a CategoricalKernel on those dims and a Matern
        kernel on the rest; the continuous dims are input-normalized to the box
        and the categorical codes pass through untouched. When `None`/empty the
        continuous `SingleTaskGP` path is unchanged.

        `check_conditioning` (default off) computes the condition number of the
        fitted K + noise*I, records it in `fit_diagnostics`, and refuses above
        `COND_REFUSE` (FitError). The portal turns it on for the proposal fit only.
        """
        Xa = np.asarray(X, float)
        ya = np.asarray(y, float).reshape(-1)
        if Xa.ndim != 2 or Xa.shape[0] == 0:
            raise ValueError("X must be a non-empty 2-D array (n x d)")
        if Xa.shape[0] != ya.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if not np.isfinite(Xa).all() or not np.isfinite(ya).all():
            raise ValueError("X and y must be finite (no NaN/inf)")
        Xt = torch.as_tensor(Xa, dtype=DTYPE, device=DEVICE)
        yt = torch.as_tensor(ya, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
        d = Xt.shape[-1]
        lower, upper = sanitize_bounds(bounds)  # widen zero-width dims so Normalize is finite
        box = torch.as_tensor(np.vstack([lower, upper]), dtype=DTYPE, device=DEVICE)

        yvar: torch.Tensor | None = None
        if noise is not None:
            # Fixed observation noise: build train_Yvar (n x 1, original y-units).
            # BoTorch scales Yvar through the Standardize outcome transform and
            # selects a FixedNoiseGaussianLikelihood automatically, so no explicit
            # likelihood override is needed on either the continuous or mixed path.
            noise_arr = np.broadcast_to(np.asarray(noise, float), (Xa.shape[0],)).astype(float)
            if not np.isfinite(noise_arr).all() or (noise_arr < 0).any():
                raise ValueError("noise must be finite and non-negative")
            yvar = torch.as_tensor(noise_arr, dtype=DTYPE, device=DEVICE).reshape(-1, 1)

        cats = [int(i) for i in cat_dims] if cat_dims else []
        if cats:
            self.model = _build_mixed_gp(Xt, yt, box, cats, d, yvar)
        else:
            normalize = Normalize(d=d, bounds=box)
            if yvar is None:
                # No explicit likelihood is passed, so SingleTaskGP uses BoTorch's
                # default noise model: a GaussianLikelihood with a
                # LogNormalPrior(-4, 1) on the noise and a positive floor. That
                # default is the right noise prior for the small-sample regime, so
                # we deliberately do not override it here.
                self.model = SingleTaskGP(Xt, yt, input_transform=normalize, outcome_transform=Standardize(m=1))
            else:
                self.model = SingleTaskGP(
                    Xt, yt, train_Yvar=yvar, input_transform=normalize, outcome_transform=Standardize(m=1)
                )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        diag = dict(_fit_mll_with_retry(mll))
        if check_conditioning:
            # Only the fit whose posterior is served (the analysis's proposal
            # surrogate) pays for the eigendecomposition and can be refused here;
            # CV folds, bench arms and auxiliary surrogates never are.
            diag.update(condition_report(kernel_condition_number(self.model)))
            if diag["condition_warning"]:
                log.warning("GP kernel matrix is ill-conditioned (cond %.1e)", diag["condition_number"])
        self.fit_diagnostics = diag
        self._y = ya
        # Keep the raw (untransformed) training inputs so the noisy-EI acquisition
        # can use them as X_baseline; the model applies its input transform to them
        # internally at acquisition time.
        self._X = Xt
        self._cat_dims = cats or None
        return self

    def posterior(self, X, *, observation_noise: bool = False):
        """Return (mean, std) in the target's original units.

        `observation_noise` selects WHICH question the `std` answers, and the two
        are not interchangeable:

          - `False` (the default, unchanged): the posterior over the LATENT
            response surface. This is uncertainty about the process itself, which
            is what an acquisition function reasons about and what a proposal's
            `+/-` describes.
          - `True`: the predictive distribution for a NEW MEASUREMENT at `X`,
            latent uncertainty plus the fitted assay noise. This is the only one
            that can be checked against a held-out reading, because a held-out
            reading carries assay noise and the latent band does not. Scoring
            observations against the latent band under-covers by construction and
            would read as a badly overconfident model on any noisy assay.
        """
        assert self.model is not None, "fit() first"
        Xt = torch.as_tensor(np.asarray(X, float), dtype=DTYPE, device=DEVICE)
        self.model.eval()
        with torch.no_grad():
            post = self.model.posterior(Xt, observation_noise=observation_noise)
            mean = post.mean.squeeze(-1).cpu().numpy()
            std = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).cpu().numpy()
        return mean, std

    @property
    def best_f(self) -> float:
        assert self._y is not None
        return float(self._y.max())
