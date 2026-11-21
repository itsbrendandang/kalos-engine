"""ml-acquire: engine-patch candidates for EXPERIMENT SELECTION (PROTOCOL.md section 9).

Every arm here is the production `_analyze` path (`KalosEngineArm`) with ONE
in-process patch around `kalos.core.optimize.propose_batch` (and, for some
variants, the names it looks up at call time: `qLogNoisyExpectedImprovement`,
`optimize_acqf`, `optimize_acqf_mixed`). The surrogate, the trust gate, the
intervals and the refusals are untouched, so a proposal's interval is the
engine's own band at whatever recipe the candidate picked.

    --arm kalos.bench.ml_v2.arms.acq:make_arm --arm-config '{"name": "acq_seq", "sequential": true}'

Configuration keys (all optional; `{}` is byte-for-byte the reference call):

- `name` (str): arm name (freeze file name). Required to be unique per config.
- Optimizer settings: `restarts` (num_restarts, production 10), `raw` (raw_samples,
  production 256), `maxiter` (L-BFGS-B iteration cap, BoTorch default 200),
  `sequential` (continuous path: greedy sequential q-batch with fantasised
  pending points instead of one joint 5-point solve; the mixed path is already
  sequential inside BoTorch's `optimize_acqf_mixed`).
- Acquisition: `acq` in {"nei" (production qLogNEI), "ucb" (qUCB, `beta`),
  "ts" (Thompson sampling on `ts_n` Sobol candidates, BoTorch-style joint
  posterior draws, without replacement)}.
- Mixed batches: `mix` = list of [slots, kind, param] run IN ORDER, each part
  seeing the earlier parts as pending (fantasised) points, e.g.
  [[4, "nei", null], [1, "ucb", 4.0]]. Kinds: "nei", "ucb" (param beta),
  "nipv" (qNegIntegratedPosteriorVariance, pure exploration), "ts",
  "tr" (qLogNEI inside a TuRBO-style box around the incumbent, param = edge
  length L in unit space, lengthscale-weighted), "out" (qLogNEI in the observed
  box EXTENDED by param x range on every free continuous dim: a deliberate,
  labelled step outside the observed range).
- `untrusted`: an alternative `mix` used only when the engine's reliability
  floor is NOT cleared (the same `clears_floor` the trust verdict uses), e.g.
  [[5, "nipv", null]] = pure exploration while untrusted.

Every variant keeps the engine's guarantees: box clip (except "out", which is
clipped to the EXTENDED box and labelled), categorical snapping, linear
constraints, fixed (held) clone and covariates, the 5 % near-duplicate rule, and
`ProposalBatch` accounting. A variant that cannot honour an input it does not
support (a constraint SURROGATE for UCB/TS/NIPV) falls back to qLogNEI for that
part and records it in `info["fallbacks"]`.
"""
from __future__ import annotations

import contextlib
import functools
import threading
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from ..adapter import ArmOutput, RoundContext
from .kalos_ref import KalosEngineArm

_STATE = threading.local()


def _st() -> dict[str, Any]:
    if not hasattr(_STATE, "d"):
        _STATE.d = {}
    return _STATE.d


# ----------------------------------------------------------------- acquisition factories
def _ucb_factory(orig_nei: Any, beta: float) -> Any:
    from botorch.acquisition.monte_carlo import qUpperConfidenceBound

    def make(model, X_baseline=None, prune_baseline=True, X_pending=None, objective=None, constraints=None, **kw):
        if constraints is not None:  # a constraint surrogate: UCB has no feasibility weighting
            _st().setdefault("fallbacks", []).append("ucb->nei (constraint surrogate)")
            return orig_nei(model, X_baseline=X_baseline, prune_baseline=prune_baseline, X_pending=X_pending,
                            objective=objective, constraints=constraints, **kw)
        return qUpperConfidenceBound(model, beta=float(beta), objective=objective, X_pending=X_pending)

    return make


def _nipv_factory(orig_nei: Any, lower: np.ndarray, upper: np.ndarray, seed: int) -> Any:
    import torch
    from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
    from botorch.utils.sampling import draw_sobol_samples

    from kalos.core.surrogate import DEVICE, DTYPE

    def make(model, X_baseline=None, prune_baseline=True, X_pending=None, objective=None, constraints=None, **kw):
        if constraints is not None:
            _st().setdefault("fallbacks", []).append("nipv->nei (constraint surrogate)")
            return orig_nei(model, X_baseline=X_baseline, prune_baseline=prune_baseline, X_pending=X_pending,
                            objective=objective, constraints=constraints, **kw)
        b = torch.stack([torch.as_tensor(lower, dtype=DTYPE, device=DEVICE),
                         torch.as_tensor(upper, dtype=DTYPE, device=DEVICE)])
        mc = draw_sobol_samples(bounds=b, n=512, q=1, seed=int(seed)).squeeze(1)
        return qNegIntegratedPosteriorVariance(model, mc_points=mc, X_pending=X_pending)

    return make


# ----------------------------------------------------------------- Thompson sampling
def _ts_batch(kw: dict[str, Any], q: int, n_cand: int, seed: int) -> Any:
    """Thompson sampling: q joint posterior draws over n_cand Sobol candidates in
    the (held-feature-collapsed) box, argmax of each draw, without replacement,
    honouring linear constraints, categorical levels and the near-duplicate rule.
    Pending points are not fantasised (TS has no X_pending); the near-duplicate
    rule still keeps proposals away from them."""
    import torch
    from botorch.utils.sampling import draw_sobol_samples

    import kalos.core.optimize as opt
    from kalos.core.surrogate import DEVICE, DTYPE, sanitize_bounds

    s = kw["surrogate"]
    lower, upper = sanitize_bounds(kw["bounds"])
    fixed = {int(k): float(v) for k, v in (kw.get("fixed_features") or {}).items()}
    cat_dims = list(kw.get("cat_dims") or [])
    for j, v in fixed.items():
        if j in cat_dims:
            v = float(np.rint(v))
        lower[j] = upper[j] = v
    cards = opt._resolve_cardinalities(cat_dims, kw.get("cat_cardinalities"), lower, upper) if cat_dims else []
    lin = opt._validate_linear_constraints(kw.get("linear_constraints"), lower, upper)
    b = torch.stack([torch.as_tensor(lower, dtype=DTYPE, device=DEVICE),
                     torch.as_tensor(upper, dtype=DTYPE, device=DEVICE)])
    pool_parts: list[np.ndarray] = []
    draws = 0
    while True:
        sob = draw_sobol_samples(bounds=b, n=n_cand, q=1, seed=int(seed) + draws).detach().cpu().numpy().reshape(
            -1, len(lower))
        for j, k in zip(cat_dims, cards):
            if j in fixed:
                continue
            unit = (sob[:, j] - lower[j]) / max(upper[j] - lower[j], 1e-12)
            sob[:, j] = np.clip(np.floor(unit * k), 0, k - 1)
        sob = np.clip(sob, lower, upper)
        pool_parts.append(sob[opt._linear_ok(sob, lin)])
        draws += 1
        if sum(len(p) for p in pool_parts) >= min(n_cand, 50 * q) or draws >= 8:
            break
    pool = np.vstack(pool_parts)[:n_cand]
    clf = kw.get("feasibility_classifier")
    if clf is not None and getattr(clf, "fitted", False) and len(pool):
        pf = np.asarray(clf.predict_proba(pool), float).reshape(-1)
        keep = pf >= 0.5
        if keep.sum() >= 10 * q:
            pool = pool[keep]
    maximize = bool(kw.get("maximize", True))
    min_sep = float(kw.get("min_separation", opt.NEAR_DUPLICATE_TOL))
    pend = kw.get("pending")
    pend_rows = [] if pend is None else list(np.asarray(pend, float).reshape(-1, len(lower)))
    accepted: list[np.ndarray] = []
    if len(pool):
        Xc = torch.as_tensor(pool, dtype=DTYPE, device=DEVICE)
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(int(seed))
            post = s.model.posterior(Xc)
            samp = post.rsample(torch.Size([q])).squeeze(-1).detach().cpu().numpy()  # q x N
        if not maximize:
            samp = -samp
        taken = np.zeros(len(pool), bool)
        for t in range(q):
            order = np.argsort(-samp[t], kind="stable")
            for i in order:
                if taken[i]:
                    continue
                if opt._is_distinct(pool[i], accepted + pend_rows, lower, upper, cat_dims, min_sep):
                    accepted.append(pool[i])
                    taken[i] = True
                    break
    X = np.vstack(accepted) if accepted else np.empty((0, len(lower)))
    for j in cat_dims:
        X[:, j] = np.rint(X[:, j])
    notice = None if X.shape[0] >= q else opt._fewer_notice(q, X.shape[0], min_sep)
    return opt.ProposalBatch(X, q, 0, min_sep, notice)


# ----------------------------------------------------------------- boxes
def _tr_box(s: Any, lower: np.ndarray, upper: np.ndarray, L: float, maximize: bool,
            free: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """TuRBO-style box: centred on the training point with the best posterior
    mean, edge L (unit space) scaled per dim by lengthscale / geometric mean."""
    import torch

    from kalos.core.surrogate import _lengthscale_modules

    X = s._X.detach().cpu().numpy()
    with torch.no_grad():
        mu = s.model.posterior(s._X).mean.detach().cpu().numpy().reshape(-1)
    c = X[int(np.argmax(mu) if maximize else np.argmin(mu))]
    span = np.where(upper - lower > 0, upper - lower, 1.0)
    w = np.ones(len(lower))
    try:
        mods = _lengthscale_modules(s.model)
        ls = mods[0].lengthscale.detach().cpu().numpy().reshape(-1) if mods else None
        if ls is not None and len(ls) == len(free):
            ww = ls / np.exp(np.mean(np.log(ls)))
            w[free] = np.clip(ww, 0.25, 4.0)
        elif ls is not None and len(ls) == len(lower):
            ww = ls[free] / np.exp(np.mean(np.log(ls[free])))
            w[free] = np.clip(ww, 0.25, 4.0)
    except Exception:  # noqa: BLE001 - fall back to an isotropic box
        pass
    lo, hi = lower.copy(), upper.copy()
    half = 0.5 * L * w * span
    lo[free] = np.maximum(lower[free], c[free] - half[free])
    hi[free] = np.minimum(upper[free], c[free] + half[free])
    return lo, hi


# ----------------------------------------------------------------- the patched propose_batch
def _make_propose_batch(orig: Any, cfg: Mapping[str, Any]) -> Any:
    import kalos.core.optimize as opt

    restarts = cfg.get("restarts")
    raw = cfg.get("raw")
    maxiter = cfg.get("maxiter")
    sequential = bool(cfg.get("sequential", False))
    acq = cfg.get("acq", "nei")
    beta = float(cfg.get("beta", 4.0))
    ts_n = int(cfg.get("ts_n", 4096))
    mix = cfg.get("mix")
    untrusted = cfg.get("untrusted")

    @contextlib.contextmanager
    def _optim_patches() -> Iterator[None]:
        with contextlib.ExitStack() as es:
            if maxiter is not None or sequential:
                o_acqf = opt.optimize_acqf

                def oa(*a, **k):  # type: ignore[no-untyped-def]
                    if maxiter is not None:
                        k["options"] = {**(k.get("options") or {}), "maxiter": int(maxiter)}
                    if sequential and int(k.get("q", 1)) > 1 and k.get("return_best_only", True):
                        k["sequential"] = True
                    return o_acqf(*a, **k)

                es.enter_context(_patched(opt, "optimize_acqf", oa))
            if maxiter is not None:
                o_mixed = opt.optimize_acqf_mixed

                def om(*a, **k):  # type: ignore[no-untyped-def]
                    k["options"] = {**(k.get("options") or {}), "maxiter": int(maxiter)}
                    return o_mixed(*a, **k)

                es.enter_context(_patched(opt, "optimize_acqf_mixed", om))
            yield

    def _call(kw: dict[str, Any], q: int, kind: str, param: Any, seed_off: int) -> Any:
        """One part of a batch: q proposals of one kind, pending = kw['pending']."""
        from kalos.core.surrogate import sanitize_bounds

        k2 = dict(kw)
        k2["q"] = q
        if restarts is not None:
            k2["num_restarts"] = int(restarts)
        if raw is not None:
            k2["raw_samples"] = int(raw)
        if k2.get("seed") is not None:
            k2["seed"] = int(k2["seed"]) + seed_off
        seed = int(k2.get("seed") or 0)
        if kind == "ts":
            return _ts_batch(k2, q, ts_n, seed)
        lower, upper = sanitize_bounds(k2["bounds"])
        cats = set(k2.get("cat_dims") or [])
        held = {int(j) for j in (k2.get("fixed_features") or {})}
        free = [j for j in range(len(lower)) if j not in cats and j not in held]
        with contextlib.ExitStack() as es:
            es.enter_context(_optim_patches())
            if kind == "ucb":
                es.enter_context(_patched(opt, "qLogNoisyExpectedImprovement",
                                          _ucb_factory(opt.qLogNoisyExpectedImprovement, float(param))))
            elif kind == "nipv":
                es.enter_context(_patched(opt, "qLogNoisyExpectedImprovement",
                                          _nipv_factory(opt.qLogNoisyExpectedImprovement, lower, upper, seed)))
            elif kind == "tr" and free:
                lo, hi = _tr_box(k2["surrogate"], lower, upper, float(param), bool(k2.get("maximize", True)), free)
                k2["bounds"] = np.vstack([lo, hi])
            elif kind == "out" and free and not k2.get("linear_constraints"):
                span = upper - lower
                lo, hi = lower.copy(), upper.copy()
                lo[free] = lower[free] - float(param) * span[free]
                hi[free] = upper[free] + float(param) * span[free]
                k2["bounds"] = np.vstack([lo, hi])
                _st()["out_box"] = (lower.copy(), upper.copy())
            return orig(**k2)

    def _mixed(kw: dict[str, Any], parts: list[Any]) -> Any:
        q_total = int(kw.get("q", 5))
        rows: list[np.ndarray] = []
        kinds: list[str] = []
        replaced = 0
        notice = None
        pend0 = kw.get("pending")
        for i, (k, kind, param) in enumerate(parts):
            k = int(k)
            if k <= 0:
                continue
            kw2 = dict(kw)
            extra = np.vstack(rows) if rows else None
            if extra is not None:
                kw2["pending"] = extra if pend0 is None else np.vstack([np.asarray(pend0, float), extra])
            pb = _call(kw2, k, str(kind), param, 101 * i)
            for r in pb.X:
                rows.append(np.asarray(r, float))
                kinds.append(str(kind))
            replaced += int(pb.near_duplicates_replaced)
            notice = notice or pb.notice
        X = np.vstack(rows) if rows else np.empty((0, np.asarray(kw["bounds"]).shape[1]))
        _st()["kinds"] = kinds
        if X.shape[0] >= q_total:
            notice = None
        elif notice is None:
            notice = opt._fewer_notice(q_total, X.shape[0], float(kw.get("min_separation", opt.NEAR_DUPLICATE_TOL)))
        return opt.ProposalBatch(X, q_total, replaced, float(kw.get("min_separation", opt.NEAR_DUPLICATE_TOL)),
                                 notice)

    @functools.wraps(orig)
    def propose_batch(surrogate, bounds, q=3, num_restarts=10, raw_samples=256, **kw):  # type: ignore[no-untyped-def]
        kw = {"surrogate": surrogate, "bounds": bounds, "q": q, "num_restarts": num_restarts,
              "raw_samples": raw_samples, **kw}
        st = _st()
        floor = st.get("clears_floor")
        parts = mix
        if untrusted is not None and floor is not True:
            parts = untrusted
            st["policy"] = "untrusted"
        else:
            st["policy"] = "trusted" if floor else "default"
        if parts is None:
            parts = [[int(q), acq, beta]]
        return _mixed(kw, [list(p) for p in parts])

    return propose_batch


@contextlib.contextmanager
def _patched(obj: Any, attr: str, value: Any) -> Iterator[None]:
    old = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, old)


class AcqArm(KalosEngineArm):
    """Reference engine with the experiment-selection step swapped (see module doc)."""

    shippable = True

    def __init__(self, config: Mapping[str, Any] | None = None):
        super().__init__(config)
        self.name = str(self.config.get("name") or "acq_unnamed")

    @contextlib.contextmanager
    def _engine_patch(self) -> Iterator[None]:
        import kalos.core.optimize as opt
        import kalos.portal.analysis as an

        st = _st()
        st.clear()
        orig_floor = an.trust_clears_floor

        def floor(*a, **k):  # type: ignore[no-untyped-def]
            out = orig_floor(*a, **k)
            try:
                st["clears_floor"] = bool(out.get("clears_floor"))
            except Exception:  # noqa: BLE001
                st["clears_floor"] = None
            return out

        with _patched(an, "trust_clears_floor", floor), \
                _patched(opt, "propose_batch", _make_propose_batch(opt.propose_batch, self.config)):
            yield

    def engine_patch(self) -> contextlib.AbstractContextManager[None]:
        return self._engine_patch()

    def propose(self, ctx: RoundContext) -> ArmOutput:
        out = super().propose(ctx)
        st = _st()
        out.info = {**out.info, "acq_kinds": st.get("kinds"), "acq_policy": st.get("policy"),
                    "acq_fallbacks": st.get("fallbacks")}
        if st.get("out_box") is not None and out.proposals:
            lo, hi = st["out_box"]
            names = list(ctx.space.continuous)
            n_out = 0
            for p in out.proposals:
                vals = [p.recipe.get(n) for n in names]
                try:
                    v = np.asarray(vals, float)
                    m = len(v)
                    tol = 1e-3 * np.maximum(hi[:m] - lo[:m], 1e-12)
                    if m <= len(lo) and (np.any(v < lo[:m] - tol) or np.any(v > hi[:m] + tol)):
                        p.label = (p.label + "|outside_observed_range").lstrip("|")
                        n_out += 1
                except (TypeError, ValueError):
                    pass
            out.info["n_outside_observed_range"] = n_out
        return out


def make_arm(config: Mapping[str, Any] | None = None) -> AcqArm:
    return AcqArm(config)
