"""The reference arm: the production analysis (`kalos.portal.analysis._analyze`)
exactly as a campaign re-analysis calls it, plus the base class for
engine-patch candidates.

    --arm kalos.bench.ml_v2.arms.kalos_ref:make_arm

A candidate that changes an engine internal subclasses `KalosEngineArm` and
overrides `engine_patch()` (see `kalos.bench.ml_v2.arms.example_patch`). Nothing
in the engine is edited: the patch is applied in-process around one analysis.
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..adapter import ArmOutput, Band, Proposal, RoundContext, Verdict

_CAPTURE: dict[str, Any] = {}


@contextlib.contextmanager
def _capture_served() -> Iterator[dict[str, Any]]:
    """Keep the served (proposal) surrogate and the design encoder's arguments of
    the next analysis. In-process wrappers, removed on exit; nothing is edited."""
    import kalos.portal.analysis as an
    from kalos.core import surrogate as s_mod

    cap: dict[str, Any] = {}
    orig_fit = s_mod.Surrogate.fit
    orig_enc = an._encode_pending

    def fit(self, *a, **kw):  # type: ignore[no-untyped-def]
        out = orig_fit(self, *a, **kw)
        if kw.get("check_conditioning"):
            cap["surrogate"] = self
        return out

    def enc(pending, cont_feats, kept_cats, code_maps, width, **kw):  # type: ignore[no-untyped-def]
        cap["encoder"] = (list(cont_feats), list(kept_cats), dict(code_maps), int(width),
                          frozenset(kw.get("absent_zero", frozenset())))
        return orig_enc(pending, cont_feats, kept_cats, code_maps, width, **kw)

    s_mod.Surrogate.fit = fit  # type: ignore[method-assign]
    an._encode_pending = enc  # type: ignore[assignment]
    try:
        yield cap
    finally:
        s_mod.Surrogate.fit = orig_fit  # type: ignore[method-assign]
        an._encode_pending = orig_enc


def _setup_for(ctx: RoundContext) -> dict[str, Any] | None:
    """A campaign setup only when the task declares something the engine needs
    from it (linear constraints); otherwise None, today's default path."""
    if not ctx.space.linear_constraints:
        return None
    from kalos.portal.setup_settings import analysis_setup, normalize_setup

    doc = normalize_setup({
        "goal": ctx.goal,
        "bounds": {k: {"lo": lo, "hi": hi} for k, (lo, hi) in ctx.space.continuous.items()},
        "linear_constraints": [dict(c) for c in ctx.space.linear_constraints],
    })
    return analysis_setup(doc)


def _roles_for(ctx: RoundContext) -> Any:
    if not ctx.roles:
        return None
    from kalos.domains import ColumnRoles

    r = dict(ctx.roles)
    return ColumnRoles(
        target=ctx.target,
        features=tuple(r.get("features", ())),
        categoricals=tuple(r.get("categoricals", ())),
        groups=r.get("groups"),
        strain=r.get("strain"),
        strain_policy=r.get("strain_policy"),
        strain_level=r.get("strain_level"),
        covariates=tuple(r.get("covariates", ())),
    )


def _band_fn(cap: Mapping[str, Any], res: Mapping[str, Any]) -> tuple[Callable | None, Callable | None]:
    s = cap.get("surrogate")
    encoder = cap.get("encoder")
    if s is None or encoder is None:
        return None, None
    import kalos.portal.analysis as an
    from kalos.portal.analysis_parts.intervals import band_from_summary

    cont, cats, code_maps, width, absent = encoder
    method = res.get("interval_method") or {}
    noise_floor = float((res.get("noise") or {}).get("noise_sd") or 0.0)

    def encode(df: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray]:
        frame = df.copy()
        ok = np.ones(len(frame), bool)
        for c in cont:
            v = pd.to_numeric(frame[c], errors="coerce") if c in frame else pd.Series(np.nan, index=frame.index)
            ok &= np.isfinite(v.to_numpy(float)) | (c in absent)
        for c in cats:
            lab = frame[c].astype(str).str.strip() if c in frame else pd.Series("", index=frame.index)
            ok &= lab.isin(code_maps[c]).to_numpy()
        if not ok.any():
            return None, ok
        x, _ = an._encode_pending(frame[ok], cont, cats, code_maps, width, absent_zero=absent)
        return x, ok

    def posterior(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, ok = encode(df)
        mu = np.full(len(df), np.nan)
        sd = np.full(len(df), np.nan)
        if x is not None:
            m, lat = s.posterior(x)
            if method.get("sigma_source") == "latent_plus_assay_noise_floor":
                sp = np.sqrt(np.asarray(lat, float) ** 2 + noise_floor ** 2)
            else:
                _, sp = s.posterior(x, observation_noise=True)
            mu[ok], sd[ok] = np.asarray(m, float).reshape(-1), np.asarray(sp, float).reshape(-1)
        return mu, sd, ok

    def band_at(df: pd.DataFrame, level: float) -> tuple[np.ndarray, np.ndarray] | None:
        mu, sd, _ok = posterior(df)
        lo, hi = band_from_summary(mu, sd, method, level)
        if lo is None or hi is None:
            return None
        return np.asarray(lo, float), np.asarray(hi, float)

    def predict(df: pd.DataFrame) -> np.ndarray:
        return posterior(df)[0]

    return band_at, predict


class KalosEngineArm:
    """The production engine path. Subclass and override `engine_patch` (and
    `name`) to make an engine-patch candidate."""

    name = "kalos_ref"
    shippable = True

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.config = dict(config or {})

    def engine_patch(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def analyze(self, df: pd.DataFrame, target: str, *, roles: Any, setup: Mapping[str, Any] | None) -> dict:
        """One production analysis (override only to change how the engine is CALLED)."""
        from kalos.portal.analysis import _analyze

        kw: dict[str, Any] = {"roles": roles}
        if setup is not None:
            kw["setup"] = setup
        return _analyze(df, target=target, **kw)

    def propose(self, ctx: RoundContext) -> ArmOutput:
        from kalos.portal.uploads import UploadRejected

        cols = [c for c in ctx.sheet.columns]
        df = ctx.sheet[cols].copy().reset_index(drop=True)
        if ctx.round_index is not None:
            df.attrs["kalos_round_index"] = [int(r) for r in ctx.round_index]
        with self.engine_patch(), _capture_served() as cap:
            try:
                res = self.analyze(df, ctx.target, roles=_roles_for(ctx), setup=_setup_for(ctx))
            except UploadRejected as rej:
                code = getattr(rej, "code", "rejected")
                return ArmOutput([], Verdict(False, f"refused_upload:{code}", True), info={"refusal": code})
            band_at, predict = _band_fn(cap, res)
        props = []
        for p in res.get("proposals") or []:
            bands = {}
            for lv in ctx.levels:
                iv = next((e for e in p.get("intervals", []) if abs(float(e["coverage"]) - lv) < 1e-9), None)
                ok = bool(iv and iv.get("available") and iv.get("lo") is not None)
                bands[lv] = Band(float(iv["lo"]), float(iv["hi"])) if ok and iv else Band(None, None)
            props.append(Proposal(dict(p["recipe"]), p.get("pred"), bands, (p.get("trust") or {}).get("status", "")))
        trust = res.get("proposal_trust") or {}
        rel = res.get("reliability") or {}
        verdict = Verdict(
            trusted=trust.get("status") == "trusted",
            state=str((res.get("verdict") or {}).get("state")),
            withheld=bool(trust.get("withheld")) or not props,
            cv_spearman=res.get("cv_spearman"),
            clears_floor=rel.get("clears_floor"),
        )
        info = {
            "cv_grouping": ((res.get("cv_grouping") or {}).get("used") or {}).get("basis"),
            "interval_method": (res.get("interval_method") or {}).get("method_by_level"),
            "replicate_aware": (res.get("noise") or {}).get("replicate_aware"),
            "n_features": len(res.get("features") or []),
        }
        return ArmOutput(props, verdict, band_at=band_at, predict=predict, info=info)


def make_arm(config: Mapping[str, Any] | None = None) -> KalosEngineArm:
    return KalosEngineArm(config)
