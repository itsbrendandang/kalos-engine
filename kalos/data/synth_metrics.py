"""Privacy and fidelity MEASUREMENTS for an anonymized or synthetic table export.

What this IS: a small set of standard, seeded, deterministic measurements that
compare a candidate synthetic (or anonymized) table against the real table it
was derived from, for a future synthetic-data export next to
`kalos.data.anonymizer`:

* privacy - distance to closest record (DCR) in a standardized space, read
  against a real-to-real reference distribution so the number means something;
  nearest-neighbour distance ratio (NNDR); and the exact-duplicate rate
  (synthetic rows identical to some real row).
* fidelity - per-column two-sample KS statistic (numeric) and total-variation
  distance (categorical); the difference between real and synthetic pairwise
  Pearson/Spearman correlation matrices; the difference between pairwise
  normalized mutual-information matrices on quantile-binned columns; and a
  classifier two-sample test (C2ST: cross-validated ROC AUC of a classifier
  telling real from synthetic, 0.5 = indistinguishable to that classifier).

What this is NOT: a privacy guarantee, a certification, or evidence of
regulatory compliance. DCR/NNDR/duplicate rates only detect the crudest leak -
synthetic rows sitting on or next to real records. They say nothing about
membership- or attribute-inference attacks, linkage with outside data, or
differential privacy, and a "clean" DCR does not make an export safe to share.
Likewise, good fidelity numbers mean the marginals and pairwise structure that
were measured match; C2ST only shows that the chosen classifier could not tell
the tables apart at this sample size. Report the numbers; do not convert them
into a claim.

Mixed tables: numeric columns (non-boolean numeric dtypes) are standardized by
the REAL table's mean and standard deviation; every other column is treated as
categorical and one-hot encoded with the real table's levels (a level unseen in
the real table encodes as all zeros). A category mismatch therefore adds 1 to
the squared distance, like a one-standard-deviation numeric difference.

NaN policy: per-column metrics (KS, TVD) drop missing cells column by column.
Row-wise metrics (DCR, NNDR, duplicates, correlations, mutual information,
C2ST) drop any row with a missing cell; the number of rows dropped per table is
reported. Nothing is imputed.

All randomness (C2ST subsampling and folds) comes from a seeded
`numpy.random.Generator` / `random_state`; same inputs and seed, same report.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

# Defaults for the report's summary flags. They are reading aids, not standards:
# each flag in the report states the threshold it was computed with.
DEFAULT_DCR_REFERENCE_QUANTILE = 0.05
DEFAULT_MEMORIZATION_SHARE = 0.15  # vs ~0.05 expected for a fresh sample
DEFAULT_C2ST_AUC_FLAG = 0.60


# ----------------------------------------------------------------------------- #
# Schema + encoding
# ----------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Schema:
    columns: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _schema(
    real: pd.DataFrame, others: Sequence[pd.DataFrame], categorical: Sequence[str] | None
) -> _Schema:
    cols = [str(c) for c in real.columns]
    if len(set(cols)) != len(cols):
        raise ValueError("real table has duplicate column names")
    for other in others:
        oc = {str(c) for c in other.columns}
        if oc != set(cols):
            missing = sorted(set(cols) - oc)
            extra = sorted(oc - set(cols))
            raise ValueError(
                f"column mismatch with the real table: missing={missing} extra={extra}"
            )
    forced = {str(c) for c in (categorical or ())}
    unknown = forced - set(cols)
    if unknown:
        raise ValueError(f"categorical columns not in the table: {sorted(unknown)}")
    numeric = tuple(c for c in cols if c not in forced and _is_numeric(real[c]))
    cats = tuple(c for c in cols if c not in numeric)
    return _Schema(tuple(cols), numeric, cats)


def _align(df: pd.DataFrame, schema: _Schema) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    out = out[list(schema.columns)]
    for c in schema.numeric:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    for c in schema.categorical:
        out[c] = out[c].astype(object).map(lambda v: np.nan if pd.isna(v) else str(v))
    return out


@dataclass(frozen=True)
class _Encoder:
    schema: _Schema
    mean: np.ndarray
    std: np.ndarray
    levels: dict[str, tuple[str, ...]]

    @classmethod
    def fit(cls, real: pd.DataFrame, schema: _Schema) -> _Encoder:
        num = real[list(schema.numeric)].to_numpy(float)
        if num.shape[1]:
            mean = num.mean(axis=0)
            std = num.std(axis=0)
            std = np.where(std > 0, std, 1.0)
        else:
            mean = std = np.zeros(0)
        levels = {c: tuple(sorted(real[c].unique())) for c in schema.categorical}
        return cls(schema, mean, std, levels)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = [(df[list(self.schema.numeric)].to_numpy(float) - self.mean) / self.std]
        scale = 1.0 / math.sqrt(2.0)  # a category mismatch adds exactly 1 to d^2
        for c in self.schema.categorical:
            lv = self.levels[c]
            vals = df[c].to_numpy(object)
            parts.append(np.column_stack(
                [(vals == level).astype(float) * scale for level in lv]
            ) if lv else np.zeros((len(df), 0)))
        return np.hstack(parts) if parts else np.zeros((len(df), 0))


def _complete(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    kept = df.dropna(axis=0, how="any")
    return kept, int(len(df) - len(kept))


def _f(x: Any) -> float | None:
    """JSON-safe float: NaN/inf -> None."""
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def _quantiles(d: np.ndarray) -> dict[str, float | None]:
    if d.size == 0:
        return {"min": None, "p05": None, "median": None, "mean": None}
    return {
        "min": _f(d.min()),
        "p05": _f(np.quantile(d, 0.05)),
        "median": _f(np.median(d)),
        "mean": _f(d.mean()),
    }


# ----------------------------------------------------------------------------- #
# Privacy
# ----------------------------------------------------------------------------- #
def _nn_dists(ref: np.ndarray, query: np.ndarray, k: int, *, exclude_self: bool) -> np.ndarray:
    kk = k + (1 if exclude_self else 0)
    kk = min(kk, len(ref))
    nn = NearestNeighbors(n_neighbors=kk).fit(ref)
    d, _ = nn.kneighbors(query)
    return d[:, 1:] if exclude_self else d


def dcr(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    holdout: pd.DataFrame | None = None,
    *,
    categorical: Sequence[str] | None = None,
    reference_quantile: float = DEFAULT_DCR_REFERENCE_QUANTILE,
) -> dict:
    """Distance to closest record: synthetic -> real, with a reference.

    Each synthetic row's Euclidean distance to its nearest real row in the
    standardized/one-hot space. The REFERENCE is how close genuinely new data
    lands to the real table: the holdout -> real distances if a holdout (real
    rows NOT given to the generator) is supplied, else real -> real leave-one-out
    distances (a weaker reference: it is computed within the very table the
    generator saw). `share_below_reference_q` is the fraction of synthetic rows
    closer to the real table than the reference's `reference_quantile`
    quantile; a fresh sample from the same distribution gives about
    `reference_quantile`, a generator that copies or jitters real rows gives
    much more.
    """
    others = [synthetic] + ([holdout] if holdout is not None else [])
    schema = _schema(real, others, categorical)
    r, r_drop = _complete(_align(real, schema))
    s, s_drop = _complete(_align(synthetic, schema))
    enc = _Encoder.fit(r, schema)
    R, S = enc.transform(r), enc.transform(s)
    if len(R) < 2 or len(S) < 1:
        raise ValueError("DCR needs at least 2 complete real rows and 1 synthetic row")
    d_syn = _nn_dists(R, S, 1, exclude_self=False)[:, 0]
    if holdout is not None:
        h, h_drop = _complete(_align(holdout, schema))
        if len(h) < 1:
            raise ValueError("holdout has no complete rows")
        d_ref = _nn_dists(R, enc.transform(h), 1, exclude_self=False)[:, 0]
        ref_kind = "holdout_to_real"
    else:
        h_drop = 0
        d_ref = _nn_dists(R, R, 1, exclude_self=True)[:, 0]
        ref_kind = "real_to_real_leave_one_out"
    thr = float(np.quantile(d_ref, reference_quantile))
    med_ref = float(np.median(d_ref))
    return {
        "synthetic": _quantiles(d_syn),
        "reference": _quantiles(d_ref),
        "reference_kind": ref_kind,
        "reference_quantile": reference_quantile,
        "reference_threshold": _f(thr),
        "share_below_reference_q": _f(np.mean(d_syn < thr)),
        "median_ratio_synthetic_to_reference": _f(np.median(d_syn) / med_ref)
        if med_ref > 0 else None,
        "rows_dropped_nan": {"real": r_drop, "synthetic": s_drop, "holdout": h_drop},
    }


def nndr(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    holdout: pd.DataFrame | None = None,
    *,
    categorical: Sequence[str] | None = None,
) -> dict:
    """Nearest-neighbour distance ratio: d(1st nearest real) / d(2nd nearest real).

    Per synthetic row, in the same encoded space as `dcr`. Values near 0 mean
    the row sits much closer to ONE real record than to any other - the
    signature of a jittered copy; values near 1 mean it is equidistant from
    several real rows. A row with both distances 0 (duplicated real record)
    gets ratio 0. The holdout (or leave-one-out real) distribution is reported
    alongside for comparison.
    """
    others = [synthetic] + ([holdout] if holdout is not None else [])
    schema = _schema(real, others, categorical)
    r, _ = _complete(_align(real, schema))
    s, _ = _complete(_align(synthetic, schema))
    enc = _Encoder.fit(r, schema)
    R = enc.transform(r)
    if len(R) < 3:
        raise ValueError("NNDR needs at least 3 complete real rows")

    def ratio(d: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            out = d[:, 0] / d[:, 1]
        return np.where(d[:, 1] > 0, out, 0.0)

    syn = ratio(_nn_dists(R, enc.transform(s), 2, exclude_self=False))
    if holdout is not None:
        h, _ = _complete(_align(holdout, schema))
        ref = ratio(_nn_dists(R, enc.transform(h), 2, exclude_self=False))
        kind = "holdout_to_real"
    else:
        ref = ratio(_nn_dists(R, R, 2, exclude_self=True))
        kind = "real_to_real_leave_one_out"
    return {"synthetic": _quantiles(syn), "reference": _quantiles(ref), "reference_kind": kind}


def _row_keys(df: pd.DataFrame, schema: _Schema) -> list[tuple]:
    cols = [df[c].to_numpy() for c in schema.columns]
    keys = []
    for row in zip(*cols, strict=True):
        keys.append(tuple(
            ("nan",) if (isinstance(v, float) and math.isnan(v))
            else (float(v) if c in schema.numeric else str(v))
            for c, v in zip(schema.columns, row, strict=True)
        ))
    return keys


def exact_duplicate_rate(
    real: pd.DataFrame, synthetic: pd.DataFrame, *, categorical: Sequence[str] | None = None
) -> dict:
    """Share of synthetic rows that exactly equal some real row (all columns).

    Exact float equality after numeric coercion; missing cells compare equal to
    each other here (so a copied row with a blank still counts). A near-copy
    with any tiny perturbation is NOT a duplicate - that is what DCR is for.
    """
    schema = _schema(real, [synthetic], categorical)
    r = _align(real, schema)
    s = _align(synthetic, schema)
    real_keys = set(_row_keys(r, schema))
    hits = [k in real_keys for k in _row_keys(s, schema)]
    n = len(hits)
    return {
        "n_synthetic": n,
        "n_duplicates": int(sum(hits)),
        "rate": _f(sum(hits) / n) if n else None,
    }


# ----------------------------------------------------------------------------- #
# Fidelity
# ----------------------------------------------------------------------------- #
def column_ks(
    real: pd.DataFrame, synthetic: pd.DataFrame, *, categorical: Sequence[str] | None = None
) -> dict[str, dict]:
    """Two-sample Kolmogorov-Smirnov statistic per numeric column (0 = same ECDF)."""
    schema = _schema(real, [synthetic], categorical)
    r, s = _align(real, schema), _align(synthetic, schema)
    out: dict[str, dict] = {}
    for c in schema.numeric:
        a, b = r[c].dropna().to_numpy(float), s[c].dropna().to_numpy(float)
        if len(a) == 0 or len(b) == 0:
            out[c] = {"statistic": None, "pvalue": None}
            continue
        # Exact p-values for small samples, asymptotic beyond; the statistic
        # itself does not depend on the method.
        res = (
            stats.ks_2samp(a, b, method="exact")
            if max(len(a), len(b)) <= 200
            else stats.ks_2samp(a, b, method="asymp")
        )
        out[c] = {"statistic": _f(res.statistic), "pvalue": _f(res.pvalue)}
    return out


def column_tvd(
    real: pd.DataFrame, synthetic: pd.DataFrame, *, categorical: Sequence[str] | None = None
) -> dict[str, float | None]:
    """Total-variation distance per categorical column: 0.5 * sum |p - q| over
    the union of levels (0 = same frequencies, 1 = disjoint supports)."""
    schema = _schema(real, [synthetic], categorical)
    r, s = _align(real, schema), _align(synthetic, schema)
    out: dict[str, float | None] = {}
    for c in schema.categorical:
        p = r[c].dropna().value_counts(normalize=True)
        q = s[c].dropna().value_counts(normalize=True)
        if p.empty or q.empty:
            out[c] = None
            continue
        idx = p.index.union(q.index)
        out[c] = _f(0.5 * np.abs(p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).sum())
    return out


def _upper(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices_from(m, k=1)]


def correlation_difference(
    real: pd.DataFrame, synthetic: pd.DataFrame, *, categorical: Sequence[str] | None = None
) -> dict:
    """Pearson and Spearman correlation-matrix differences over numeric columns.

    Reports the Frobenius norm of (corr_real - corr_synth) over the full matrix
    and the mean / max absolute difference over the distinct pairs. An
    undefined correlation (a constant column) is set to 0 on both sides before
    differencing. Fewer than two numeric columns -> all None.
    """
    schema = _schema(real, [synthetic], categorical)
    cols = list(schema.numeric)
    out: dict[str, Any] = {"columns": cols}
    if len(cols) < 2:
        for m in ("pearson", "spearman"):
            out[m] = {"frobenius": None, "mean_abs": None, "max_abs": None}
        return out
    r, _ = _complete(_align(real, schema)[cols])
    s, _ = _complete(_align(synthetic, schema)[cols])
    methods: tuple[Literal["pearson"], Literal["spearman"]] = ("pearson", "spearman")
    for m in methods:
        with np.errstate(invalid="ignore", divide="ignore"):
            cr = np.nan_to_num(r.corr(method=m).to_numpy(float))
            cs = np.nan_to_num(s.corr(method=m).to_numpy(float))
        diff = cr - cs
        ud = np.abs(_upper(diff))
        out[m] = {
            "frobenius": _f(np.linalg.norm(diff)),
            "mean_abs": _f(ud.mean()),
            "max_abs": _f(ud.max()),
        }
    return out


def mutual_information_difference(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    categorical: Sequence[str] | None = None,
    bins: int = 5,
) -> dict:
    """Pairwise normalized-mutual-information difference across ALL columns.

    Numeric columns are discretized into `bins` quantile bins whose edges come
    from the REAL table (synthetic values outside fall into the end bins);
    categoricals are used as-is. NMI is in [0, 1]. A plug-in estimate on
    binned data: biased upward at small n, so compare real vs synthetic at
    similar sizes rather than reading the absolute values.
    """
    schema = _schema(real, [synthetic], categorical)
    cols = list(schema.columns)
    if len(cols) < 2:
        return {"columns": cols, "frobenius": None, "mean_abs": None, "max_abs": None}
    r, _ = _complete(_align(real, schema))
    s, _ = _complete(_align(synthetic, schema))

    def codes(df: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {}
        for c in cols:
            if c in schema.numeric:
                edges = np.unique(np.quantile(r[c].to_numpy(float), np.linspace(0, 1, bins + 1)))
                out[c] = np.digitize(df[c].to_numpy(float), edges[1:-1])
            else:
                out[c] = df[c].astype(str).to_numpy()
        return out

    cr, cs = codes(r), codes(s)
    k = len(cols)
    mr, ms = np.eye(k), np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            mr[i, j] = mr[j, i] = normalized_mutual_info_score(cr[cols[i]], cr[cols[j]])
            ms[i, j] = ms[j, i] = normalized_mutual_info_score(cs[cols[i]], cs[cols[j]])
    diff = mr - ms
    ud = np.abs(_upper(diff))
    return {
        "columns": cols,
        "bins": bins,
        "frobenius": _f(np.linalg.norm(diff)),
        "mean_abs": _f(ud.mean()),
        "max_abs": _f(ud.max()),
    }


def c2st(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    categorical: Sequence[str] | None = None,
    classifier: str = "logistic",
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    """Classifier two-sample test: cross-validated ROC AUC for real vs synthetic.

    The larger table is subsampled (seeded) to the size of the smaller so the
    classes are balanced; stratified K-fold out-of-fold probabilities give one
    pooled AUC (plus per-fold mean/std). 0.5 means THIS classifier could not
    tell the tables apart; 1.0 means it separates them perfectly. An AUC below
    0.5 is noise around 0.5 (e.g. an exact copy, where every row appears in
    both classes).

    `classifier="logistic"`: logistic regression on the encoded features plus
    squared numeric features, so it sees location AND scale shifts but not
    general interactions. `classifier="gbm"`: sklearn histogram gradient
    boosting, which sees interactions but needs more rows to be stable.
    """
    schema = _schema(real, [synthetic], categorical)
    r, _ = _complete(_align(real, schema))
    s, _ = _complete(_align(synthetic, schema))
    n = min(len(r), len(s))
    if n < n_splits * 2:
        raise ValueError(f"C2ST needs at least {n_splits * 2} complete rows per table")
    rng = np.random.default_rng(seed)
    r = r.iloc[np.sort(rng.choice(len(r), n, replace=False))]
    s = s.iloc[np.sort(rng.choice(len(s), n, replace=False))]
    enc = _Encoder.fit(r, schema)
    X = np.vstack([enc.transform(r), enc.transform(s)])
    y = np.r_[np.zeros(n), np.ones(n)]
    if classifier == "logistic":
        k = len(schema.numeric)
        X = np.hstack([X, X[:, :k] ** 2])
        model_factory = lambda: make_pipeline(  # noqa: E731
            StandardScaler(), LogisticRegression(max_iter=2000)
        )
    elif classifier == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier

        model_factory = lambda: HistGradientBoostingClassifier(  # noqa: E731
            max_iter=100, random_state=seed
        )
    else:
        raise ValueError("classifier must be 'logistic' or 'gbm'")
    oof = np.zeros(len(y))
    fold_auc: list[float] = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # Single-threaded: these tables are small, and OpenMP oversubscription
    # (e.g. alongside torch on a busy host) can make the GBM 100x slower.
    with threadpool_limits(limits=1):
        for tr, te in skf.split(X, y):
            model = model_factory().fit(X[tr], y[tr])
            p = model.predict_proba(X[te])[:, 1]
            oof[te] = p
            fold_auc.append(float(roc_auc_score(y[te], p)))
    return {
        "classifier": classifier,
        "n_per_class": int(n),
        "n_splits": n_splits,
        "auc": _f(roc_auc_score(y, oof)),
        "fold_auc_mean": _f(np.mean(fold_auc)),
        "fold_auc_std": _f(np.std(fold_auc)),
    }


# ----------------------------------------------------------------------------- #
# Report
# ----------------------------------------------------------------------------- #
def synth_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    holdout: pd.DataFrame | None = None,
    *,
    categorical: Sequence[str] | None = None,
    classifier: str = "logistic",
    seed: int = 0,
    mi_bins: int = 5,
    memorization_share: float = DEFAULT_MEMORIZATION_SHARE,
    c2st_auc_flag: float = DEFAULT_C2ST_AUC_FLAG,
) -> dict:
    """All measurements as one plain, JSON-serializable dict.

    `summary` holds the headline numbers and two reading flags, each with the
    threshold it used: `near_copy_signal` (share of synthetic rows closer to
    the real table than the reference's 5th percentile exceeds
    `memorization_share`; exact duplicates are reported separately because on
    coarse or all-categorical tables they occur naturally) and
    `distinguishable_by_c2st` (C2ST AUC >= `c2st_auc_flag`). These are
    measurements to read, not a pass/fail certification of privacy or quality.
    Raises ValueError if the tables do not have the same columns.
    """
    others = [synthetic] + ([holdout] if holdout is not None else [])
    schema = _schema(real, others, categorical)
    cat = list(schema.categorical)
    d = dcr(real, synthetic, holdout, categorical=cat)
    nn = nndr(real, synthetic, holdout, categorical=cat)
    dup = exact_duplicate_rate(real, synthetic, categorical=cat)
    ks = column_ks(real, synthetic, categorical=cat)
    tvd = column_tvd(real, synthetic, categorical=cat)
    corr = correlation_difference(real, synthetic, categorical=cat)
    mi = mutual_information_difference(real, synthetic, categorical=cat, bins=mi_bins)
    clf = c2st(real, synthetic, categorical=cat, classifier=classifier, seed=seed)

    ks_vals = [v["statistic"] for v in ks.values() if v["statistic"] is not None]
    tvd_vals = [v for v in tvd.values() if v is not None]
    share = d["share_below_reference_q"]
    auc = clf["auc"]
    return {
        "kind": "synthetic_data_measurements",
        "disclaimer": (
            "Measurements comparing a synthetic/anonymized table to its source. "
            "They detect near-copies and distribution mismatch; they are not a "
            "privacy guarantee, a certification, or evidence of regulatory compliance."
        ),
        "seed": seed,
        "n_rows": {
            "real": len(real),
            "synthetic": len(synthetic),
            "holdout": None if holdout is None else len(holdout),
        },
        "columns": {"numeric": list(schema.numeric), "categorical": cat},
        "nan_policy": (
            "KS/TVD drop missing cells per column; all other metrics drop rows "
            "with any missing cell; nothing is imputed"
        ),
        "summary": {
            "exact_duplicate_rate": dup["rate"],
            "dcr_share_below_reference_p05": share,
            "dcr_reference_kind": d["reference_kind"],
            "near_copy_signal": bool(share is not None and share > memorization_share),
            "near_copy_threshold": memorization_share,
            "max_ks": _f(max(ks_vals)) if ks_vals else None,
            "max_tvd": _f(max(tvd_vals)) if tvd_vals else None,
            "pearson_mean_abs_diff": corr["pearson"]["mean_abs"],
            "c2st_auc": auc,
            "distinguishable_by_c2st": bool(auc is not None and auc >= c2st_auc_flag),
            "c2st_auc_threshold": c2st_auc_flag,
        },
        "privacy": {"dcr": d, "nndr": nn, "exact_duplicates": dup},
        "fidelity": {
            "ks": ks,
            "tvd": tvd,
            "correlation": corr,
            "mutual_information": mi,
            "c2st": clf,
        },
    }


__all__ = [
    "c2st",
    "column_ks",
    "column_tvd",
    "correlation_difference",
    "dcr",
    "exact_duplicate_rate",
    "mutual_information_difference",
    "nndr",
    "synth_report",
]
