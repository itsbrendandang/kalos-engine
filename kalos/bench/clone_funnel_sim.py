"""Seeded clone-selection funnel with a KNOWN latent productivity.

STRUCTURE ONLY. This generator has the shape of a clone-selection funnel
dataset (504 clones x 49 per-clone features, clones grouped by parental
strain, a multi-stage screen where each stage gates which clones advance)
as described in `/tmp/redesign/research/bioqore-mine.md`. Every VALUE comes
from the latent function below; nothing was read from, fitted on, or
calibrated to any real funnel data. Column names are generic.

The latent model:

    productivity(clone) = strain_effect[strain] + 3 * exp(-|| (x_a - c) / w ||^2 / 2)

where `x_a` are 4 of the 49 features (the informative ones). Another 6
features are a strain FINGERPRINT (strain mean plus small noise): they carry
strain identity, so a model can memorise strains. That is the leakage trap a
grouped split by strain must catch: random-split CV reads optimistic, CV
grouped by `strain` does not. The remaining features are pure noise.

Stages: stage 1 measures every clone (noise 0.6), the top half advances;
stage 2 measures those (noise 0.35), the top quarter of the original funnel
advances; stage 3 measures the finalists (noise 0.15). A clone's measurement
at a stage it never reached is NaN, never imputed.

A benchmark fixture. Not evidence about any real cell line.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_CLONES = 504
N_FEATURES = 49
N_STRAINS = 12
ACTIVE = (0, 1, 2, 3)
FINGERPRINT = tuple(range(4, 10))
STAGE_NOISE = (0.6, 0.35, 0.15)
STAGE_KEEP = (0.5, 0.25)  # share of ALL clones that advance past stage 1 and stage 2

FEATURE_COLUMNS = tuple(f"feature_{i + 1:02d}" for i in range(N_FEATURES))


@dataclass(frozen=True)
class Funnel:
    frame: pd.DataFrame
    productivity: np.ndarray  # noiseless latent, aligned to frame rows
    strain: np.ndarray

    @property
    def best_clone(self) -> int:
        return int(np.argmax(self.productivity))

    def features(self) -> np.ndarray:
        return self.frame[list(FEATURE_COLUMNS)].to_numpy(float)


def make_funnel(seed: int = 0, *, n_clones: int = N_CLONES, n_strains: int = N_STRAINS) -> Funnel:
    """Build one seeded funnel. Deterministic per `seed`."""
    rng = np.random.default_rng(seed)
    strain = np.sort(rng.integers(0, n_strains, size=n_clones))
    strain_effect = rng.normal(0.0, 0.8, size=n_strains)
    strain_center = rng.uniform(0.2, 0.8, size=(n_strains, len(FINGERPRINT)))

    X = rng.uniform(0.0, 1.0, size=(n_clones, N_FEATURES))
    X[:, list(FINGERPRINT)] = strain_center[strain] + rng.normal(0.0, 0.03, (n_clones, len(FINGERPRINT)))
    c = rng.uniform(0.35, 0.65, size=len(ACTIVE))
    w = 0.25
    xa = X[:, list(ACTIVE)]
    productivity = strain_effect[strain] + 3.0 * np.exp(-0.5 * (((xa - c) / w) ** 2).sum(axis=1))

    s1 = productivity + rng.normal(0.0, STAGE_NOISE[0], n_clones)
    adv1 = np.zeros(n_clones, bool)
    adv1[np.argsort(-s1)[: int(STAGE_KEEP[0] * n_clones)]] = True
    s2 = np.where(adv1, productivity + rng.normal(0.0, STAGE_NOISE[1], n_clones), np.nan)
    adv2 = np.zeros(n_clones, bool)
    order2 = np.argsort(-np.where(adv1, s2, -np.inf))
    adv2[order2[: int(STAGE_KEEP[1] * n_clones)]] = True
    s3 = np.where(adv2, productivity + rng.normal(0.0, STAGE_NOISE[2], n_clones), np.nan)

    df = pd.DataFrame(X, columns=list(FEATURE_COLUMNS))
    df.insert(0, "strain", [f"strain_{g:02d}" for g in strain])
    df.insert(0, "clone_id", [f"clone_{i:03d}" for i in range(n_clones)])
    df["stage1_readout"] = s1
    df["stage1_advanced"] = adv1
    df["stage2_readout"] = s2
    df["stage2_advanced"] = adv2
    df["stage3_readout"] = s3
    return Funnel(df, productivity, strain)


__all__ = ["FEATURE_COLUMNS", "N_CLONES", "N_FEATURES", "Funnel", "make_funnel"]
