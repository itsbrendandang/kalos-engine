"""ML benchmark v2: the machine-readable half of the pre-registered protocol.

`benchmarks/ml_v2/PROTOCOL.md` is the normative text. This module holds every
number that text fixes (seeds, tiers, budgets, levels, thresholds), so the
harness cannot drift from it. Nothing here may change after the protocol commit
except through a dated amendment in PROTOCOL.md that names the constant.
"""
from __future__ import annotations

PROTOCOL_VERSION = "ml_v2.0"

# --------------------------------------------------------------------- seeds
# Seeds are per-instance draws: one seed = one randomized problem instance (and
# its initial design, noise stream and regime). Tuning and held-out ranges never
# overlap. Candidates may look at tuning seeds as often as they like. Held-out
# seeds are run once per FROZEN arm (benchmarks/ml_v2/freeze/<arm>.json) and
# every held-out run is appended to the ledger (see `harness.guard_heldout`).
PILOT_SEEDS: tuple[int, ...] = (9001,)  # timing only; never reported
TUNING_SEEDS: dict[str, tuple[int, ...]] = {
    "closed_loop": tuple(range(0, 12)),
    "pool": tuple(range(0, 6)),
    "emulator": tuple(range(0, 4)),
    "snapshot": tuple(range(0, 12)),
    "snapshot_real": tuple(range(0, 6)),
}
HELDOUT_SEEDS: dict[str, tuple[int, ...]] = {
    "closed_loop": tuple(range(1000, 1030)),
    "pool": tuple(range(1000, 1016)),
    "emulator": tuple(range(1000, 1010)),
    "snapshot": tuple(range(1000, 1040)),
    "snapshot_real": tuple(range(1000, 1016)),
}
SMOKE_SEEDS: dict[str, tuple[int, ...]] = {k: (9002,) for k in TUNING_SEEDS}
TUNING_SEEDS["internal"] = (0, 1, 2)
HELDOUT_SEEDS["internal"] = (1000,)
SMOKE_SEEDS["internal"] = (9002,)
for _s in ("backtest", "scale"):
    TUNING_SEEDS[_s] = (0, 1, 2, 3, 4, 5) if _s == "backtest" else (0,)
    HELDOUT_SEEDS[_s] = tuple(range(1000, 1030)) if _s == "backtest" else (1000,)
    SMOKE_SEEDS[_s] = (9002,)

# CI / scoreboard: a disjoint seed set and a fast task subset (closed loop only,
# n0 = 12 because seed % 3 == 0). Never used for decisions; a drift signal only.
CI_SEEDS: dict[str, tuple[int, ...]] = {"closed_loop": (5001, 5004)}
CI_TASKS: tuple[str, ...] = ("heteroscedastic", "bumps")

# Dataset-level split for real data (PROTOCOL.md section 4). Seeds protect
# against tuning to a split, not against tuning to a dataset, so the real and
# out-of-domain datasets are divided: TUNING datasets may be run on tuning seeds
# at will; HELD-OUT-ONLY datasets are never run before an arm is frozen.
# Fixtures (synthetic sheets) and the information-only Kaggle sheet are in both.
TUNING_DATASETS: tuple[str, ...] = (
    "amii/DBO_rat_myocyte", "amii/HeLa_timesaving", "amii/Tcell_expansion", "olympus/hplc",
    "fixture/synthetic_bioprocess", "fixture/synthetic_scaleup", "kaggle/fermentation",
)
HELDOUT_DATASETS: tuple[str, ...] = (
    "amii/MOBO_rat_myocyte", "amii/HeLa_regular", "amii/TF1_expansion", "zenodo/refold_scfv", "olympus/lnp3",
    "fixture/synthetic_bioprocess", "fixture/synthetic_scaleup", "kaggle/fermentation",
)

# --------------------------------------------------------------- the loop
Q = 5  # proposals per round (the engine's fixed batch size)
ROUNDS = 5  # closed loop / pool / emulator: rounds after the initial design
# Low-data regimes. Closed-loop, pool and emulator instances start from n0 in
# N0_REGIMES (assigned by seed: seed % 3). Snapshot instances use n in
# SNAPSHOT_SIZES (assigned by seed: seed % 4).
N0_REGIMES: tuple[int, ...] = (12, 20, 40)
SNAPSHOT_SIZES: tuple[int, ...] = (12, 20, 40, 80)
LEVELS: tuple[float, ...] = (0.90, 0.95)
N_HELDOUT_POINTS = 200  # fresh in-box points (synthetic/emulator) or unrevealed rows (pool)

# Experiments-to-target: normalized-regret thresholds (synthetic / emulator) and
# the pool's top fraction (pool replay).
REGRET_TARGETS: tuple[float, ...] = (0.25, 0.10)
POOL_TOP_FRACTION = 0.05

# --------------------------------------------------------------- statistics
BOOT_SEED = 20261120
N_BOOT = 4000
ALPHA = 0.05

# --------------------------------------------------------------- criteria
# A candidate ships only if every one of these holds on the HELD-OUT seeds
# (PROTOCOL.md section 8). The numbers are fixed here.
CRITERIA = {
    # C1 (improvement, at least one of C1a / C1b):
    #   C1a regret: paired mean difference (candidate - reference) of normalized
    #       simple regret at the final round, pooled over the synthetic closed-loop
    #       instances, has a 95% CI entirely below 0; same sign required on the
    #       pooled real pool-replay group (point estimate <= 0).
    #   C1b width: ratio of median interval width at proposals (candidate /
    #       reference, 90% level, synthetic closed loop) has a 95% CI entirely below 1.
    "width_ratio_max": 1.0,
    # C2 coverage at proposals, both levels, synthetic closed loop + emulator:
    #   pooled point estimate >= nominal, and no family/dataset whose
    #   seed-cluster 95% CI upper bound is below nominal.
    "coverage_tolerance": 0.0,
    # C3 held-out runs coverage (guard): pooled >= nominal - this.
    "heldout_coverage_slack": 0.02,
    # C4 false trust: pooled over null snapshot scenarios, candidate rate <=
    #   reference rate; per null scenario, candidate <= reference + this.
    "false_trust_slack_per_scenario": 0.05,
    # C5 planted faults: `kalos.bench.faults.score.regressions(reference, candidate)` empty.
    # C6 no hidden loss: no family or dataset (synthetic, pool, emulator) where the
    #   candidate is significantly worse on regret (paired, Holm-corrected over
    #   all families and datasets at ALPHA) AND worse by more than this.
    "loss_margin_regret": 0.05,
    # C7 non-inferiority on the other axis: shipping on regret, the width ratio CI
    #   must not lie entirely above this; shipping on width, the regret difference
    #   CI must not lie entirely above 0.
    "width_ratio_noninferior": 1.10,
    # Wall time is reported; a median per-analysis time above this multiple of the
    # reference needs the user's sign-off (not an automatic block).
    "walltime_signoff_ratio": 2.0,
}

# Groups that enter the criteria. `info_only` (the Kaggle sheet) and `user`
# (a user-supplied folder) are reported but never decide anything.
DECISION_GROUPS = ("synthetic", "real_pool", "real_emulator", "ood_pool", "ood_emulator", "fixture_pool")
REPORT_ONLY_GROUPS = ("info_only", "user")
