# Validation on real data

> **The honest bottom line.** On real, published cell-culture media datasets,
> Kalos's Bayesian optimization reaches the top formulations in **far fewer
> experiments than space-filling search on the one large, hard dataset** (TF-1
> cell expansion: ~14 vs ~62 experiments, a statistically significant 15/20-seed
> win). On the smaller or smoother datasets it ties — and on one it loses to
> space-filling. This is a conditional, measured result, not a blanket claim that
> Kalos beats everything everywhere.

This is the first validation of Kalos against **real measured laboratory data**
(not synthetic surfaces). It is a *retrospective pool replay*, not a live wet-lab
trial — see "What this is and isn't" below.

## The datasets

Six published, open-access cell-culture media-optimization datasets, bundled by
the Amii "Benchmarking active learning for cell-culture media" project
([github.com/Amii-Applied-AI/amii-cell-ag-tools](https://github.com/Amii-Applied-AI/amii-cell-ag-tools)).
They are **real measured experiments** from three peer-reviewed papers (cite the
originals if you reuse them):

| Dataset | n rows | features | readout | source |
|---|---|---|---|---|
| DBO_rat_myocyte | 209 | 15 | rat myoblast growth | Cosenza et al., Eng. Life Sci. 2023 |
| MOBO_rat_myocyte | 182 | 26 | rat myoblast growth (multi-fidelity) | Cosenza et al., Eng. Life Sci. 2023 |
| HeLa_regular | 308 | 29 | HeLa-S3 A450 | Hashizume et al., npj Syst. Biol. Appl. 2023 |
| HeLa_timesaving | 403 | 29 | HeLa-S3 A450 | Hashizume et al., npj Syst. Biol. Appl. 2023 |
| Tcell_expansion | 2124 | 14 | primary T-cell count | Kim & Audet, Commun. Biol. 2019 |
| TF1_expansion | 1562 | 15 | TF-1 fold-change | Kim & Audet, Commun. Biol. 2019 |

The datasets are **not** redistributed in this repo (their licences keep them
under the original papers). Point the benchmark at a local copy with
`KALOS_AMII_DATA`; see `examples/datasets/README.md`.

## The protocol

Pool replay: each dataset's rows are a fixed pool. Each round the optimizer
proposes `q` recipes in the normalized design box; each is snapped to the nearest
not-yet-picked recorded row and "measured" by revealing that row's recorded
outcome. We measure **how many experiments it takes to reach the pool's top 5%**
(lower is better), over 20 seeds, at batch sizes q=1 and q=4. The paired per-seed
BO-vs-baseline difference is tested with a sign test and a Holm correction across
the comparisons. Baselines: uniform random and Latin-hypercube (LHS), both from
the same seeded initial design. `python examples/amii_benchmark.py --seeds 20`.

## Results (20 seeds)

Median experiments to the top 5% (lower is better):

| Dataset (n, d) | Kalos BO | Random | LHS | Verdict |
|---|---|---|---|---|
| **TF1_expansion** (1562, 15) | **14** | 20 | 62 | **BO wins vs LHS 15/20, p=0.001, Holm 0.002** (vs random p=0.035) |
| Tcell_expansion (2124, 14) | 15–16 | 16 | 18 | slight edge, not significant |
| DBO_rat_myocyte (209, 15) | 14–16 | 14 | 16 | tie |
| MOBO_rat_myocyte (182, 26) | 12 | 13 | 14 | slight edge, not significant |
| HeLa_timesaving (403, 29) | 8 | 8 | 8 | tie (an easy pool) |
| **HeLa_regular** (308, 29) | 26–27 | 20 | **14** | **BO loses to LHS** |

## What this shows — honestly

- **A real, significant win where BO should win.** TF-1 is the largest, highest-
  variance pool. There Kalos reaches the top 5% in ~14 experiments versus LHS's
  ~62 — a ~4× reduction, winning 15 of 20 seeds, significant after correction
  (Holm p = 0.002). This is the defensible claim.
- **Ties on small / smooth pools.** On the rat-myocyte and T-cell pools BO is
  level with random/LHS: with a few hundred points and a smooth response, a
  space-filling design is hard to beat, and BO's model advantage has little room.
- **One honest loss.** On HeLa_regular BO is *worse* than LHS. That pool is small
  (308), was itself collected by an active-learning campaign (rows cluster near
  good regions), so uniform sampling lands on winners by chance while BO spends a
  fit optimizing a surrogate that has not yet earned trust. The engine now says so
  in the response: when the model does not clear its cross-validation trust floor,
  `reliability.space_filling_advice` tells the user the batch is exploration and a
  space-filling design does about as well (see CHANGELOG / `kalos/portal/analysis.py`).

## What this is and isn't

- **It is** retrospective replay on real measured data, with proper baselines,
  20 seeds, and multiple-comparison correction.
- **It is not** a live wet-lab trial. Snapping proposals to the nearest recorded
  row is a stand-in for running the assay; it cannot measure recipes that were
  never in the original study.
- **It does not** claim Kalos improves outcomes in a lab, or beats every method
  on every dataset. The win is conditional on the problem being large and hard
  enough for a model to help.

## Reproduce

```bash
export KALOS_AMII_DATA=/path/to/the/six/CSVs   # not shipped; see examples/datasets/README.md
python examples/amii_benchmark.py --seeds 20   # ~30-40 min; writes a JSON report
```

---

# Interval coverage at proposals

The per-proposal 90% and 95% ranges are sized from grouped cross-validation on past
runs. Proposals are not like past runs: they are chosen *because* the model
predicts them high, often where it is least sure. This section measures how often
the measurement of a **proposed** recipe lands inside the range shown next to it
(ML audit B1, REORG F02/F15).

**Protocol** (`python -m kalos.bench.coverage_at_proposals --seeds 40 --seed-offset 100`): Latin-hypercube
design from a known simulator, the production analysis (`_analyze`), then each of the
5 proposals is measured once with fresh simulated noise and checked against the shipped
`proposals[].intervals` `[lo, hi]`. The plain normalized band (`pred +/- q_norm * sd`, the
fallback) is rebuilt at the same proposals, and the shipped band at 300 random points in
the design box ("runs like past runs"), for comparison. 40 seeds x 5 proposals = 200
measured proposals per scenario; the 95% CI is a seed-cluster bootstrap (the 5 proposals
of one seed are correlated). Widths are a fraction of the training target's range. All
simulators are synthetic; Branin is a control where selection hardly matters.
Each results file records its commit and environment in `meta.environment`.

## Before F15 (engine @ `0fc05c3`, normalized conformal band; `coverage_at_proposals_before.json`)

Historical: recorded before F15, seeds 0-39; the file records no environment. Not rerun on `main` (that code path no longer exists); the plain band at `main`'s proposals is in the tables below.

| Scenario | 90%: at proposals [95% CI] | 90%: random in-box | 95%: at proposals [95% CI] | 95%: random in-box | GP band 90% at proposals | `conformal_q` at proposals |
|---|---|---|---|---|---|---|
| Hartmann-6, n=30 | **0.60** [0.54, 0.67] | 0.90 | **0.83** [0.77, 0.89] | 0.98 | 0.50 | 0.68 |
| Hartmann-6, n=80 | **0.65** [0.58, 0.72] | 0.91 | **0.76** [0.69, 0.82] | 0.96 | 0.57 | 0.77 |
| Bioprocess, n=40 | **0.74** [0.67, 0.81] | 0.91 | **0.83** [0.77, 0.89] | 0.96 | 0.64 | 0.79 |
| Bioprocess, n=100 | **0.81** [0.77, 0.86] | 0.90 | **0.86** [0.82, 0.90] | 0.95 | 0.73 | 0.83 |
| Heteroscedastic 3-D, n=40 | **0.69** [0.63, 0.74] | 0.91 | **0.81** [0.75, 0.87] | 0.96 | 0.53 | 0.77 |
| Branin, n=20 (control) | 0.94 [0.87, 0.98] | 0.91 | 0.96 [0.91, 0.99] | 0.96 | 0.80 | 0.97 |

The calibration step works where it was calibrated (0.90-0.91 and 0.95-0.98 on random
in-box points). At the proposals it does not: a nominal 90% range contained the
measurement for 60-81% of proposals, a 95% range for 76-86%. That measurement led to F02
(every proposal interval carries `coverage_basis` and a `caveat` with measured
numbers) and F15 (below).

## Selection-aware bands (F15) on `main`, both locked stacks (seeds 100-139)

The proposal bands are sized on the held-out runs the model ranked highest in each
fold (top 30%, pooled over both CV repeats), and the proposal's sd is floored at the
90th percentile of the held-out sds (`calibrate_selection_aware`,
`kalos/portal/analysis_parts/intervals.py`). The two parameters were chosen on seeds
0-39 (`coverage_at_proposals_after_selection_seeds.json`, no environment recorded);
the numbers below are on **fresh seeds 100-139**.

`uv.lock` pins two numerical stacks: Python < 3.12 gets numpy 2.4.6 and scipy 1.17.1
(CI runs this one), Python >= 3.12 gets numpy 2.5.1 and scipy 1.18.0 (the engine image,
`python:3.12`, runs this one). The benchmark ran on both, on `main` @ `cf7105c` (bench
commit `84f044a`: `cf7105c` plus results metadata), on macOS arm64 at 1 torch thread per
worker:

- `coverage_at_proposals_main_py311_seeds100.json`: Python 3.11.17, numpy 2.4.6, scipy 1.17.1.
- `coverage_at_proposals_main_py312_seeds100.json`: Python 3.12.15, numpy 2.5.1, scipy 1.18.0.
  Row for row (240/240) the file first committed with F15, which was recorded on
  Python 3.14.7 on the same branch of the lock.

**90% nominal**

| Scenario | Python 3.11: at proposals [95% CI] | Python >= 3.12: at proposals [95% CI] | plain band, same proposals (3.11 / >=3.12) | random in-box (3.11 / >=3.12) | median width, x y range (3.11 / >=3.12) | width vs plain band | GP band (3.11 / >=3.12) | `conformal_q` (3.11 / >=3.12) |
|---|---|---|---|---|---|---|---|---|
| Hartmann-6, n=30 | **0.91** [0.855, 0.955] | **0.935** [0.895, 0.97] | 0.58 / 0.64 | 0.98 / 0.98 | 1.76 / 1.76 | x2.6 / x2.5 | 0.42 / 0.485 | 0.685 / 0.735 |
| Hartmann-6, n=80 | **0.94** [0.89, 0.98] | **0.94** [0.89, 0.975] | 0.545 / 0.595 | 0.99 / 0.99 | 0.98 / 0.98 | x2.7 / x2.6 | 0.5 / 0.56 | 0.7 / 0.71 |
| Bioprocess, n=40 | **0.86** [0.79, 0.92] | **0.92** [0.87, 0.965] | 0.58 / 0.66 | 0.99 / 0.99 | 1.82 / 1.82 | x2.8 / x2.5 | 0.415 / 0.53 | 0.66 / 0.77 |
| Bioprocess, n=100 | **0.95** [0.905, 0.985] | **0.99** [0.975, 1] | 0.705 / 0.835 | 0.99 / 0.99 | 1.30 / 1.31 | x3.7 / x2.9 | 0.655 / 0.795 | 0.77 / 0.87 |
| Heteroscedastic 3-D, n=40 | **0.965** [0.94, 0.985] | **0.96** [0.93, 0.985] | 0.65 / 0.65 | 0.99 / 0.99 | 1.71 / 1.71 | x3.1 / x3.2 | 0.51 / 0.51 | 0.685 / 0.71 |
| Branin, n=20 (control) | 0.975 [0.955, 0.995] | 0.98 [0.96, 0.995] | 0.92 / 0.915 | 0.98 / 0.98 | 0.84 / 0.84 | x1.2 / x1.4 | 0.795 / 0.765 | 0.97 / 0.965 |

**95% nominal**

| Scenario | Python 3.11: at proposals [95% CI] | Python >= 3.12: at proposals [95% CI] | plain band, same proposals (3.11 / >=3.12) | random in-box (3.11 / >=3.12) | median width, x y range (3.11 / >=3.12) | width vs plain band | GP band (3.11 / >=3.12) |
|---|---|---|---|---|---|---|---|
| Hartmann-6, n=30 | **0.965** [0.93, 0.995] | **0.97** [0.95, 0.99] | 0.76 / 0.8 | 0.99 / 0.99 | 2.40 / 2.40 | x2.1 / x1.9 | 0.47 / 0.52 |
| Hartmann-6, n=80 | **0.96** [0.925, 0.99] | **0.97** [0.94, 0.995] | 0.68 / 0.72 | 1.00 / 1.00 | 1.54 / 1.54 | x3.1 / x2.9 | 0.55 / 0.58 |
| Bioprocess, n=40 | **0.935** [0.875, 0.98] | **0.965** [0.92, 1] | 0.645 / 0.765 | 1.00 / 1.00 | 2.63 / 2.64 | x2.7 / x2.5 | 0.45 / 0.565 |
| Bioprocess, n=100 | **0.97** [0.94, 0.99] | **0.995** [0.985, 1] | 0.83 / 0.915 | 1.00 / 1.00 | 1.81 / 1.82 | x2.9 / x2.3 | 0.705 / 0.815 |
| Heteroscedastic 3-D, n=40 | **0.99** [0.975, 1] | **0.995** [0.985, 1] | 0.75 / 0.765 | 1.00 / 1.00 | 2.57 / 2.57 | x3.1 / x3.2 | 0.56 / 0.565 |
| Branin, n=20 (control) | 0.99 [0.975, 1] | 0.995 [0.985, 1] | 0.95 / 0.95 | 0.99 / 0.99 | 0.97 / 0.97 | x1.2 / x1.3 | 0.83 / 0.815 |

Width vs plain band: the median width of the selection-aware band over the median width
of the plain band at the same proposals.

What this shows, and what it does not:

- **Nominal is not reached everywhere.** At 90% the band covered 0.86-0.965 on
  Python 3.11 and 0.92-0.99 on Python 3.12; at 95%, 0.935-0.99 and 0.965-0.995.
  Bioprocess n=40 at 90% on Python 3.11 is 0.86 [0.79, 0.92]. The shipped caveat spans
  both stacks and says so: "covered 86-99% in simulation, below 90% in some cases"
  (`coverage_lower_at_proposals`); at 95%, "93-99%, below 95% in some cases".
- **The library stack moves the number.** Same code, same seeds: bioprocess n=40 at 90%
  is 0.86 on one locked stack and 0.92 on the other; bioprocess n=100 0.95 vs 0.99.
  Bisected: the F15 commit `0bbf5ae` and `main` give identical rows on each stack;
  swapping only scipy 1.17.1 -> 1.18.0 (Python 3.14, numpy 2.4.6) turns one stack's rows
  into the other's. The surrogate fit is the same (identical coverage on random in-box
  points); the acquisition optimizer lands on different proposals. Read the per-scenario
  numbers with that run-to-run spread in mind, on top of the seed CI.
- **The price is width.** At the same proposals the bands are 2.5-3.7x (90%) and
  1.9-3.2x (95%) wider than the plain band (median), often wider than the whole range of
  the training target when the model is weak (Hartmann-6 n=30 has a held-out Spearman
  near 0.3). They are honest, not informative: a model that cannot rank its own held-out
  runs cannot say where a proposal will land.
- **They over-cover runs like past runs.** 0.98-1.00 on random in-box points: the
  90th-percentile sd floor is conservative there. A less conservative floor (75th
  percentile) measured 0.87 on bioprocess n=40 on the selection seeds; the 90th was kept.
- **No guarantee.** The pooled CV repeats re-use each run once per repeat and the
  top-ranked held-out runs are only a proxy for proposals. The claim is the measured
  coverage on these synthetic simulators, nothing more; CRUX_PORT_PLAN Rank 2 (F12)
  measures it prospectively on real campaigns.
- **Fallback.** A level with too few top-ranked held-out runs (90% needs 9, 95% needs
  19; e.g. a sheet whose binding CV grouping is 5 batches) falls back to the plain
  normalized band, which keeps its own caveat (at `main`'s proposals it covered
  0.545-0.835 at 90% and 0.645-0.915 at 95% over both stacks) and a
  `selection_aware_unavailable` reason.
- Cost: no extra model fits (arithmetic on the CV predictions already computed).
- The served numbers are generated from the two results files
  (`python -m kalos.bench.coverage_at_proposals --print-evidence FILE`);
  `tests/test_interval_labels.py` fails when the caveat and the files disagree.

---

# Trust verdict under leaks (false-trust rates)

Every scenario below has **no learnable input signal**, so any "trusted" verdict is
false trust (ML audit c1, c2, f1). `python -m kalos.bench.trust_leaks --seeds 20`;
results in `benchmarks/results/trust_leaks_*.json`. Batch scenarios: 6 batches x 8
runs clustered in space (as campaign rounds are), batch offset sd 1.0, noise sd 0.2.
Jitter: 15 recipes x 4 replicates recorded as run, recipe effect sd 1.0.

| Scenario (20 seeds) | `clears_floor` before | after F05 (leak-free CV) | after F04 (gates) | verdict `reliable` after F04 | proposals withheld after F04 | batch proposed as a setpoint before -> after |
|---|---|---|---|---|---|---|
| text `batch` column | 1.00 | 0.10 | 0.05 | 0.00 | 0.15 | 1.00 -> 0.00 |
| numeric `batch` column | 1.00 | 0.10 | 0.05 | 0.00 | 0.15 | 1.00 -> 0.00 |
| campaign rounds (fold history) | 0.95 | 0.10 | 0.05 | 0.00 | 0.15 | - |
| no batch column (structure invisible) | 0.95 | 0.95 | 0.80 | 0.80 | 0.00 | - |
| as-run replicates, 0.2% jitter | 1.00 | 0.15 | 0.00 | 0.00 | 0.00 | - |
| as-run replicates, 1% jitter | 1.00 | 0.15 | 0.00 | 0.00 | 0.10 | - |
| pure noise, n=12 | 0.30 | 0.30 | 0.05 | 0.00 | 0.10 | - |
| pure noise, n=20 | 0.10 | 0.10 | 0.00 | 0.00 | 0.10 | - |
| pure noise, n=40 | 0.05 | 0.10 | 0.05 | 0.05 | 0.10 | - |

F04 makes the verdict act: `refused` withholds the batch, `not_yet` labels every
proposal `exploration_untrusted`. The price is power: the audit measured that
requiring the interval above 0 drops `clears_floor` on a real 6-D bump at n=20
from 70% to about 30% (f2); small, learnable sheets now read `not_yet` more often
and their proposals are labelled exploration, not withheld.

When the batch structure is not on the sheet at all, no grouping can see it: record
the batch, plate, lot or date column, or run the campaign through the fold (rounds are
carried automatically).

## Planted-fault acceptance (feat/planted-faults suite, full tier)

Run against this branch on a scratch merge (`scratch/trust-faults`, not for merge):
`python -m kalos.bench.faults all --tier full --workers 4 --datasets ~/kalos/datasets`.
Totals: 20 caught / 4 noticed / 12 missed of 36 at the baseline -> **23 / 3 / 10**;
false alarms on clean sheets unchanged (6 of 8; the outlier check on the clean sheet
was left alone).

| Case | Before (baseline scorecard) | After (this branch) |
|---|---|---|
| batch offsets, batch column, no signal | missed: CV 0.93 [0.87, 0.96] trusted, `batch: "B4"` proposed | **caught**: CV 0.01 [-0.71, 0.70] by batch, not trusted, batch not proposed |
| batch offsets, no batch column | missed: CV 0.75 [0.49, 0.88] trusted | **still missed**: the structure is not on the sheet (rows shuffled, no round index) |
| jittered replicates (0.2%), no signal | missed: CV 0.78 [0.61, 0.97] trusted | **caught**: CV -0.40 [-0.74, 0.37] over 15 replicate groups |
| 8 rows, 11 inputs | noticed: verdict issued, 5 proposals | **caught**: `reliability.verdict: not_judged`; refused (interval at or below 0), 0 proposals |
| clean Kaggle slice (150 rows) | trusted with CV 0.27 [-0.06, 0.38] | trusted with CV 0.22 [0.04, 0.36]: one near-duplicate pair now shares a fold, the interval moved above 0, so the stricter F04 rule passes; a weak, borderline ranking |
| clean scale-up sheet (90 rows) | trusted, CV 0.77 | trusted, CV 0.46 [0.17, 0.69]: leave-one-scale-out (9 scales) binds |
