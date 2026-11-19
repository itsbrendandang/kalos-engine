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

The per-proposal 90% and 95% ranges are calibrated by grouped cross-validation on
past runs. Proposals are not like past runs: they are chosen *because* the model
predicts them high, often where it is least sure. This section measures how often
the measurement of a **proposed** recipe lands inside the range shown next to it
(ML audit B1, REORG F02/F15).

**Protocol** (`python -m kalos.bench.coverage_at_proposals --seeds 40`): Latin-hypercube
design from a known simulator, the production analysis (`_analyze`), then each of the
5 proposals is measured once with fresh simulated noise and checked against the shipped
`proposals[].intervals` `[lo, hi]`. The same band is rebuilt at 300 random points in the
design box ("runs like past runs") for comparison. 40 seeds x 5 proposals = 200 measured
proposals per scenario; the 95% CI is a seed-cluster bootstrap (the 5 proposals of one
seed are correlated). Widths are a fraction of the training target's range. All
simulators are synthetic; Branin is a control where selection hardly matters.
Results files: `benchmarks/results/coverage_at_proposals_*.json`.

## Before (engine @ `0fc05c3`, normalized conformal band; `coverage_at_proposals_before.json`)

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
measurement for 60-81% of proposals, a 95% range for 76-86%. The engine says so: every
proposal interval carries `coverage_basis: "held_out_past_runs"` and a `caveat` with
these measured numbers (`docs/CAMPAIGNS_API.md`, "Coverage basis and the proposal
caveat").

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
