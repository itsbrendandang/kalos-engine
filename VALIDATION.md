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
