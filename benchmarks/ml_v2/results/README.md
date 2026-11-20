# ML benchmark v2: reference-arm results (TUNING split only)

`kalos_ref` = the production `_analyze` path at commit `2338a63` (engine identical to
`main` @ `4bb477d`; the branch adds only bench files), torch 2.13.0 / botorch 0.18.1 /
gpytorch 1.15.2, KALOS_TORCH_THREADS=1. Tuning seeds and tuning datasets only
(PROTOCOL.md section 4); 438 jobs, 7,118 s wall with 10 workers on a shared Mac Studio
(load average 230-350 from other agents: wall times are inflated about 5x over CPU
time; both are recorded). The `constrained` jobs were re-run under amendment A2.
Random and LHS baselines ran on the same instances (`baselines_tuning/`).

Files: `reference_tuning/summary.md` (every task, regime and group),
`summary.json`, raw per-job JSON for the synthetic suites (real-data raw records are not
committed, `.gitignore`). These are TUNING numbers: they are what candidates tune
against, not a held-out claim. Simulations and retrospective replays only.

## Regret: Kalos vs random and LHS (paired by instance, 95% bootstrap CI)

Normalized simple regret after 5 rounds of 5, judged on the truth at the best
MEASURED recipe (0 = optimum, 1 = a typical recipe). Negative difference = Kalos better.
Runs-to-target = median runs to regret <= 0.25 (pool: the pool's top 5 %), censored at
budget + 1, so medians near the budget mean "mostly not reached".

| suite | task | n | Kalos regret | random | LHS | Kalos - random [95% CI] | Kalos - LHS [95% CI] | runs-to-target K/R/L (median) |
|---|---|---|---|---|---|---|---|---|
| closed_loop | batch | 12 | 0.338 | 0.359 | 0.471 | -0.020 [-0.1495, 0.1371] | -0.132 [-0.248, -0.0047] | 39/40/42 |
| closed_loop | bioprocess | 12 | 0.363 | 0.497 | 0.501 | -0.134 [-0.2074, -0.0645] | -0.139 [-0.2503, -0.0321] | 46/46/46 |
| closed_loop | bumps | 12 | 0.494 | 0.604 | 0.564 | -0.110 [-0.2065, -0.0224] | -0.070 [-0.1727, 0.0233] | 43/43/43 |
| closed_loop | clones | 12 | 0.205 | 0.424 | 0.373 | -0.219 [-0.375, -0.0614] | -0.168 [-0.3344, 0.0215] | 39/39/43 |
| closed_loop | constrained | 12 | 0.354 | 0.429 | 0.446 | -0.075 [-0.2494, 0.1057] | -0.092 [-0.2587, 0.0869] | 40/46/46 |
| closed_loop | hartmann6 | 12 | 0.330 | 0.563 | 0.555 | -0.233 [-0.3726, -0.0957] | -0.225 [-0.3622, -0.0687] | 43/46/46 |
| closed_loop | heteroscedastic | 12 | 0.135 | 0.330 | 0.311 | -0.195 [-0.3114, -0.0886] | -0.177 [-0.2606, -0.0927] | 39/43/43 |
| closed_loop | mixed | 12 | 0.155 | 0.345 | 0.370 | -0.189 [-0.2633, -0.1292] | -0.215 [-0.3032, -0.1272] | 38/46/46 |
| emulator | gp:olympus/hplc | 4 | 0.177 | 0.283 | 0.129 | -0.106 [-0.2576, 0.1296] | +0.048 [-0.1298, 0.1996] | 39/38/36 |
| emulator | rf:olympus/hplc | 4 | 0.262 | 0.233 | 0.231 | +0.029 [-0.0171, 0.0978] | +0.031 [0.0053, 0.0658] | 24/29/18 |
| emulator | gp:amii/DBO_rat_myocyte | 4 | 0.347 | 0.416 | 0.374 | -0.069 [-0.1297, -0.0089] | -0.028 [-0.0812, 0.0191] | 42/42/42 |
| emulator | gp:amii/HeLa_timesaving | 4 | 1.000 | 1.000 | 1.000 | +0.000 [0.0, 0.0] | +0.000 [0.0, 0.0] | 42/42/42 |
| emulator | gp:amii/Tcell_expansion | 4 | 0.463 | 0.750 | 0.667 | -0.287 [-0.3583, -0.2363] | -0.204 [-0.3059, -0.1016] | 42/42/42 |
| emulator | rf:amii/DBO_rat_myocyte | 4 | 0.419 | 0.778 | 0.722 | -0.359 [-0.4357, -0.2792] | -0.303 [-0.4524, -0.1677] | 38/42/42 |
| emulator | rf:amii/HeLa_timesaving | 4 | 0.566 | 0.623 | 0.669 | -0.057 [-0.1965, 0.0309] | -0.103 [-0.2509, 0.0448] | 42/42/42 |
| emulator | rf:amii/Tcell_expansion | 4 | 0.788 | 0.739 | 0.796 | +0.049 [-0.0449, 0.1603] | -0.007 [-0.331, 0.3025] | 42/42/42 |
| pool | fixture/synthetic_bioprocess | 6 | 0.146 | 0.207 | 0.162 | -0.061 [-0.255, 0.172] | -0.017 [-0.254, 0.1697] | 14/14/14 |
| pool | fixture/synthetic_scaleup | 6 | 0.013 | 0.012 | 0.010 | +0.001 [-0.0243, 0.0327] | +0.003 [0.0, 0.0083] | 24/19/26 |
| pool | kaggle/fermentation | 6 | 0.350 | 0.244 | 0.273 | +0.106 [-0.1149, 0.2935] | +0.077 [-0.1495, 0.2856] | 36/30/24 |
| pool | olympus/hplc | 6 | 0.128 | 0.204 | 0.177 | -0.076 [-0.1885, 0.0149] | -0.049 [-0.1943, 0.1285] | 18/20/24 |
| pool | amii/DBO_rat_myocyte | 6 | 0.134 | 0.234 | 0.521 | -0.100 [-0.4069, 0.1868] | -0.387 [-0.6455, -0.1051] | 20/24/28 |
| pool | amii/HeLa_timesaving | 6 | 0.230 | 0.347 | 0.059 | -0.116 [-0.1885, -0.0441] | +0.172 [-0.0321, 0.3591] | 12/12/12 |
| pool | amii/Tcell_expansion | 6 | 0.170 | 0.162 | 0.265 | +0.008 [-0.0992, 0.1098] | -0.095 [-0.2049, -0.0198] | 14/14/14 |

Pooled (Kalos - baseline): synthetic closed loop -0.147 [-0.194, -0.101] vs random and
-0.152 [-0.199, -0.105] vs LHS (96 instances); real emulators -0.121 [-0.190, -0.053] and
-0.107 [-0.187, -0.025] (24); real pool replay -0.070 [-0.185, +0.047] and
-0.103 [-0.264, +0.053] (18, not significant). Per-dataset losses are visible above:
Kaggle (information only) +0.106 vs random; the `gp:HeLa_timesaving` emulator is
degenerate (every arm sits at regret 1.0: the GP emulator is flat away from the rows in
29 dimensions, so f* at a data row is unreachable); `rf:olympus/hplc` LHS beats Kalos
(+0.031 [0.005, 0.066]).

## Intervals (coverage of the shipped 90/95 % bands)

| Group | n | coverage at proposals 90 % [CI] | 95 % [CI] | held-out runs 90 / 95 % | median width at proposals 90 % (x (f* - f_med)) |
|---|---|---|---|---|---|
| closed loop, synthetic | 96 | 0.957 [0.941, 0.971] | **0.948 [0.933, 0.961]** | 0.984 / 0.991 | 1.94 |
| emulator, real | 24 | 0.930 [0.904, 0.956] | **0.908 [0.881, 0.935]** | 0.972 / 0.991 | 1.98 |
| emulator, ood | 8 | 0.970 [0.930, 1.0] | 0.963 [0.929, 1.0] | 0.996 / 0.994 | 2.46 |
| pool replay, real | 18 | 0.987 [0.976, 0.996] | 0.985 [0.974, 0.995] | 0.984 / 0.974 | 3.19 |

The 90 % band is conservative everywhere; the **95 % band is at or below nominal at the
proposals in the closed loop (0.948) and below it on the real emulators (0.908)**:
under 19 top-ranked held-out runs the 95 % level falls back to the plain normalized band
(`interval_method` shows `conformal_normalized` at 0.95 while 0.90 is selection-aware),
and that fallback under-covers at proposals. At n = 12 the snapshot has no 95 % band at
all (unavailable). Bands are about 2x the (f* - f_med) range wide. Note C2 (coverage >=
nominal at proposals) is a bar the reference itself does not clear at 95 %.

## Trust verdict

- False trust (`proposal_trust.status == trusted` with no learnable signal), pooled over
  60 synthetic null snapshots and 36 permuted-target real snapshots (decision groups):
  0.083 [0.043, 0.156] (8/96). Seven are `null_batch_hidden` (7/12: 0.67 at n = 20/40,
  1.0 at n = 80): batch structure that is not on the sheet. With a batch column or round
  index, and for noise and jittered replicates: 0/12 each. Permuted real targets: 1/42
  (DBO, n = 20; Kaggle included).
- Power (trusted on signal snapshots): 0 at n = 12 by design (no `reliable` below 20
  runs); 0.17-0.67 at n = 20; 0.33-1.0 at n = 40-80; `mixed` stays at 0.33. Trusted in
  24 % of closed-loop rounds.

## Backtest (retrospective, pool-restricted; tuning datasets)

| Dataset | Order | Cuts | M1 | M1 - random [CI] | M1 - OFAT [CI] | M1 - lab | M3 |
|---|---|---|---|---|---|---|---|
| DBO_rat_myocyte | seeded orderings | 18 | 0.161 | -0.016 [-0.043, 0.010] tie | +0.080 [0.036, 0.121] **loss** | n/a | 0.84 |
| HeLa_timesaving | real (`Round`) | 2 | 0.410 | +0.094 [-0.31, 0.50] tie | +0.138 tie | +0.005 tie | -0.30 |
| Tcell_expansion | real (`gen #`) | 3 | 0.070 | -0.104 [-0.115, -0.097] win | -0.096 tie | -0.184 [-0.43, -0.06] win | 0.14 |

Anagram (internal, held-out only, by agent real-backtest with its own frozen plan,
`~/.kalos-dev/REAL_BACKTEST.md`, valid under A1): M1 vs random +0.007 [-0.053, +0.070]
tie; vs GP+UCB -0.066 [-0.101, -0.008] win; vs OFAT -0.093 [-0.160, +0.028] tie; trust 2
trusted / 78 exploration / 10 withheld of 90 replays.

## Scale and clones (internal, data rights unconfirmed; report-only)

- Clone ranking 96w -> 24w, tuning folds (S1/S2 only), 3 seeds. Across campaigns (train
  on S1, test on S2 and the reverse): the served model's predictions were flat on the
  other campaign (Spearman undefined, AUC 0.50, top-8 precision 0.125, verdict
  `not_yet`), likely because the other campaign's readouts sit outside the training
  range and the GP reverts to its mean; carry-forward (96w signal) gave Spearman 0.37 and
  0.61, AUC 0.70 and 0.76, top-8 precision 0.56. Within S1+S2 (4-fold by clone): Kalos
  Spearman 0.31 mean over 12 folds (range -0.16 to 0.82), AUC 0.60 vs carry-forward 0.75.
  Interval coverage on held-out clones 0.885 (90 %) / 0.931 (95 %). Kalos does not beat
  the mandatory baseline (agrees with voyager-port's leave-one-campaign-out: 0.30 vs 0.53).
- Scale transfer, leave-one-transition-out on tuning folds: Spearman 0.29 vs 0.35
  carry-forward; coverage 0.96 / 0.98.
- 24w -> SF125 (11 clones, LOO): 0/33 folds trusted (correct); 90 % coverage 1.0.
- Synthetic scale-up LOSO (fixture): interior Spearman 0.71; the up/down folds have too
  few rows for a Spearman.

## Wall time per analysis (shared machine)

Median wall / CPU seconds per analysis: synthetic closed loop 31 / 5.7; `mixed` 257 / 31
and `clones` 148 / 26 (the mixed-categorical optimizer); real pool 85 / 9.3.

## Reproducibility notes

- The engine is not bit-reproducible across runs here: the A2 re-run of `constrained`
  gave 40 strict constraint misses where the first run gave 44 on the same instances
  (identical data until the proposals differ), consistent with CAMPAIGNS_API.md
  "Determinism". Treat differences below the CIs as noise.
- Strict constraint misses (A2): 40 of 60 proposals on an active linear constraint
  exceed it in the last displayed digit (~1e-4). An engine issue for ml-acquire.
