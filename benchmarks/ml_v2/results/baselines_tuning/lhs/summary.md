# ML benchmark v2: lhs (tuning split)

Protocol ml_v2.0; spec `kalos.bench.ml_v2.arms.baselines:make_lhs`; config hash `bef341e959bfb8b0`; commit `2338a63549`; started 2026-11-20T04:54:15, wall 9.7 s with 4 workers.

Regret is normalized: (f* - f(best measured recipe)) / (f* - median f); 0 = optimum, 1 = a typical recipe. Coverage CIs are instance-cluster bootstraps; rate CIs are Wilson. Synthetic and emulator results are simulations; pool replay is retrospective. Report-only groups are marked.

## Closed loop, emulator and pool replay (per task)

| suite | task | group | n | regret final [95% CI] | by n0 (12/20/40) | cov@prop 90 / 95 | cov held-out 90 / 95 | width@prop 90 | trusted (rounds) | median s/analysis |
|---|---|---|---|---|---|---|---|---|---|---|
| closed_loop | batch | synthetic | 12 | 0.471 [0.316, 0.627] | 0.616/0.385/0.411 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| closed_loop | bioprocess | synthetic | 12 | 0.501 [0.424, 0.574] | 0.565/0.443/0.496 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| closed_loop | bumps | synthetic | 12 | 0.564 [0.393, 0.747] | 0.613/0.591/0.488 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| closed_loop | clones | synthetic | 12 | 0.373 [0.273, 0.469] | 0.297/0.387/0.436 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| closed_loop | constrained | synthetic | 12 | 0.446 [0.298, 0.607] | 0.429/0.528/0.381 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| closed_loop | hartmann6 | synthetic | 12 | 0.555 [0.495, 0.615] | 0.604/0.554/0.509 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| closed_loop | heteroscedastic | synthetic | 12 | 0.311 [0.207, 0.422] | 0.211/0.441/0.282 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| closed_loop | mixed | synthetic | 12 | 0.37 [0.267, 0.489] | 0.399/0.473/0.238 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| emulator | gp:amii/DBO_rat_myocyte | real_emulator | 4 | 0.374 [0.243, 0.506] | 0.348/0.288/0.514 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| emulator | gp:amii/HeLa_timesaving | real_emulator | 4 | 1 [1, 1] | 1/1/1 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| emulator | gp:amii/Tcell_expansion | real_emulator | 4 | 0.667 [0.591, 0.781] | 0.627/0.837/0.575 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| emulator | gp:olympus/hplc | ood_emulator | 4 | 0.129 [0.0431, 0.214] | 0.113/0.084/0.206 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| emulator | rf:amii/DBO_rat_myocyte | real_emulator | 4 | 0.722 [0.693, 0.749] | 0.696/0.75/0.748 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| emulator | rf:amii/HeLa_timesaving | real_emulator | 4 | 0.669 [0.533, 0.805] | 0.679/0.809/0.511 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| emulator | rf:amii/Tcell_expansion | real_emulator | 4 | 0.796 [0.539, 0.982] | 0.954/0.44/0.835 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| emulator | rf:olympus/hplc | ood_emulator | 4 | 0.231 [0.216, 0.245] | 0.229/0.213/0.252 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| pool | amii/DBO_rat_myocyte | real_pool | 6 | 0.52 [0.275, 0.695] | 0.579/0.402/0.581 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| pool | amii/HeLa_timesaving | real_pool | 6 | 0.0585 [0, 0.176] | 0/0.176/0 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| pool | amii/Tcell_expansion | real_pool | 6 | 0.265 [0.214, 0.319] | 0.28/0.319/0.194 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| pool | fixture/synthetic_bioprocess | fixture_pool | 6 | 0.162 [0.0077, 0.358] | 0.193/0/0.294 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| pool | fixture/synthetic_scaleup | fixture_pool | 6 | 0.0102 [0, 0.0305] | 0.0305/0/0 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| pool | kaggle/fermentation | info_only (report-only) | 6 | 0.273 [0.142, 0.408] | 0.117/0.304/0.399 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| pool | olympus/hplc | ood_pool | 6 | 0.177 [0.0843, 0.276] | 0.238/0.227/0.0657 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |

## Pooled per suite and group

| suite / group | n | regret final [95% CI] | cov@prop 90 [CI] | cov@prop 95 [CI] | cov held-out 90 / 95 | width@prop 90 |
|---|---|---|---|---|---|---|
| closed_loop|synthetic | 96 | 0.449 [0.4, 0.497] | n/a | n/a | n/a / n/a | n/a |
| emulator|real_emulator | 24 | 0.705 [0.609, 0.794] | n/a | n/a | n/a / n/a | n/a |
| emulator|ood_emulator | 8 | 0.18 [0.117, 0.229] | n/a | n/a | n/a / n/a | n/a |
| pool|real_pool | 18 | 0.281 [0.165, 0.401] | n/a | n/a | n/a / n/a | n/a |
| pool|fixture_pool | 12 | 0.0862 [0.0051, 0.204] | n/a | n/a | n/a / n/a | n/a |
| pool|info_only | 6 | 0.273 [0.142, 0.408] | n/a | n/a | n/a / n/a | n/a |
| pool|ood_pool | 6 | 0.177 [0.0843, 0.276] | n/a | n/a | n/a / n/a | n/a |

False trust over all null snapshots: 0 [0, 0.0385] (n=96).

## Snapshots (trust and calibration by data size)

| suite | task | signal | n | trusted [Wilson CI] | cov@prop 90 | cov held-out 90 / 95 | width held-out 90 | prospective Spearman |
|---|---|---|---|---|---|---|---|---|
| snapshot | batch | True | 12 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | batch | True | 20 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | batch | True | 40 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | batch | True | 80 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | bioprocess | True | 12 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | bioprocess | True | 20 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | bioprocess | True | 40 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | bioprocess | True | 80 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | bumps | True | 12 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | bumps | True | 20 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | bumps | True | 40 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | bumps | True | 80 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | clones | True | 12 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | clones | True | 20 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | clones | True | 40 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | clones | True | 80 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | constrained | True | 12 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | constrained | True | 20 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | constrained | True | 40 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | constrained | True | 80 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | hartmann6 | True | 12 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | hartmann6 | True | 20 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | hartmann6 | True | 40 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | hartmann6 | True | 80 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | heteroscedastic | True | 12 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | heteroscedastic | True | 20 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | heteroscedastic | True | 40 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | heteroscedastic | True | 80 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | mixed | True | 12 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | mixed | True | 20 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | mixed | True | 40 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | mixed | True | 80 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_batch_column | False | 12 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_batch_column | False | 20 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_batch_column | False | 40 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_batch_column | False | 80 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_batch_hidden | False | 12 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_batch_hidden | False | 20 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_batch_hidden | False | 40 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_batch_hidden | False | 80 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_batch_rounds | False | 12 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_batch_rounds | False | 20 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_batch_rounds | False | 40 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_batch_rounds | False | 80 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_jitter | False | 12 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_jitter | False | 20 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_jitter | False | 40 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_jitter | False | 80 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_noise | False | 12 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_noise | False | 20 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_noise | False | 40 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot | null_noise | False | 80 | 0 [0, 0.561] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:amii/DBO_rat_myocyte | False | 12 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:amii/DBO_rat_myocyte | False | 20 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:amii/DBO_rat_myocyte | False | 40 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:amii/DBO_rat_myocyte | False | 80 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:amii/HeLa_timesaving | False | 12 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:amii/HeLa_timesaving | False | 20 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:amii/HeLa_timesaving | False | 40 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:amii/HeLa_timesaving | False | 80 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:amii/Tcell_expansion | False | 12 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:amii/Tcell_expansion | False | 20 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:amii/Tcell_expansion | False | 40 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:amii/Tcell_expansion | False | 80 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:fixture/synthetic_bioprocess | False | 12 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:fixture/synthetic_bioprocess | False | 20 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:fixture/synthetic_bioprocess | False | 40 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:fixture/synthetic_bioprocess | False | 80 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:fixture/synthetic_scaleup | False | 12 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:fixture/synthetic_scaleup | False | 20 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:fixture/synthetic_scaleup | False | 40 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:fixture/synthetic_scaleup | False | 45 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:kaggle/fermentation | False | 12 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:kaggle/fermentation | False | 20 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:kaggle/fermentation | False | 40 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:kaggle/fermentation | False | 80 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:olympus/hplc | False | 12 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:olympus/hplc | False | 20 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:olympus/hplc | False | 40 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:olympus/hplc | False | 80 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:amii/DBO_rat_myocyte | True | 12 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:amii/DBO_rat_myocyte | True | 20 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:amii/DBO_rat_myocyte | True | 40 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:amii/DBO_rat_myocyte | True | 80 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:amii/HeLa_timesaving | True | 12 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:amii/HeLa_timesaving | True | 20 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:amii/HeLa_timesaving | True | 40 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:amii/HeLa_timesaving | True | 80 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:amii/Tcell_expansion | True | 12 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:amii/Tcell_expansion | True | 20 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:amii/Tcell_expansion | True | 40 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:amii/Tcell_expansion | True | 80 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:fixture/synthetic_bioprocess | True | 12 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:fixture/synthetic_bioprocess | True | 20 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:fixture/synthetic_bioprocess | True | 40 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:fixture/synthetic_bioprocess | True | 80 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:fixture/synthetic_scaleup | True | 12 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:fixture/synthetic_scaleup | True | 20 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:fixture/synthetic_scaleup | True | 40 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:fixture/synthetic_scaleup | True | 45 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:kaggle/fermentation | True | 12 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:kaggle/fermentation | True | 20 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:kaggle/fermentation | True | 40 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:kaggle/fermentation | True | 80 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:olympus/hplc | True | 12 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:olympus/hplc | True | 20 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:olympus/hplc | True | 40 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | true:olympus/hplc | True | 80 | 0 [0, 0.793] | n/a | n/a / n/a | n/a | n/a |

## Retrospective backtest (pool-restricted; never a wet-lab win)

| dataset | real order | cuts | M1 [CI] | M1 - random [CI] | M1 - OFAT | M1 - lab | M3 | M4 | trusted |
|---|---|---|---|---|---|---|---|---|---|
| amii/DBO_rat_myocyte | False | 18 | 0.283 [0.21, 0.355] | 0.106 [0.0557, 0.155] loss | 0.201 | n/a | n/a | 1.07 | 0 |
| amii/HeLa_timesaving | True | 2 | 0.0697 [0, 0.139] | -0.246 [-0.314, -0.178] win | -0.202 | -0.336 | n/a | 1 | 0 |
| amii/Tcell_expansion | True | 3 | 0.147 [0.083, 0.255] | -0.0259 [-0.0793, 0.0727] tie | -0.018 | -0.106 | n/a | 0.992 | 0 |

## Scale transfer, clone ranking (internal and fixture)

- **internal / flask_small** (internal, data rights unconfirmed): group=internal, label=internal, data rights unconfirmed, n_folds=33, trusted_folds=0, refused_or_withheld_folds=0, coverage_0.9={"coverage": null, "n": 0, "ci95": null}, coverage_0.95={"coverage": null, "n": 0, "ci95": null}, median_wall_s=0, p90_wall_s=0, median_cpu_s=0, n_calls=33, arm_errors=0
- **internal / rank_96to24** (internal, data rights unconfirmed): group=internal, label=internal, data rights unconfirmed, n_folds=18, auc_top_third_carry_forward=0.745, trusted_folds=0, refused_or_withheld_folds=0, coverage_0.9={"coverage": null, "n": 0, "ci95": null}, coverage_0.95={"coverage": null, "n": 0, "ci95": null}, median_wall_s=0, p90_wall_s=0, median_cpu_s=0, n_calls=18, arm_errors=0
- **internal / transfer_loso** (internal, data rights unconfirmed): group=internal, label=internal, data rights unconfirmed, n_folds=12, auc_top_third_carry_forward=0.689, spearman_carry_forward=0.35, trusted_folds=0, refused_or_withheld_folds=0, coverage_0.9={"coverage": null, "n": 0, "ci95": null}, coverage_0.95={"coverage": null, "n": 0, "ci95": null}, median_wall_s=0, p90_wall_s=0, median_cpu_s=0, n_calls=12, arm_errors=0
- **scale / loso:fixture/synthetic_scaleup** (synthetic (generated) scale-up fixture): group=fixture_scale, label=synthetic (generated) scale-up fixture, n_folds=9, direction_up={"n": 1, "spearman_mean": null, "trusted": 0}, direction_down={"n": 1, "spearman_mean": null, "trusted": 0}, direction_interior={"n": 7, "spearman_mean": null, "trusted": 0}, trusted_folds=0, refused_or_withheld_folds=0, coverage_0.9={"coverage": null, "n": 0, "ci95": null}, coverage_0.95={"coverage": null, "n": 0, "ci95": null}, median_wall_s=0, p90_wall_s=0, median_cpu_s=0, n_calls=9, arm_errors=0
