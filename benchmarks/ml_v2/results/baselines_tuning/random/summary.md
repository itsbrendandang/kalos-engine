# ML benchmark v2: random (tuning split)

Protocol ml_v2.0; spec `kalos.bench.ml_v2.arms.baselines:make_random`; config hash `ecabdb716924309a`; commit `2338a63549`; started 2026-11-20T04:53:58, wall 12.0 s with 4 workers.

Regret is normalized: (f* - f(best measured recipe)) / (f* - median f); 0 = optimum, 1 = a typical recipe. Coverage CIs are instance-cluster bootstraps; rate CIs are Wilson. Synthetic and emulator results are simulations; pool replay is retrospective. Report-only groups are marked.

## Closed loop, emulator and pool replay (per task)

| suite | task | group | n | regret final [95% CI] | by n0 (12/20/40) | cov@prop 90 / 95 | cov held-out 90 / 95 | width@prop 90 | trusted (rounds) | median s/analysis |
|---|---|---|---|---|---|---|---|---|---|---|
| closed_loop | batch | synthetic | 12 | 0.359 [0.238, 0.482] | 0.424/0.265/0.387 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| closed_loop | bioprocess | synthetic | 12 | 0.497 [0.415, 0.572] | 0.561/0.486/0.443 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| closed_loop | bumps | synthetic | 12 | 0.604 [0.414, 0.8] | 0.59/0.733/0.488 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| closed_loop | clones | synthetic | 12 | 0.424 [0.284, 0.578] | 0.407/0.415/0.449 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| closed_loop | constrained | synthetic | 12 | 0.429 [0.298, 0.577] | 0.392/0.477/0.418 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| closed_loop | hartmann6 | synthetic | 12 | 0.563 [0.453, 0.681] | 0.67/0.524/0.495 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| closed_loop | heteroscedastic | synthetic | 12 | 0.33 [0.198, 0.464] | 0.236/0.365/0.388 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| closed_loop | mixed | synthetic | 12 | 0.345 [0.236, 0.463] | 0.443/0.382/0.209 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| emulator | gp:amii/DBO_rat_myocyte | real_emulator | 4 | 0.416 [0.243, 0.589] | 0.431/0.288/0.514 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| emulator | gp:amii/HeLa_timesaving | real_emulator | 4 | 1 [1, 1] | 1/1/1 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| emulator | gp:amii/Tcell_expansion | real_emulator | 4 | 0.75 [0.596, 0.904] | 0.755/0.837/0.654 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| emulator | gp:olympus/hplc | ood_emulator | 4 | 0.282 [0.12, 0.416] | 0.416/0.0919/0.206 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| emulator | rf:amii/DBO_rat_myocyte | real_emulator | 4 | 0.778 [0.673, 0.895] | 0.859/0.648/0.748 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| emulator | rf:amii/HeLa_timesaving | real_emulator | 4 | 0.623 [0.556, 0.737] | 0.57/0.797/0.556 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| emulator | rf:amii/Tcell_expansion | real_emulator | 4 | 0.739 [0.622, 0.897] | 0.782/0.656/0.735 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| emulator | rf:olympus/hplc | ood_emulator | 4 | 0.233 [0.213, 0.255] | 0.229/0.267/0.206 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| pool | amii/DBO_rat_myocyte | real_pool | 6 | 0.234 [0.0446, 0.452] | 0.378/0/0.326 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| pool | amii/HeLa_timesaving | real_pool | 6 | 0.347 [0.231, 0.469] | 0.402/0.422/0.215 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| pool | amii/Tcell_expansion | real_pool | 6 | 0.162 [0.085, 0.232] | 0.0426/0.249/0.194 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| pool | fixture/synthetic_bioprocess | fixture_pool | 6 | 0.207 [0.0642, 0.358] | 0.17/0.427/0.023 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| pool | fixture/synthetic_scaleup | fixture_pool | 6 | 0.0122 [0, 0.0243] | 0.0182/0.0182/0 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| pool | kaggle/fermentation | info_only (report-only) | 6 | 0.244 [0.115, 0.366] | 0.164/0.284/0.284 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |
| pool | olympus/hplc | ood_pool | 6 | 0.204 [0.13, 0.281] | 0.238/0.146/0.228 | n/a / n/a | n/a / n/a | n/a | 0 | 0 |

## Pooled per suite and group

| suite / group | n | regret final [95% CI] | cov@prop 90 [CI] | cov@prop 95 [CI] | cov held-out 90 / 95 | width@prop 90 |
|---|---|---|---|---|---|---|
| closed_loop|synthetic | 96 | 0.444 [0.392, 0.495] | n/a | n/a | n/a / n/a | n/a |
| emulator|real_emulator | 24 | 0.718 [0.628, 0.803] | n/a | n/a | n/a / n/a | n/a |
| emulator|ood_emulator | 8 | 0.258 [0.189, 0.335] | n/a | n/a | n/a / n/a | n/a |
| pool|real_pool | 18 | 0.248 [0.159, 0.343] | n/a | n/a | n/a / n/a | n/a |
| pool|fixture_pool | 12 | 0.109 [0.0183, 0.216] | n/a | n/a | n/a / n/a | n/a |
| pool|info_only | 6 | 0.244 [0.115, 0.366] | n/a | n/a | n/a / n/a | n/a |
| pool|ood_pool | 6 | 0.204 [0.13, 0.281] | n/a | n/a | n/a / n/a | n/a |

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
| amii/DBO_rat_myocyte | False | 18 | 0.197 [0.163, 0.231] | 0.02 [-0.0169, 0.0512] tie | 0.116 | n/a | n/a | 0.594 | 0 |
| amii/HeLa_timesaving | True | 2 | 0.286 [0, 0.573] | -0.0292 [-0.314, 0.256] tie | 0.0147 | -0.119 | n/a | 0.0922 | 0 |
| amii/Tcell_expansion | True | 3 | 0.178 [0.114, 0.215] | 0.0048 [-0.0406, 0.0324] tie | 0.0127 | -0.0756 | n/a | 0.513 | 0 |

## Scale transfer, clone ranking (internal and fixture)

- **internal / flask_small** (internal, data rights unconfirmed): group=internal, label=internal, data rights unconfirmed, n_folds=33, trusted_folds=0, refused_or_withheld_folds=0, coverage_0.9={"coverage": null, "n": 0, "ci95": null}, coverage_0.95={"coverage": null, "n": 0, "ci95": null}, median_wall_s=0, p90_wall_s=0, median_cpu_s=0, n_calls=33, arm_errors=0
- **internal / rank_96to24** (internal, data rights unconfirmed): group=internal, label=internal, data rights unconfirmed, n_folds=18, auc_top_third_carry_forward=0.745, trusted_folds=0, refused_or_withheld_folds=0, coverage_0.9={"coverage": null, "n": 0, "ci95": null}, coverage_0.95={"coverage": null, "n": 0, "ci95": null}, median_wall_s=0, p90_wall_s=0, median_cpu_s=0, n_calls=18, arm_errors=0
- **internal / transfer_loso** (internal, data rights unconfirmed): group=internal, label=internal, data rights unconfirmed, n_folds=12, auc_top_third_carry_forward=0.689, spearman_carry_forward=0.35, trusted_folds=0, refused_or_withheld_folds=0, coverage_0.9={"coverage": null, "n": 0, "ci95": null}, coverage_0.95={"coverage": null, "n": 0, "ci95": null}, median_wall_s=0, p90_wall_s=0.01, median_cpu_s=0, n_calls=12, arm_errors=0
- **scale / loso:fixture/synthetic_scaleup** (synthetic (generated) scale-up fixture): group=fixture_scale, label=synthetic (generated) scale-up fixture, n_folds=9, direction_up={"n": 1, "spearman_mean": null, "trusted": 0}, direction_down={"n": 1, "spearman_mean": null, "trusted": 0}, direction_interior={"n": 7, "spearman_mean": null, "trusted": 0}, trusted_folds=0, refused_or_withheld_folds=0, coverage_0.9={"coverage": null, "n": 0, "ci95": null}, coverage_0.95={"coverage": null, "n": 0, "ci95": null}, median_wall_s=0, p90_wall_s=0, median_cpu_s=0, n_calls=9, arm_errors=0
