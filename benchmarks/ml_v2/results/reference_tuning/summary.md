# ML benchmark v2: kalos_ref (tuning split)

Protocol ml_v2.0; spec `kalos.bench.ml_v2.arms.kalos_ref:make_arm`; config hash `98c2bcaf9584b22a`; commit `2338a63549`; started 2026-11-20T02:35:17, wall 7117.6 s with 10 workers.

Regret is normalized: (f* - f(best measured recipe)) / (f* - median f); 0 = optimum, 1 = a typical recipe. Coverage CIs are instance-cluster bootstraps; rate CIs are Wilson. Synthetic and emulator results are simulations; pool replay is retrospective. Report-only groups are marked.

## Closed loop, emulator and pool replay (per task)

| suite | task | group | n | regret final [95% CI] | by n0 (12/20/40) | cov@prop 90 / 95 | cov held-out 90 / 95 | width@prop 90 | trusted (rounds) | median s/analysis |
|---|---|---|---|---|---|---|---|---|---|---|
| closed_loop | batch | synthetic | 12 | 0.338 [0.18, 0.501] | 0.311/0.447/0.256 | 0.888 / 0.922 | 0.928 / 0.966 | 2 | 0.283 | 12.9 |
| closed_loop | bioprocess | synthetic | 12 | 0.363 [0.284, 0.451] | 0.388/0.448/0.252 | 0.932 / 0.91 | 0.993 / 0.998 | 1.89 | 0.283 | 48 |
| closed_loop | bumps | synthetic | 12 | 0.494 [0.293, 0.719] | 0.469/0.612/0.401 | 0.939 / 0.973 | 0.988 / 0.995 | 1.62 | 0.2 | 20.3 |
| closed_loop | clones | synthetic | 12 | 0.205 [0.0787, 0.365] | 0.23/0.158/0.227 | 0.973 / 0.954 | 0.993 / 0.991 | 1.93 | 0.167 | 148 |
| closed_loop | constrained | synthetic | 12 | 0.354 [0.143, 0.594] | 0.316/0.434/0.313 | 0.968 / 0.955 | 0.982 / 0.988 | 1.28 | 0.233 | 36.8 |
| closed_loop | hartmann6 | synthetic | 12 | 0.33 [0.195, 0.496] | 0.403/0.378/0.209 | 0.99 / 0.95 | 0.992 / 0.995 | 1.69 | 0.267 | 17.8 |
| closed_loop | heteroscedastic | synthetic | 12 | 0.135 [0.0719, 0.205] | 0.14/0.164/0.1 | 0.98 / 0.946 | 0.997 / 0.996 | 3.1 | 0.35 | 10.4 |
| closed_loop | mixed | synthetic | 12 | 0.155 [0.0913, 0.224] | 0.23/0.17/0.0663 | 0.983 / 0.969 | 0.996 / 0.998 | 2.34 | 0.217 | 257 |
| emulator | gp:amii/DBO_rat_myocyte | real_emulator | 4 | 0.347 [0.169, 0.525] | 0.301/0.254/0.53 | 0.93 / 0.95 | 0.977 / 0.996 | 2.11 | 0.2 | 37.5 |
| emulator | gp:amii/HeLa_timesaving | real_emulator | 4 | 1 [1, 1] | 1/1/1 | 0.859 / 0.857 | 0.966 / 0.984 | 0.747 | 0 | 68.7 |
| emulator | gp:amii/Tcell_expansion | real_emulator | 4 | 0.463 [0.353, 0.566] | 0.432/0.553/0.433 | 0.95 / 0.963 | 0.995 / 0.999 | 2.56 | 0.05 | 40.2 |
| emulator | gp:olympus/hplc | ood_emulator | 4 | 0.177 [0.0495, 0.304] | 0.187/0.333/0.0003 | 0.98 / 0.95 | 0.996 / 0.989 | 2.03 | 0.15 | 24.9 |
| emulator | rf:amii/DBO_rat_myocyte | real_emulator | 4 | 0.419 [0.275, 0.563] | 0.468/0.223/0.517 | 0.98 / 0.912 | 0.991 / 0.997 | 1.81 | 0.15 | 46.2 |
| emulator | rf:amii/HeLa_timesaving | real_emulator | 4 | 0.566 [0.545, 0.589] | 0.565/0.535/0.601 | 0.947 / 0.9 | 0.969 / 0.986 | 1.93 | 0.15 | 77.4 |
| emulator | rf:amii/Tcell_expansion | real_emulator | 4 | 0.788 [0.616, 0.961] | 0.787/0.883/0.696 | 0.905 / 0.853 | 0.935 / 0.981 | 2.11 | 0.1 | 37.5 |
| emulator | rf:olympus/hplc | ood_emulator | 4 | 0.262 [0.233, 0.312] | 0.235/0.241/0.336 | 0.96 / 0.975 | 0.996 / 0.998 | 2.91 | 0.3 | 31.9 |
| pool | amii/DBO_rat_myocyte | real_pool | 6 | 0.134 [0, 0.402] | 0.402/0/0 | 0.987 / 0.977 | 0.997 / 0.998 | 3.66 | 0.3 | 70.8 |
| pool | amii/HeLa_timesaving | real_pool | 6 | 0.23 [0.0998, 0.362] | 0.318/0.266/0.108 | 1 / 0.992 | 0.974 / 0.948 | 2.32 | 0.167 | 124 |
| pool | amii/Tcell_expansion | real_pool | 6 | 0.17 [0.0877, 0.235] | 0.205/0.144/0.16 | 0.973 / 0.985 | 0.984 / 0.98 | 3.43 | 0.233 | 62.4 |
| pool | fixture/synthetic_bioprocess | fixture_pool | 6 | 0.145 [0.0153, 0.293] | 0.244/0.17/0.023 | 0.953 / 0.961 | 0.972 / 0.984 | 1.15 | 0.2 | 361 |
| pool | fixture/synthetic_scaleup | fixture_pool | 6 | 0.0129 [0, 0.0388] | 0.0388/0/0 | 0.973 / 0.992 | 0.978 / 0.994 | 9.18 | 0.4 | 65.4 |
| pool | kaggle/fermentation | info_only (report-only) | 6 | 0.35 [0.196, 0.45] | 0.406/0.416/0.226 | 0.942 / 0.971 | 0.966 / 0.992 | 1.48 | 0 | 77.1 |
| pool | olympus/hplc | ood_pool | 6 | 0.128 [0.0411, 0.229] | 0.0706/0.115/0.199 | 0.967 / 0.962 | 0.983 / 0.991 | 2.43 | 0.267 | 29.3 |

## Pooled per suite and group

| suite / group | n | regret final [95% CI] | cov@prop 90 [CI] | cov@prop 95 [CI] | cov held-out 90 / 95 | width@prop 90 |
|---|---|---|---|---|---|---|
| closed_loop|synthetic | 96 | 0.297 [0.24, 0.356] | 0.957 [0.941, 0.971] | 0.948 [0.933, 0.961] | 0.984 / 0.991 | 1.94 |
| emulator|real_emulator | 24 | 0.597 [0.493, 0.704] | 0.93 [0.904, 0.956] | 0.907 [0.881, 0.935] | 0.972 / 0.991 | 1.98 |
| emulator|ood_emulator | 8 | 0.219 [0.138, 0.286] | 0.97 [0.93, 1] | 0.963 [0.929, 1] | 0.996 / 0.994 | 2.46 |
| pool|real_pool | 18 | 0.178 [0.0922, 0.283] | 0.987 [0.976, 0.996] | 0.985 [0.974, 0.995] | 0.984 / 0.974 | 3.19 |
| pool|fixture_pool | 12 | 0.0792 [0.0115, 0.171] | 0.963 [0.943, 0.98] | 0.977 [0.956, 0.993] | 0.974 / 0.987 | 5.75 |
| pool|info_only | 6 | 0.35 [0.196, 0.45] | 0.942 [0.904, 0.981] | 0.971 [0.939, 0.993] | 0.966 / 0.992 | 1.48 |
| pool|ood_pool | 6 | 0.128 [0.0411, 0.229] | 0.967 [0.947, 0.987] | 0.962 [0.939, 0.985] | 0.983 / 0.991 | 2.43 |

False trust over all null snapshots: 0.0833 [0.0428, 0.156] (n=96).

## Snapshots (trust and calibration by data size)

| suite | task | signal | n | trusted [Wilson CI] | cov@prop 90 | cov held-out 90 / 95 | width held-out 90 | prospective Spearman |
|---|---|---|---|---|---|---|---|---|
| snapshot | batch | True | 12 | 0 [0, 0.561] | 0.867 | 0.993 / n/a | 1.31 | 0.693 |
| snapshot | batch | True | 20 | 0.667 [0.208, 0.939] | 0.933 | 0.988 / 0.995 | 1.67 | 0.719 |
| snapshot | batch | True | 40 | 1 [0.439, 1] | 0.733 | 0.997 / 0.998 | 0.67 | 0.977 |
| snapshot | batch | True | 80 | 1 [0.439, 1] | 0.6 | 0.993 / 0.998 | 0.851 | 0.813 |
| snapshot | bioprocess | True | 12 | 0 [0, 0.561] | 0.867 | 0.977 / n/a | 1.03 | -0.0112 |
| snapshot | bioprocess | True | 20 | 0 [0, 0.561] | 1 | 0.995 / 0.997 | 1.39 | 0.175 |
| snapshot | bioprocess | True | 40 | 0.667 [0.208, 0.939] | 0.867 | 0.982 / 0.988 | 1.03 | 0.337 |
| snapshot | bioprocess | True | 80 | 1 [0.439, 1] | 0.933 | 0.99 / 0.998 | 0.773 | 0.496 |
| snapshot | bumps | True | 12 | 0 [0, 0.561] | 1 | 0.982 / n/a | 1.18 | 0.332 |
| snapshot | bumps | True | 20 | 0 [0, 0.561] | 0.8 | 0.918 / 0.973 | 0.696 | 0.11 |
| snapshot | bumps | True | 40 | 1 [0.439, 1] | 1 | 0.988 / 1 | 0.629 | 0.947 |
| snapshot | bumps | True | 80 | 0.667 [0.208, 0.939] | 0.733 | 0.98 / 0.992 | 0.6 | 0.738 |
| snapshot | clones | True | 12 | 0 [0, 0.561] | 0.933 | 0.992 / n/a | 1.45 | 0.628 |
| snapshot | clones | True | 20 | 0.667 [0.208, 0.939] | 1 | 0.998 / 1 | 1.3 | 0.687 |
| snapshot | clones | True | 40 | 0.667 [0.208, 0.939] | 1 | 0.995 / 1 | 1.15 | 0.84 |
| snapshot | clones | True | 80 | 0.667 [0.208, 0.939] | 1 | 0.995 / 1 | 0.962 | 0.645 |
| snapshot | constrained | True | 12 | 0 [0, 0.561] | 0.6 | 0.917 / n/a | 1.12 | 0.316 |
| snapshot | constrained | True | 20 | 0.667 [0.208, 0.939] | 1 | 0.988 / 0.998 | 1.49 | 0.618 |
| snapshot | constrained | True | 40 | 0.333 [0.0615, 0.792] | 0.933 | 0.993 / 1 | 0.927 | 0.535 |
| snapshot | constrained | True | 80 | 1 [0.439, 1] | 0.933 | 0.992 / 1 | 0.573 | 0.893 |
| snapshot | hartmann6 | True | 12 | 0 [0, 0.561] | 1 | 1 / n/a | 1.66 | 0.0503 |
| snapshot | hartmann6 | True | 20 | 0 [0, 0.561] | 1 | 0.992 / 1 | 1.01 | 0.317 |
| snapshot | hartmann6 | True | 40 | 1 [0.439, 1] | 0.933 | 0.997 / 0.997 | 1.22 | 0.61 |
| snapshot | hartmann6 | True | 80 | 0.667 [0.208, 0.939] | 0.867 | 0.988 / 0.995 | 0.82 | 0.409 |
| snapshot | heteroscedastic | True | 12 | 0 [0, 0.561] | 0.933 | 0.98 / n/a | 1.31 | 0.495 |
| snapshot | heteroscedastic | True | 20 | 1 [0.439, 1] | 1 | 0.993 / 0.998 | 1.78 | 0.876 |
| snapshot | heteroscedastic | True | 40 | 0.667 [0.208, 0.939] | 0.933 | 0.998 / 1 | 1.59 | 0.864 |
| snapshot | heteroscedastic | True | 80 | 1 [0.439, 1] | 0.867 | 0.968 / 0.987 | 0.996 | 0.966 |
| snapshot | mixed | True | 12 | 0 [0, 0.561] | 1 | 0.997 / n/a | 1.96 | 0.231 |
| snapshot | mixed | True | 20 | 0.333 [0.0615, 0.792] | 1 | 0.998 / 1 | 2.23 | 0.509 |
| snapshot | mixed | True | 40 | 0.333 [0.0615, 0.792] | 1 | 0.978 / 1 | 1.44 | 0.321 |
| snapshot | mixed | True | 80 | 0.333 [0.0615, 0.792] | 1 | 0.992 / 0.998 | 1.46 | 0.592 |
| snapshot | null_batch_column | False | 12 | 0 [0, 0.561] | 1 | 0.998 / n/a | 21.1 | n/a |
| snapshot | null_batch_column | False | 20 | 0 [0, 0.561] | 0.7 | 0.928 / 0.942 | 8.07 | n/a |
| snapshot | null_batch_column | False | 40 | 0 [0, 0.561] | 1 | 1 / 0.982 | 10.4 | n/a |
| snapshot | null_batch_column | False | 80 | 0 [0, 0.561] | 1 | 1 / 1 | 8.87 | n/a |
| snapshot | null_batch_hidden | False | 12 | 0 [0, 0.561] | 0.733 | 0.962 / n/a | 5.03 | n/a |
| snapshot | null_batch_hidden | False | 20 | 0.667 [0.208, 0.939] | 0.933 | 0.978 / 0.997 | 6.29 | n/a |
| snapshot | null_batch_hidden | False | 40 | 0.667 [0.208, 0.939] | 1 | 0.953 / 1 | 5.13 | n/a |
| snapshot | null_batch_hidden | False | 80 | 1 [0.439, 1] | 0.6 | 0.778 / 0.975 | 2.77 | n/a |
| snapshot | null_batch_rounds | False | 12 | 0 [0, 0.561] | 0.867 | 0.945 / n/a | 9.99 | n/a |
| snapshot | null_batch_rounds | False | 20 | 0 [0, 0.561] | 0.8 | 0.927 / 0.94 | 7.78 | n/a |
| snapshot | null_batch_rounds | False | 40 | 0 [0, 0.561] | 1 | 0.992 / 0.945 | 6.64 | n/a |
| snapshot | null_batch_rounds | False | 80 | 0 [0, 0.561] | 1 | 0.998 / 1 | 6.95 | n/a |
| snapshot | null_jitter | False | 12 | 0 [0, 0.561] | 0.933 | 0.987 / n/a | 9.23 | n/a |
| snapshot | null_jitter | False | 20 | 0 [0, 0.561] | 0.9 | 0.97 / 0.928 | 5.12 | n/a |
| snapshot | null_jitter | False | 40 | 0 [0, 0.561] | 1 | 0.99 / 0.998 | 7.93 | n/a |
| snapshot | null_jitter | False | 80 | 0 [0, 0.561] | 0.6 | 0.958 / 0.993 | 5.1 | n/a |
| snapshot | null_noise | False | 12 | 0 [0, 0.561] | 0.9 | 1 / n/a | 9.3 | n/a |
| snapshot | null_noise | False | 20 | 0 [0, 0.561] | 0.867 | 0.993 / 1 | 6.75 | n/a |
| snapshot | null_noise | False | 40 | 0 [0, 0.561] | 1 | 0.978 / 0.998 | 5.27 | n/a |
| snapshot | null_noise | False | 80 | 0 [0, 0.561] | 0.867 | 0.972 / 1 | 4.9 | n/a |
| snapshot_real | perm:amii/DBO_rat_myocyte | False | 12 | 0 [0, 0.658] | n/a | n/a / n/a | n/a | n/a |
| snapshot_real | perm:amii/DBO_rat_myocyte | False | 20 | 0.5 [0.0945, 0.905] | n/a | n/a / n/a | n/a | n/a |
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
| snapshot_real | true:amii/DBO_rat_myocyte | True | 12 | 0 [0, 0.658] | n/a | 0.995 / n/a | 2.31 | 0.618 |
| snapshot_real | true:amii/DBO_rat_myocyte | True | 20 | 0.5 [0.0945, 0.905] | n/a | 0.992 / 1 | 1.46 | 0.702 |
| snapshot_real | true:amii/DBO_rat_myocyte | True | 40 | 1 [0.206, 1] | n/a | 1 / 1 | 2.34 | 0.687 |
| snapshot_real | true:amii/DBO_rat_myocyte | True | 80 | 1 [0.206, 1] | n/a | 0.985 / 0.992 | 1.09 | 0.775 |
| snapshot_real | true:amii/HeLa_timesaving | True | 12 | 0 [0, 0.658] | n/a | 0.975 / n/a | 1.48 | 0.239 |
| snapshot_real | true:amii/HeLa_timesaving | True | 20 | 1 [0.342, 1] | n/a | 0.945 / 0.975 | 1.29 | 0.3 |
| snapshot_real | true:amii/HeLa_timesaving | True | 40 | 0 [0, 0.793] | n/a | 1 / 1 | 3.79 | 0.117 |
| snapshot_real | true:amii/HeLa_timesaving | True | 80 | 1 [0.206, 1] | n/a | 0.99 / 1 | 1.83 | 0.143 |
| snapshot_real | true:amii/Tcell_expansion | True | 12 | 0 [0, 0.658] | n/a | 0.995 / n/a | 2.69 | 0.333 |
| snapshot_real | true:amii/Tcell_expansion | True | 20 | 0 [0, 0.658] | n/a | 1 / 1 | 4.92 | 0.364 |
| snapshot_real | true:amii/Tcell_expansion | True | 40 | 0 [0, 0.793] | n/a | 1 / 1 | 5.17 | 0.253 |
| snapshot_real | true:amii/Tcell_expansion | True | 80 | 1 [0.206, 1] | n/a | 0.985 / 1 | 3.17 | 0.313 |
| snapshot_real | true:fixture/synthetic_bioprocess | True | 12 | 0 [0, 0.658] | n/a | 0.905 / n/a | 0.542 | 0.155 |
| snapshot_real | true:fixture/synthetic_bioprocess | True | 20 | 0 [0, 0.658] | n/a | 0.95 / 0.975 | 0.587 | 0.382 |
| snapshot_real | true:fixture/synthetic_bioprocess | True | 40 | 1 [0.206, 1] | n/a | 1 / 1 | 2.03 | 0.613 |
| snapshot_real | true:fixture/synthetic_bioprocess | True | 80 | 1 [0.206, 1] | n/a | 0.975 / 1 | 0.992 | 0.494 |
| snapshot_real | true:fixture/synthetic_scaleup | True | 12 | 0 [0, 0.658] | n/a | 0.994 / n/a | 7.45 | 0.402 |
| snapshot_real | true:fixture/synthetic_scaleup | True | 20 | 0 [0, 0.658] | n/a | 0.964 / 0.964 | 10.1 | 0.527 |
| snapshot_real | true:fixture/synthetic_scaleup | True | 40 | 1 [0.206, 1] | n/a | 0.98 / 1 | 9.49 | 0.35 |
| snapshot_real | true:fixture/synthetic_scaleup | True | 45 | 0 [0, 0.793] | n/a | 0.956 / 1 | 7.14 | 0.545 |
| snapshot_real | true:kaggle/fermentation | True | 12 | 0 [0, 0.658] | n/a | 0.983 / n/a | 1.7 | -0.0541 |
| snapshot_real | true:kaggle/fermentation | True | 20 | 0 [0, 0.658] | n/a | 0.99 / 1 | 1.59 | 0.0042 |
| snapshot_real | true:kaggle/fermentation | True | 40 | 0 [0, 0.793] | n/a | 0.985 / 1 | 1.73 | -0.0622 |
| snapshot_real | true:kaggle/fermentation | True | 80 | 0 [0, 0.793] | n/a | 0.955 / 0.97 | 1.14 | 0.0288 |
| snapshot_real | true:olympus/hplc | True | 12 | 0 [0, 0.658] | n/a | 1 / n/a | 3.24 | 0.355 |
| snapshot_real | true:olympus/hplc | True | 20 | 0.5 [0.0945, 0.905] | n/a | 1 / 1 | 4.12 | 0.476 |
| snapshot_real | true:olympus/hplc | True | 40 | 1 [0.206, 1] | n/a | 0.98 / 1 | 1.47 | 0.606 |
| snapshot_real | true:olympus/hplc | True | 80 | 1 [0.206, 1] | n/a | 0.99 / 1 | 1.65 | 0.781 |

## Retrospective backtest (pool-restricted; never a wet-lab win)

| dataset | real order | cuts | M1 [CI] | M1 - random [CI] | M1 - OFAT | M1 - lab | M3 | M4 | trusted |
|---|---|---|---|---|---|---|---|---|---|
| amii/DBO_rat_myocyte | False | 18 | 0.161 [0.12, 0.203] | -0.0159 [-0.0433, 0.0099] tie | 0.0795 | n/a | 0.835 | 0.83 | 1 |
| amii/HeLa_timesaving | True | 2 | 0.41 [0, 0.82] | 0.0944 [-0.314, 0.503] tie | 0.138 | 0.0049 | -0.296 | 0.879 | 1 |
| amii/Tcell_expansion | True | 3 | 0.0696 [0.0398, 0.0862] | -0.104 [-0.115, -0.0968] win | -0.0957 | -0.184 | 0.136 | 0.724 | 1 |

## Scale transfer, clone ranking (internal and fixture)

- **internal / flask_small** (internal, data rights unconfirmed): group=internal, label=internal, data rights unconfirmed, n_folds=33, abs_err=66.9, cv_spearman=0.409, trusted_folds=0, refused_or_withheld_folds=0, coverage_0.9={"coverage": 1.0, "n": 33, "ci95": [0.8957, 1.0]}, coverage_0.95={"coverage": null, "n": 0, "ci95": null}, median_wall_s=18.6, p90_wall_s=25.5, median_cpu_s=2.14, n_calls=33, arm_errors=0
- **internal / rank_96to24** (internal, data rights unconfirmed): group=internal, label=internal, data rights unconfirmed, n_folds=18, auc_top_third=0.569, auc_top_third_carry_forward=0.745, cv_spearman=0.293, spearman=0.312, spearman_carry_forward=0.491, top8_precision=0.125, top8_precision_carry_forward=0.562, trusted_folds=7, refused_or_withheld_folds=0, coverage_0.9={"coverage": 0.8854, "n": 288, "ci95": [0.8434, 0.9172]}, coverage_0.95={"coverage": 0.9306, "n": 288, "ci95": [0.8952, 0.9546]}, median_wall_s=14.4, p90_wall_s=19.4, median_cpu_s=1.82, n_calls=18, arm_errors=0
- **internal / transfer_loso** (internal, data rights unconfirmed): group=internal, label=internal, data rights unconfirmed, n_folds=12, auc_top_third=0.661, auc_top_third_carry_forward=0.689, cv_spearman=0.251, spearman=0.293, spearman_carry_forward=0.35, trusted_folds=10, refused_or_withheld_folds=0, coverage_0.9={"coverage": 0.9627, "n": 777, "ci95": [0.9469, 0.9739]}, coverage_0.95={"coverage": 0.9833, "n": 777, "ci95": [0.9716, 0.9902]}, median_wall_s=20, p90_wall_s=24.1, median_cpu_s=2.17, n_calls=12, arm_errors=0
- **scale / loso:fixture/synthetic_scaleup** (synthetic (generated) scale-up fixture): group=fixture_scale, label=synthetic (generated) scale-up fixture, n_folds=9, cv_spearman=0.401, mae_norm=0.168, spearman=0.69, direction_up={"n": 1, "spearman_mean": null, "trusted": 1}, direction_down={"n": 1, "spearman_mean": 0.6, "trusted": 1}, direction_interior={"n": 7, "spearman_mean": 0.7051, "trusted": 5}, trusted_folds=7, refused_or_withheld_folds=0, coverage_0.9={"coverage": 0.9778, "n": 90, "ci95": [0.9226, 0.9939]}, coverage_0.95={"coverage": 0.9889, "n": 90, "ci95": [0.9397, 0.998]}, median_wall_s=62.3, p90_wall_s=71.7, median_cpu_s=6.41, n_calls=9, arm_errors=0
