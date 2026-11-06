# Benchmark: DataHow's public in-silico fed-batch runs

Simulated runs only. Results here describe how Kalos does on this simulator.
They are not evidence about any real process.

## Source and licence

- `hybrid-modeling-summer-school` by DataHow AG,
  https://github.com/DataHow/hybrid-modeling-summer-school, commit
  `a002db21de9f42a9fb87da22f860e70c0cc953e9`.
- MIT License, Copyright (c) 2023 DataHow. Kalos does not copy their code. The
  data files are not in this repository.
- Files used: `data/insilico_noisy_A_train.csv` (20 runs) and
  `data/insilico_noisy_A_test.csv` (100 runs); the `fixed` files are the same
  runs without measurement noise. Each run has 15 daily steps.

## How Kalos reads it

`kalos/bench/insilico_hybrid.py` turns each run into one row:

- Settings: temperature and pH at the start and after their one shift,
  stirring, DO, starting glucose and glutamine, and the size of the glucose and
  glutamine bolus feeds.
- Target: the titer at the last step (harvest).

Stirring is the same in every run, so it carries no signal.

## Reproduce

```bash
git clone https://github.com/DataHow/hybrid-modeling-summer-school /tmp/hms
KALOS_DATAHOW_DATA=/tmp/hms/data python examples/insilico_hybrid_benchmark.py --seeds 20
```

## Results (noisy variant, torch 2.13.0, BoTorch 0.18.1, 20 seeds)

| Question | Result |
| --- | --- |
| Fit on the 20 train runs and rank the 100 test runs by predicted harvest titer | Spearman 0.20, 95% interval -0.02 to 0.40 |
| Pool replay over all 120 runs (8 random starting runs, up to 40 more): runs needed to reach 90% of the best titer in the pool | Kalos BO 31.4, random 32.8; paired difference -1.5 runs, 95% interval -11.8 to 8.0 |

What this shows:

- With 20 runs and 9 varying settings, the endpoint model ranks unseen runs only weakly. The interval includes zero.
- Over this pool, BO and random picks are not separated.

The source course treats this data as a time-series problem. A model of the trajectories (the hybrid models it teaches) may do better than an endpoint table. Kalos does not have one yet.
