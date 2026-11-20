# ML benchmark v2: how to run an arm

Read `PROTOCOL.md` first (normative), then `DATASETS.md`. Code: `kalos/bench/ml_v2/`.

## Setup

```bash
uv venv && uv pip install -e ".[ml,portal,bench,dev]"   # `bench` brings xgboost (tree baselines)
python -m kalos.bench.ml_v2 fetch                         # Zenodo + Olympus open data -> ~/kalos-bench-data
export KALOS_AMII_DATA=~/AI/workspace/data/amii/active-learning-for-cell-media/Data
export KALOS_EXTERNAL_DATA=~/kalos/external               # internal data (held-out only, report-only)
export KALOS_STATE_DIR=$(mktemp -d) KALOS_TORCH_THREADS=1 OMP_NUM_THREADS=1
python -m kalos.bench.ml_v2 datasets                      # what is available
```

macOS + XGBoost needs an OpenMP runtime: `brew install libomp`, or run with
`DYLD_FALLBACK_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.11/site-packages/torch/lib`
(not through `nohup`, which drops DYLD variables).

## Write an arm (new file under `kalos/bench/ml_v2/arms/`)

Engine-patch candidate (can ship), see `arms/example_patch.py`:

```python
from kalos.bench.ml_v2.adapter import patched
from kalos.bench.ml_v2.arms.kalos_ref import KalosEngineArm

class MyArm(KalosEngineArm):
    name = "surr_saas_v1"                      # unique; frozen arms are never renamed
    def engine_patch(self):
        import kalos.core.surrogate as s
        return patched(s, "Surrogate", MySurrogate)   # anything _analyze looks up at call time

def make_arm(config=None):
    return MyArm(config)
```

Standalone arm (BayBE, Ax, GP+UCB, trees; cannot ship): implement `name`,
`shippable = False`, `propose(ctx) -> ArmOutput`, `engine_patch() -> nullcontext()`.
Return `band_at` and `predict` so held-out coverage and ranking are scored.

## Run

```bash
# tuning (as often as you like; tuning seeds and tuning datasets only)
python -m kalos.bench.ml_v2 run --arm kalos.bench.ml_v2.arms.my_arm:make_arm --arm-config '{}' \
    --split tuning --out /tmp/mlv2/my_arm_tuning --workers 8 [--suites closed_loop snapshot] [--tasks bumps]
python -m kalos.bench.ml_v2 report /tmp/mlv2/my_arm_tuning
python -m kalos.bench.ml_v2 compare /tmp/mlv2/my_arm_tuning benchmarks/ml_v2/results/reference_tuning \
    --baseline /tmp/mlv2/random_tuning
# planted faults with your engine patch (criterion C5)
python -m kalos.bench.ml_v2 faults --arm kalos.bench.ml_v2.arms.my_arm:make_arm --out /tmp/mlv2/my_arm_faults
```

The reference arm's tuning results are committed in `results/reference_tuning/`
(`summary.md`, `summary.json`, raw per-job JSON except internal data). Jobs are
resumable: rerunning the same command skips finished jobs. `--split smoke` runs one
pilot seed for debugging.

## Held-out (once per frozen arm, max 3 per agent)

```bash
python -m kalos.bench.ml_v2 freeze --arm kalos.bench.ml_v2.arms.my_arm:make_arm --arm-config '{}' --note "why"
git add benchmarks/ml_v2/freeze/<name>.json && git commit -m "freeze <name>"
python -m kalos.bench.ml_v2 run --arm ... --split heldout --out /tmp/mlv2/my_arm_heldout --leadgene ...
python -m kalos.bench.ml_v2 run --arm kalos.bench.ml_v2.arms.kalos_ref:make_arm --split heldout --out /tmp/mlv2/ref_heldout
python -m kalos.bench.ml_v2 compare /tmp/mlv2/my_arm_heldout /tmp/mlv2/ref_heldout \
    --faults-cand <cand>/scorecard_full.json --faults-ref <ref>/scorecard_full.json
```

The guard refuses a held-out run without the committed freeze file or with a dirty
tree, and appends every held-out run to `results/heldout_ledger.jsonl`. Run the
reference arm's held-out evaluation from `main`'s engine code (a freeze file for
`kalos_ref` is committed); it is deterministic, so one shared copy serves every agent.

## CI / scoreboard

`kalos.bench.ml_v2.ci.ci_regret(out_dir, workers=4)`: closed loop on
`protocol.CI_TASKS` x `protocol.CI_SEEDS` (disjoint from tuning and held-out). A drift
signal, never a decision.
