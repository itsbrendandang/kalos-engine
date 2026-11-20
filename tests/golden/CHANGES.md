# Golden output changes

One line per intended change to the engine's output (scripts/golden.py record --reason). A PR whose output differs from its base must add a line here; the CI gate (golden.yml, base vs head on one runner) fails an undeclared difference.

The records under `darwin-arm64-…/` are a local pre-push check (`scripts/golden.py check` on an arm64 Mac with the uv.lock stack, Python 3.11), NOT a CI gate: float results differ across CPUs.

- 2026-11-20 (darwin-arm64-py3.11-torch2.13.0-botorch0.18.1-gpytorch1.15.2-numpy2.4.6-scipy1.17.1-pandas3.0.3-sklearn1.9.0): Initial recording (reorg guard R02) on main 8e33a37: the baseline every reorg move must reproduce byte for byte.
