### Fixed - a grouping whose CV skipped a fold no longer sets the reliability score

- A campaign whose newest rounds held 1-2 runs got a round grouping in which holding out round 0 left fewer than 4 training runs, so that fold was skipped and the grouping's held-out Spearman was a rank over 2-3 runs (only ±0.5 or ±1 possible) with no interval. As the lowest score it replaced the primary score, and the reliability floor showed "n/a" (seen in QA: 162 -> 163 rows moved ρ 0.65 -> -0.50).
- `leak_free_cv` now counts a candidate grouping as evaluable only when every run was held out once and its 95% interval exists; otherwise the candidate is listed with the reason and a `structure_not_checkable` warning. `require_full_coverage=False` restores the old rule.
- Trust goldens: no change from this fix (`scripts/trust_golden.py diff` output is identical to main's).
