"""Before/after probe for the scale readout honesty checks (PR #180).

Run it twice, once per state of the switch, to see what each check changes:

    KALOS_SCALE_READOUT_CHECKS=off uv run --no-sync python scripts/scale_readout_honesty_probe.py   # before
    KALOS_SCALE_READOUT_CHECKS=on  uv run --no-sync python scripts/scale_readout_honesty_probe.py   # after

`off` gives the same decisions, reasons and numbers as main before the checks
(tests/test_scale_readout_checks_setting.py). Every sheet here is synthetic,
with the same shape as the real sheets that exposed the gaps and invented
values.

`--external-ladder PATH` adds one local, non-synthetic case: a transposed
scale-ladder CSV (first column = parameter names, one column per scale) with
rows `scale_L`, `--target` and `--method`. Its output is labelled with
`--rights` (default "internal, data rights unconfirmed"). Nothing is written
anywhere; never commit such a file or its output.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd

from kalos.scale.evaluation import leave_one_scale_out_report
from kalos.scale.readout import TargetSpec, build_readout


def _short(r: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "decision": r["decision"],
        "reasons": r["reasons"],
        "gate": {k: r["gate"][k] for k in ("passed", "failed_check")},
        "assay_confound_severity": (r.get("assay_confound") or {}).get("severity"),
        "assumed_volumes_L": {k: v["value"] for k, v in (r.get("working_volumes") or {}).get("assumptions", {}).items()},
    }
    return out


def _loso(df: pd.DataFrame, target: str) -> Any:
    try:
        rep = leave_one_scale_out_report(df, target, [])
    except ValueError as exc:
        return f"ValueError: {exc}"
    return {k: rep[k] for k in ("n_rows_used", "n_rows_dropped", "n_scales")}


def _synthetic() -> dict[str, Any]:
    ladder = pd.DataFrame(
        {
            "scale_L": [1.5, 4.0, 20.0, 2500.0, 30000.0],
            "titer_g_L": [0.72, 0.70, 1.84, 0.69, 1.52],
            "titer_method": ["SDS-PAGE", None, "RP-HPLC", "SDS-PAGE", "RP-HPLC"],
        }
    )
    rng = np.random.default_rng(11)
    labels = ["384w", "96w", "24w", "SF125"]
    funnel = pd.DataFrame(
        {
            "scale_L": [lbl for lbl in labels for _ in range(6)],
            "ph": np.round(rng.uniform(6.8, 7.2, 24), 2),
            "titer": np.round(rng.uniform(1.0, 3.0, 24), 2),
        }
    )
    demo = pd.read_csv("examples/synthetic_scaleup/synthetic_scaleup.csv")
    demo["titer_method"] = np.where(demo["titer_g_per_L"] > demo["titer_g_per_L"].median(), "hplc", "gel")
    demo_target = TargetSpec(
        scale_L=7500.0, agitation_rpm=150.0, airflow_L_per_min=10.0,
        process_params={"ph_setpoint": 7.0, "temperature_C": 36.7},
    )
    demo_out = build_readout(demo, "titer_g_per_L", ["ph_setpoint", "temperature_C"], demo_target)
    return {
        "S1_5scale_ladder_assay_switch": _short(
            build_readout(ladder, "titer_g_L", [], TargetSpec(scale_L=90000.0, process_params={}))
        ),
        "S2_loso_default_config_no_agitation_airflow": _loso(ladder[["scale_L", "titer_g_L"]], "titer_g_L"),
        "S3_plate_flask_labels": _short(
            build_readout(funnel, "titer", ["ph"], TargetSpec(scale_L=0.25, process_params={"ph": 7.0}))
        ),
        "S4_demo_with_planted_method_column": {**_short(demo_out), "prediction": demo_out["prediction"]},
    }


def _external(path: str, target: str, method: str, target_scale: float) -> dict[str, Any]:
    raw = pd.read_csv(path, index_col=0).T
    df = pd.DataFrame(
        {
            "scale_L": pd.to_numeric(raw["scale_L"]).to_numpy(),
            target: pd.to_numeric(raw[target]).to_numpy(),
            method: raw[method].to_numpy(),
        }
    )
    return {
        "readout": _short(build_readout(df, target, [], TargetSpec(scale_L=target_scale, process_params={}))),
        "loso_default_config": _loso(df[["scale_L", target]], target),
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--external-ladder")
    ap.add_argument("--target", default="lipase_titer_gL")
    ap.add_argument("--method", default="lipase_titer_method")
    ap.add_argument("--target-scale", type=float, default=100000.0)
    ap.add_argument("--rights", default="internal, data rights unconfirmed")
    args = ap.parse_args()
    out: dict[str, Any] = {
        "KALOS_SCALE_READOUT_CHECKS": os.environ.get("KALOS_SCALE_READOUT_CHECKS", "") or "on (default)",
        "synthetic": _synthetic(),
    }
    if args.external_ladder:
        out["external"] = {
            "label": args.rights,
            **_external(args.external_ladder, args.target, args.method, args.target_scale),
        }
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
