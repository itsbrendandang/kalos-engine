"""Trust-layer goldens: the full `_analyze` response on eight fixed synthetic
sheets, recorded so every flagged trust change (F02/F04/F05/F15) ships with a
reviewable before/after diff.

    python scripts/trust_golden.py record --reason "F05: ..."   # rewrite the fixture
    python scripts/trust_golden.py diff                          # current code vs fixture
    python scripts/trust_golden.py diff --against OLD.json       # two fixtures

`record` keeps the previous fixture's reasons and appends the new one, so the
fixture carries its own change log. `diff` prints changed, added and removed
leaf paths (old -> new), truncated per sheet, which is what goes in the commit
message. Same-stack claim only (see tests/test_golden_determinism.py): the
fixture records the OS, architecture, Python, library versions and the torch
thread count. The committed fixture is recorded on the CI stack (linux x86_64,
Python 3.11, uv.lock, CPU torch): `scripts/record_goldens.py`.

Sheets (all synthetic, generated here from fixed seeds; no real data):
  two_feature       24 rows, 2 continuous inputs, smooth signal
  bioprocess_subset first 30 rows / 3 inputs of examples/synthetic_bioprocess
  batch_text        6 batches x 8 runs, rounds clustered in space, batch offset
                    sd 1.0 and NO input signal; text `batch` column (audit c1)
  jitter_reps       15 recipes x 4 as-run replicates, 0.2% jitter, recipe
                    effect sd 1.0, no input signal (audit c2)
  pure_noise        20 rows, 3 inputs, target independent of inputs (audit f1)
  mixed_medium      24 rows, a 3-level text factor plus 2 continuous inputs
  minimize          24 rows, 2 inputs, goal minimize
  campaign_rounds   20 base rows + 3 folded rounds of 5, round index attached
                    the way a campaign re-analysis passes it (F05)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "trust_golden.json"
EXAMPLE = ROOT / "examples" / "synthetic_bioprocess" / "synthetic_bioprocess.csv"
# Wall clock only; everything else is compared.
VOLATILE = {"timestamp", "fit_seconds", "created_at"}
ROUND_ATTR = "kalos_round_index"


def _two_feature() -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(0)
    m = rng.uniform(0, 4, 24)
    ph = rng.uniform(5, 7, 24)
    df = pd.DataFrame({
        "Methanol": m.round(3), "pH": ph.round(2),
        "lipase_titer": (1.5 * m - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.1, 24)).round(3),
    })
    return df, {"target": "lipase_titer"}


def _bioprocess_subset() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(EXAMPLE).head(30)[["pH", "temperature_C", "glucose_gL", "titer_g_L"]]
    return df, {"target": "titer_g_L"}


def _batch_text() -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(11)
    xs, ys, bs = [], [], []
    for b in range(6):
        c = rng.uniform(0.15, 0.85, 3)
        x = np.clip(c + rng.normal(0, 0.12, (8, 3)), 0, 1)
        xs.append(x)
        ys.append(rng.normal(0, 1.0) + rng.normal(0, 0.2, 8))
        bs += [f"B{b + 1}"] * 8
    x = np.vstack(xs)
    df = pd.DataFrame(x.round(4), columns=["feed_a", "feed_b", "temp_offset"])
    df["batch"] = bs
    df["titer"] = np.concatenate(ys).round(4)
    return df, {"target": "titer"}


def _jitter_reps() -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(12)
    base = rng.uniform(0, 1, (15, 3))
    u = rng.normal(0, 1.0, 15)
    x = np.repeat(base, 4, axis=0) + rng.normal(0, 0.002, (60, 3))
    y = np.repeat(u, 4) + rng.normal(0, 0.2, 60)
    df = pd.DataFrame(x.round(5), columns=["feed_a", "feed_b", "temp_offset"])
    df["titer"] = y.round(4)
    return df, {"target": "titer"}


def _pure_noise() -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(13)
    x = rng.uniform(0, 1, (20, 3))
    df = pd.DataFrame(x.round(4), columns=["feed_a", "feed_b", "temp_offset"])
    df["titer"] = rng.normal(5, 1, 20).round(4)
    return df, {"target": "titer"}


def _mixed_medium() -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(1)
    n = 24
    medium = np.tile(["MedA", "MedB", "MedC"], 8)
    ph = rng.uniform(6.8, 7.2, n)
    t = rng.uniform(33, 37, n)
    off = {"MedA": 0.0, "MedB": 0.4, "MedC": 0.9}
    y = np.array([off[m] for m in medium]) + 2 - 0.3 * (t - 35) ** 2 / 4 + rng.normal(0, 0.15, n)
    df = pd.DataFrame({"medium": medium, "pH": ph.round(3), "temp_C": t.round(2), "titer_g_L": y.round(3)})
    return df, {"target": "titer_g_L"}


def _minimize() -> tuple[pd.DataFrame, dict]:
    from kalos.portal.setup_settings import analysis_setup, normalize_setup

    rng = np.random.default_rng(14)
    a = rng.uniform(0, 5, 24)
    b = rng.uniform(30, 37, 24)
    y = (a - 2.0) ** 2 + 0.2 * (b - 34) ** 2 + rng.normal(0, 0.2, 24)
    df = pd.DataFrame({"feed_a": a.round(3), "temp_C": b.round(2), "impurity_pct": y.round(3)})
    return df, {"target": "impurity_pct", "setup": analysis_setup(normalize_setup({"goal": "minimize"}))}


def _campaign_rounds() -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(15)

    def f(x: np.ndarray) -> np.ndarray:
        return 3 * np.exp(-6 * ((x - 0.65) ** 2).sum(1))

    x0 = rng.uniform(0, 1, (20, 2))
    rows, rounds = [x0], [0] * 20
    for r in range(1, 4):  # each round's batch clusters near the running best, as BO rounds do
        c = np.clip(0.65 + rng.normal(0, 0.15, 2), 0, 1)
        rows.append(np.clip(c + rng.normal(0, 0.05, (5, 2)), 0, 1))
        rounds += [r] * 5
    x = np.vstack(rows)
    y = f(x) + rng.normal(0, 0.15, len(x)) + np.array([0.0, 0.5, -0.4, 0.3])[rounds]
    df = pd.DataFrame(x.round(4), columns=["feed_a", "feed_b"])
    df["titer"] = y.round(4)
    df.attrs[ROUND_ATTR] = list(rounds)
    return df, {"target": "titer"}


SHEETS = {
    "two_feature": _two_feature,
    "bioprocess_subset": _bioprocess_subset,
    "batch_text": _batch_text,
    "jitter_reps": _jitter_reps,
    "pure_noise": _pure_noise,
    "mixed_medium": _mixed_medium,
    "minimize": _minimize,
    "campaign_rounds": _campaign_rounds,
}


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _env() -> dict:
    """The stack the goldens are valid on. Stable fields only: the OS and CPU
    architecture (not the kernel build, which changes with every CI runner
    image), Python major.minor and the numerical libraries. A fixture is
    compared only on a matching stack (tests/test_trust_golden.py)."""
    import botorch
    import gpytorch
    import scipy
    import torch

    return {"system": platform.system(), "machine": platform.machine(),
            "python": ".".join(platform.python_version_tuple()[:2]),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "botorch": botorch.__version__,
            "gpytorch": gpytorch.__version__, "torch_threads": torch.get_num_threads()}


def compute(names: list[str] | None = None) -> dict:
    import torch

    from kalos.portal.analysis import _analyze

    torch.set_num_threads(1)
    out: dict[str, Any] = {}
    for name in names or list(SHEETS):
        df, kw = SHEETS[name]()
        target = kw.pop("target")
        res = _analyze(df, target=target, **kw)
        out[name] = json.loads(json.dumps(_strip(res), default=str))
    return out


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        if not obj:
            flat[prefix] = {}
        for k, v in obj.items():
            flat.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        if not obj:
            flat[prefix] = []
        for i, v in enumerate(obj):
            flat.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        flat[prefix] = obj
    return flat


def _short(v: Any, n: int = 70) -> str:
    s = json.dumps(v, default=str)
    return s if len(s) <= n else s[: n - 3] + "..."


def diff(old: dict, new: dict, *, max_lines: int = 25) -> str:
    lines: list[str] = []
    for sheet in sorted(set(old) | set(new)):
        fo, fn = _flatten(old.get(sheet, {})), _flatten(new.get(sheet, {}))
        changed = [k for k in fo if k in fn and fo[k] != fn[k]]
        added = [k for k in fn if k not in fo]
        removed = [k for k in fo if k not in fn]
        if not (changed or added or removed):
            lines.append(f"{sheet}: unchanged")
            continue
        lines.append(f"{sheet}: {len(changed)} changed, {len(added)} added, {len(removed)} removed")
        shown = 0
        for tag, keys in (("~", changed), ("+", added), ("-", removed)):
            for k in keys:
                if shown >= max_lines:
                    break
                if tag == "~":
                    lines.append(f"  ~ {k}: {_short(fo[k])} -> {_short(fn[k])}")
                elif tag == "+":
                    lines.append(f"  + {k}: {_short(fn[k])}")
                else:
                    lines.append(f"  - {k}: {_short(fo[k])}")
                shown += 1
        hidden = len(changed) + len(added) + len(removed) - shown
        if hidden > 0:
            lines.append(f"  ... {hidden} more")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--reason", required=True)
    rec.add_argument("--out", type=Path, default=FIXTURE)
    df_ = sub.add_parser("diff")
    df_.add_argument("--against", type=Path, default=None, help="compare this fixture to FIXTURE instead of running")
    df_.add_argument("--max-lines", type=int, default=25)
    args = ap.parse_args(argv)

    if args.cmd == "record":
        prev = json.loads(args.out.read_text()) if args.out.exists() else {}
        reasons = [*prev.get("reasons", []), args.reason]
        payload = {"env": _env(), "reasons": reasons, "sheets": compute()}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        if prev:
            print(diff(prev.get("sheets", {}), payload["sheets"]))
        print(f"wrote {args.out}")
        return 0
    base = json.loads(FIXTURE.read_text())
    if args.against is not None:
        other = json.loads(args.against.read_text())
        print(diff(other["sheets"], base["sheets"], max_lines=args.max_lines))
    else:
        print(diff(base["sheets"], compute(), max_lines=args.max_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
