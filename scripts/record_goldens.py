"""Record the two golden fixtures on THIS stack.

    python scripts/record_goldens.py --out DIR --reason "why"   # write both fixtures to DIR

Writes `DIR/trust_golden.json` (scripts/trust_golden.py) and
`DIR/strain_golden_no_clone.json` (tests/test_strain_golden_no_clone.py), each
computed by the same code the tests compare with. The committed fixtures in
tests/fixtures/ are recorded on the CI stack (linux x86_64, Python 3.11, uv.lock,
CPU torch): the CI serial job runs this script when the serial tests fail and
uploads DIR as the `goldens-ci-stack` artifact. To re-record, download it, review
the diff (`python scripts/trust_golden.py diff --against NEW.json`), copy the two
files into tests/fixtures/, and state the reason in the commit and the PR
(AGENTS.md 8: never re-record a golden without a stated reason).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="directory to write the two fixtures to")
    ap.add_argument("--reason", required=True, help="why the fixtures are (re-)recorded")
    args = ap.parse_args(argv)

    import torch

    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)

    tg = _load("trust_golden", ROOT / "scripts" / "trust_golden.py")
    prev = json.loads(tg.FIXTURE.read_text()) if tg.FIXTURE.exists() else {}
    trust = {"env": tg._env(), "reasons": [*prev.get("reasons", []), args.reason], "sheets": tg.compute()}
    (args.out / "trust_golden.json").write_text(json.dumps(trust, indent=1, sort_keys=True) + "\n")
    if prev:
        print(tg.diff(prev.get("sheets", {}), trust["sheets"]))

    sg = _load("strain_golden_no_clone", ROOT / "tests" / "test_strain_golden_no_clone.py")
    prev_strain = json.loads(sg.FIXTURE.read_text()) if sg.FIXTURE.exists() else {}
    history = prev_strain.get("recorded_on")
    strain = sg.record(f"{history}; {args.reason}" if history else args.reason)
    (args.out / "strain_golden_no_clone.json").write_text(json.dumps(strain, sort_keys=True, indent=1) + "\n")
    print(f"wrote {args.out}/trust_golden.json and {args.out}/strain_golden_no_clone.json on {trust['env']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
