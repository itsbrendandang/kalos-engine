"""Build the assistant fixture for the LLM evaluation (run once; output committed).

Runs one REAL engine analysis on the first 60 rows of the repo's example sheet
(examples/synthetic_bioprocess, generated data), seeds a campaign in a temp
state dir, starts the first three proposed recipes, records two results (one of
them corrected with a reason), and writes what the assistant would retrieve
from: the campaign's index chunks and the guide step. Every fact the eval's
questions ask about is in these chunks, computed by the engine.

    KALOS_TORCH_THREADS=1 .venv/bin/python benchmarks/llm/build_fixture.py
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "fixtures" / "assistant_campaign.json"


def main() -> None:
    os.environ["KALOS_STATE_DIR"] = tempfile.mkdtemp(prefix="kalos-llm-eval-")
    os.environ["KALOS_ASSIST_EMBED"] = "0"
    from kalos.assist.guide import next_step
    from kalos.assist.index import campaign_chunks
    from kalos.assist.state import campaign_inputs
    from kalos.portal.analysis import _analyze
    from kalos.portal.campaign import CampaignStore

    df = pd.read_csv(ROOT / "examples/synthetic_bioprocess/synthetic_bioprocess.csv").head(60)
    analysis = _analyze(df, "titer_g_L")
    store = CampaignStore(Path(os.environ["KALOS_STATE_DIR"]))
    cid = store.create_campaign(tenant="eval", name="Fed-batch eval", created_by="eval")["id"]
    store.seed(df, analysis["target"], list(analysis.get("proposal_features") or analysis["features"]),
               tenant="eval", campaign_id=cid, analysis=analysis, dataset="synthetic_bioprocess.csv")
    recipes = [{"recipe": p["recipe"], "mode": p.get("mode")} for p in analysis["proposals"][:3]]
    runs = store.start(recipes, tenant="eval", campaign_id=cid)
    store.set_result(runs[0]["id"], 0.981, tenant="eval", campaign_id=cid)
    store.set_result(runs[1]["id"], 0.915, tenant="eval", campaign_id=cid)
    store.set_result(runs[1]["id"], 0.903, tenant="eval", campaign_id=cid, reason="plate re-read")
    info = store.campaign_info(tenant="eval", campaign_id=cid)
    state = store.get(tenant="eval", campaign_id=cid)
    stored = store.load_analysis(tenant="eval", campaign_id=cid)
    chunks = campaign_chunks(info, state, stored, None)
    _, summary, a, refusal = campaign_inputs(store, "eval", cid)
    guide = next_step(summary, a, refusal, campaign_id=cid)
    OUT.write_text(json.dumps({
        "source": "examples/synthetic_bioprocess/synthetic_bioprocess.csv, first 60 rows (generated data)",
        "campaign_id": cid,
        "run_ids": [r["id"] for r in runs],
        "guide": guide,
        "chunks": [asdict(c) for c in chunks],
    }, indent=1, default=str) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
