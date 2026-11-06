"""SOP drafts (`kalos.assist.sop`): the pure checks a revision passes through."""
from __future__ import annotations

import pandas as pd
import pytest

from kalos.assist import sop
from kalos.assist.words import banned_in, claims_in
from kalos.portal.campaign import CampaignStore


def _seed(store: CampaignStore) -> str:
    cid = store.create_campaign(tenant="acme", name="c", created_by="x")["id"]
    df = pd.DataFrame({"Methanol": [1.0, 2.0, 3.0], "pH": [6.0, 6.5, 7.0], "lipase_titer": [1.1, 2.2, 3.3]})
    store.seed(df, "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=cid,
               analysis={"target": "lipase_titer", "proposals": [{"recipe": {"Methanol": 2.5, "pH": 6.2}, "pred": 3.1, "std": 0.4}],
                         "feature_ranges": {"Methanol": [0.0, 4.0]}})
    store.start([{"recipe": {"Methanol": 2.5, "pH": 6.2}, "pred": 3.1, "std": 0.4}], tenant="acme", campaign_id=cid)
    return cid


def test_draft_from_the_module_is_sourced_and_clean(tmp_path):
    store = CampaignStore(tmp_path)
    cid = _seed(store)
    d = sop.draft(store, "acme", cid, {"round": 0}, "batch_recipe", None, "alice")
    assert d.markdown.startswith(sop.HEADER_PREFIX) and d.version == 1 and d.model is None
    assert {f["value"] for f in d.fields_from_record} >= {"2.5", "6.2", "3.1", "0.4", "0.0", "4.0", "lipase_titer"}
    assert set(sop.PLACEHOLDER_RE.findall(d.markdown)) == {p["name"] for p in d.placeholders}
    assert not banned_in(d.markdown) and not claims_in(d.markdown)
    assert sop.load(store, "acme", d.draft_id).markdown == d.markdown
    assert sop.load(store, "globex", d.draft_id) is None
    with pytest.raises(sop.SourceError):
        sop.draft(store, "acme", cid, {"round": 3}, "custom", None, "alice")


def test_check_revision_refuses_every_kind_of_drift(tmp_path):
    store = CampaignStore(tmp_path)
    d = sop.draft(store, "acme", _seed(store), {"proposal_index": [0]}, "media_prep", "T", "alice")
    md = d.markdown
    assert sop.check_revision(d, md) == []
    assert sop.check_revision(d, md.replace("It is a starting point", "Treat it as a starting point")) == []
    for bad in (
        md.split("\n", 1)[1],
        md.replace("| 2.5 |", "| 2.50 |"),
        md.replace("[[TO FILL: safety_notes]]", "gloves"),
        md.replace("[[TO FILL: safety_notes]]", "[[TO FILL: safety]]"),
        md + "[[TO FILL: more]]\n",
        md + "Hold 30 minutes.\n",
        md.replace("| 3.1 | Predicted |", "| 3.1 | Measured |"),
        md + "This will work.\n",
    ):
        assert sop.check_revision(d, bad), bad[-80:]
