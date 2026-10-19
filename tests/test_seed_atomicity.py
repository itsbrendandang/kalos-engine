"""`CampaignStore.seed(..., analysis=...)` publishes the campaign state and its
analysis in ONE SQLite transaction (docs/CAMPAIGN_LOOP.md, "Analysis/campaign
coherence"). A crash between the two steps must leave the previous state AND
the previous analysis exactly as they were: never a new campaign without its
analysis, never a new analysis without its campaign.

Crash injection: the real `_cas_write` / `_write_analysis` are wrapped so the
process "dies" (a BaseException, like KeyboardInterrupt or SystemExit, which
the transaction helper must also roll back) at each point of the transaction.
"""
from __future__ import annotations

import pandas as pd
import pytest

from kalos.portal.campaign import CampaignStore

TENANT = "acme"
OLD_ANALYSIS = {"target": "old_target", "marker": "OLD"}
NEW_ANALYSIS = {"target": "lipase_titer", "marker": "NEW"}


def _sheet(offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "Methanol": [1.0 + offset, 2.0, 3.0],
        "pH": [6.0, 6.5, 7.0 + offset],
        "lipase_titer": [1.0, 2.0, 3.0],
    })


def _seed(store: CampaignStore, df: pd.DataFrame, analysis: dict, dataset: str) -> str:
    return store.seed(
        df, "lipase_titer", ["Methanol", "pH"], tenant=TENANT,
        analysis=analysis, dataset=dataset,
    )


class _Crash(BaseException):
    """Stands in for the process dying mid-transaction (not a normal error)."""


def _snapshot(store: CampaignStore) -> tuple:
    state = store.get(tenant=TENANT)
    analysis = store.load_analysis(tenant=TENANT)
    return state, analysis


@pytest.fixture
def seeded(tmp_path):
    """A campaign that already has an analysis published (generation g1)."""
    store = CampaignStore(tmp_path)
    gen = _seed(store, _sheet(), OLD_ANALYSIS, "old.csv")
    return tmp_path, store, gen


@pytest.mark.parametrize("step", ["after_cas_write", "before_analysis_write", "during_analysis_write"])
def test_crash_inside_seed_leaves_old_state_and_old_analysis(seeded, monkeypatch, step):
    tmp_path, store, old_gen = seeded
    before = _snapshot(store)
    assert before[1] is not None and before[1]["marker"] == "OLD"

    real_cas = CampaignStore.__dict__["_cas_write"].__func__
    real_analysis = CampaignStore.__dict__["_write_analysis"].__func__

    def crash_after_cas(*a, **k):
        real_cas(*a, **k)  # the campaign row IS written inside the transaction...
        raise _Crash(step)  # ...and then the process dies before the analysis

    def crash_before_analysis(*a, **k):
        raise _Crash(step)

    def crash_during_analysis(conn, *a, **k):
        real_analysis(conn, *a, **k)  # the analysis row is written...
        raise _Crash(step)  # ...but the process dies before COMMIT

    injected = {
        "after_cas_write": (crash_after_cas, real_analysis),
        "before_analysis_write": (real_cas, crash_before_analysis),
        "during_analysis_write": (real_cas, crash_during_analysis),
    }[step]
    monkeypatch.setattr(CampaignStore, "_cas_write", staticmethod(injected[0]))
    monkeypatch.setattr(CampaignStore, "_write_analysis", staticmethod(injected[1]))

    with pytest.raises(_Crash):
        _seed(store, _sheet(offset=0.5), NEW_ANALYSIS, "new.csv")

    monkeypatch.undo()
    # Same connection: the transaction was rolled back, nothing leaked.
    assert _snapshot(store) == before
    # The analysis is still the one published with the old generation, so the
    # campaign still reports itself in sync with /api/latest.
    summary = store.summary(tenant=TENANT, analysis_generation=old_gen)
    assert summary["analysis_in_sync"] is True

    # A fresh connection (the restarted process) sees the same old state.
    reopened = CampaignStore(tmp_path)
    assert _snapshot(reopened) == before


def test_store_is_usable_after_a_crashed_seed(seeded, monkeypatch):
    _tmp, store, _gen = seeded

    def boom(*_a, **_k):
        raise _Crash("boom")

    monkeypatch.setattr(CampaignStore, "_write_analysis", staticmethod(boom))
    with pytest.raises(_Crash):
        _seed(store, _sheet(offset=0.5), NEW_ANALYSIS, "new.csv")
    monkeypatch.undo()

    new_gen = _seed(store, _sheet(offset=0.5), NEW_ANALYSIS, "new.csv")
    assert store.load_analysis(tenant=TENANT)["marker"] == "NEW"
    assert store.summary(tenant=TENANT, analysis_generation=new_gen)["analysis_in_sync"] is True


def test_successful_seed_publishes_state_and_analysis_together(tmp_path):
    store = CampaignStore(tmp_path)
    gen = _seed(store, _sheet(), NEW_ANALYSIS, "new.csv")
    assert store.get(tenant=TENANT)["target"] == "lipase_titer"
    assert store.load_analysis(tenant=TENANT)["marker"] == "NEW"
    assert store.summary(tenant=TENANT, analysis_generation=gen)["analysis_in_sync"] is True
