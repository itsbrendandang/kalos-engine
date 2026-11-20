"""Records above and between campaigns, ported from the Voyager backend (NOTICE).

- `projects`: a grouping above campaigns (objective, status, optional sponsor);
- `factors`: a project's factor definitions, mapped onto campaign setup bounds;
- `lineage`: parent -> child links between campaigns (and their runs) across
  rounds and scale stages, with bounded traversal and cycle refusal;
- `outcomes`: read views joining runs, their measured results (every version) and
  the prediction recorded at start, without copying either.

Domain rules only: no SQL (kalos.storage.records_repo) and no HTTP
(kalos.api.routers). Functions take plain values and return plain dicts.
"""
