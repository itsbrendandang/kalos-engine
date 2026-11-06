"""Side-effect import: puts `ContractHeaderMiddleware` on `kalos.portal.app.app`.

Imported once, from the bottom of `kalos/portal/app.py`, so wiring the header
costs app.py one import line while the live branches still edit that file
(docs plan §6.4). Added last, so it is the outermost user middleware and the
header is on CORS, size-limit and auth refusals too. When app.py is split
(plan S1) this becomes one `app.add_middleware(...)` in the app factory.
"""
from __future__ import annotations

from kalos.portal.app import app
from kalos.portal.contract import ContractHeaderMiddleware

app.add_middleware(ContractHeaderMiddleware)
