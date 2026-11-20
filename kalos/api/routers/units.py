"""The units registry over HTTP (read-only): ``GET /api/units`` (read scope).

Voyager's data-unit routes let anyone add a unit without a token; in Kalos the
registry is code (kalos/normalize/unit_registry.py), the same for every tenant,
and this route only reads it.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from kalos.normalize.unit_registry import all_units, dimensions
from kalos.portal.auth import READ, Principal, require_scope

router = APIRouter()


@router.get("/api/units")
def get_units(principal: Principal = Depends(require_scope(READ))) -> JSONResponse:
    """Every registered unit: its canonical symbol, dimension, name, accepted
    spellings, and whether a cell value in it is converted on upload."""
    return JSONResponse({
        "units": [u.as_dict() for u in all_units()],
        "dimensions": dimensions(),
        "notes": "factor and target units must be one of these symbols or spellings; "
                 "conversions happen only where converts_in_cells is true",
    })


__all__ = ["router"]
