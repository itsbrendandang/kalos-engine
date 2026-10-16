"""Demo: generate synthetic response surface data and open the HTML widget.

Usage:
    cd kalos-engine
    python examples/response_surface_demo.py

This builds a fake _analyze result from a synthetic bioprocess surface,
computes the 21×21 GP posterior surface, then launches the response surface
widget in the default browser so you can see it immediately.

Requires: python -m pip install -e ".[ml,portal]"
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Synthetic objective — titer with a clear peak at (0.3, 0.5) in a 3-d design
# --------------------------------------------------------------------------- #

TITER_OPT = np.array([0.3, 0.5, 0.2])  # [Param_0, Param_1, Param_2]


def _titer(X: np.ndarray) -> np.ndarray:
    d = np.sum((np.atleast_2d(X) - TITER_OPT) ** 2, axis=1)
    return 22.0 * np.exp(-3.0 * d)


# --------------------------------------------------------------------------- #
# Fit surrogate and sweep surfaces
# --------------------------------------------------------------------------- #

def _build_surfaces():
    """Return (primary_surface, proposals, extra_pairs, incumbent, X, y) ready
    for the HTML template."""
    from kalos.core.surrogate import Surrogate
    from kalos.portal.viz_utils import (
        _response_surface_gp,
        _response_surface_extra_pairs,
    )

    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (25, 3))
    y = _titer(X)
    bounds = np.array([[0, 0, 0], [1, 1, 1]], float)

    cont_feats = ["Param_0", "Param_1", "Param_2"]

    s = Surrogate().fit(X, y, bounds=bounds)
    incumbent = float(y.max())

    # Dummy drivers — top 2 continuous features
    drv = [
        {"name": "Param_0", "abs_rho": 0.8},
        {"name": "Param_1", "abs_rho": 0.6},
        {"name": "Param_2", "abs_rho": 0.4},
    ]

    surface = _response_surface_gp(
        s, X, y, cont_feats, [], {}, {}, bounds, drv, None, incumbent, False
    )

    # Proposals — optimize_acqf-style batch
    from kalos.core.optimize import propose
    proposals = propose(s, bounds, q=3)
    mean, std = s.posterior(proposals)

    proposal_list = []
    for i in range(len(proposals)):
        proposal_list.append({
            "vals": [round(float(proposals[i, j]), 3) for j in range(3)],
            "pred": round(float(mean[i]), 3),
            "std": round(float(std[i]), 3),
            "mode": "explore" if std[i] > 1.5 else "exploit",
            "reason": "next recommendation",
        })

    extra = _response_surface_extra_pairs(
        s, X, y, cont_feats, None, bounds, drv, incumbent
    )
    extra_pairs = []
    if extra:
        for ep in extra:
            extra_pairs.append({
                "x_feature": ep["x_feature"],
                "y_feature": ep["y_feature"],
            })

    return surface, proposal_list, extra_pairs, incumbent, X, y


# --------------------------------------------------------------------------- #
# Write HTML with data baked in
# --------------------------------------------------------------------------- #

def main():
    surface, proposals, extra_pairs, incumbent, X, y = _build_surfaces()

    if surface is None:
        print("ERROR: could not compute response surface (surrogate fit failed)")
        sys.exit(1)
    if "reason" in surface:
        print("ERROR: could not compute response surface")
        print(f"  reason: {surface['reason']}")
        sys.exit(1)

    # Bake data into template
    template_path = Path(__file__).parent.parent / "kalos" / "portal" / "templates" / "response_surface.html"
    template = template_path.read_text(encoding="utf-8")

    surface_payload = json.dumps({
        "x_feature": surface["x_feature"],
        "y_feature": surface["y_feature"],
        "x_vals": surface["x_vals"],
        "y_vals": surface["y_vals"],
        "z": surface["z"],
        "std": surface["std"],
        "runs": surface["runs"],
        "fixed_recipe": surface["fixed_recipe"],
        "incumbent_basis": surface["incumbent_basis"],
    })

    proposals_payload = json.dumps(proposals)
    extra_pairs_payload = json.dumps(extra_pairs)

    x_name = surface["x_feature"]

    script = "\n".join([
        f'var SURFACE_DATA = {surface_payload};',
        f'var SURFACE_PROPOSALS = {proposals_payload};',
        f'var SURFACE_EXTRA_PAIRS = {extra_pairs_payload};',
        'var SURFACE_TARGET = "titer (mg/L)"',
        'var SURFACE_ROUND = 0',
        f'var SURFACE_N_RUNS = {len(surface["runs"])}',
        f'var SURFACE_INCUMBENT_VALUE = {incumbent:.4f}',
    ])

    body = ""  # No extra body content
    html = template.replace("${data_injection}", script)

    # Write to a temp file and open in browser
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        f.write(html)
        path = f.name

    print(f"Response surface written to: {path}")
    print(f"  Sweep: {x_name} × {surface['y_feature']}")
    print("  Grid: 21×21")
    print(f"  Observed runs: {len(surface['runs'])}")
    print(f"  Proposals: {len(proposals)}")
    print(f"  Incumbent: {incumbent:.4f}")
    print(f"  Open with: open {path} (macOS) or xdg-open {path} (Linux)")
    print()

    # Try to open in browser automatically
    try:
        url = "file://" + os.path.abspath(path)
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif sys.platform == "linux":
            subprocess.Popen(["xdg-open", path])
        else:
            # Windows
            import webbrowser
            webbrowser.open(url)
        print("Browser opened.")
    except Exception as e:
        print(f"Could not open browser: {e}")
        print("Open the file manually to see the widget.")


if __name__ == "__main__":
    main()
