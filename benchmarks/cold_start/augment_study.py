"""Gap-filling study: how should new runs be placed next to runs that already exist?

Criterion asked for: maximise the minimum distance to the measured points. The
risk of plain greedy maximin over a large candidate set is that it sends new
points to the corners of the box (poor uniformity: high discrepancy). Compared,
on [0,1]^d with existing points drawn uniformly (30 seeds per cell):

  maximin_full      greedy maximin over the full 1024-point scrambled-Sobol pool
  maximin_reflected the same, with a point's distance to a face counted double
  maximin_prefix    greedy maximin over a Sobol prefix of 4 * (existing + new)
                    points (what `kalos.core.design` ships)
  random            uniform new points
  greedy_cd         each new point minimises the centered L2 discrepancy

Scores: min distance from the new points to the existing ones (to_existing),
min distance over all pairs involving a new point (min_all), and the centered
L2 discrepancy of existing + new (CD; lower is more uniform).

    .venv/bin/python benchmarks/cold_start/augment_study.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import qmc

from kalos.core.design import ContinuousFactor, _greedy_maximin, _sobol, generate_design


def _cd(x: np.ndarray) -> float:
    return float(qmc.discrepancy(np.clip(x, 0, 1), method="CD"))


def _metrics(ex: np.ndarray, new: np.ndarray) -> tuple[float, float, float]:
    return (float(cdist(new, ex).min()),
            float(min(pdist(new).min(), cdist(new, ex).min())),
            _cd(np.vstack([ex, new])))


def _reflected(pool: np.ndarray, ex: np.ndarray, k: int) -> list[int]:
    face = 2.0 * np.minimum(pool, 1.0 - pool).min(axis=1)
    dmin = cdist(pool, ex).min(axis=1)
    out: list[int] = []
    for _ in range(k):
        score = np.minimum(dmin, face)
        score[out] = -np.inf
        i = int(np.argmax(score))
        out.append(i)
        dmin = np.minimum(dmin, np.linalg.norm(pool - pool[i], axis=1))
    return out


def main() -> dict:
    rows = []
    for d in (2, 4, 6, 10):
        for n_ex, n_new in ((3, 9), (10, 6)):
            acc: dict[str, list[tuple[float, float, float]]] = {
                k: [] for k in ("maximin_full", "maximin_reflected", "maximin_prefix", "random", "greedy_cd")}
            for s in range(30):
                rng = np.random.default_rng(s)
                ex = rng.random((n_ex, d))
                pool = _sobol(d, 1024, s)
                acc["maximin_full"].append(_metrics(ex, pool[_greedy_maximin(pool, ex, n_new)]))
                acc["maximin_reflected"].append(_metrics(ex, pool[_reflected(pool, ex, n_new)]))
                res = generate_design([ContinuousFactor(f"x{j}", 0.0, 1.0) for j in range(d)],
                                      n=n_new, seed=s, existing=ex)
                new = np.array([[r[f"x{j}"] for j in range(d)] for r in res.rows])
                acc["maximin_prefix"].append(_metrics(ex, new))
                acc["random"].append(_metrics(ex, rng.random((n_new, d))))
                cur, pick = ex.copy(), []
                for _ in range(n_new):
                    best, bv = -1, np.inf
                    for i in range(0, 1024, 4):
                        if i in pick:
                            continue
                        v = _cd(np.vstack([cur, pool[i]]))
                        if v < bv:
                            bv, best = v, i
                    pick.append(best)
                    cur = np.vstack([cur, pool[best]])
                acc["greedy_cd"].append(_metrics(ex, pool[pick]))
            for k, v in acc.items():
                a = np.array(v)
                rows.append({"d": d, "n_existing": n_ex, "n_new": n_new, "method": k,
                             "to_existing": round(float(a[:, 0].mean()), 3),
                             "min_all": round(float(a[:, 1].mean()), 3),
                             "cd_with_existing": round(float(a[:, 2].mean()), 3)})
    return {"seeds": 30, "rows": rows}


if __name__ == "__main__":
    out = main()
    path = Path(__file__).resolve().parent / "results" / "augment_study.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("| d | existing+new | method | to_existing | min_all | CD with existing |")
    print("|---|---|---|---|---|---|")
    for r in out["rows"]:
        print(f"| {r['d']} | {r['n_existing']}+{r['n_new']} | {r['method']} | {r['to_existing']} | "
              f"{r['min_all']} | {r['cd_with_existing']} |")
