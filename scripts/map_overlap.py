"""Do test or dev MetaDrive routes retrace training roads? Geometry only, never outcomes.

Measures, per family, the share of a route within 2 m of its single most similar training route. The
baseline is the same measure between training maps of different families, which shows how much overlap
unrelated maps have anyway: every map starts at the same origin. Writes runs/analysis/map_overlap.json.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from eval_lab.adapters.metadrive import MetaDriveSim
from eval_lab.scenarios import TRAIN, build_metadrive_manifest

sim = MetaDriveSim()
routes = {}
for s in sorted(build_metadrive_manifest(12), key=lambda v: v.family):
    sim.reset(s, 0)
    routes[s.id] = (s, sim.route[: int(sim.goal_s)].copy())
sim.close()


def best_share(route: np.ndarray, others: list[np.ndarray]) -> float:
    return max(float((cKDTree(t).query(route)[0] < 2.0).mean()) for t in others)


train = [(s, r) for s, r in routes.values() if s.split == TRAIN]
out = {}
for family in sorted({s.family for s, _ in routes.values()}):
    mine = [(s, r) for s, r in routes.values() if s.family == family]
    if mine[0][0].split == TRAIN:  # baseline: other training family only
        others = [r for s, r in train if s.family != family]
        label = "baseline_train_vs_other_train"
    else:
        others = [r for _, r in train]
        label = f"{mine[0][0].split}_vs_train"
    shares = [best_share(r, others) for _, r in mine]
    out[family] = {
        "kind": label,
        "mean": round(float(np.mean(shares)), 3),
        "max": round(max(shares), 3),
    }
Path("runs/analysis").mkdir(exist_ok=True)
Path("runs/analysis/map_overlap.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=1))
