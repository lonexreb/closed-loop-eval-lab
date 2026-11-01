"""Check the M2 families before freezing them, using the expert only and spare draws.

Fixture: spare manifest seed. MetaDrive: map seeds 900 and up, which the held-out manifest cannot use.
No candidate policy runs, so nothing here reveals how a candidate scores. Map overlap with training
uses the real held-out maps, but only their geometry. Writes runs/analysis/m2_feasibility.json.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from eval_lab.adapters.kinematic import KinematicSim
from eval_lab.adapters.metadrive import MetaDriveSim
from eval_lab.metrics import score_episode
from eval_lab.runner import expert_rollout
from eval_lab.scenarios import (
    MD_M2_FAMILIES,
    TRAIN,
    Scenario,
    build_m2_manifest,
    build_metadrive_m2_manifest,
    build_metadrive_manifest,
)

SPARE_SEED = 777


def expert_safe(sim, scenario, seed):
    states = expert_rollout(sim, scenario, seed)
    return bool(score_episode(sim.trace(states)).safe_completed)


out = {"kinematic_expert_safe": {}, "metadrive_expert_safe": {}, "metadrive_overlap": {}}
sim = KinematicSim(disturbance=1.0)
spare = build_m2_manifest(30, SPARE_SEED)
for family in sorted({s.family for s in spare}):
    ok = [expert_safe(sim, s, x) for s in spare if s.family == family for x in (0, 1, 2)]
    out["kinematic_expert_safe"][family] = round(float(np.mean(ok)), 3)

md = MetaDriveSim(disturbance=1.0)
for n, family in enumerate(MD_M2_FAMILIES):
    probes = [Scenario(f"{family}-{900 + 10 * n + i}", family, "probe", (), 2.0, 10.0) for i in range(6)]
    out["metadrive_expert_safe"][family] = round(float(np.mean([expert_safe(md, s, 0) for s in probes])), 3)


def route_of(s):
    md.reset(s, 0)
    return md.route[: int(md.goal_s)].copy()


train = [route_of(s) for s in sorted(build_metadrive_manifest(12), key=lambda v: v.family) if s.split == TRAIN]
for s in build_metadrive_m2_manifest(12):
    r = route_of(s)
    share = max(float((cKDTree(t).query(r)[0] < 2.0).mean()) for t in train)
    out["metadrive_overlap"].setdefault(s.family, []).append(share)
md.close()
out["metadrive_overlap"] = {
    k: {"mean": round(float(np.mean(v)), 3), "max": round(max(v), 3)} for k, v in out["metadrive_overlap"].items()
}
Path("runs/analysis").mkdir(exist_ok=True)
Path("runs/analysis/m2_feasibility.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=1))
