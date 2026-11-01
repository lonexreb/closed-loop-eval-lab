"""POST-HOC, added after seeing the final S3 table, so it is not part of the pre-registered protocol.

Question: when our trace says off-road and MetaDrive does not, which side did the car leave on? The
left side of a one-lane MetaDrive road is the opposite-direction lane, which still counts as on-lane.
Reads the retained final traces and rebuilds each route. Writes runs/analysis/s3_offroad_side.json.
"""

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from eval_lab.adapters.metadrive import HALF_WIDTH_M, MetaDriveSim
from eval_lab.metrics import station_lateral
from eval_lab.scenarios import read_manifest

run = Path("runs/metadrive-final-v2")
with gzip.open(run / "episodes.jsonl.gz", "rt") as f:
    rows = [json.loads(x) for x in f]
traces = np.load(run / "traces.npz")
scenarios = {s.id: s for s in read_manifest(Path("experiments/scenarios_metadrive.json"))}

sim = MetaDriveSim()
routes = {}
sides: Counter = Counter()
sim_only_collision = []
for r in sorted(rows, key=lambda v: (v["family"], v["scenario"])):
    score = r["score"]
    key = f"{r['policy']}|{r['scenario']}|{r['seed']}"
    if score["collision"] is not True and score["extra"].get("crash"):
        sim_only_collision.append({"episode": key, "outcome": score["outcome"]})
    disagree = score["offroad"] and not score["extra"].get("out_of_road")
    agree = score["offroad"] and score["extra"].get("out_of_road")
    if not (disagree or agree):
        continue
    if r["scenario"] not in routes:
        sim.reset(scenarios[r["scenario"]], 0)
        routes[r["scenario"]] = sim.route.copy()
    _, lat = station_lateral(routes[r["scenario"]], traces[key][:, 1:3].astype(float))
    first = lat[np.argmax(np.abs(lat) > HALF_WIDTH_M)]
    sides["trace_only" if disagree else "both", "left" if first > 0 else "right"] += 1
sim.close()
out = {
    "offroad_first_exit_side": {f"{a}:{b}": n for (a, b), n in sorted(sides.items())},
    "sim_only_collisions": sim_only_collision,
}
Path("runs/analysis").mkdir(exist_ok=True)
Path("runs/analysis/s3_offroad_side.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=1))
