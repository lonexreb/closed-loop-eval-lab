# closed-loop-eval-lab

An auditable evaluation toolkit that tests whether better open-loop trajectory scores predict better
closed-loop behavior. It uses only generated scenarios and public simulator maps.

**New here? Read [MINI_BOOK.md](MINI_BOOK.md).** It explains everything in plain language.

## Result in one table

Safe completion on held-out families, for the model chosen by each rule on development data only:

| World | Chosen by best ADE | Chosen by closed-loop result | Paired difference (95% CI) |
|---|---|---|---|
| Kinematic fixture (toy), 7,200 episodes | 0.002 | 0.467 | +0.465 (+0.392, +0.545) |
| MetaDrive 0.4.3, 864 episodes | 0.000 | 0.889 | +0.889 (+0.778, +0.972) |

These are pilot numbers. The harder M2 study below changed part of the story, so read both tables.

## M2 study: new families, chosen to be further from training

Pre-registered in [experiments/protocol_m2.md](experiments/protocol_m2.md). Full results are in
[reports/study.md](reports/study.md).

| World | Closed-loop pick | Clean-ADE pick | Difference (95% CI) | Disturbed-ADE pick | Difference (95% CI) |
|---|---|---|---|---|---|
| Kinematic fixture (toy), 14,400 episodes | 0.469 | 0.010 | +0.459 (+0.403, +0.517) | 0.428 | +0.041 (-0.018, +0.107) |
| MetaDrive 0.4.3, 3,456 episodes | 0.156 | 0.000 | +0.156 (+0.083, +0.240) | 0.007 | +0.149 (+0.080, +0.229) |

- Clean-log ADE selection lost again in both worlds.
- Choosing by ADE on disturbed logs nearly matched closed-loop selection on the fixture, so that pilot claim did not
  replicate there.
- The pilot's backwards ranking did not replicate on MetaDrive: rho was -0.708, the right direction.
- On roads further from training, every learned model did badly (best 0.156), while `rule_center` scored 1.000.
- If families rather than scenarios are the unit, both MetaDrive intervals reach 0 (post-hoc).

Mined regression packs beat random ones at the registered test, by +0.141 at budget 16, but one regression supplies
most of that gap. Dev validation also missed a large regression that only held-out families exposed.

Development happened in a private repository. Run manifests cite its commit hashes, so this public history cannot
itself show that protocols came before results.

These are toy policies in simulation, and nothing here measures a real vehicle. Pilot details are in
[reports/pilot.md](reports/pilot.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is pinned because MetaDrive 0.4.3 needs below 3.12.

```bash
uv sync --extra sim
.venv/bin/python -m pytest tests -q
.venv/bin/python -m eval_lab.analytic reports/analytic.md
.venv/bin/python -m eval_lab.report --out reports/pilot.md
```

To reproduce runs, use the commands below on a clean tree. `final` stages refuse a dirty tree or an existing run.

```bash
.venv/bin/python -m eval_lab.experiment --world kinematic --stage dev
.venv/bin/python -m eval_lab.experiment --world kinematic --stage final
.venv/bin/python -m eval_lab.experiment --world metadrive --stage dev --suffix=-v2
.venv/bin/python -m eval_lab.experiment --world metadrive --stage final --suffix=-v2
```

## Layout

| Path | What it is |
|---|---|
| `docs/METRICS.md` | Definitions, units, frames and denominators |
| `experiments/protocol.md`, `amendment_1.md` | The analysis plan, and every later deviation |
| `src/eval_lab/metrics/` | ADE/FDE, collision, off-road, completion, comfort |
| `src/eval_lab/adapters/` | Toy fixture and real MetaDrive, behind one interface |
| `src/eval_lab/statistics.py` | Scenario-clustered intervals and paired comparisons |
| `src/eval_lab/mining.py` | Dev-only failure mining with lineage |
| `reports/pilot.md` | Generated results. Do not edit by hand |
| `runs/` | Raw evidence and manifests. `traces.npz` files stay local because of size |
| `CLAIMS_LEDGER.md` | What can and cannot be claimed |
