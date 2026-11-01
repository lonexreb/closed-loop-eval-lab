# Decisions

Reversible choices, with what would change each one.

| Decision | Why | Revisit when |
|---|---|---|
| Python 3.11 | `metadrive-simulator` 0.4.3 requires `<3.12` | MetaDrive supports newer Python |
| MetaDrive is an optional extra | The fixture and metrics must run without a simulator | Never needed for core metrics |
| Ridge regression policies, not neural networks | Exact, inspectable and CPU-fast, and the question is about evaluation rather than model quality | A study needs realistic planner capacity |
| Every policy emits a plan and shares one tracker | Open-loop and closed-loop scores then judge one object, and differences cannot come from different controllers | Comparing control stacks themselves |
| Disjoint scenario **families** across splits | Scenarios within a family share road shape and hazard, so per-scenario splits leak | A public dataset defines its own splits |
| Scenario is the independent unit for CIs | Seeds within a scenario are correlated | Scenario families become numerous enough to cluster on |
| Safe completion is the primary endpoint | A stationary policy cannot win on it | A release decision needs a different risk weighting |
| First second is expert-driven for every policy | History models were trained with one second of history. Without this they start in a state they have never seen, which was unfair and which I found in my own pipeline | Policies do not use history |
| Collision and off-road come from our trace geometry | One definition across both worlds. MetaDrive flags are kept as a cross-check | Simulator flags are required by a downstream consumer |
| MetaDrive early stop on road departure is off; off-road is the vehicle centre past the 2.0 m road edge, as in the fixture | MetaDrive stops when a wheel crosses a line, about 1.07 m from centre, before a centre test can fire. Dev S3 showed 245 sim-only off-roads, all mislabelled as timeouts | Wheel-based definition adopted in both worlds |
| Traces are saved as float32 | Compact, and precise enough for positions under a few km | Sub-millimetre analysis |
| GitHub Actions enabled after M2 | The token first lacked the `workflow` scope, so pilot and M2 checks ran locally. The scope arrived on 2026-11-01 and the parked config moved to `.github/workflows/ci.yml` | Not applicable |

| M2 uses new held-out families instead of the pilot test families | Pilot test outcomes had been seen, so tuning regressions afterwards and reusing them would leak | Not applicable |
| M2 MetaDrive families chosen by geometry and expert only | Maps opening with S share the training start and reached 0.75 overlap. Choosing without outcomes keeps the split clean | A better distance measure exists |
| Scenario-level CIs kept as registered, family-level added as post-hoc | Changing the registered unit after seeing results would be a deviation. Family CIs are shown beside them | The next protocol registers family CIs |
| Commit `789b8b6` kept as the run revision despite one over-long comment | A piped lint check failed to stop the commit. The later fix touches only that comment, and all three held-out runs are clean at `789b8b6` | Not applicable |
| Post-run commits add reporting only | `study_report.py`, tests and docs came after the runs. Run code is unchanged except that comment and formatting of it | Not applicable |

## Measured simulator behavior

- **Fixture determinism:** bit-exact for the same scenario and seed (asserted in tests).
- **MetaDrive determinism:** the same scenario and seed gives the same outcome and length, with position
  drift of up to about 0.2 mm across repeats and after visiting another map. It is not bit-exact. Drift
  values fall on float32 steps, which points at float32 physics state carried across resets. A rerun can
  therefore differ in the last digits, and a truly marginal outcome could in principle flip.
- **MetaDrive startup:** measured 0.8 s after assets are cached, then about 3,000 steps/s. The first load in
  a fresh process took far longer.

## Fixture bugs found and fixed on train and dev families only

Each was caught because the results looked wrong for a careful driver, and each now has a regression test.

1. The episode ended one metre before the goal, so finished roads scored as timeouts.
2. Actors clamped at the end of the road while still reporting their speed, so followers hit them.
3. A lead that stopped for good blocked the single lane for every policy.
4. Pure pursuit went to full lock when a plan collapsed to a sideways point at low speed.
5. On MetaDrive the goal sat 6 m from the route end, so plans collapsed and the driver braked at
   12 to 15 m/s^2 for no reason.
6. MetaDrive ended runs on a wheel crossing, so cars leaving the road were scored as timeouts. Found by
   the S3 cross-check on dev data.

## Found after the first final runs, by independent review

7. MetaDrive map seeds repeated across families, so held-out maps retraced training roads. Seeds are now separate per
   family. `metadrive-final` is kept and superseded. See `experiments/amendment_1.md`.
8. The false-alarm control was identical to the baseline, so it could not fire. It now uses different seeds.
9. S1 compared different seeds, the P1 bootstrap resampled axes separately, and open-loop and closed-loop used different
   disturbance. Each is fixed or measured in amendment 1.
