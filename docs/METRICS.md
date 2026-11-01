# Metric definitions (v1)

Written before any model or simulator code. `METRIC_VERSION` in `src/eval_lab/schemas.py` changes
whenever a definition here changes, and it is stamped on every result.

## Units and frames

| Quantity | Unit | Convention |
|---|---|---|
| time | s | strictly increasing; `t_s[0]` is the first *future* sample of a prediction |
| position | m | |
| yaw | rad | counter-clockwise from +x, wrapped to (-pi, pi] |
| speed | m/s | non-negative in the kinematic fixture |
| acceleration / jerk | m/s^2, m/s^3 | |

Frames: `world`, or `ego:<id>`, a right-handed frame with origin at the reference vehicle, +x forward
and +y left. Two trajectories are compared only when their frame strings are equal. A mismatch is
UNASSESSABLE, never silently transformed.

## The rule that overrides the others

Missing or invalid data yields **UNASSESSABLE**. It is never a pass, never a zero and never a failure.
It stays visible in every denominator table.

## Open-loop metrics (a prediction compared with a logged future)

`H` is the horizon in seconds, measured from the first ground-truth sample.

| Metric | Definition | Denominator |
|---|---|---|
| ADE@H | mean L2 distance over all ground-truth samples in `[t0, t0+H]` | samples in the window |
| FDE@H | L2 distance at the last sample of that window | 1 |

UNASSESSABLE when: frames differ; timestamps differ by more than 1e-6 s; any sample in the window is
masked or non-finite in either trajectory; or the ground truth does not reach `t0+H`. A truncated
ground truth is rejected because it would flatter a prediction by dropping its largest errors.

## Closed-loop metrics (one episode)

| Metric | Definition | Unit |
|---|---|---|
| collision | oriented ego box overlaps an actor box at any logged step (separating-axis test) | 0/1 |
| off-road | ego centre has lateral distance from the route centreline > road half width at any step | 0/1 |
| completion | route station reaches the goal station within the time budget | 0/1 |
| safe completion | completion and no collision and no off-road. **Primary endpoint** | 0/1 |
| distance | sum of step displacements | m |
| peak longitudinal accel / jerk | max absolute first / second finite difference of speed | m/s^2, m/s^3 |
| peak lateral accel | max of speed times absolute yaw rate | m/s^2 |

Collision is computed from the trace by this repository's own geometry. The simulator's flags are
retained as a cross-check and are not the source of truth. An unknown actor pose makes "no collision"
UNASSESSABLE, while a collision observed at a known step still counts.

## Aggregates and their denominators

Every episode that was **planned** is in the denominator, including timeouts, crashed runs and
unassessable episodes.

| Aggregate | Numerator | Denominator |
|---|---|---|
| safe completion rate | safe completions | planned episodes |
| completion rate | completions | planned episodes |
| collisions per episode | observed collisions | planned episodes |
| worst-case collisions per episode | observed collisions + unassessable + failed | planned episodes |
| collisions per km | observed collisions | total assessed km |
| off-road rate | off-road episodes | planned episodes |

Why both per-episode and per-km: a policy that never moves has zero collisions per episode, which looks
perfect, and an undefined per-km rate. Collisions per km on zero km is UNASSESSABLE, and safe completion
is 0 because it timed out. A stationary policy therefore cannot look superior on the primary endpoint.

Why per-km alone is also not enough: a policy that drives quickly into a crash has a small denominator
inflating its rate while a policy that survives has a large one, so distance exposure differs by policy.

## Uncertainty

Episodes sharing a scenario are correlated, so the independent unit is the **scenario**, not the
episode. Confidence intervals resample scenarios (`src/eval_lab/statistics.py`). Observing zero events
gives an upper bound (Clopper-Pearson on scenarios), never a claim of zero.

## What these metrics cannot tell you

A simulation pass is not a real-world or fleet-safety guarantee. These are fixture and simulator
measurements of toy policies. Nothing here measures a real vehicle.
