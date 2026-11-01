"""M0: fixtures with answers worked out by hand, plus a report comparing them with the code.

Run: python -m eval_lab.analytic reports/analytic.md
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from eval_lab.metrics import COLLISION, COMPLETED, TIMEOUT, Trace, ade, fde, score_episode
from eval_lab.schemas import METRIC_VERSION, Trajectory

T = np.arange(0.5, 3.01, 0.5)  # 6 samples over a 2.5 s window


def straight(speed: float = 10.0, lateral: float = 0.0) -> Trajectory:
    return Trajectory(T, np.stack([speed * T, np.full_like(T, lateral)], 1), np.zeros_like(T))


def drive_trace(actor_x: float | None, speed: float = 10.0, steps: int = 101) -> Trace:
    """Straight drive along +x at constant speed, with one parked car or none."""
    t = np.arange(steps) * 0.1
    ego = Trajectory(t, np.stack([speed * t, np.zeros(steps)], 1), np.zeros(steps))
    if actor_x is None:
        actors = np.zeros((steps, 0, 5))
    else:
        actors = np.tile([[actor_x, 0.0, 0.0, 4.5, 1.9]], (steps, 1, 1))
    route = np.array([[0.0, 0.0], [200.0, 0.0]])
    return Trace(ego, np.full(steps, speed), actors, route, 3.5, 90.0, 20.0)


def cases() -> list[tuple[str, str, object, object]]:
    """(case, metric, hand-calculated expectation, code result)."""
    gt = straight()
    missing = Trajectory(gt.t_s, gt.xy_m, gt.yaw_rad, valid=np.array([1, 1, 0, 1, 1, 1], bool))
    gap = Trajectory(gt.t_s + np.array([0, 0, 0.2, 0, 0, 0]), gt.xy_m, gt.yaw_rad)
    rows: list[tuple[str, str, object, object]] = [
        ("zero error", "ADE@2.5", 0.0, ade(gt, gt, 2.5).value),
        ("zero error", "FDE@2.5", 0.0, fde(gt, gt, 2.5).value),
        # A constant 1 m lateral shift is 1 m at every sample.
        ("1 m lateral offset", "ADE@2.5", 1.0, ade(straight(lateral=1.0), gt, 2.5).value),
        ("1 m lateral offset", "FDE@2.5", 1.0, fde(straight(lateral=1.0), gt, 2.5).value),
        # 11 vs 10 m/s: errors 0.5, 1, ... 3 m. Mean of 0.5k for k=1..6 is 1.75, last is 3.
        ("1 m/s speed error", "ADE@2.5", 1.75, ade(straight(11.0), gt, 2.5).value),
        ("1 m/s speed error", "FDE@2.5", 3.0, fde(straight(11.0), gt, 2.5).value),
        ("masked sample", "ADE@2.5", "UNASSESSABLE", ade(missing, gt, 2.5).status.value),
        ("shifted timestamp", "ADE@2.5", "UNASSESSABLE", ade(gap, gt, 2.5).status.value),
    ]
    # Parked car centre x=50, length 4.5 on both: boxes meet when ego x reaches 45.5, which is
    # step 46 (x=46.0) on a 1 m grid at 10 m/s, 0.1 s.
    hit = score_episode(drive_trace(50.0))
    rows += [
        ("parked car", "outcome", COLLISION, hit.outcome),
        ("parked car", "first collision step", 46, hit.collision_step),
    ]
    # 90 m goal at 10 m/s, no actors: reaches 100 m in 10 s, all inside the 20 s budget.
    clear = score_episode(drive_trace(None))
    rows += [
        ("clear road", "outcome", COMPLETED, clear.outcome),
        ("clear road", "distance_m", 100.0, clear.distance_m),
    ]
    # Never moves: no collision, cannot complete, so it is a timeout rather than a success.
    still = score_episode(drive_trace(None, speed=0.0))
    rows += [
        ("stationary", "outcome", TIMEOUT, still.outcome),
        ("stationary", "safe_completed", False, still.safe_completed),
    ]
    return rows


def main(out: str) -> int:
    rows = cases()
    lines = [
        "# Analytic metric check (M0)",
        "",
        f"Metric version {METRIC_VERSION}. Expected values are hand calculations, shown as comments"
        " in `src/eval_lab/analytic.py`. No simulator or model is involved.",
        "",
        "| Case | Metric | Expected | Measured | Match |",
        "|---|---|---|---|---|",
    ]
    bad = 0
    for case, metric, want, got in rows:
        same = (
            math.isclose(want, float(str(got)), abs_tol=1e-9)
            if isinstance(want, float)
            else want == got
        )
        bad += not same
        lines.append(f"| {case} | {metric} | {want} | {got} | {'yes' if same else 'NO'} |")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"{len(rows) - bad}/{len(rows)} analytic cases match; wrote {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "reports/analytic.md"))
