"""Episode outcomes computed from the trace, and denominators that keep every planned run.

See docs/METRICS.md. A simulator's own flags are a cross-check, not the source of truth.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from eval_lab.schemas import METRIC_VERSION, WORLD, Status, Trajectory

# Episode outcomes. Exactly one per planned run.
COMPLETED = "COMPLETED"
COLLISION = "COLLISION"
OFFROAD = "OFFROAD"
TIMEOUT = "TIMEOUT"
FAILED = "FAILED"  # the run itself crashed
UNASSESSABLE = "UNASSESSABLE"  # the run produced telemetry that cannot be judged


@dataclass(frozen=True)
class Box:
    """Oriented rectangle: centre, yaw, full length along yaw, full width across it."""

    x: float
    y: float
    yaw: float
    length: float
    width: float

    def corners(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        fwd = np.array([c, s]) * self.length / 2
        left = np.array([-s, c]) * self.width / 2
        centre = np.array([self.x, self.y])
        return np.array(
            [centre + fwd + left, centre + fwd - left, centre - fwd - left, centre - fwd + left]
        )


def boxes_overlap(a: Box, b: Box) -> bool:
    """Separating-axis test. Touching edges do not count as an overlap."""
    ca, cb = a.corners(), b.corners()
    for yaw in (a.yaw, b.yaw):
        for axis in (np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])):
            pa, pb = ca @ axis, cb @ axis
            if pa.max() <= pb.min() or pb.max() <= pa.min():
                return False
    return True


def station_lateral(route: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project points on a polyline. Returns station (m along) and signed lateral (m, left +)."""
    a, b = route[:-1], route[1:]
    seg = b - a
    seg_len = np.linalg.norm(seg, axis=1)
    keep = seg_len > 0
    a, seg, seg_len = a[keep], seg[keep], seg_len[keep]
    start_s = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])
    d = xy[:, None, :] - a[None, :, :]
    u = np.clip((d * seg).sum(-1) / seg_len**2, 0.0, 1.0)
    foot = a[None] + u[..., None] * seg[None]
    dist = np.linalg.norm(xy[:, None, :] - foot, axis=-1)
    i = dist.argmin(axis=1)
    rows = np.arange(len(xy))
    cross = seg[i, 0] * d[rows, i, 1] - seg[i, 1] * d[rows, i, 0]
    return start_s[i] + u[rows, i] * seg_len[i], np.sign(cross) * dist[rows, i]


@dataclass(frozen=True, eq=False)
class Trace:
    """What actually happened in one run. Enough to recompute every closed-loop metric."""

    ego: Trajectory
    speed_mps: np.ndarray
    actors: np.ndarray  # (N, K, 5): x, y, yaw, length, width. NaN means telemetry is unknown.
    route: np.ndarray  # (M, 2) centreline in world frame
    half_width_m: float
    goal_s_m: float
    time_budget_s: float
    ego_length_m: float = 4.5
    ego_width_m: float = 1.9


@dataclass
class EpisodeScore:
    outcome: str
    collision: bool | None  # None: cannot be determined
    offroad: bool | None
    completed: bool
    safe_completed: bool
    distance_m: float
    duration_s: float
    collision_step: int | None = None
    peak_long_accel: float | None = None
    peak_jerk: float | None = None
    peak_lat_accel: float | None = None
    reason: str = ""
    metric_version: str = METRIC_VERSION
    extra: dict = field(default_factory=dict)


def _first_collision(trace: Trace) -> tuple[int | None, bool]:
    """First colliding step, and whether any actor was unknown at a step with no collision."""
    unknown = False
    length, width = trace.ego_length_m, trace.ego_width_m
    for i in range(len(trace.ego.t_s)):
        row = trace.actors[i]
        if len(row) == 0:
            continue
        known = np.isfinite(row).all(axis=1)
        unknown = unknown or not known.all()
        x, y, yaw = trace.ego.xy_m[i, 0], trace.ego.xy_m[i, 1], trace.ego.yaw_rad[i]
        ego = Box(float(x), float(y), float(yaw), length, width)
        reach = np.hypot(length, width) / 2 + np.hypot(row[:, 3], row[:, 4]) / 2
        near = known & (np.hypot(row[:, 0] - x, row[:, 1] - y) < reach)
        for a in row[near]:
            if boxes_overlap(ego, Box(*(float(v) for v in a))):
                return i, unknown
    return None, unknown


def _comfort(t: np.ndarray, speed: np.ndarray, yaw: np.ndarray) -> tuple:
    if len(t) < 4:
        return None, None, None
    dt = np.diff(t)
    if np.ptp(dt) > 1e-6:  # finite differences below assume a uniform clock
        return None, None, None
    h = float(dt[0])
    accel = np.diff(speed) / h
    jerk = np.diff(accel) / h
    yaw_rate = np.diff(np.unwrap(yaw)) / h
    lat = np.abs(speed[:-1] * yaw_rate)
    return float(np.abs(accel).max()), float(np.abs(jerk).max()), float(lat.max())


def score_episode(trace: Trace) -> EpisodeScore:
    """One outcome from one trace. Bad telemetry gives UNASSESSABLE, not a pass."""
    ego = trace.ego
    problems = ego.problems()
    if ego.frame != WORLD:
        problems.append(f"ego trace must be world frame, got {ego.frame}")
    if len(trace.speed_mps) != len(ego.t_s) or not np.isfinite(trace.speed_mps).all():
        problems.append("speed telemetry missing or misaligned")
    if problems:
        return EpisodeScore(
            UNASSESSABLE, None, None, False, False, 0.0, 0.0, reason="; ".join(problems)
        )

    collide_at, unknown_actor = _first_collision(trace)
    collided = collide_at is not None
    s, lat = station_lateral(trace.route, ego.xy_m)
    offroad = bool((np.abs(lat) > trace.half_width_m).any())
    distance = float(np.linalg.norm(np.diff(ego.xy_m, axis=0), axis=1).sum())
    duration = float(ego.t_s[-1] - ego.t_s[0])
    in_time = ego.t_s - ego.t_s[0] <= trace.time_budget_s + 1e-9
    completed = bool((s[in_time] >= trace.goal_s_m).any())
    a, j, y = _comfort(ego.t_s, trace.speed_mps, ego.yaw_rad)
    collision: bool | None = True if collided else (None if unknown_actor else False)

    if collided:
        outcome = COLLISION
    elif unknown_actor:
        outcome = UNASSESSABLE
    elif offroad:
        outcome = OFFROAD
    elif completed:
        outcome = COMPLETED
    else:
        outcome = TIMEOUT
    return EpisodeScore(
        outcome,
        collision,
        offroad,
        completed,
        completed and collision is False and not offroad,
        distance,
        duration,
        collide_at,
        a,
        j,
        y,
        "unknown actor telemetry" if unknown_actor else "",
    )


def failed_episode(reason: str) -> EpisodeScore:
    """The run crashed. It stays in every denominator."""
    return EpisodeScore(FAILED, None, None, False, False, 0.0, 0.0, reason=reason)


def aggregate(scores: Iterable[EpisodeScore]) -> dict:
    """Denominators are planned episodes. A run that crashed still counts against the policy."""
    rows = list(scores)
    if not rows:
        return {"status": Status.UNASSESSABLE.value, "reason": "no episodes", "planned": 0}
    n = len(rows)
    collisions = sum(r.collision is True for r in rows)
    unknown = sum(r.collision is None for r in rows)
    km = sum(r.distance_m for r in rows) / 1000.0
    return {
        "status": Status.OK.value,
        "planned": n,
        "outcomes": {
            k: sum(r.outcome == k for r in rows)
            for k in (COMPLETED, COLLISION, OFFROAD, TIMEOUT, FAILED, UNASSESSABLE)
        },
        "safe_completion_rate": sum(r.safe_completed for r in rows) / n,
        "completion_rate": sum(r.completed for r in rows) / n,
        "collisions_per_episode": collisions / n,
        "worst_case_collisions_per_episode": (collisions + unknown) / n,
        "collisions_per_km": collisions / km if km > 0 else None,
        "offroad_rate": sum(r.offroad is True for r in rows) / n,
        "total_km": km,
    }
