"""Candidate policies. Every one outputs the same plan and is driven by the same tracker.

A plan is 6 ego-frame waypoints, at 0.5 s to 3.0 s ahead. That makes the open-loop score (plan vs
logged future) and the closed-loop score (tracked plan in the world) measure one object, not two.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from eval_lab.adapters.base import Action, WorldState

PLAN_T = np.arange(1, 7) * 0.5
WHEELBASE_M = 2.7
HISTORY_STEPS = 10  # 1.0 s at 0.1 s per step


class Policy(Protocol):
    name: str

    def reset(self) -> None: ...

    def plan(self, st: WorldState) -> np.ndarray: ...


def to_ego_xy(st: WorldState, xy: np.ndarray) -> np.ndarray:
    c, s = np.cos(st.yaw), np.sin(st.yaw)
    d = np.atleast_2d(xy) - [st.x, st.y]
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)


def track(plan: np.ndarray, st: WorldState) -> Action:
    """The one fixed low-level controller. Tracks a plan; it never sees which policy made it."""
    target = plan[1] if np.hypot(*plan[1]) > 3.0 else plan[-1]
    d2 = float(target @ target)
    # A point beside or behind the car gives no usable pure-pursuit arc: hold the wheel straight.
    steer = np.arctan2(2 * WHEELBASE_M * target[1], d2) if target[0] > 1.0 else 0.0
    speed_cmd = float(np.hypot(*plan[0])) / 0.5
    return Action(1.5 * (speed_cmd - st.speed), float(steer))


def safe_speed(st: WorldState) -> float:
    """Speed the expert aims for: limit, curve comfort and an IDM-style gap term."""
    d = np.diff(np.vstack([[st.x, st.y], st.ref]), axis=0)
    yaws = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    seg = np.linalg.norm(d, axis=1)
    kappa = np.abs(np.diff(yaws)) / np.maximum((seg[:-1] + seg[1:]) / 2, 1e-3)
    curve = np.sqrt(2.5 / max(float(kappa.max()), 1e-4))
    v = min(st.speed_limit, curve)
    if st.gap_m is not None and st.lead_speed is not None:
        wanted_gap = 5.0 + 1.5 * st.speed
        v = min(v, max(0.0, st.lead_speed + 0.5 * (st.gap_m - wanted_gap)))
    return float(v)


def rule_plan(st: WorldState, offset_m: float = 0.0) -> np.ndarray:
    """Waypoints along the centreline ahead, at the safe speed, optionally shifted sideways."""
    ref = to_ego_xy(st, st.ref)
    pts = np.vstack([[0.0, 0.0], ref])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    # Known limit: interp clamps at the last reference, 40 m, which shortens plans above 13.3 m/s.
    want = safe_speed(st) * PLAN_T
    plan = np.stack([np.interp(want, s, pts[:, 0]), np.interp(want, s, pts[:, 1])], axis=1)
    plan[:, 1] += offset_m
    return plan


def expert_action(st: WorldState) -> Action:
    """Hand-written driver that generates the logs. Uses rule_plan with a stronger speed loop."""
    return expert_track(rule_plan(st), st)


def expert_track(plan: np.ndarray, st: WorldState) -> Action:
    """The tracker that made the logs: `track` steering with the expert's stronger speed loop."""
    target_speed = float(np.hypot(*plan[0])) / 0.5
    steer = track(plan, st).steer_rad
    return Action(float(np.clip(2.0 * (target_speed - st.speed), -6.0, 2.0)), steer)


@dataclass
class RuleCenter:
    name: str = "rule_center"

    def reset(self) -> None:
        pass

    def plan(self, st: WorldState) -> np.ndarray:
        return rule_plan(st)


@dataclass
class RuleOffset:
    """Built on purpose: a fixed sideways bias costs ADE and is harmless on a wide enough road."""

    offset_m: float = 0.4
    name: str = "rule_offset"

    def reset(self) -> None:
        pass

    def plan(self, st: WorldState) -> np.ndarray:
        return rule_plan(st, self.offset_m)


@dataclass
class ConstantVelocity:
    name: str = "constant_velocity"

    def reset(self) -> None:
        pass

    def plan(self, st: WorldState) -> np.ndarray:
        return np.stack([st.speed * PLAN_T, np.zeros_like(PLAN_T)], axis=1)


@dataclass
class Stationary:
    name: str = "stationary"

    def reset(self) -> None:
        pass

    def plan(self, st: WorldState) -> np.ndarray:
        return np.zeros((len(PLAN_T), 2))


def base_features(st: WorldState) -> np.ndarray:
    ref = to_ego_xy(st, st.ref).ravel()
    gap = 80.0 if st.gap_m is None else min(st.gap_m, 80.0)
    closing = 0.0 if st.lead_speed is None else st.speed - st.lead_speed
    return np.concatenate(
        [ref, [st.speed, st.lateral_m, st.heading_error, st.speed_limit, gap, closing]]
    )


def history_features(past: deque, st: WorldState) -> np.ndarray:
    """Ego motion over the last second, in the current ego frame."""
    a, b = past[len(past) // 2], past[0]
    return np.concatenate(
        [to_ego_xy(st, np.array([[a.x, a.y], [b.x, b.y]])).ravel(), [a.speed, b.speed]]
    )


def expand(x: np.ndarray, degree: int) -> np.ndarray:
    if degree == 1:
        return x
    iu = np.triu_indices(x.shape[-1])
    return np.concatenate([x, x[..., iu[0]] * x[..., iu[1]]], axis=-1)


@dataclass
class Ridge:
    """Multi-output ridge regression from features to waypoints. Small, exact and inspectable."""

    name: str
    use_history: bool
    degree: int
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    lam: float

    def reset(self) -> None:
        self.past: deque = deque(maxlen=HISTORY_STEPS + 1)

    def plan(self, st: WorldState) -> np.ndarray:
        self.past.append(st)
        x = base_features(st)
        if self.use_history:
            x = np.concatenate([x, history_features(self.past, st)])
        z = (expand(x, self.degree) - self.mean) / self.scale
        return (np.append(z, 1.0) @ self.weights).reshape(len(PLAN_T), 2)


def fit_ridge(
    name: str, x: np.ndarray, y: np.ndarray, use_history: bool, degree: int, lam: float
) -> Ridge:
    """Closed form. Scaling statistics come from the training rows only."""
    phi = expand(x, degree)
    mean, scale = phi.mean(0), phi.std(0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    z = np.column_stack([(phi - mean) / scale, np.ones(len(phi))])
    reg = lam * len(z) * np.eye(z.shape[1])
    reg[-1, -1] = 0.0  # never shrink the intercept
    weights = np.linalg.solve(z.T @ z + reg, z.T @ y.reshape(len(y), -1))
    return Ridge(name, use_history, degree, mean, scale, weights, lam)
