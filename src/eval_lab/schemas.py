"""Typed trajectories and metric results. Definitions live in docs/METRICS.md."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

METRIC_VERSION = "1"
WORLD = "world"


class Status(StrEnum):
    OK = "OK"
    UNASSESSABLE = "UNASSESSABLE"


@dataclass(frozen=True)
class MetricResult:
    name: str
    unit: str
    status: Status
    value: float | None = None
    numerator: float | None = None
    denominator: float | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status is Status.OK


def assessed(name: str, unit: str, numerator: float, denominator: float) -> MetricResult:
    return MetricResult(
        name,
        unit,
        Status.OK,
        float(numerator) / float(denominator),
        float(numerator),
        float(denominator),
    )


def unassessable(name: str, unit: str, reason: str) -> MetricResult:
    return MetricResult(name, unit, Status.UNASSESSABLE, reason=reason)


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap to (-pi, pi]."""
    return np.pi - (np.pi - np.asarray(a)) % (2 * np.pi)


@dataclass(frozen=True, eq=False)
class Trajectory:
    """N samples of pose. Bad data is representable on purpose: metrics must see it and refuse.

    Wrong shapes raise (a programming error). NaNs, gaps and disordered time are data, and they are
    reported by problems().
    """

    t_s: np.ndarray
    xy_m: np.ndarray
    yaw_rad: np.ndarray
    frame: str = WORLD
    valid: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.t_s)
        if self.xy_m.shape != (n, 2) or self.yaw_rad.shape != (n,):
            raise ValueError(f"bad shapes: t={n}, xy={self.xy_m.shape}, yaw={self.yaw_rad.shape}")
        if self.valid is not None and self.valid.shape != (n,):
            raise ValueError("valid mask must match t_s")
        if not (self.frame == WORLD or self.frame.startswith("ego:")):
            raise ValueError(f"unknown frame {self.frame!r}")

    @property
    def mask(self) -> np.ndarray:
        """True where a sample is usable: not masked and fully finite."""
        declared = np.ones(len(self.t_s), bool) if self.valid is None else self.valid
        finite = (
            np.isfinite(self.t_s) & np.isfinite(self.xy_m).all(axis=1) & np.isfinite(self.yaw_rad)
        )
        return declared & finite

    def problems(self) -> list[str]:
        out = []
        if len(self.t_s) == 0:
            return ["empty trajectory"]
        if not np.isfinite(self.t_s).all():
            out.append("non-finite timestamp")
        elif (np.diff(self.t_s) <= 0).any():
            out.append("timestamps not strictly increasing")
        if not self.mask.all():
            out.append(f"{int((~self.mask).sum())} missing or invalid samples")
        return out


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float


def to_ego(traj: Trajectory, pose: Pose, frame_id: str) -> Trajectory:
    """World to ego frame of `pose`."""
    if traj.frame != WORLD:
        raise ValueError(f"expected world frame, got {traj.frame}")
    c, s = np.cos(pose.yaw), np.sin(pose.yaw)
    d = traj.xy_m - [pose.x, pose.y]
    xy = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)
    yaw = np.asarray(wrap_angle(traj.yaw_rad - pose.yaw))
    return Trajectory(traj.t_s, xy, yaw, f"ego:{frame_id}", traj.valid)


def to_world(traj: Trajectory, pose: Pose) -> Trajectory:
    """Ego frame of `pose` back to world."""
    if not traj.frame.startswith("ego:"):
        raise ValueError(f"expected ego frame, got {traj.frame}")
    c, s = np.cos(pose.yaw), np.sin(pose.yaw)
    x, y = traj.xy_m[:, 0], traj.xy_m[:, 1]
    xy = np.stack([pose.x + c * x - s * y, pose.y + s * x + c * y], axis=1)
    yaw = np.asarray(wrap_angle(traj.yaw_rad + pose.yaw))
    return Trajectory(traj.t_s, xy, yaw, WORLD, traj.valid)
