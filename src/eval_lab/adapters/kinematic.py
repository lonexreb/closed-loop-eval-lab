"""TOY FIXTURE: a kinematic bicycle on a curvature-defined road, with scripted non-reactive actors.

This is deterministic, fast and hand-checkable. It is not a physics simulator. It has no tyres, no
sensors and no reactive traffic, and results from it are labelled `kinematic_fixture` everywhere.
"""

from __future__ import annotations

import zlib

import numpy as np

from eval_lab.adapters.base import DT_S, REF_STATIONS_M, Action, WorldState, lead_on_route
from eval_lab.metrics import Box, Trace, boxes_overlap
from eval_lab.scenarios import ActorSpec, Scenario
from eval_lab.schemas import Trajectory, wrap_angle

BACKEND = "kinematic_fixture"
WHEELBASE_M = 2.7
LENGTH_M, WIDTH_M = 4.5, 1.9
ACCEL_LIMITS = (-6.0, 3.0)
STEER_LIMIT = 0.6
STEER_RATE_LIMIT = 0.6  # rad/s
STEP_M = 1.0
RUNOUT_M = 80.0  # straight road past the goal, so plans and actors do not hit a world edge
BRAKE_HOLD_S = 3.0
RESUME_MPS2 = 1.5


def route_polyline(segments: tuple[tuple[float, float], ...]) -> tuple[np.ndarray, np.ndarray]:
    """Integrate piecewise-constant curvature at STEP_M spacing. Returns points and tangent yaw."""
    kappa = np.concatenate([np.full(max(1, int(round(n / STEP_M))), k) for n, k in segments])
    yaw = np.concatenate([[0.0], np.cumsum(kappa * STEP_M)])
    mid = (yaw[:-1] + yaw[1:]) / 2  # midpoint rule is exact on a circle up to second order
    step = STEP_M * np.stack([np.cos(mid), np.sin(mid)], axis=1)
    return np.vstack([[0.0, 0.0], np.cumsum(step, axis=0)]), yaw


def _actor_track(spec: ActorSpec, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Station and lateral per step, integrated in closed form so it is independent of the ego."""
    s, v, lat = spec.s0_m, spec.v0_mps, spec.lateral_m
    stations, laterals, speeds = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    for i in range(steps):
        t = i * DT_S
        if spec.brake_at_s is not None and t >= spec.brake_at_s:
            # Brake, hold, then resume: a lead that stopped for good would block every policy.
            drop = (spec.v0_mps - spec.v_after_mps) / max(spec.brake_mps2, 1e-6)
            if t < spec.brake_at_s + max(drop, 0.0) + BRAKE_HOLD_S:
                v = max(spec.v_after_mps, v - spec.brake_mps2 * DT_S)
            else:
                v = min(spec.v0_mps, v + RESUME_MPS2 * DT_S)
        if spec.cut_in_at_s is not None:
            # Smooth two-second lane change towards the ego lane.
            u = np.clip((t - spec.cut_in_at_s) / 2.0, 0.0, 1.0)
            lat = spec.lateral_m * (1 - (3 * u**2 - 2 * u**3))
        stations[i], laterals[i], speeds[i] = s, lat, v
        s += v * DT_S
    return np.stack([stations, laterals]), speeds


class KinematicSim:
    """Reset with a scenario and seed, then step. Noise is fixed at reset, which pairs policies."""

    backend = BACKEND

    def __init__(self, disturbance: float = 1.0):
        self.disturbance = disturbance

    def reset(self, sc: Scenario, seed: int) -> WorldState:
        self.sc = sc
        self.route, self.route_yaw = route_polyline((*sc.segments, (RUNOUT_M, 0.0)))
        self.budget_s = 2.5 * sc.length_m / sc.speed_limit_mps
        self.max_steps = int(self.budget_s / DT_S) + 1
        rng = np.random.default_rng([zlib.crc32(sc.id.encode()), seed])
        # AR(1) steering disturbance, the same for every policy on this scenario and seed.
        white = rng.normal(0.0, 1.0, self.max_steps + 1)
        noise = np.zeros(self.max_steps + 1)
        for i in range(1, self.max_steps + 1):
            noise[i] = 0.9 * noise[i - 1] + np.sqrt(1 - 0.9**2) * white[i]
        self.noise = 0.02 * self.disturbance * noise
        lat0 = float(rng.normal(0.0, 0.2)) * self.disturbance
        err0 = float(rng.normal(0.0, 0.03)) * self.disturbance
        self.speed = sc.speed_limit_mps * float(rng.uniform(0.85, 1.0))

        c, s = np.cos(self.route_yaw[0]), np.sin(self.route_yaw[0])
        self.x, self.y = -s * lat0, c * lat0
        self.yaw = self.route_yaw[0] + err0
        self.steer = 0.0
        self.step_i = 0
        self.station_i = 0

        tracks = [_actor_track(a, self.max_steps + 1) for a in sc.actors]
        self.actor_arrays = self._actor_world(tracks)
        return self.observe()

    def _actor_world(self, tracks) -> np.ndarray:
        """(steps, K, 6): x, y, yaw, length, width, speed."""
        k = len(tracks)
        out = np.zeros((self.max_steps + 1, k, 6))
        for j, (pos, speeds) in enumerate(tracks):
            ds = np.clip(pos[0], 0, len(self.route) - 1)
            i = np.minimum(ds.astype(int), len(self.route) - 2)
            u = (ds - i)[:, None]
            base = self.route[i] * (1 - u) + self.route[i + 1] * u
            yaw = self.route_yaw[i]
            normal = np.stack([-np.sin(yaw), np.cos(yaw)], axis=1)
            xy = base + normal * pos[1][:, None]
            out[:, j] = np.column_stack(
                [xy, yaw, np.full_like(yaw, 4.5), np.full_like(yaw, 1.9), speeds]
            )
        return out

    def _project(self) -> tuple[float, float, int]:
        """Local nearest point on the route. Window search because ego moves about 1 m a step."""
        lo = max(0, self.station_i - 5)
        hi = min(len(self.route), self.station_i + 25)
        d = self.route[lo:hi] - [self.x, self.y]
        i = lo + int(np.argmin((d * d).sum(1)))
        i = min(i, len(self.route) - 2)
        self.station_i = i
        tangent = self.route[i + 1] - self.route[i]
        rel = np.array([self.x, self.y]) - self.route[i]
        along = float(rel @ tangent)  # tangent has length STEP_M = 1
        lat = float(tangent[0] * rel[1] - tangent[1] * rel[0])
        return i + along, lat, i

    def observe(self) -> WorldState:
        s, lat, i = self._project()
        last = len(self.route) - 1
        idx = np.clip(np.array([s + d for d in REF_STATIONS_M]).astype(int), 0, last)
        gap, lead_speed = lead_on_route(self.route, s, self.actor_arrays[self.step_i], LENGTH_M)
        return WorldState(
            self.step_i * DT_S,
            self.x,
            self.y,
            self.yaw,
            self.speed,
            self.route[idx],
            lat,
            float(wrap_angle(self.yaw - self.route_yaw[i])),
            self.sc.speed_limit_mps,
            gap,
            lead_speed,
        )

    def collided(self) -> bool:
        ego = Box(self.x, self.y, self.yaw, LENGTH_M, WIDTH_M)
        for a in self.actor_arrays[self.step_i]:
            if np.hypot(a[0] - self.x, a[1] - self.y) < 7.0 and boxes_overlap(
                ego, Box(float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]))
            ):
                return True
        return False

    def done(self, st: WorldState) -> bool:
        return (
            self.station_i >= self.sc.length_m + 1
            or abs(st.lateral_m) > self.sc.half_width_m + 3.0
            or self.collided()
            or self.step_i >= self.max_steps - 1
        )

    def step(self, action: Action) -> WorldState:
        accel = float(np.clip(action.accel_mps2, *ACCEL_LIMITS))
        wanted = float(np.clip(action.steer_rad, -STEER_LIMIT, STEER_LIMIT))
        change = np.clip(wanted - self.steer, -STEER_RATE_LIMIT * DT_S, STEER_RATE_LIMIT * DT_S)
        self.steer += float(change)
        steer = self.steer + float(self.noise[self.step_i])
        self.x += self.speed * np.cos(self.yaw) * DT_S
        self.y += self.speed * np.sin(self.yaw) * DT_S
        self.yaw += self.speed / WHEELBASE_M * np.tan(steer) * DT_S
        self.speed = max(0.0, self.speed + accel * DT_S)
        self.step_i += 1
        return self.observe()

    def trace(self, states: list[WorldState]) -> Trace:
        n = len(states)
        t = np.array([st.t_s for st in states])
        xy = np.array([[st.x, st.y] for st in states])
        yaw = np.array([st.yaw for st in states])
        speed = np.array([st.speed for st in states])
        actors = self.actor_arrays[:n, :, :5]
        goal = self.sc.length_m
        return Trace(
            Trajectory(t, xy, yaw),
            speed,
            actors,
            self.route,
            self.sc.half_width_m,
            goal,
            self.budget_s,
            LENGTH_M,
            WIDTH_M,
        )
