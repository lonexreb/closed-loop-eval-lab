"""REAL SIMULATOR adapter: MetaDrive 0.4.3 (Apache-2.0), headless, with procedural public maps.

Same reset/step/done/trace interface as the fixture. A scenario is a block-sequence family plus a
map seed. Collision and off-road are recomputed from the trace by this repository; MetaDrive's own
flags are retained in `flags` as a cross-check.
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np

from eval_lab.adapters.base import DT_S, REF_STATIONS_M, Action, WorldState, lead_on_route
from eval_lab.metrics import Trace, station_lateral
from eval_lab.scenarios import Scenario
from eval_lab.schemas import Trajectory, wrap_angle

BACKEND = "metadrive-0.4.3"
LANE_WIDTH_M = 4.0
HALF_WIDTH_M = LANE_WIDTH_M / 2  # own lane edge. Left of it is the oncoming lane, not grass.
MAX_ACCEL, MAX_BRAKE = 3.0, 6.0  # scale from our m/s^2 command to MetaDrive's [-1, 1] pedal
RUNOUT_M = 50
ABSENT = 1.0e6  # traffic that has despawned is known to be absent, which is not unknown telemetry


class MetaDriveSim:
    backend = BACKEND

    def __init__(self, disturbance: float = 1.0, traffic_density: float = 0.1):
        self.disturbance = disturbance
        self.traffic_density = traffic_density
        self.env: Any = None
        self.env_family: str | None = None

    def _env_for(self, family: str):
        if self.env_family == family:
            return self.env
        self.close()
        from metadrive.envs.metadrive_env import MetaDriveEnv

        self.env = MetaDriveEnv(
            dict(
                use_render=False,
                image_observation=False,
                log_level=50,
                num_scenarios=1000,
                start_seed=0,
                horizon=None,
                traffic_density=self.traffic_density,
                accident_prob=0.0,
                random_traffic=False,
                random_lane_width=False,
                random_agent_model=False,
                physics_world_step_size=0.02,
                decision_repeat=5,
                # Our trace decides off-road, with the same definition as the fixture. Left on,
                # MetaDrive ends the run when a wheel crosses a line, before our centre test fires.
                out_of_road_done=False,
                on_continuous_line_done=False,
                map_config=dict(
                    type="block_sequence",
                    config=family.removeprefix("md_"),
                    lane_num=1,
                    lane_width=LANE_WIDTH_M,
                    exit_length=50,
                ),
            )
        )
        self.env_family = family
        return self.env

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
            self.env_family = None

    def _route(self) -> np.ndarray:
        """Centreline of the navigation route, resampled to 1 m."""
        nav = self.env.agent.navigation
        graph = self.env.current_map.road_network.graph
        pts = []
        for a, b in zip(nav.checkpoints[:-1], nav.checkpoints[1:]):
            lane = graph[a][b][0]
            pts += [lane.position(s, 0) for s in np.arange(0.0, lane.length, 0.5)]
        raw = np.array(pts, float)
        keep = np.concatenate([[True], np.linalg.norm(np.diff(raw, axis=0), axis=1) > 1e-6])
        raw = raw[keep]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))])
        grid = np.arange(0.0, s[-1], 1.0)
        return np.stack([np.interp(grid, s, raw[:, 0]), np.interp(grid, s, raw[:, 1])], axis=1)

    def reset(self, sc: Scenario, seed: int) -> WorldState:
        self.sc = sc
        env = self._env_for(sc.family)
        map_seed = int(sc.id.rsplit("-", 1)[1])
        env.reset(seed=map_seed)
        self.route = self._route()
        self.goal_s = float(len(self.route) - RUNOUT_M)  # plans need road beyond the goal
        self.budget_s = 2.5 * self.goal_s / sc.speed_limit_mps
        self.max_steps = int(self.budget_s / DT_S) + 1
        rng = np.random.default_rng([zlib.crc32(sc.id.encode()), seed])
        white = rng.normal(0.0, 1.0, self.max_steps + 1)
        noise = np.zeros(self.max_steps + 1)
        for i in range(1, self.max_steps + 1):
            noise[i] = 0.9 * noise[i - 1] + np.sqrt(1 - 0.9**2) * white[i]
        self.noise = 0.02 * self.disturbance * noise
        self.step_i = 0
        self.ended = False
        self.flags = {"crash": False, "out_of_road": False, "arrive_dest": False}
        self.actor_log: list[np.ndarray] = []
        self.ego_size = (float(env.agent.LENGTH), float(env.agent.WIDTH))
        self.max_steer = float(np.deg2rad(env.agent.max_steering))
        return self.observe()

    def _actors(self) -> np.ndarray:
        """Rows of x, y, yaw, length, width, speed for traffic, excluding ego."""
        ego = self.env.agent
        rows = [
            [*v.position, v.heading_theta, v.LENGTH, v.WIDTH, v.speed]
            for v in self.env.engine.traffic_manager.vehicles
            if v is not ego
        ]
        return np.array(rows, float).reshape(-1, 6)

    def observe(self) -> WorldState:
        v = self.env.agent
        xy = np.array(v.position, float)
        s_arr, lat_arr = station_lateral(self.route, xy[None])
        s, lat = float(s_arr[0]), float(lat_arr[0])
        i = min(int(s), len(self.route) - 2)
        tangent = self.route[i + 1] - self.route[i]
        idx = np.clip(np.array([s + d for d in REF_STATIONS_M]).astype(int), 0, len(self.route) - 1)
        actors = self._actors()
        self.actor_log.append(actors)
        gap, lead_speed = lead_on_route(self.route, s, actors, self.ego_size[0])
        self.station = s
        return WorldState(
            self.step_i * DT_S,
            float(xy[0]),
            float(xy[1]),
            float(v.heading_theta),
            float(v.speed),
            self.route[idx],
            lat,
            float(wrap_angle(v.heading_theta - np.arctan2(tangent[1], tangent[0]))),
            self.sc.speed_limit_mps,
            gap,
            lead_speed,
        )

    def done(self, st: WorldState) -> bool:
        return (
            self.ended
            or self.station >= self.goal_s
            or abs(st.lateral_m) > HALF_WIDTH_M + 3.0
            or self.step_i >= self.max_steps - 1
        )

    def step(self, action: Action) -> WorldState:
        steer = action.steer_rad + float(self.noise[self.step_i])
        pedal = action.accel_mps2 / (MAX_ACCEL if action.accel_mps2 >= 0 else MAX_BRAKE)
        cmd = [float(np.clip(steer / self.max_steer, -1, 1)), float(np.clip(pedal, -1, 1))]
        _, _, terminated, truncated, info = self.env.step(cmd)
        self.step_i += 1
        self.flags["crash"] |= bool(info.get("crash_vehicle") or info.get("crash_object"))
        self.flags["out_of_road"] |= bool(info.get("out_of_road"))
        self.flags["arrive_dest"] |= bool(info.get("arrive_dest"))
        self.ended = bool(terminated or truncated)
        return self.observe()

    def trace(self, states: list[WorldState]) -> Trace:
        t = np.array([st.t_s for st in states])
        xy = np.array([[st.x, st.y] for st in states])
        yaw = np.array([st.yaw for st in states])
        speed = np.array([st.speed for st in states])
        k = max((len(a) for a in self.actor_log[: len(states)]), default=0)
        actors = np.full((len(states), k, 5), ABSENT)
        for i, a in enumerate(self.actor_log[: len(states)]):
            actors[i, : len(a)] = a[:, :5]
        return Trace(
            Trajectory(t, xy, yaw),
            speed,
            actors,
            self.route,
            HALF_WIDTH_M,
            self.goal_s,
            self.budget_s,
            *self.ego_size,
        )
