"""What a policy sees and what every adapter must provide. Both worlds use this one interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DT_S = 0.1
REF_STATIONS_M = (5.0, 10.0, 20.0, 30.0, 40.0)


@dataclass(frozen=True, eq=False)
class WorldState:
    t_s: float
    x: float
    y: float
    yaw: float
    speed: float
    ref: np.ndarray  # (len(REF_STATIONS_M), 2) route centreline ahead of ego, world frame
    lateral_m: float  # signed offset from centreline, left +
    heading_error: float  # ego yaw minus route tangent yaw
    speed_limit: float
    gap_m: float | None  # bumper to bumper gap to the lead on the route, None when no lead
    lead_speed: float | None


@dataclass(frozen=True)
class Action:
    accel_mps2: float
    steer_rad: float


def lead_on_route(
    route: np.ndarray, station_m: float, actors: np.ndarray, ego_length_m: float
) -> tuple[float | None, float | None]:
    """Nearest actor ahead on the route, as (gap, speed). Actors: rows of x, y, ..., speed last.

    Route points must be 1 m apart, so an index is a station.
    """
    best_gap, best_speed = None, None
    for a in actors:
        d = route - a[:2]
        dist2 = (d * d).sum(1)
        j = int(np.argmin(dist2))
        gap = j - station_m - ego_length_m
        if dist2[j] < 1.9**2 and gap > -ego_length_m and (best_gap is None or gap < best_gap):
            best_gap, best_speed = gap, float(a[-1])
    return best_gap, best_speed
